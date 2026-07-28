"""Evaluate a trained model: recall@{1,2,4,8,16,32,64} against each test
year's full entity pool.

Usage:
    python -m cycle.evaluate --run-dir outputs/new_2019_default_seed52313 \
        --test-years 2019,2020,2021,2022
"""

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from cycle.config import Config
from cycle.data.documents import load_candidate_pool
from cycle.data.graphs import _read_qid_mapping, load_year_graph
from cycle.data.mentions import load_mention_dataset
from cycle.models.biencoder import BiEncoder
from cycle.models.fusion import ScoreFusion
from cycle.models.graph_paper import PaperGraphModule
from cycle.utils import (configure_tf32, get_logger, oom_safe_map,
                         resolve_device, set_all_seeds)

RANKS = (1, 2, 4, 8, 16, 32, 64)


def load_candidate_enhancer(cfg, ckpt, device):
    """Rebuild the trained graph module + fusion for candidate-mode
    checkpoints and return the projected per-node enhancement table
    (train-year graph, train-split id space); None for other modes."""
    if cfg.fusion.get("mode") != "candidate" or ckpt.get("graph_module") is None:
        return None, None
    graph_path = os.path.join(
        cfg.data.root, "processed", "graphs", f"{cfg.data.train_year}.npz"
    )
    graph = load_year_graph(graph_path, device)
    g = cfg.graph
    module = PaperGraphModule(
        feat_dim=graph["X"].shape[1], hidden_dim=g.hidden_dim,
        feat_drop=g.feat_drop, attn_drop=g.attn_drop, tau=g.tau,
        neighbor_threshold=g.neighbor_threshold,
        isolated_fallback=g.isolated_fallback,
        feature_negatives=g.feature_negatives,
        relation_extra_negatives=g.get("relation_extra_negatives", 0),
    ).to(device)
    module.load_state_dict(ckpt["graph_module"])
    module.eval()
    fusion = ScoreFusion(
        graph_dim=g.hidden_dim, bert_dim=512, weight=cfg.fusion.weight,
        projection=cfg.fusion.projection, alignment=cfg.fusion.alignment,
        mode="candidate",
    ).to(device)
    # bert_dim from the checkpoint, not hardcoded
    fusion.linear = torch.nn.Linear(
        ckpt["fusion"]["linear.weight"].shape[1],
        ckpt["fusion"]["linear.weight"].shape[0],
    ).to(device)
    fusion.load_state_dict(ckpt["fusion"])
    fusion.eval()
    gen = torch.Generator().manual_seed(cfg.run.seed)
    with torch.no_grad():
        module.resample_neighbors(graph, device, generator=gen)
        z_r, z_f = module(graph)
        src = cfg.fusion.get("source", "mean")
        z = z_r if src == "rel" else z_f if src == "feat" else (z_r + z_f) / 2
    train_map = _read_qid_mapping(os.path.join(
        cfg.data.root, "raw", "relationship", str(cfg.data.train_year),
        f"{cfg.data.train_year}_train_qid_id_mapping.txt",
    ))
    return (z, fusion), train_map


def enhance_pool(pool_encs, enhancer, train_map, cfg, year, split, device):
    """pool_encs + P z for pool entities present in the train-year graph."""
    (z, fusion) = enhancer
    test_map = _read_qid_mapping(os.path.join(
        cfg.data.root, "raw", "relationship", str(year),
        f"{year}_{split}_qid_id_mapping.txt",
    ))
    idx = torch.zeros(len(test_map), dtype=torch.long)
    present = torch.zeros(len(test_map), dtype=torch.float32)
    for qid, pool_i in test_map.items():
        node = train_map.get(qid, -1)
        if node >= 0:
            idx[pool_i] = node
            present[pool_i] = 1.0
    with torch.no_grad():
        return fusion.enhance_candidates(
            pool_encs, z[idx.to(device)], present.to(device)
        ), int(present.sum().item())


@torch.no_grad()
def encode_pool(biencoder, pool_ids, device, batch_size, amp_dtype):
    def enc(chunk):
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            return biencoder.encode_candidate(chunk.to(device)).float()
    return oom_safe_map(enc, pool_ids, batch_size)


@torch.no_grad()
def recall_at_k(biencoder, dataset, pool_encs, device, batch_size, top_k, amp_dtype):
    def enc(chunk):
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            return biencoder.encode_context(chunk.to(device)).float()

    ctx_encs = oom_safe_map(enc, dataset.tensors[0], batch_size)
    label_all = dataset.tensors[2].to(device)
    hits = torch.zeros(len(RANKS), dtype=torch.long)
    total = ctx_encs.size(0)
    for s in tqdm(range(0, total, batch_size), desc="score mentions"):
        scores = ctx_encs[s: s + batch_size] @ pool_encs.t()
        label_idx = label_all[s: s + batch_size]
        _, top = scores.topk(top_k, dim=1)
        rank = (top == label_idx.unsqueeze(1)).float().argmax(dim=1)
        found = (top == label_idx.unsqueeze(1)).any(dim=1)
        for ri, r in enumerate(RANKS):
            hits[ri] += ((rank < r) & found).sum().item()
    return {f"recall@{r}": hits[ri].item() / total for ri, r in enumerate(RANKS)}, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--test-years", required=True,
                        help="comma-separated, e.g. 2019,2020,2021,2022")
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--limit", type=int, default=None,
                        help="cap mentions per year (smoke tests)")
    args = parser.parse_args()

    with open(os.path.join(args.run_dir, "config_used.json"), encoding="utf-8") as f:
        cfg = Config.wrap(json.load(f))
    logger = get_logger(args.run_dir)
    configure_tf32(cfg.run.tf32)
    set_all_seeds(cfg.run.seed)
    device = resolve_device()
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.run.amp)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.bert_model, do_lower_case=cfg.model.lowercase
    )
    biencoder = BiEncoder(cfg.model.bert_model).to(device)
    ckpt = torch.load(os.path.join(args.run_dir, args.checkpoint),
                      map_location=device, weights_only=True)
    biencoder.load_state_dict(ckpt["biencoder"])
    biencoder.eval()
    enhancer, train_map = load_candidate_enhancer(cfg, ckpt, device)

    results_path = os.path.join(args.run_dir, "eval_results.jsonl")
    for year in args.test_years.split(","):
        year = year.strip()
        pool_ids, num_entities = load_candidate_pool(cfg, tokenizer, year, args.split)
        pool_encs = encode_pool(
            biencoder, pool_ids, device, cfg.eval.encode_batch_size, amp_dtype
        )
        if enhancer is not None:
            pool_encs, n_present = enhance_pool(
                pool_encs, enhancer, train_map, cfg, year, args.split, device
            )
            logger.info(f"candidate enhancement: {n_present}/{num_entities} "
                        f"pool entities mapped to train-year graph nodes")

        dataset = load_mention_dataset(
            cfg, tokenizer, cfg.data.entity_set, year, args.split, args.limit
        )
        res, total = recall_at_k(
            biencoder, dataset, pool_encs, device,
            cfg.eval.batch_size, cfg.eval.top_k, amp_dtype,
        )
        del pool_encs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        record = {
            "train_year": cfg.data.train_year,
            "entity_set": cfg.data.entity_set,
            "config": cfg.run.config_name,
            "seed": cfg.run.seed,
            "test_year": int(year),
            "split": args.split,
            "num_mentions": total,
            "num_entities": num_entities,
            **res,
        }
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(json.dumps(record))


if __name__ == "__main__":
    main()
