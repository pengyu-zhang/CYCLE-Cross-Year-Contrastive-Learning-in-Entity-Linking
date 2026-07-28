"""Train a CYCLE model on one (entity_set, train_year).

Usage:
    python -m cycle.train --config configs/default.yaml \
        --set data.entity_set=new --set data.train_year=2019
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from cycle.config import load_config
from cycle.data.documents import load_candidate_pool
from cycle.data.graphs import load_samples2015, load_year_graph
from cycle.data.mentions import load_mention_dataset
from cycle.models.biencoder import BiEncoder
from cycle.models.fusion import ScoreFusion
from cycle.models.graph_paper import PaperGraphModule
from cycle.models.heco import HeCo, pretrain_heco
from cycle.optim import build_optimizer, build_scheduler
from cycle.utils import (MetricsWriter, atomic_torch_save, configure_tf32,
                         get_logger, oom_safe_map, resolve_device,
                         set_all_seeds)


def make_run_dir(cfg):
    name = (
        f"{cfg.data.entity_set}_{cfg.data.train_year}"
        f"_{cfg.run.config_name}_seed{cfg.run.seed}"
    )
    run_dir = os.path.join(cfg.run.output_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def build_graph_state(cfg, device, logger):
    """Returns (node_embeds_or_None, graph_module_or_None, graph_data_or_None)."""
    g = cfg.graph
    if not g.enabled:
        return None, None, None

    if g.source == "samples2015":
        if g.mode != "heco_pretrain":
            raise ValueError("samples2015 source implies heco_pretrain mode")
        # The pretraining inputs are identical for every run of a given graph
        # config (fixed 2015 arrays, fixed seed), so the resulting embeddings
        # are cached across runs, keyed by the graph config.
        import hashlib
        key_src = json.dumps({k: g[k] for k in sorted(g) if k != "enabled"},
                             sort_keys=True)
        key = hashlib.md5(key_src.encode()).hexdigest()[:12]
        cache = os.path.join(cfg.data.root, "processed", "cache", f"heco_{key}.pt")
        if os.path.exists(cache):
            logger.info(f"Loading cached HeCo embeddings ({cache})")
            return torch.load(cache, weights_only=True).to(device), None, None

        samples_dir = os.path.join(cfg.data.root, "raw", "samples2015")
        inputs = load_samples2015(samples_dir, device)
        set_all_seeds(g.pretrain.seed)
        heco = HeCo(
            hidden_dim=g.hidden_dim,
            feats_dim_list=[f.shape[1] for f in inputs["feats"]],
            feat_drop=g.feat_drop,
            attn_drop=g.attn_drop,
            num_paths=len(inputs["mps"]),
            sample_rate=list(g.sample_rate),
            nei_num=len(inputs["nei"]),
            tau=g.tau,
            lam=g.lam,
        ).to(device)
        logger.info("Pretraining HeCo graph module (two-stage mode)")
        node_embeds = pretrain_heco(
            heco, inputs,
            lr=g.pretrain.lr, weight_decay=g.pretrain.weight_decay,
            patience=g.pretrain.patience, max_epochs=g.pretrain.max_epochs,
            logger=logger,
        )
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        atomic_torch_save(node_embeds.cpu(), cache)
        return node_embeds, None, None

    # per_year / joint
    graph_path = os.path.join(
        cfg.data.root, "processed", "graphs", f"{cfg.data.train_year}.npz"
    )
    graph = load_year_graph(graph_path, device)
    module = PaperGraphModule(
        feat_dim=graph["X"].shape[1],
        hidden_dim=g.hidden_dim,
        feat_drop=g.feat_drop,
        attn_drop=g.attn_drop,
        tau=g.tau,
        neighbor_threshold=g.neighbor_threshold,
        isolated_fallback=g.isolated_fallback,
        feature_negatives=g.feature_negatives,
        relation_extra_negatives=g.get("relation_extra_negatives", 0),
    ).to(device)
    logger.info(
        f"Per-year graph {cfg.data.train_year}: {graph['num_nodes']} nodes, "
        f"{graph['pos_src'].numel()} cross-year positives, "
        f"{graph['neg_src'].numel()} negatives (vs {graph['prev_year']})"
    )
    return None, module, graph


def graph_fusion_embeds(cfg, z_r, z_f):
    src = cfg.fusion.get("source", "mean")
    if src == "rel":
        return z_r
    if src == "feat":
        return z_f
    return (z_r + z_f) / 2


def in_batch_scores(cfg, biencoder, fusion, batch, node_embeds, step, training):
    """Returns (in-batch score matrix, context embeddings)."""
    ctx_ids, cand_ids, label_idx = batch
    ctx_emb = biencoder.encode_context(ctx_ids)
    cand_emb = biencoder.encode_candidate(cand_ids)
    if (fusion is not None and node_embeds is not None
            and fusion.mode == "candidate" and training):
        # graph-enhanced candidate representations (train/eval consistent;
        # validation batches index a different id space, so train-only here)
        cand_emb = fusion.enhance_candidates(cand_emb, node_embeds[label_idx])
        return ctx_emb @ cand_emb.t(), ctx_emb
    scores = ctx_emb @ cand_emb.t()
    if fusion is not None and node_embeds is not None and fusion.mode == "gold_mention":
        # In qid alignment, validation batches index a different id space than
        # the train graph, so fusion is train-only there; sequential alignment
        # applies fusion during validation too.
        if training or fusion.alignment == "sequential":
            z = fusion.select_batch_embeddings(
                node_embeds, label_idx, step, cfg.train.eval_batch_size
                if not training else cfg.train.batch_size,
            )
            scores = fusion(scores, cand_emb, z)
    return scores, ctx_emb


@torch.no_grad()
def encode_train_pool(cfg, biencoder, tokenizer, fusion, node_embeds, device):
    """Encode the training year's full entity pool with the current model
    (used as cached full-pool negatives, refreshed every epoch). In candidate
    fusion mode the cached encodings include the graph enhancement, matching
    how the pool is scored at inference. Train-split pool line ids equal the
    graph node ids, so the enhancement is a direct table lookup."""
    biencoder.eval()
    pool_ids, _ = load_candidate_pool(
        cfg, tokenizer, cfg.data.train_year, "train"
    )
    pool_encs = oom_safe_map(
        lambda ids: biencoder.encode_candidate(ids.to(device)).float(),
        pool_ids, cfg.eval.encode_batch_size,
    )
    if (fusion is not None and node_embeds is not None
            and fusion.mode == "candidate"):
        pool_encs = fusion.enhance_candidates(
            pool_encs, node_embeds[: pool_encs.size(0)].float()
        ).detach()
    biencoder.train()
    return pool_encs


@torch.no_grad()
def evaluate_in_batch(cfg, biencoder, fusion, node_embeds, valid_data, device):
    """In-batch validation accuracy over groups of eval_batch_size.

    Embeddings are encoded in large chunks first (no dropout in eval mode, so
    this is numerically identical to per-group encoding) and only the B×B
    scoring — including the fusion term with its per-step slicing
    semantics — happens per group.
    """
    biencoder.eval()
    encode_bs = cfg.eval.encode_batch_size
    ctx_all = oom_safe_map(
        lambda ids: biencoder.encode_context(ids.to(device)),
        valid_data.tensors[0], encode_bs,
    )
    cand_all = oom_safe_map(
        lambda ids: biencoder.encode_candidate(ids.to(device)),
        valid_data.tensors[1], encode_bs,
    )
    label_idx_all = valid_data.tensors[2].to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    bs = cfg.train.eval_batch_size
    correct, total = 0, 0
    for step, s in enumerate(range(0, ctx_all.size(0), bs)):
        ctx_emb, cand_emb = ctx_all[s: s + bs], cand_all[s: s + bs]
        scores = ctx_emb @ cand_emb.t()
        if fusion is not None and node_embeds is not None and fusion.alignment == "sequential":
            z = fusion.select_batch_embeddings(
                node_embeds, label_idx_all[s: s + bs], step, bs
            )
            scores = fusion(scores, cand_emb, z)
        preds = scores.argmax(dim=1)
        target = torch.arange(scores.size(0), device=device)
        correct += (preds == target).sum().item()
        total += scores.size(0)
    biencoder.train()
    return correct / max(1, total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides",
                        help="Config override, e.g. data.train_year=2020")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    cfg.run["config_name"] = os.path.splitext(os.path.basename(args.config))[0]
    run_dir = make_run_dir(cfg)
    logger = get_logger(run_dir)
    metrics = MetricsWriter(os.path.join(run_dir, "metrics.jsonl"))

    with open(os.path.join(run_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    configure_tf32(cfg.run.tf32)
    set_all_seeds(cfg.run.seed)
    device = resolve_device()
    logger.info(f"run dir: {run_dir}")

    # ---- data ----
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.bert_model, do_lower_case=cfg.model.lowercase
    )
    limit = cfg.data.debug_n
    valid_limit = cfg.data.get("valid_n") or limit
    train_data = load_mention_dataset(
        cfg, tokenizer, cfg.data.entity_set, cfg.data.train_year, "train", limit
    )
    valid_data = load_mention_dataset(
        cfg, tokenizer, cfg.data.entity_set, cfg.data.train_year, "validation",
        valid_limit,
    )
    logger.info(f"train samples: {len(train_data)}, valid samples: {len(valid_data)}")

    sampler = RandomSampler(train_data) if cfg.train.shuffle else SequentialSampler(train_data)
    train_loader = DataLoader(train_data, sampler=sampler, batch_size=cfg.train.batch_size)

    # ---- models ----
    biencoder = BiEncoder(cfg.model.bert_model).to(device)
    node_embeds, graph_module, graph = build_graph_state(cfg, device, logger)

    fusion = None
    if cfg.fusion.enabled and cfg.graph.enabled:
        fusion = ScoreFusion(
            graph_dim=cfg.graph.hidden_dim,
            bert_dim=biencoder.output_dim,
            weight=cfg.fusion.weight,
            projection=cfg.fusion.projection,
            alignment=cfg.fusion.alignment,
            mode=cfg.fusion.get("mode", "gold_mention"),
            init_seed=cfg.graph.pretrain.seed if "pretrain" in cfg.graph else 0,
        ).to(device)

    # reseed after graph setup so bi-encoder batches start from the run seed
    set_all_seeds(cfg.run.seed)

    extra_modules = [m for m in (graph_module, fusion) if m is not None]
    optimizer = build_optimizer(
        biencoder, cfg.train.type_optimization, cfg.train.learning_rate,
        extra_modules=extra_modules, extra_lr=cfg.train.graph_learning_rate,
    )
    steps_per_epoch = max(1, len(train_data) // cfg.train.batch_size)
    total_steps = steps_per_epoch * cfg.train.num_epochs
    scheduler = build_scheduler(optimizer, total_steps, cfg.train.warmup_proportion)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.run.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.run.amp == "fp16")

    loss_weights = cfg.loss
    joint = graph_module is not None

    best_acc, best_epoch = -1.0, -1
    pool_cfg = cfg.train.get("pool_loss") or {}
    pool_enabled = bool(pool_cfg.get("enabled"))
    pool_weight = float(pool_cfg.get("weight", 1.0))

    global_step = 0
    biencoder.train()
    for epoch in range(cfg.train.num_epochs):
        if joint:
            graph_module.resample_neighbors(graph, device)
        pool_cached = None
        if pool_enabled:
            # full-pool negatives: cache the whole training-year entity pool
            # with the current model (refreshed every epoch)
            with torch.no_grad():
                epoch_embeds = node_embeds
                if joint:
                    ez_r, ez_f = graph_module(graph)
                    epoch_embeds = graph_fusion_embeds(cfg, ez_r, ez_f)
            pool_cached = encode_train_pool(
                cfg, biencoder, tokenizer, fusion, epoch_embeds, device
            )
        epoch_loss, window_loss = 0.0, 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}")):
            if cfg.train.reseed_each_batch:
                # baseline behavior: reset all global seeds on every batch,
                # freezing the dropout mask sequence
                seed = cfg.graph.pretrain.seed if "pretrain" in cfg.graph else 0
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
            batch = tuple(t.to(device) for t in batch)

            # graph module always runs fp32 outside autocast: it is tiny
            # (10k×64), sparse mm has no low-precision CUDA kernel, and the
            # contrastive exp-sums are numerically happier in fp32
            if joint:
                z_r, z_f = graph_module(graph)
                node_embeds = graph_fusion_embeds(cfg, z_r, z_f)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                scores, ctx_emb = in_batch_scores(
                    cfg, biencoder, fusion, batch, node_embeds, step, training=True
                )
                target = torch.arange(scores.size(0), device=device)
                loss_e = F.cross_entropy(scores, target)
            loss = loss_weights.a * loss_e
            extra = {}
            if pool_cached is not None:
                # retrieval objective over the whole entity pool (cached
                # candidate encodings; gradient flows through the mention
                # encoder only)
                pool_scores = ctx_emb.float() @ pool_cached.t()
                loss_pool = F.cross_entropy(pool_scores, batch[2])
                loss = loss + pool_weight * loss_pool
                extra["loss_pool"] = loss_pool.item()
            if joint:
                loss_f = graph_module.loss_f(z_r, z_f, graph)
                loss_r = graph_module.loss_r(z_r, z_f, graph)
                loss = loss + loss_weights.b * loss_f + loss_weights.c * loss_r
                extra = {**extra, "loss_f": loss_f.item(), "loss_r": loss_r.item()}

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for g_ in optimizer.param_groups for p in g_["params"]],
                cfg.train.max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            window_loss += loss.item()
            global_step += 1
            if (step + 1) % cfg.train.print_interval == 0:
                metrics.write({
                    "type": "train_step", "epoch": epoch, "step": global_step,
                    "loss": window_loss / cfg.train.print_interval,
                    "loss_e": loss_e.item(), **extra,
                    "lr": scheduler.get_last_lr()[0],
                })
                window_loss = 0.0
            if (step + 1) % cfg.train.eval_interval == 0:
                acc = evaluate_in_batch(
                    cfg, biencoder, fusion, node_embeds, valid_data, device
                )
                metrics.write({"type": "valid_inbatch", "epoch": epoch,
                               "step": global_step, "accuracy": acc})

        acc = evaluate_in_batch(cfg, biencoder, fusion, node_embeds, valid_data, device)
        peak_gb = (torch.cuda.max_memory_allocated() / 2**30
                   if torch.cuda.is_available() else 0.0)
        metrics.write({"type": "epoch_end", "epoch": epoch,
                       "mean_loss": epoch_loss / max(1, steps_per_epoch),
                       "valid_inbatch_accuracy": acc,
                       "gpu_peak_gb": round(peak_gb, 2)})
        logger.info(f"epoch {epoch}: valid in-batch accuracy {acc:.5f}")

        state = {
            "biencoder": biencoder.state_dict(),
            "graph_module": graph_module.state_dict() if joint else None,
            "fusion": fusion.state_dict() if fusion is not None else None,
            "epoch": epoch,
        }
        torch.save(state, os.path.join(run_dir, f"epoch_{epoch}.pt"))
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            torch.save(state, os.path.join(run_dir, "best.pt"))
        if cfg.train.early_stopping.enabled and epoch - best_epoch >= cfg.train.early_stopping.patience:
            logger.info(f"early stopping at epoch {epoch} (best {best_epoch})")
            break

    metrics.write({"type": "done", "best_epoch": best_epoch, "best_acc": best_acc})
    metrics.close()
    logger.info(f"best epoch {best_epoch} (valid in-batch acc {best_acc:.5f})")
    # keep only best + last epoch checkpoints to save disk
    for epoch in range(cfg.train.num_epochs):
        p = os.path.join(run_dir, f"epoch_{epoch}.pt")
        if epoch not in (best_epoch, cfg.train.num_epochs - 1) and os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    main()
