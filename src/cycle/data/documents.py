"""Entity-side (candidate pool) loading.

Each year+split has a documents file ``{year}_{split}.json`` (JSONL) with one
entity per line: ``{"title": str, "text": [tokens...]}``. Line index == entity
node id == ``label_id`` in the mention files.
"""

import json
import os

import torch

from cycle.data.mentions import build_candidate_ids
from cycle.utils import atomic_torch_save


def read_documents(path, limit=None):
    docs = []
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
            if limit is not None and len(docs) >= limit:
                break
    return docs


def build_candidate_pool(
    docs, tokenizer, max_cand_length, include_title, max_desc_tokens=128
):
    """Token-id tensor (num_entities, max_cand_length) for the whole pool.

    ``include_title=False`` (configs/baseline.yaml) encodes only the first
    128 description tokens without the entity title (unlike training, where
    candidates are title + [unused2] + description).
    """
    rows = []
    for doc in docs:
        desc = doc["text"][:max_desc_tokens]
        title = doc["title"] if include_title else None
        rows.append(build_candidate_ids(desc, tokenizer, max_cand_length, title=title))
    return torch.tensor(rows, dtype=torch.long)


def load_candidate_pool(cfg, tokenizer, year, split):
    """Tokenized entity pool with an on-disk cache (see load_mention_dataset).

    Returns (pool_ids, num_entities).
    """
    model_slug = cfg.model.bert_model.split("/")[-1]
    title_tag = "title" if cfg.data.candidate_include_title else "notitle"
    cache_dir = os.path.join(cfg.data.root, "processed", "cache")
    cache = os.path.join(
        cache_dir,
        f"pool_{year}_{split}_cand{cfg.data.max_cand_length}"
        f"_{title_tag}_{model_slug}.pt",
    )
    if os.path.exists(cache):
        pool_ids = torch.load(cache, weights_only=True)
    else:
        docs = read_documents(
            os.path.join(cfg.data.root, "raw", "documents", f"{year}_{split}.json")
        )
        pool_ids = build_candidate_pool(
            docs, tokenizer, cfg.data.max_cand_length,
            include_title=cfg.data.candidate_include_title,
        )
        os.makedirs(cache_dir, exist_ok=True)
        atomic_torch_save(pool_ids, cache)
    return pool_ids, pool_ids.size(0)
