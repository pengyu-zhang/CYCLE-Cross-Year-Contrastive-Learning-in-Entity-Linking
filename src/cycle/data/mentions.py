"""Mention-side data loading.

GCL-TempEL mention files are BLINK-format JSONL where ``context_left``,
``context_right`` and ``label`` are already WordPiece token lists (the
BLINK "token_only" format); ``mention`` is a raw string.
``label_id`` is the line index of the target entity in that year+split's
entity documents file, which is also the node id in that year+split's graph.
"""

import json
import os

import torch
from torch.utils.data import TensorDataset

from cycle import ENT_END_TAG, ENT_START_TAG, ENT_TITLE_TAG
from cycle.utils import atomic_torch_save


def read_mentions(path, limit=None):
    samples = []
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
            if limit is not None and len(samples) >= limit:
                break
    return samples


def build_context_ids(sample, tokenizer, max_seq_length):
    """[CLS] ctxt_left [unused0] mention [unused1] ctxt_right [SEP] (Eq. 1)."""
    mention_tokens = []
    if sample["mention"]:
        mention_tokens = tokenizer.tokenize(sample["mention"])
        mention_tokens = [ENT_START_TAG] + mention_tokens + [ENT_END_TAG]

    context_left = sample["context_left"]
    context_right = sample["context_right"]

    left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
    right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    tokens = (
        ["[CLS]"]
        + context_left[-left_quota:]
        + mention_tokens
        + context_right[:right_quota]
        + ["[SEP]"]
    )
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids += [0] * (max_seq_length - len(ids))
    assert len(ids) == max_seq_length
    return ids


def build_candidate_ids(desc_tokens, tokenizer, max_seq_length, title=None):
    """[CLS] title [unused2] description [SEP] (Eq. 2); title optional."""
    cand_tokens = list(desc_tokens)
    if title is not None:
        cand_tokens = tokenizer.tokenize(title) + [ENT_TITLE_TAG] + cand_tokens
    cand_tokens = ["[CLS]"] + cand_tokens[: max_seq_length - 2] + ["[SEP]"]
    ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    ids += [0] * (max_seq_length - len(ids))
    assert len(ids) == max_seq_length
    return ids


def process_mentions(samples, tokenizer, max_context_length, max_cand_length):
    """Tokenized tensors for training/validation.

    Returns a TensorDataset of (context_ids, candidate_ids, label_idx) where
    label_idx is the entity node id used for graph alignment.
    """
    ctx, cand, label_idx = [], [], []
    for sample in samples:
        ctx.append(build_context_ids(sample, tokenizer, max_context_length))
        cand.append(
            build_candidate_ids(
                sample["label"], tokenizer, max_cand_length,
                title=sample.get("label_title"),
            )
        )
        label_idx.append(int(sample["label_id"]))
    return TensorDataset(
        torch.tensor(ctx, dtype=torch.long),
        torch.tensor(cand, dtype=torch.long),
        torch.tensor(label_idx, dtype=torch.long),
    )


def load_mention_dataset(cfg, tokenizer, entity_set, year, split, limit=None):
    """process_mentions with an on-disk tensor cache.

    Tokenizing 48k test mentions costs ~30 s and every test year is scored by
    dozens of runs across the grid, so the full tokenized tensors are cached
    once per (entity_set, year, split, lengths, model) and sliced afterwards.
    """
    path = os.path.join(
        cfg.data.root, "raw", "mentions", entity_set, str(year), f"{split}.jsonl"
    )
    model_slug = cfg.model.bert_model.split("/")[-1]
    cache_dir = os.path.join(cfg.data.root, "processed", "cache")
    cache = os.path.join(
        cache_dir,
        f"mentions_{entity_set}_{year}_{split}"
        f"_ctx{cfg.data.max_context_length}_cand{cfg.data.max_cand_length}"
        f"_{model_slug}.pt",
    )
    if os.path.exists(cache):
        tensors = torch.load(cache, weights_only=True)
    else:
        samples = read_mentions(path)
        dataset = process_mentions(
            samples, tokenizer,
            cfg.data.max_context_length, cfg.data.max_cand_length,
        )
        tensors = dataset.tensors
        os.makedirs(cache_dir, exist_ok=True)
        atomic_torch_save(tensors, cache)
    if limit is not None:
        tensors = tuple(t[:limit] for t in tensors)
    return TensorDataset(*tensors)
