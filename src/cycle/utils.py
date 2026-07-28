"""Seeding, device selection, logging and incremental metrics."""

import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(no_cuda=False):
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        print(f"[cycle] device: {device} ({name})", flush=True)
    else:
        print(f"[cycle] device: {device}", flush=True)
    return device


def configure_tf32(enabled):
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)


def get_logger(output_dir=None, name="cycle"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "log.txt"), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def _is_cuda_oom(exc):
    return "out of memory" in str(exc).lower()


def oom_safe_map(fn, ids, batch_size, min_batch=16):
    """Apply ``fn`` to ``ids`` in batches and concatenate the results,
    halving the batch size (down to ``min_batch``) on CUDA out-of-memory —
    e.g. when another process is sharing the GPU."""
    import torch
    out = []
    bs = batch_size
    s = 0
    while s < ids.size(0):
        try:
            out.append(fn(ids[s: s + bs]))
            s += bs
        except Exception as e:  # torch.cuda.OutOfMemoryError or AcceleratorError
            if not _is_cuda_oom(e) or bs <= min_batch:
                raise
            bs = max(min_batch, bs // 2)
            torch.cuda.empty_cache()
            print(f"[cycle] CUDA OOM during batched inference; retrying with batch {bs}",
                  flush=True)
    return torch.cat(out)


def atomic_torch_save(obj, path):
    """torch.save via a temp file + rename, so concurrent builders of the
    same cache file can never leave a truncated artifact."""
    import torch
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


class MetricsWriter:
    """Append-only JSONL metrics writer; every record is flushed to disk so an
    interrupted run loses nothing."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f = open(path, "a", encoding="utf-8")
        self._t0 = time.time()

    def write(self, record):
        record = dict(record)
        record.setdefault("elapsed_s", round(time.time() - self._t0, 2))
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self):
        self._f.close()
