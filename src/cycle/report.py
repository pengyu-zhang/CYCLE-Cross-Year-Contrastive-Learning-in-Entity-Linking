"""Aggregate eval_results.jsonl files across runs into gap tables.

Table layout mirrors the paper's Table 2: rows are recall@k, columns are the
|train_year - test_year| gap (0-3), one block per entity set. Cells average
over all (train, test) pairs with that gap; for multi-seed configs the mean
of per-seed aggregates is reported with its std.

Usage:
    python -m cycle.report --outputs outputs [--config default] [--markdown]
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

RANKS = (1, 2, 4, 8, 16, 32, 64)


def load_records(outputs_dir):
    records = []
    for path in glob.glob(os.path.join(outputs_dir, "*", "eval_results.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
    return records


def gap_table(records, entity_set, config, split="test"):
    """Returns {rank: {gap: (mean, std_or_None)}}.

    Per seed: average recall over all (train, test) pairs with the given gap.
    Across seeds: mean and std of those per-seed aggregates.
    """
    by_seed = defaultdict(lambda: defaultdict(list))  # seed -> gap -> [values per rank]
    for r in records:
        if (r["entity_set"], r["config"], r["split"]) != (entity_set, config, split):
            continue
        gap = abs(r["train_year"] - r["test_year"])
        by_seed[r["seed"]][gap].append([r[f"recall@{k}"] for k in RANKS])

    table = {k: {} for k in RANKS}
    gaps = sorted({g for s in by_seed.values() for g in s})
    for gi, gap in enumerate(gaps):
        per_seed = []
        for seed, gapmap in sorted(by_seed.items()):
            if gap in gapmap:
                per_seed.append(np.mean(gapmap[gap], axis=0))
        if not per_seed:
            continue
        arr = np.stack(per_seed)
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        for ki, k in enumerate(RANKS):
            table[k][gap] = (mean[ki], std[ki] if len(per_seed) > 1 else None)
    return table, len(by_seed)


def fmt_cell(mean, std, markdown):
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f}±{std:.3f}" if markdown else f"{mean:.3f}+-{std:.3f}"


def print_table(table, n_seeds, title, markdown=False):
    gaps = sorted({g for row in table.values() for g in row})
    print(f"\n### {title} ({n_seeds} seed{'s' if n_seeds != 1 else ''})"
          if markdown else f"\n== {title} ({n_seeds} seeds) ==")
    header = ["recall"] + [f"gap {g}" for g in gaps]
    if markdown:
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
    else:
        print("  ".join(f"{h:>14}" for h in header))
    for k, row in table.items():
        cells = [fmt_cell(*row[g], markdown) if g in row else "-" for g in gaps]
        if markdown:
            print(f"| @{k} | " + " | ".join(cells) + " |")
        else:
            print("  ".join(f"{c:>14}" for c in [f"@{k}"] + cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--config", default=None,
                        help="limit to one config (baseline/paper/default)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    records = load_records(args.outputs)
    if not records:
        print("no eval_results.jsonl found under", args.outputs)
        return
    configs = sorted({r["config"] for r in records})
    if args.config:
        configs = [c for c in configs if c == args.config]
    for entity_set in ("new", "continual"):
        for config in configs:
            table, n_seeds = gap_table(records, entity_set, config, args.split)
            if n_seeds:
                print_table(table, n_seeds, f"{entity_set} entities — {config}",
                            markdown=args.markdown)


if __name__ == "__main__":
    main()
