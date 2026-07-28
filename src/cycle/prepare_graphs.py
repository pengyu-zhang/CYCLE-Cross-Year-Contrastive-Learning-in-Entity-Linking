"""Build processed per-year graph files for paper/default configs.

For each year Y (2014-2022; 2013 has no previous year for cross-year
samples) this produces data/processed/graphs/Y.npz containing the year's
relation adjacency, kNN adjacency, token feature matrix and the cross-year
positive/negative sample lists (paper Eq. 10/11, computed in QID space
against year Y-1).

Usage:
    python -m cycle.prepare_graphs --data-root data [--years 2019,2020]
"""

import argparse
import os

from cycle.data.graphs import build_year_graph, save_year_graph

ALL_YEARS = list(range(2014, 2023))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--years", default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    years = ALL_YEARS if args.years == "all" else [
        int(y) for y in args.years.split(",")
    ]
    raw_root = os.path.join(args.data_root, "raw")
    out_dir = os.path.join(args.data_root, "processed", "graphs")

    for year in years:
        out_path = os.path.join(out_dir, f"{year}.npz")
        if os.path.exists(out_path) and not args.force:
            print(f"[prepare_graphs] {year}: exists, skipping")
            continue
        graph = build_year_graph(raw_root, year)
        n_pos = sum(len(p) for p in graph["pos_lists"])
        n_neg = sum(len(m) for m in graph["neg_lists"])
        save_year_graph(graph, out_path)
        print(
            f"[prepare_graphs] {year}: {graph['n']} nodes, "
            f"rel nnz {graph['rel'].nnz}, knn nnz {graph['knn'].nnz}, "
            f"X nnz {graph['X'].nnz}, cross-year +{n_pos}/-{n_neg} (vs {graph['prev_year']})"
        )


if __name__ == "__main__":
    main()
