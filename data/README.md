# Data

All dataset files live under this directory but are **not** tracked by git
(`data/raw/`, `data/processed/` are gitignored). One command prepares
everything:

```bash
bash scripts/prepare_data.sh
```

The script downloads GCL-TempEL (915 MB) from the official Zenodo record,
falls back to this repository's GitHub Release mirror if Zenodo is
unavailable, verifies the MD5 checksum, extracts the nested 7z files and
arranges them as:

```
data/
├── raw/
│   ├── documents/            # {year}_{split}.json — entity pool, 10,373 entities/year
│   ├── mentions/
│   │   ├── new/{year}/        # train/validation/test.jsonl (new entities)
│   │   └── continual/{year}/  # train/validation/test.jsonl (continual entities)
│   ├── relationship/{year}/   # yearly Wikidata5M relation edges + QID↔id mappings
│   ├── feature/{year}/        # token feature matrices (10,373 × 2,914)
│   ├── knn/{year}/            # kNN (k=3) description-similarity graphs
│   └── samples2015/           # preprocessed arrays consumed by configs/baseline.yaml
└── processed/
    └── graphs/{year}.npz      # per-year graphs + cross-year positive/negative samples
                               # built by python -m cycle.prepare_graphs
```

## Sources, attribution and licensing

| Resource | Where | License / citation |
| --- | --- | --- |
| GCL-TempEL | [Zenodo 12794944](https://doi.org/10.5281/zenodo.12794944) (official) / GitHub Release mirror | CC BY 4.0; cite the CYCLE paper |
| Graph-TempEL (same data, unbundled) | [Zenodo 10977757](https://doi.org/10.5281/zenodo.10977757) | CC BY 4.0 |
| TempEL (upstream benchmark) | [Zaporojets et al., NeurIPS 2022](https://doi.org/10.48550/arXiv.2302.02500) | CYCLE's text data derives from it |
| Wikidata5M (upstream KG) | [deepgraphlearning.github.io/project/wikidata5m](https://deepgraphlearning.github.io/project/wikidata5m) | relation edges derive from it |

## Datasets from the paper that are not wired into this pipeline

The paper additionally reports results on two static EL benchmarks. They do
not exercise CYCLE's cross-year mechanism (no temporal graphs; the model
reduces to the BLINK bi-encoder there), so this repository does not include
training pipelines for them. To obtain the data:

* **ZESHEL** — run
  [`get_zeshel_data.sh`](https://github.com/facebookresearch/BLINK/tree/main/examples/zeshel)
  from the BLINK repository.
* **WikiLinksNED (Unseen-Mentions)** — download from the
  [ET4EL repository](https://github.com/yasumasaonoe/ET4EL).
