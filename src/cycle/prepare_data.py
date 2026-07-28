"""Download and lay out the GCL-TempEL dataset.

Downloads the official Zenodo bundle (with a GitHub-Release mirror as
fallback), verifies its MD5, extracts the nested 7z files and arranges
everything under data/raw/:

    documents/{year}_{split}.json      entity pools (10,373 entities/year)
    mentions/{new,continual}/{year}/{train,validation,test}.jsonl
    relationship/{year}/*.txt          yearly relation graphs + QID mappings
    feature/{year}/*.txt               token feature matrices
    knn/{year}/*.txt                   kNN (k=3) description graphs
    samples2015/*                      preprocessed arrays used by baseline.yaml

Usage:
    python -m cycle.prepare_data --data-root data [--bundle path/to/GCL-TempEL.7z]
"""

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request

SOURCES = [
    # (name, url) — tried in order
    ("zenodo", "https://zenodo.org/records/12794944/files/GCL-TempEL.7z?download=1"),
    ("github-mirror",
     "https://github.com/pengyu-zhang/CYCLE-Cross-Year-Contrastive-Learning-in-Entity-Linking/"
     "releases/download/v1.0/GCL-TempEL.7z"),
]
BUNDLE_MD5 = "a3e59f2871fb853d7bd1864d69b18e48"
BUNDLE_NAME = "GCL-TempEL.7z"


def md5sum(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(dest):
    for name, url in SOURCES:
        try:
            print(f"[prepare_data] downloading from {name}: {url}")
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as e:  # noqa: BLE001 — fall through to the mirror
            print(f"[prepare_data] {name} failed ({e}); trying next source")
    raise RuntimeError("all download sources failed")


def extract_7z(path, dest):
    import py7zr
    with py7zr.SevenZipFile(path, mode="r") as z:
        z.extractall(path=dest)


def layout(extract_root, raw_root):
    """Rearrange the extracted GCL-TempEL folder into data/raw/."""
    src = os.path.join(extract_root, "GCL-TempEL")

    inner = {
        "01_documents_no_duplicates.7z": "01",
        "02_continual_entities_1764.7z": "02c",
        "02_new_entities_1764.7z": "02n",
        "03_relationship.7z": "03",
        "04_feature.7z": "04",
        "05_knn_relation.7z": "05",
    }
    work = os.path.join(extract_root, "_inner")
    for fname in inner:
        print(f"[prepare_data] extracting {fname}")
        extract_7z(os.path.join(src, fname), work)

    def move(a, b):
        os.makedirs(os.path.dirname(b), exist_ok=True)
        if os.path.exists(b):
            shutil.rmtree(b) if os.path.isdir(b) else os.remove(b)
        shutil.move(a, b)

    move(os.path.join(work, "01_documents_no_duplicates"),
         os.path.join(raw_root, "documents"))
    move(os.path.join(work, "new_entities_1764"),
         os.path.join(raw_root, "mentions", "new"))
    move(os.path.join(work, "continual_entities_1764"),
         os.path.join(raw_root, "mentions", "continual"))
    move(os.path.join(work, "03_relationship"),
         os.path.join(raw_root, "relationship"))
    move(os.path.join(work, "04_feature", "01_keyword"),
         os.path.join(raw_root, "feature"))
    move(os.path.join(work, "05_knn_relation"),
         os.path.join(raw_root, "knn"))
    move(os.path.join(src, "06_samples"),
         os.path.join(raw_root, "samples2015"))
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(src, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--bundle", default=None,
                        help="use an already-downloaded GCL-TempEL.7z")
    parser.add_argument("--keep-bundle", action="store_true")
    args = parser.parse_args()

    raw_root = os.path.join(args.data_root, "raw")
    if os.path.isdir(os.path.join(raw_root, "documents")):
        print("[prepare_data] data/raw already prepared; nothing to do")
        return

    os.makedirs(raw_root, exist_ok=True)
    bundle = args.bundle or os.path.join(args.data_root, BUNDLE_NAME)
    if not os.path.exists(bundle):
        download(bundle)
    actual = md5sum(bundle)
    if actual != BUNDLE_MD5:
        print(f"[prepare_data] MD5 mismatch: expected {BUNDLE_MD5}, got {actual}")
        sys.exit(1)
    print("[prepare_data] bundle MD5 verified")

    extract_root = os.path.join(args.data_root, "_extract")
    print("[prepare_data] extracting outer bundle (this takes a few minutes)")
    extract_7z(bundle, extract_root)
    layout(extract_root, raw_root)
    shutil.rmtree(extract_root, ignore_errors=True)
    if not args.keep_bundle and args.bundle is None:
        os.remove(bundle)
    print("[prepare_data] done — data/raw is ready")


if __name__ == "__main__":
    main()
