"""Graph-side data loading.

Two sources are supported:

* ``samples2015`` — the preprocessed arrays published on Zenodo
  (``06_samples``). They encode the 2015 train-split graphs regardless of
  the experiment year. Used by configs/baseline.yaml.

* ``per_year`` — graphs built from the raw per-year files (03_relationship,
  04_feature, 05_knn_relation) in the train-split id space of the requested
  year, including cross-year positive/negative samples (paper Eq. 10/11)
  computed in QID space against the previous year. Built once by
  ``python -m cycle.prepare_graphs`` and cached under data/processed/graphs.
"""

import os

import numpy as np
import scipy.sparse as sp
import torch


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def normalize_adj(adj):
    """Symmetric degree normalization D^-1/2 A D^-1/2 (epsilon-stabilized)."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum + 1e-10, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def _row_normalize_sparse(features):
    rowsum = np.array(features.sum(1))
    with np.errstate(divide="ignore"):
        r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    return sp.diags(r_inv).dot(features)


def row_normalize_features(features):
    """Row-normalize a (sparse) feature matrix, returning a dense array."""
    return np.asarray(_row_normalize_sparse(features).todense())


def to_torch_sparse(mx, device=None):
    mx = sp.coo_matrix(mx).astype(np.float32)
    indices = torch.from_numpy(np.vstack((mx.row, mx.col)).astype(np.int64))
    values = torch.from_numpy(mx.data)
    t = torch.sparse_coo_tensor(indices, values, torch.Size(mx.shape)).coalesce()
    return t.to(device) if device is not None else t


# ---------------------------------------------------------------------------
# baseline-mode inputs (06_samples)
# ---------------------------------------------------------------------------

def load_samples2015(samples_dir, device):
    """Load the published preprocessed arrays (06_samples).

    Returns a dict with HeCo inputs:
      feats: [feat_p (n×2914 row-normalized dense), eye(828), eye(828)]
      mps:   [normalized relation adjacency, normalized kNN(k=3) adjacency]
      pos:   kNN(k=4) positive-pair matrix (unnormalized)
      nei:   [kNN neighbor lists, relation neighbor lists] (per-node LongTensor)
    """
    def p(name):
        return os.path.join(samples_dir, name)

    feat_p = sp.load_npz(p("converted_sparse_matrix_float.npz"))
    rel = sp.load_npz(p("2015_train_id_relation.npz"))
    knn = sp.load_npz(p("converted_knn_graph_3_coo.npz"))
    pos = sp.load_npz(p("converted_knn_graph_4_coo.npz"))
    nei_knn = np.load(p("my_nei_a_knn.npy"), allow_pickle=True)
    nei_rel = np.load(p("my_nei_a_relation.npy"), allow_pickle=True)

    n_side = 828  # one-hot neighbor-universe features
    feats = [
        torch.FloatTensor(row_normalize_features(feat_p)).to(device),
        torch.FloatTensor(row_normalize_features(sp.eye(n_side))).to(device),
        torch.FloatTensor(row_normalize_features(sp.eye(n_side))).to(device),
    ]
    mps = [
        to_torch_sparse(normalize_adj(rel), device),
        to_torch_sparse(normalize_adj(knn), device),
    ]
    return {
        "feats": feats,
        "mps": mps,
        "pos": to_torch_sparse(pos, device),
        # neighbor lists stay on CPU: sampling uses np.random.choice, and the
        # sampled index tensor is moved to the device afterwards
        "nei": [
            [torch.LongTensor(np.asarray(x, dtype=np.int64)) for x in nei_knn],
            [torch.LongTensor(np.asarray(x, dtype=np.int64)) for x in nei_rel],
        ],
        "num_nodes": rel.shape[0],
    }


# ---------------------------------------------------------------------------
# per-year inputs (paper/default mode)
# ---------------------------------------------------------------------------

def _read_qid_mapping(path):
    qid2id = {}
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            qid, idx = line.split()
            qid2id[qid] = int(idx)
    return qid2id


def _read_edges(path):
    edges = []
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            a, b = line.split()
            edges.append((a, b))
    return edges


def _undirected_qid_set(edges):
    return {frozenset((a, b)) for a, b in edges if a != b}


def build_year_graph(raw_root, year, knn_k=3):
    """Build train-split graph arrays for one year from the raw files.

    Returns a dict of scipy/numpy objects (no torch), suitable for np.savez.
    Cross-year positive/negative samples follow paper Eq. 10/11 with
    t1 = year-1, t2 = year, computed in QID space (each year+split has its
    own id ordering, so id-space diffs would be meaningless).
    """
    rel_dir = os.path.join(raw_root, "relationship")
    mapping = _read_qid_mapping(
        os.path.join(rel_dir, str(year), f"{year}_train_qid_id_mapping.txt")
    )
    n = len(mapping)

    # relation graph (undirected 0/1 adjacency in this year's train id space)
    rel_edges_qid = _undirected_qid_set(
        _read_edges(os.path.join(rel_dir, str(year), f"{year}_train_qid_relation.txt"))
    )
    rows, cols = [], []
    for pair in rel_edges_qid:
        a, b = tuple(pair)
        if a in mapping and b in mapping:
            i, j = mapping[a], mapping[b]
            rows += [i, j]
            cols += [j, i]
    rel = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)
    ).tocsr()
    rel.data[:] = 1.0  # deduplicate parallel edges

    # feature (kNN) graph, symmetrized (the files may be stored flat or in
    # per-year folders — support both)
    knn_name = f"{year}_knn_graph_{knn_k}.txt"
    knn_path = os.path.join(raw_root, "knn", str(year), knn_name)
    if not os.path.exists(knn_path):
        knn_path = os.path.join(raw_root, "knn", knn_name)
    knn_edges = np.loadtxt(knn_path, dtype=np.int64)
    kr = np.concatenate([knn_edges[:, 0], knn_edges[:, 1]])
    kc = np.concatenate([knn_edges[:, 1], knn_edges[:, 0]])
    knn = sp.coo_matrix(
        (np.ones(len(kr), dtype=np.float32), (kr, kc)), shape=(n, n)
    ).tocsr()
    knn.data[:] = 1.0

    # feature matrix X (n × vocab-slice), 0/1
    x_path = os.path.join(
        raw_root, "feature", str(year), f"{year}_train_token_id_after_filter_bin.txt"
    )
    X = sp.csr_matrix(np.loadtxt(x_path, dtype=np.float32))

    # cross-year positive/negative samples (Eq. 10/11), QID space
    prev_year = year - 1
    prev_path = os.path.join(
        rel_dir, str(prev_year), f"{prev_year}_train_qid_relation.txt"
    )
    prev_edges_qid = _undirected_qid_set(_read_edges(prev_path))
    pos_lists = [[] for _ in range(n)]
    neg_lists = [[] for _ in range(n)]
    for pair in rel_edges_qid - prev_edges_qid:  # newly added → positives
        a, b = tuple(pair)
        if a in mapping and b in mapping:
            pos_lists[mapping[a]].append(mapping[b])
            pos_lists[mapping[b]].append(mapping[a])
    for pair in prev_edges_qid - rel_edges_qid:  # newly removed → negatives
        a, b = tuple(pair)
        if a in mapping and b in mapping:
            neg_lists[mapping[a]].append(mapping[b])
            neg_lists[mapping[b]].append(mapping[a])

    return {
        "n": n,
        "rel": rel,
        "knn": knn,
        "X": X,
        "pos_lists": pos_lists,
        "neg_lists": neg_lists,
        "prev_year": prev_year,
    }


def _ragged_to_arrays(lists):
    flat = np.array([j for lst in lists for j in lst], dtype=np.int64)
    offsets = np.zeros(len(lists) + 1, dtype=np.int64)
    np.cumsum([len(lst) for lst in lists], out=offsets[1:])
    return flat, offsets


def save_year_graph(graph, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pos_flat, pos_off = _ragged_to_arrays(graph["pos_lists"])
    neg_flat, neg_off = _ragged_to_arrays(graph["neg_lists"])
    np.savez_compressed(
        path,
        n=graph["n"],
        prev_year=graph["prev_year"],
        rel_indptr=graph["rel"].indptr, rel_indices=graph["rel"].indices,
        knn_indptr=graph["knn"].indptr, knn_indices=graph["knn"].indices,
        X_indptr=graph["X"].indptr, X_indices=graph["X"].indices,
        X_shape=np.array(graph["X"].shape),
        pos_flat=pos_flat, pos_off=pos_off,
        neg_flat=neg_flat, neg_off=neg_off,
    )


def load_year_graph(path, device):
    """Load a processed per-year graph as torch tensors.

    Adjacencies are degree-normalized once here and cached on the device —
    they are constant for the whole run.
    """
    z = np.load(path)
    n = int(z["n"])

    def csr(prefix, shape):
        indptr, indices = z[f"{prefix}_indptr"], z[f"{prefix}_indices"]
        data = np.ones(len(indices), dtype=np.float32)
        return sp.csr_matrix((data, indices, indptr), shape=shape)

    rel = csr("rel", (n, n))
    knn = csr("knn", (n, n))
    X = csr("X", tuple(z["X_shape"]))

    def flat_edges(flat_key, off_key):
        flat = z[flat_key]
        off = z[off_key]
        src = np.repeat(np.arange(n, dtype=np.int64), np.diff(off))
        return (
            torch.from_numpy(src).to(device),
            torch.from_numpy(flat.astype(np.int64)).to(device),
        )

    pos_src, pos_dst = flat_edges("pos_flat", "pos_off")
    neg_src, neg_dst = flat_edges("neg_flat", "neg_off")
    knn_coo = knn.tocoo()
    knn_src = torch.from_numpy(knn_coo.row.astype(np.int64)).to(device)
    knn_dst = torch.from_numpy(knn_coo.col.astype(np.int64)).to(device)

    return {
        "num_nodes": n,
        "prev_year": int(z["prev_year"]),
        "rel_adj": to_torch_sparse(normalize_adj(rel), device),
        "knn_adj": to_torch_sparse(normalize_adj(knn), device),
        "rel_neighbors": [torch.from_numpy(np.asarray(r)) for r in _csr_neighbor_lists(rel)],
        "knn_neighbors": [torch.from_numpy(np.asarray(r)) for r in _csr_neighbor_lists(knn)],
        # kept sparse: the paper module consumes it via torch.sparse.mm,
        # numerically identical to a dense matmul but ~25x smaller
        "X": to_torch_sparse(_row_normalize_sparse(X), device),
        "pos_src": pos_src, "pos_dst": pos_dst,
        "neg_src": neg_src, "neg_dst": neg_dst,
        "knn_src": knn_src, "knn_dst": knn_dst,
    }


def _csr_neighbor_lists(m):
    m = m.tocsr()
    return [m.indices[m.indptr[i]: m.indptr[i + 1]] for i in range(m.shape[0])]
