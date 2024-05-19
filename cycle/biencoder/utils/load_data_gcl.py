import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import OneHotEncoder


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    # d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt = np.power(rowsum + 1e-10, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """Convert a scipy sparse matrix to a torch sparse tensor."""
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     indices = torch.from_numpy(
#         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse.FloatTensor(indices, values, shape)
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)




def load_acm(ratio, type_num):
    # The order of node types: 0 p 1 a 2 s
    path = "/gpfs/home5/pzhang/code/dataset/acm/"
    label = np.load(path + "labels_new.npy").astype('int32')
    label = encode_onehot(label)
    nei_a = np.load(path + "my_nei_a_knn.npy", allow_pickle=True)
    nei_s = np.load(path + "my_nei_a_relation.npy", allow_pickle=True)
    feat_p = sp.load_npz(path + "converted_sparse_matrix_float.npz")
    feat_a = sp.eye(type_num[1])
    feat_s = sp.eye(type_num[2])
    pap = sp.load_npz(path + "2015_train_id_relation.npz")
    psp = sp.load_npz(path + "converted_knn_graph_3_coo.npz")
    pos = sp.load_npz(path + "converted_knn_graph_4_coo.npz")
    train = [np.load(path + "modified_train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "modified_test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "modified_val_" + str(i) + ".npy") for i in ratio]

    label = torch.FloatTensor(label)
    nei_a = [torch.LongTensor(i) for i in nei_a]
    nei_s = [torch.LongTensor(i) for i in nei_s]
    feat_p = torch.FloatTensor(preprocess_features(feat_p))
    feat_a = torch.FloatTensor(preprocess_features(feat_a))
    feat_s = torch.FloatTensor(preprocess_features(feat_s))
    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    psp = sparse_mx_to_torch_sparse_tensor(normalize_adj(psp))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]

    # print("label shape: ", label.shape, "label type: ", type(label))
    # print("nei_a length: ", len(nei_a), "nei_a type: ", type(nei_a))
    # print("nei_s length: ", len(nei_s), "nei_s type: ", type(nei_s))
    # print("feat_p shape: ", feat_p.shape, "feat_p type: ", type(feat_p), "feat_p: ", feat_p)
    # # 获取feat_p中非零元素的行和列索引以及对应的值
    # print("feat_p nonzero: ", feat_p.nonzero())
    # print("feat_p nonzero row: ", feat_p.nonzero()[0])
    # print("feat_p nonzero col: ", feat_p.nonzero()[1])
    # print("feat_p nonzero value: ", feat_p.data)
    # print("feat_p nonzero value type: ", type(feat_p.data))
    #
    # print("feat_a shape: ", feat_a.shape, "feat_a type: ", type(feat_a))
    # print("feat_s shape: ", feat_s.shape, "feat_s type: ", type(feat_s))
    # print("pap shape: ", pap.shape, "pap type: ", type(pap))
    # print("psp shape: ", psp.shape, "psp type: ", type(psp))
    # print("pos shape: ", pos.shape, "pos type: ", type(pos))
    # print("train length: ", len(train), "train type: ", type(train))
    # print("val length: ", len(val), "val type: ", type(val))
    # print("test length: ", len(test), "test type: ", type(test))

    return [nei_a, nei_s], [feat_p, feat_a, feat_s], [pap, psp], pos, label, train, val, test


def load_dblp(ratio, type_num):
    # The order of node types: 0 a 1 p 2 c 3 t
    path = "../data/dblp/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_p = np.load(path + "nei_p.npy", allow_pickle=True)
    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_p = sp.eye(type_num[1])
    apa = sp.load_npz(path + "apa.npz")
    apcpa = sp.load_npz(path + "apcpa.npz")
    aptpa = sp.load_npz(path + "aptpa.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    
    label = torch.FloatTensor(label)
    nei_p = [torch.LongTensor(i) for i in nei_p]
    feat_p = torch.FloatTensor(preprocess_features(feat_p))
    feat_a = torch.FloatTensor(preprocess_features(feat_a))
    apa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apa))
    apcpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apcpa))
    aptpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(aptpa))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]
    return [nei_p], [feat_a, feat_p], [apa, apcpa, aptpa], pos, label, train, val, test


def load_aminer(ratio, type_num):
    # The order of node types: 0 p 1 a 2 r
    path = "../data/aminer/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_r = np.load(path + "nei_r.npy", allow_pickle=True)
    # Because none of P, A or R has features, we assign one-hot encodings to all of them.
    feat_p = sp.eye(type_num[0])
    feat_a = sp.eye(type_num[1])
    feat_r = sp.eye(type_num[2])
    pap = sp.load_npz(path + "pap.npz")
    prp = sp.load_npz(path + "prp.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = torch.FloatTensor(label)
    nei_a = [torch.LongTensor(i) for i in nei_a]
    nei_r = [torch.LongTensor(i) for i in nei_r]
    feat_p = torch.FloatTensor(preprocess_features(feat_p))
    feat_a = torch.FloatTensor(preprocess_features(feat_a))
    feat_r = torch.FloatTensor(preprocess_features(feat_r))
    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    prp = sparse_mx_to_torch_sparse_tensor(normalize_adj(prp))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]
    return [nei_a, nei_r], [feat_p, feat_a, feat_r], [pap, prp], pos, label, train, val, test


def load_freebase(ratio, type_num):
    # The order of node types: 0 m 1 d 2 a 3 w
    path = "../data/freebase/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_d = np.load(path + "nei_d.npy", allow_pickle=True)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_w = np.load(path + "nei_w.npy", allow_pickle=True)
    feat_m = sp.eye(type_num[0])
    feat_d = sp.eye(type_num[1])
    feat_a = sp.eye(type_num[2])
    feat_w = sp.eye(type_num[3])
    # Because none of M, D, A or W has features, we assign one-hot encodings to all of them.
    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    mwm = sp.load_npz(path + "mwm.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = torch.FloatTensor(label)
    nei_d = [torch.LongTensor(i) for i in nei_d]
    nei_a = [torch.LongTensor(i) for i in nei_a]
    nei_w = [torch.LongTensor(i) for i in nei_w]
    feat_m = torch.FloatTensor(preprocess_features(feat_m))
    feat_d = torch.FloatTensor(preprocess_features(feat_d))
    feat_a = torch.FloatTensor(preprocess_features(feat_a))
    feat_w = torch.FloatTensor(preprocess_features(feat_w))
    mam = sparse_mx_to_torch_sparse_tensor(normalize_adj(mam))
    mdm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mdm))
    mwm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mwm))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]
    return [nei_d, nei_a, nei_w], [feat_m, feat_d, feat_a, feat_w], [mdm, mam, mwm], pos, label, train, val, test


def load_data_gcl(dataset, ratio, type_num):
    if dataset == "acm":
        data = load_acm(ratio, type_num)
    elif dataset == "dblp":
        data = load_dblp(ratio, type_num)
    elif dataset == "aminer":
        data = load_aminer(ratio, type_num)
    elif dataset == "freebase":
        data = load_freebase(ratio, type_num)
    return data
