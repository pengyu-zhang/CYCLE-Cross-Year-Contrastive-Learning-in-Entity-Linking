import scipy.sparse as sp
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD


def common_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cost = torch.mean((cov1 - cov2)**2)
    return cost


def loss_dependence(emb1, emb2, dim):
    R = torch.eye(dim).cuda() - (1/dim) * torch.ones(dim, dim).cuda()
    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_data(feature_path):
    f = np.loadtxt(feature_path, dtype=float)
    # l = np.loadtxt(label_path, dtype=int)
    # test = np.loadtxt(test_path, dtype=int)
    # train = np.loadtxt(train_path, dtype=int)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))


    # idx_test = test.tolist()
    # idx_train = train.tolist()

    # idx_train = torch.LongTensor(idx_train)
    # idx_test = torch.LongTensor(idx_test)

    # label = torch.LongTensor(np.array(l))

    return features


# def load_graph(featuregraph_path, structgraph_path, featuregraph_size, structgraph_size):
def load_graph(featuregraph_path, structgraph_path, featuregraph_size, structgraph_size):
    featuregraph_path = featuregraph_path
    structgraph_path = structgraph_path

    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(featuregraph_size, featuregraph_size), dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))

    struct_edges = np.genfromtxt(structgraph_path, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(structgraph_size, structgraph_size), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    nsadj = normalize(sadj + sp.eye(sadj.shape[0]))

    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

    # # 将nsadj转化为(5000, 128)的tensor开始
    # # 从文件中读取 node_features
    # node_features_np = np.genfromtxt("C:/code/blink/data/tempel_token/acm/acm.feature", dtype=np.float32)
    # # 将 NumPy 数组转换为 PyTorch 张量
    # node_features = torch.tensor(node_features_np)
    # input_features = node_features.shape[1]  # 根据输入节点特征的维度更新输入特征数量
    # hidden_features = 256  # 或其他合适的值
    # output_features = 128
    # dropout = 0.5
    # # 实例化 GCN 模型
    # gcn_model = GCN(input_features, hidden_features, output_features, dropout)
    # # 计算节点嵌入
    # node_embeddings = gcn_model(node_features, nsadj)
    # # 将节点嵌入的值限制在 0 和 1 之间
    # node_embeddings_clamped = node_embeddings.clamp(min=0, max=1)
    # # 将节点嵌入四舍五入为 0 或 1
    # node_embeddings_binary = node_embeddings_clamped.round().to(torch.int64)
    # # 将nsadj转化为(5000, 128)的tensor结束

    # 转化为0-1矩阵 截取前1024个
    # a = nsadj._values()
    # b = a.numpy()
    # # 将b中所有元素扩大100倍
    # b = b * 100
    # # 将b中所有元素转化为整数
    # b = b.astype(int)
    # # c = np.select([b <= .05, b > .05], [np.zeros_like(b), np.ones_like(b)])
    # nsadj = b.tolist()
    # nsadj = nsadj[0:512]

    return nsadj, nfadj
    # return node_embeddings
    # return nsadj


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc2(x, adj)
        return x


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class SFGCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout):
        super(SFGCN, self).__init__()

        self.SGCN1 = GCN(nfeat, nhid1, nhid2, dropout)
        self.SGCN2 = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)

        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(nhid2)
        self.tanh = nn.Tanh()

        # self.MLP = nn.Sequential(
        #     nn.Linear(nhid2, nclass),
        #     nn.LogSoftmax(dim=1)
        # )

    def forward(self, x, sadj, fadj):
        emb1 = self.SGCN1(x, sadj)  # Special_GCN out1 -- sadj structure graph
        com1 = self.CGCN(x, sadj)  # Common_GCN out1 -- sadj structure graph
        com2 = self.CGCN(x, fadj)  # Common_GCN out2 -- fadj feature graph
        emb2 = self.SGCN2(x, fadj)  # Special_GCN out2 -- fadj feature graph
        Xcom = (com1 + com2) / 2
        # attention
        emb = torch.stack([emb1, emb2, Xcom], dim=1)
        emb, att = self.attention(emb)
        # output = self.MLP(emb)
        return att, emb1, com1, com2, emb2, emb
