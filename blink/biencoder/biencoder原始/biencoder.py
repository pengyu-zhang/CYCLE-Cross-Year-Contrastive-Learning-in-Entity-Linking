# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# from pytorch_transformers.modeling_bert import (
#     BertPreTrainedModel,
#     BertConfig,
#     BertModel,
# )
from transformers import AutoTokenizer, AutoModel
# from pytorch_transformers.tokenization_bert import BertTokenizer

from blink.common.ranker_base import BertEncoder, get_model_obj
from blink.common.optimizer import get_bert_optimizer
from gcn_utils import *  # 加入GCN
import torch.optim as optim  # 加入GCN


def load_biencoder(params):
    # Init model
    biencoder = BiEncoderRanker(params)
    return biencoder


class BiEncoderModule(torch.nn.Module):
    def __init__(self, params):
        super(BiEncoderModule, self).__init__()
        ctxt_bert = AutoModel.from_pretrained(params["bert_model"])
        cand_bert = AutoModel.from_pretrained(params['bert_model'])
        self.context_encoder = BertEncoder(
            ctxt_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.config = ctxt_bert.config

    def forward(
            self,
            token_idx_ctxt,
            segment_idx_ctxt,
            mask_ctxt,
            token_idx_cands,
            segment_idx_cands,
            mask_cands,
    ):
        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt
            )
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )
        return embedding_ctxt, embedding_cands


class BiEncoderRanker(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(BiEncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = AutoTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )
        # init model
        self.build_model()
        model_path = params.get("path_to_model", None)
        if model_path is not None:
            self.load_model(model_path)

        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict)

    def build_model(self):
        self.model = BiEncoderModule(self.params)

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model)
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )

    def encode_context(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        embedding_context, _ = self.model(
            token_idx_cands, segment_idx_cands, mask_cands, None, None, None
        )
        return embedding_context.cpu().detach()

    def encode_candidate(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        return embedding_cands.cpu().detach()
        # TODO: why do we need cpu here?
        # return embedding_cands

    # Score candidates given context input and label input
    # If cand_encs is provided (pre-computed), cand_ves is ignored
    def score_candidate(
            self,
            text_vecs,
            cand_vecs,
            # relation_vec,  # 加入关系图1/4
            random_negs=True,
            cand_encs=None,  # pre-computed candidate encoding.
    ):
        # 将text_vecs的后128维替换为relation_vec的前128维
        # text_vecs[:, -128:] = relation_vec[:, :128]
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
        )

        # 添加relation_vec
        # relation_vec, segment_idx_ctxt, mask_ctxt = to_bert_input(
        #     relation_vec, self.NULL_IDX
        # )
        # relation_vec, _ = self.model(
        #     relation_vec, segment_idx_ctxt, mask_ctxt, None, None, None
        # )

        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        if cand_encs is not None:
            return embedding_ctxt.mm(cand_encs.t())

        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        # 将embedding_cands保存为txt文件，只保存小数点后四位，并且保存方式为不断添加，而不是替换
        # embedding_cands_txt = embedding_cands.cpu().detach().numpy()
        # with open('embedding_cands.txt', 'a') as f:
        #     np.savetxt(f, embedding_cands_txt, fmt='%.2f')
        # embedding_cands_txt = (embedding_cands > 0.0).cpu().detach().numpy().astype(int)
        # with open('embedding_cands2.txt', 'a') as f:
        #     np.savetxt(f, embedding_cands_txt, fmt='%d')

        if random_negs:
            # train on random negatives
            return embedding_ctxt.mm(embedding_cands.t())  # 加入关系图 blink原始代码
            # 加入关系图2/4开始
            # scores1 = embedding_ctxt.mm(embedding_cands.t())
            # scores2 = embedding_cands.mm(relation_vec.t())
            # scores = scores1 + scores2
            # scores = torch.mul(embedding_cands, relation_vec)
            # scores = torch.mm(embedding_ctxt, scores.t())
            # return scores
            # 加入关系图2/4结束

        else:
            # train on hard negatives
            embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
            embedding_cands = embedding_cands.unsqueeze(2)  # batchsize x embed_size x 2
            scores = torch.bmm(embedding_ctxt, embedding_cands)  # batchsize x 1 x 1
            scores = torch.squeeze(scores)
            return scores

    # label_input -- negatives provided
    # If label_input is None, train on in-batch negatives
    def forward(self, context_input, cand_input, label_input=None):  # 加入关系图 blink原始代码
    # def forward(self, context_input, cand_input, relation_vec, label_input=None):  # 加入关系图3/4
        # 加入GCN开始
        featuregraph_path = self.params["featuregraph_path"]
        structgraph_path = self.params["structgraph_path"]
        featuregraph_size = self.params["featuregraph_size"]
        structgraph_size = self.params["structgraph_size"]
        feature_path = self.params["feature_path"]
        nfeat = self.params["nfeat"]
        nhid1 = self.params["nhid1"]
        nhid2 = self.params["nhid2"]
        dropout = self.params["dropout"]
        beta = self.params["beta"]
        theta = self.params["theta"]
        model1 = SFGCN(nfeat, nhid1, nhid2, dropout)
        model1.cuda()
        # optimizer_gcn = optim.Adam(model1.parameters(), lr=0.0005, weight_decay=5e-4)
        # optimizer_gcn.zero_grad()
        sadj, fadj = load_graph(featuregraph_path, structgraph_path, featuregraph_size, structgraph_size)
        features = load_data(feature_path)
        features = features.cuda()
        sadj = sadj.cuda()
        fadj = fadj.cuda()
        att, emb1, com1, com2, emb2, emb = model1(features, sadj, fadj)
        emb = emb1 + emb2 + com1 + com2
        # 加入GCN结束
        flag = label_input is None
        scores = self.score_candidate(context_input, cand_input, flag)  # 加入关系图 blink原始代码
        # scores = self.score_candidate(context_input, cand_input, relation_vec, flag)  # 加入关系图4/4
        bs = scores.size(0)
        if label_input is None:
            target = torch.LongTensor(torch.arange(bs))
            target = target.to(self.device)
            # loss = F.cross_entropy(scores, target, reduction="mean")  # 加入GCN blink原始代码
            loss_og = F.cross_entropy(scores, target, reduction="mean")
            # 加入GCN开始
            loss_dep = (loss_dependence(emb1, com1, structgraph_size) + loss_dependence(emb2, com2, structgraph_size)) / 2
            loss_com = common_loss(com1, com2)
            loss = loss_og + beta * loss_dep + theta * loss_com
            # 加入GCN结束
        else:
            loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
            # TODO: add parameters?
            loss = loss_fct(scores, label_input)
        return loss, scores


def to_bert_input(token_idx, null_idx):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask
