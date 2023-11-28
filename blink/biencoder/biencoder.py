# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import load_data_gcl, set_params, evaluate_gcl
from module import HeCo
import datetime
import random
import pickle as pkl


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
            final_embeds_batch,
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
            # return embedding_ctxt.mm(embedding_cands.t())  # train on random negatives 加入关系图 blink原始代码
            # 方式1 将final升至512维计算score2
            scores1 = embedding_ctxt.mm(embedding_cands.t())
            embedding_cands_transposed = embedding_cands.transpose(0, 1)  # 结果形状是 (512, 16)
            linear_layer = nn.Linear(64, 512).to(self.device)  # 使用一个线性层将 tensor_b 映射到 (16, 512)
            final_embeds_batch = final_embeds_batch.to(self.device)
            final_embeds_batch_mapped = linear_layer(final_embeds_batch)  # 结果形状是 (16, 512)
            scores2 = torch.matmul(final_embeds_batch_mapped, embedding_cands_transposed)  # 结果形状是 (16, 16)
            scores2_normalized = torch.tanh(scores2) * 10  # 将 scores2 归一化到 -1 到 1 之间，然后乘以 X 来放大 13年最佳是20
            if scores1.shape != scores2_normalized.shape:
                scores = scores1  # 如果形状不一致，直接将scores设置为scores1
            else:
                scores = scores1 + scores2_normalized  # 如果形状一致，将scores1和scores2_normalized相加
            return scores
            # 方式1结束


        else:
            # train on hard negatives
            embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
            embedding_cands = embedding_cands.unsqueeze(2)  # batchsize x embed_size x 2
            scores = torch.bmm(embedding_ctxt, embedding_cands)  # batchsize x 1 x 1
            scores = torch.squeeze(scores)
            return scores

    # label_input -- negatives provided
    # If label_input is None, train on in-batch negatives
    def forward(self, context_input, cand_input, step, label_input=None):  # 加入关系图 blink原始代码
    # def forward(self, context_input, cand_input, relation_vec, label_input=None):  # 加入关系图3/4
        # 加入GCN开始
        args = set_params()
        own_str = args.dataset
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed = args.seed
        numpy.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test = \
            load_data_gcl(args.dataset, args.ratio, args.type_num)
        nb_classes = label.shape[-1]
        feats_dim_list = [i.shape[1] for i in feats]
        P = int(len(mps))
        # print("seed ", args.seed)
        # print("Dataset: ", args.dataset)
        # print("The number of meta-paths: ", P)

        # model1 = HeCo(args.hidden_dim, feats_dim_list, args.feat_drop, args.attn_drop,
        #               P, args.sample_rate, args.nei_num, args.tau, args.lam, loss_og)
        model1 = HeCo(args.hidden_dim, feats_dim_list, args.feat_drop, args.attn_drop,
                      P, args.sample_rate, args.nei_num, args.tau, args.lam)
        optimiser = torch.optim.Adam(model1.parameters(), lr=args.lr, weight_decay=args.l2_coef)

        if torch.cuda.is_available():
            # print('Using CUDA')
            model1.cuda()
            feats = [feat.cuda() for feat in feats]
            mps = [mp.cuda() for mp in mps]
            pos = pos.cuda()
            label = label.cuda()
            idx_train = [i.cuda() for i in idx_train]
            idx_val = [i.cuda() for i in idx_val]
            idx_test = [i.cuda() for i in idx_test]

        cnt_wait = 0
        best = 1e9
        best_t = 0

        # 在训练之前获取和保存初始嵌入
        # model1.eval()  # 确保模型处于评估模式
        # initial_embeds = model1.get_embeds(feats, mps)
        # np.savetxt('initial_embeds_' + own_str + '.txt', initial_embeds.cpu().detach().numpy())

        starttime = datetime.datetime.now()
        for epoch in range(args.nb_epochs):
            model1.train()
            optimiser.zero_grad()
            # loss = model1(feats, pos, mps, nei_index, loss_og)
            loss1 = model1(feats, pos, mps, nei_index)
            # print("loss ", loss.data.cpu())
            if loss1 < best:
                best = loss1
                best_t = epoch
                cnt_wait = 0
                torch.save(model1.state_dict(), 'HeCo_' + own_str + '.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                # print('Early stopping!')
                break
            # loss1.backward()
            # optimiser.step()

        # print('Loading {}th epoch'.format(best_t))
        model1.load_state_dict(torch.load('HeCo_' + own_str + '.pkl'))
        model1.eval()

        # 在训练之后获取和保存最终嵌入
        final_embeds = model1.get_embeds(feats, mps)
        # np.savetxt('final_embeds_' + own_str + '.txt', final_embeds.cpu().detach().numpy())

        os.remove('HeCo_' + own_str + '.pkl')
        embeds = model1.get_embeds(feats, mps)
        # for i in range(len(idx_train)):
        #     evaluate_gcl(embeds, args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label, nb_classes, device, args.dataset,
        #              args.eva_lr, args.eva_wd)

        endtime = datetime.datetime.now()
        time = (endtime - starttime).seconds
        # print("Total time: ", time, "s")

        if args.save_emb:
            f = open("./embeds/" + args.dataset + "/" + str(args.turn) + ".pkl", "wb")
            pkl.dump(embeds.cpu().data.numpy(), f)
            f.close()
        # 加入GCN结束
        flag = label_input is None

        batch_size = context_input.size(0)  # 获取当前批次的大小
        start_index = step * batch_size  # 计算当前批次在 final_embeds 中的起始索引
        end_index = start_index + batch_size  # 计算结束索引
        final_embeds_batch = final_embeds[start_index:end_index, :]  # 提取对应当前批次的嵌入

        # scores = self.score_candidate(context_input, cand_input, flag)  # 加入关系图 blink原始代码
        scores = self.score_candidate(context_input, cand_input, final_embeds_batch, flag)  # 加入关系图4/4
        bs = scores.size(0)
        if label_input is None:
            target = torch.LongTensor(torch.arange(bs))
            target = target.to(self.device)
            # loss = F.cross_entropy(scores, target, reduction="mean")  # 加入GCN blink原始代码
            loss = F.cross_entropy(scores, target, reduction="mean")
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
