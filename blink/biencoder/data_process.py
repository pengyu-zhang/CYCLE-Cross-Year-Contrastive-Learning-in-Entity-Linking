# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import torch
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset
# from pytorch_transformers.tokenization_bert import BertTokenizer

from blink.biencoder.zeshel_utils import world_to_id
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG
from gcn_utils import *  # 加入GCN


# def select_field(data, key1, key2=None):
#     if key2 is None:
#         if key1 == "relation_vec":
#             return [example[key1]["relation_vec"] for example in data]
#         else:
#             return [example[key1] for example in data]
#     else:
#         return [example[key1][key2] for example in data]
def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]


# def get_relation_vec(sample):
#     relation_vec_str = sample['relation_vec']
#     relation_vec = list(map(int, relation_vec_str.split()))
#     return {
#         "relation_vec": relation_vec,
#     }

def get_context_representation(
    sample,
    tokenizer,
    max_seq_length,
    # mention_key="mention",  # 输入改为token_only blink原始代码1/1
    # context_key="context",  # 输入改为token_only blink原始代码1/1
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
):
    mention_tokens = []

    # 输入改为token_only blink原始代码1/1开始
    # if sample[mention_key] and len(sample[mention_key]) > 0:
        # mention_tokens = tokenizer.tokenize(sample[mention_key])
        # mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]
    # 输入改为token_only blink原始代码1/1结束

    # 输入改为token_only1/1开始
    if sample['mention'] and len(sample['mention']) > 0:
        mention_tokens = tokenizer.tokenize(sample['mention'])
        mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]
    # 输入改为token_only1/1结束

    # 输入改为token_only blink原始代码1/1开始
    # context_left = sample[context_key + "_left"]
    # context_right = sample[context_key + "_right"]
    # context_left = tokenizer.tokenize(context_left)
    # context_right = tokenizer.tokenize(context_right)
    # 输入改为token_only blink原始代码1/1结束

    # 输入改为token_only1/1开始
    context_left = sample["context_left"]
    context_right = sample["context_right"]
    # 输入改为token_only1/1结束

    left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
    right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    context_tokens = (context_left[-left_quota:] + mention_tokens + context_right[:right_quota])

    context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": context_tokens,
        "ids": input_ids,
    }


def get_candidate_representation(
    candidate_desc, 
    tokenizer, 
    max_seq_length, 
    candidate_title=None,
    title_tag=ENT_TITLE_TAG,
):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    # cand_tokens = tokenizer.tokenize(candidate_desc)  # 输入改为token_only blink原始代码1/1
    cand_tokens = candidate_desc  # 输入改为token_only1/1
    if candidate_title is not None:
        title_tokens = tokenizer.tokenize(candidate_title)
        cand_tokens = title_tokens + [title_tag] + cand_tokens

    cand_tokens = cand_tokens[: max_seq_length - 2]
    cand_tokens = [cls_token] + cand_tokens + [sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": cand_tokens,
        "ids": input_ids,
    }


# 加入关系图2/8开始
# def get_candidate_graph(
#         structgraph_path,
#         structgraph_size,
# ):
#     relation_vec = load_graph(structgraph_path, structgraph_size)
#     return relation_vec
# 加入关系图2/8结束


def process_mention_data(
    samples,
    tokenizer,
    # 加入关系图3/8开始
    # structgraph_path,
    # structgraph_size,
    # 加入关系图3/8结束
    max_context_length,
    max_cand_length,
    silent,
    # mention_key="mention",  # 输入改为token_only blink原始代码1/1
    # context_key="context",  # 输入改为token_only blink原始代码1/1
    label_key="label",
    title_key='label_title',
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    title_token=ENT_TITLE_TAG,
    debug=False,
    logger=None,
):
    processed_samples = []

    if debug:
        samples = samples[:200]

    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)

    use_world = True

    # 加入关系图4/8开始
    # relation_vec = get_candidate_graph(
    #     structgraph_path,
    #     structgraph_size,
    # )
    # 加入关系图4/8结束

    for idx, sample in enumerate(iter_):
        # 先读128维的矩阵
        # relation_vec = get_relation_vec(sample)

        # 再读context_tokens
        context_tokens = get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            # mention_key,  # 输入改为token_only blink原始代码1/1
            # context_key,  # 输入改为token_only blink原始代码1/1
            ent_start_token,
            ent_end_token,
        )

        label = sample[label_key]
        title = sample.get(title_key, None)
        label_tokens = get_candidate_representation(
            label, tokenizer, max_cand_length, title,
        )

        label_idx = int(sample["label_id"])

        record = {
            "context": context_tokens,
            "label": label_tokens,
            # "relation_vec": relation_vec,  # 加入关系图5/8
            "label_idx": [label_idx],
        }

        if "world" in sample:
            src = sample["world"]
            src = world_to_id[src]
            record["src"] = [src]
            use_world = True
        else:
            use_world = False

        processed_samples.append(record)

    if debug and logger:
        logger.info("====Processed samples: ====")
        for sample in processed_samples[:5]:
            logger.info("Context tokens : " + " ".join(sample["context"]["tokens"]))
            logger.info(
                "Context ids : " + " ".join([str(v) for v in sample["context"]["ids"]])
            )
            logger.info("Label tokens : " + " ".join(sample["label"]["tokens"]))
            logger.info(
                "Label ids : " + " ".join([str(v) for v in sample["label"]["ids"]])
            )
            logger.info("Src : %d" % sample["src"][0])
            logger.info("Label_id : %d" % sample["label_idx"][0])

    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    cand_vecs = torch.tensor(
        select_field(processed_samples, "label", "ids"), dtype=torch.long,
    )
    # 加入关系图6/8开始
    # relation_vec = torch.tensor(
    #     select_field(processed_samples, "relation_vec"), dtype=torch.long,
    # )
    # 加入关系图6/8结束
    if use_world:
        src_vecs = torch.tensor(
            select_field(processed_samples, "src"), dtype=torch.long,
        )
    label_idx = torch.tensor(
        select_field(processed_samples, "label_idx"), dtype=torch.long,
    )
    data = {
        "context_vecs": context_vecs,
        "cand_vecs": cand_vecs,
        # "relation_vec": relation_vec,  # 加入关系图7/8
        "label_idx": label_idx,
    }
    # print("context_vecs shape:", context_vecs.shape)
    # print("cand_vecs shape:", cand_vecs.shape)
    # print("src_vecs shape:", src_vecs.shape)
    # print("relation_vec shape:", relation_vec.shape)
    # print("label_idx shape:", label_idx.shape)
    # print("context_vecs:", context_vecs)
    # print("cand_vecs:", cand_vecs)
    # print("src_vecs:", src_vecs)
    # print("relation_vec:", relation_vec)
    # print("label_idx:", label_idx)

    if use_world:
        data["src"] = src_vecs
        tensor_data = TensorDataset(context_vecs, cand_vecs, src_vecs, label_idx)  # 加入关系图 blink原始代码
        # tensor_data = TensorDataset(context_vecs, cand_vecs, src_vecs, relation_vec, label_idx)  # 加入关系图8/8
    else:
        tensor_data = TensorDataset(context_vecs, cand_vecs, label_idx)
    return data, tensor_data
