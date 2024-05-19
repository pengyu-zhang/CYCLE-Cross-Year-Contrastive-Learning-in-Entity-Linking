# # Copyright (c) Facebook, Inc. and its affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.
# #
#
# # Utility code for zeshel dataset
# import json
# import torch
# from blink.common.params import BlinkParser
#
# # DOC_PATH = "./data/zeshel/documents/"
# DOC_PATH = "./data/tempel/documents/"
#
# WORLDS = [
#     # 'american_football',
#     # 'doctor_who',
#     # 'fallout',
#     # 'final_fantasy',
#     # 'military',
#     # 'pro_wrestling',
#     # 'starwars',
#     # 'world_of_warcraft',
#     # 'coronation_street',
#     # 'muppets',
#     # 'ice_hockey',
#     # 'elder_scrolls',
#     # 'forgotten_realms',
#     # 'lego',
#     # 'star_trek',
#     # 'yugioh'
#     '2014_train',
#     '2014_validation',
#     '2022_test'
# ]
#
# world_to_id = {src: k for k, src in enumerate(WORLDS)}
#
#
# def load_entity_dict_zeshel(logger, params):
#     entity_dict = {}
#     # different worlds in train/valid/test
#     # if params["mode"] == "train":
#     #     start_idx = 0
#     #     end_idx = 8
#     # elif params["mode"] == "valid":
#     #     start_idx = 8
#     #     end_idx = 12
#     # else:
#     #     start_idx = 12
#     #     end_idx = 16
#     if params["mode"] == "train":
#         start_idx = 0
#         end_idx = 1
#     elif params["mode"] == "validation":
#         start_idx = 1
#         end_idx = 2
#     else:
#         start_idx = 2
#         end_idx = 3
#     # load data
#     for i, src in enumerate(WORLDS[start_idx:end_idx]):
#         fname = DOC_PATH + src + ".json"
#         cur_dict = {}
#         doc_list = []
#         src_id = world_to_id[src]
#         with open(fname, 'rt', encoding='utf-8') as f:
#             for line in f:
#                 line = line.rstrip()
#                 item = json.loads(line)
#                 text = item["text"]
#                 doc_list.append(text[:256])
#
#                 if params["debug"]:
#                     if len(doc_list) > 200:
#                         break
#
#         logger.info("Load for world %s." % src)
#         entity_dict[src_id] = doc_list
#     return entity_dict
#
#
# class Stats():
#     def __init__(self, top_k=64):
#         self.cnt = 0
#         self.hits = []
#         self.top_k = top_k
#         self.rank = [1, 2, 4, 8, 16, 32, 64]
#         self.LEN = len(self.rank)
#         for i in range(self.LEN):
#             self.hits.append(0)
#
#     def add(self, idx):
#         self.cnt += 1
#         if idx == -1:
#             return
#         for i in range(self.LEN):
#             if idx < self.rank[i]:
#                 self.hits[i] += 1
#
#     def extend(self, stats):
#         self.cnt += stats.cnt
#         for i in range(self.LEN):
#             self.hits[i] += stats.hits[i]
#
#     def output(self):
#         output_json = "Total: %d examples." % self.cnt
#         for i in range(self.LEN):
#             if self.top_k < self.rank[i]:
#                 break
#             output_json += " r@%d: %.4f" % (self.rank[i], self.hits[i] / float(self.cnt))
#         return output_json
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Utility code for zeshel dataset
import json
import torch

# DOC_PATH = "./data/zeshel/documents/"
# DOC_PATH = "./data/tempel/documents/"
DOC_PATH = "/gpfs/home5/pzhang/code/dataset/01_blink_baseline/documents/"

WORLDS = [
    '2013_test',
    '2014_test',
    '2015_test',
    '2016_test',
    '2017_test',
    '2018_test',
    '2019_test',
    '2020_test',
    '2021_test',
    '2022_test',
    '2013_train',
    '2013_validation',
    '2014_train',
    '2014_validation',
    '2015_train',
    '2015_validation',
    '2016_train',
    '2016_validation',
    '2017_train',
    '2017_validation',
    '2018_train',
    '2018_validation',
    '2019_train',
    '2019_validation',
    '2020_train',
    '2020_validation',
    '2021_train',
    '2021_validation',
    '2022_train',
    '2022_validation'
]

world_to_id = {src: k for k, src in enumerate(WORLDS)}


def load_entity_dict_zeshel(logger, params):
    entity_dict = {}
    if params["mode"] == "2013":
        start_idx = 0
        end_idx = 1
    elif params["mode"] == "2014":
        start_idx = 1
        end_idx = 2
    elif params["mode"] == "2015":
        start_idx = 2
        end_idx = 3
    elif params["mode"] == "2016":
        start_idx = 3
        end_idx = 4
    elif params["mode"] == "2017":
        start_idx = 4
        end_idx = 5
    elif params["mode"] == "2018":
        start_idx = 5
        end_idx = 6
    elif params["mode"] == "2019":
        start_idx = 6
        end_idx = 7
    elif params["mode"] == "2020":
        start_idx = 7
        end_idx = 8
    elif params["mode"] == "2021":
        start_idx = 8
        end_idx = 9
    elif params["mode"] == "2022":
        start_idx = 9
        end_idx = 10
    # load data
    for i, src in enumerate(WORLDS[start_idx:end_idx]):
        fname = DOC_PATH + src + ".json"
        cur_dict = {}
        doc_list = []
        src_id = world_to_id[src]
        with open(fname, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                item = json.loads(line)
                text = item["text"]
                doc_list.append(text[:128])

                if params["debug"]:
                    if len(doc_list) > 200:
                        break

        logger.info("Load for world %s." % src)
        entity_dict[src_id] = doc_list
    return entity_dict


class Stats():
    def __init__(self, top_k=64):
        self.cnt = 0
        self.hits = []
        self.top_k = top_k
        self.rank = [1, 2, 4, 8, 16, 32, 64]
        self.LEN = len(self.rank)
        for i in range(self.LEN):
            self.hits.append(0)

    def add(self, idx):
        self.cnt += 1
        if idx == -1:
            return
        for i in range(self.LEN):
            if idx < self.rank[i]:
                self.hits[i] += 1

    def extend(self, stats):
        self.cnt += stats.cnt
        for i in range(self.LEN):
            self.hits[i] += stats.hits[i]

    def output(self):
        output_json = "Total: %d examples." % self.cnt
        for i in range(self.LEN):
            if self.top_k < self.rank[i]:
                break
            # output_json += " r@%d: %.4f" % (self.rank[i], self.hits[i] / float(self.cnt))
            if self.cnt == 0:
                output_json += " r@%d: 0" % self.rank[i]
            else:
                output_json += " r@%d: %.4f" % (self.rank[i], self.hits[i] / float(self.cnt))

        return output_json
