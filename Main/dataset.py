# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 18:59
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : dataset.py
# @Software: PyCharm
# @Note    :
import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from Main.utils import clean_comment


class WeiboDataset(Dataset):
    def __init__(self, root, word2vec, clean=True, tddroprate=0.0, budroprate=0.0):
        self.root = root
        self.raw_dir = os.path.join(self.root, 'raw')
        self.word2vec = word2vec
        self.clean = clean
        self.tddroprate = tddroprate
        self.budroprate = budroprate
        self.data_list = self.process()

    def process(self):
        data_list = []
        raw_file_names = os.listdir(self.raw_dir)

        if self.clean:
            limit_num = 600
            pass_comment = ['', '转发微博', '转发微博。', '轉發微博', '轉發微博。']
            for filename in raw_file_names:
                y = []
                row = []
                col = []
                filepath = os.path.join(self.raw_dir, filename)
                post = json.load(open(filepath, 'r', encoding='utf-8'))
                x = self.word2vec.get_sentence_embedding(post['source']['content']).view(1, -1)
                y.append(post['source']['label'])
                pass_num = 0
                id_to_index = {}
                for i, comment in enumerate(post['comment']):
                    if i == limit_num:
                        break
                    id_to_index[comment['comment id']] = i
                for i, comment in enumerate(post['comment']):
                    if i == limit_num:
                        break
                    if comment['content'] in pass_comment and comment['children'] == []:
                        pass_num += 1
                        continue
                    post['comment'][i]['comment id'] -= pass_num
                for i, comment in enumerate(post['comment']):
                    if i == limit_num:
                        break
                    if comment['content'] in pass_comment and comment['children'] == []:
                        continue
                    x = torch.cat(
                        [x, self.word2vec.get_sentence_embedding(clean_comment(comment['content'])).view(1, -1)], 0)
                    if comment['parent'] == -1:
                        row.append(0)
                    else:
                        row.append(post['comment'][id_to_index[comment['parent']]]['comment id'] + 1)
                    col.append(comment['comment id'] + 1)

                tdrow = row
                tdcol = col
                if self.tddroprate > 0:
                    length = len(row)
                    poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
                    poslist = sorted(poslist)
                    tdrow = list(np.array(tdrow)[poslist])
                    tdcol = list(np.array(tdcol)[poslist])
                    edge_index = [tdrow, tdcol]
                else:
                    edge_index = [tdrow, tdcol]

                burow = col
                bucol = row
                if self.budroprate > 0:
                    length = len(burow)
                    poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
                    poslist = sorted(poslist)
                    burow = list(np.array(burow)[poslist])
                    bucol = list(np.array(bucol)[poslist])
                    bu_edge_index = [burow, bucol]
                else:
                    bu_edge_index = [burow, bucol]

                y = torch.LongTensor(y)
                edge_index = torch.LongTensor(edge_index)
                BU_edge_index = torch.LongTensor(bu_edge_index)
                # one_data = Data(x=x, y=y, root_feat=root_feat, edge_index=edge_index, BU_edge_index=BU_edge_index)
                one_data = Data(x=x, y=y, edge_index=edge_index, BU_edge_index=BU_edge_index)
                data_list.append(one_data)
        else:
            for filename in raw_file_names:
                y = []
                row = []
                col = []
                filepath = os.path.join(self.raw_dir, filename)
                post = json.load(open(filepath, 'r', encoding='utf-8'))
                x = self.word2vec.get_sentence_embedding(post['source']['content']).view(1, -1)
                y.append(post['source']['label'])
                for i, comment in enumerate(post['comment']):
                    x = torch.cat(
                        [x, self.word2vec.get_sentence_embedding(clean_comment(comment['content'])).view(1, -1)], 0)
                    row.append(comment['parent'] + 1)
                    col.append(comment['comment id'] + 1)

                tdrow = row
                tdcol = col
                if self.tddroprate > 0:
                    length = len(row)
                    poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
                    poslist = sorted(poslist)
                    tdrow = list(np.array(tdrow)[poslist])
                    tdcol = list(np.array(tdcol)[poslist])
                    edge_index = [tdrow, tdcol]
                else:
                    edge_index = [tdrow, tdcol]

                burow = col
                bucol = row
                if self.budroprate > 0:
                    length = len(burow)
                    poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
                    poslist = sorted(poslist)
                    burow = list(np.array(burow)[poslist])
                    bucol = list(np.array(bucol)[poslist])
                    bu_edge_index = [burow, bucol]
                else:
                    bu_edge_index = [burow, bucol]

                y = torch.LongTensor(y)
                edge_index = torch.LongTensor(edge_index)
                BU_edge_index = torch.LongTensor(bu_edge_index)
                # one_data = Data(x=x, y=y, root_feat=root_feat, edge_index=edge_index, BU_edge_index=BU_edge_index)
                one_data = Data(x=x, y=y, edge_index=edge_index, BU_edge_index=BU_edge_index)
                data_list.append(one_data)

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]
