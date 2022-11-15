# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 18:59
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : dataset.py
# @Software: PyCharm
# @Note    :
import os
import json
import torch
from torch_geometric.data import Data, InMemoryDataset


class TreeDataset(InMemoryDataset):
    def __init__(self, root, word_embedding, word2vec, transform=None, pre_transform=None,
                 pre_filter=None):
        self.word_embedding = word_embedding
        self.word2vec = word2vec
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        raw_file_names = self.raw_file_names

        for filename in raw_file_names:
            y = []
            row = []
            col = []

            filepath = os.path.join(self.raw_dir, filename)
            post = json.load(open(filepath, 'r', encoding='utf-8'))
            if self.word_embedding == 'word2vec':
                x = self.word2vec.get_sentence_embedding(post['source']['content']).view(1, -1)
            elif self.word_embedding == 'tfidf':
                tfidf = post['source']['content']
                indices = [[0, int(index_freq.split(':')[0])] for index_freq in tfidf.split()]
                values = [int(index_freq.split(':')[1]) for index_freq in tfidf.split()]
            if 'label' in post['source'].keys():
                y.append(post['source']['label'])
            for i, comment in enumerate(post['comment']):
                if self.word_embedding == 'word2vec':
                    x = torch.cat(
                        [x, self.word2vec.get_sentence_embedding(comment['content']).view(1, -1)], 0)
                elif self.word_embedding == 'tfidf':
                    indices += [[i + 1, int(index_freq.split(':')[0])] for index_freq in comment['content'].split()]
                    values += [int(index_freq.split(':')[1]) for index_freq in comment['content'].split()]
                row.append(comment['parent'] + 1)
                col.append(comment['comment id'] + 1)

            edge_index = [row, col]
            y = torch.LongTensor(y)
            edge_index = torch.LongTensor(edge_index)
            if self.word_embedding == 'tfidf':
                x = torch.sparse_coo_tensor(torch.tensor(indices).t(), values, (len(post['comment']) + 1, 5000),
                                            dtype=torch.float32).to_dense()
            one_data = Data(x=x, y=y, edge_index=edge_index) if 'label' in post['source'].keys() else \
                Data(x=x, edge_index=edge_index)
            data_list.append(one_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        all_data, slices = self.collate(data_list)
        torch.save((all_data, slices), self.processed_paths[0])
