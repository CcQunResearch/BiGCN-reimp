# -*- coding: utf-8 -*-
# @Time    : 2022/8/23 18:22
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : model.py
# @Software: PyCharm
# @Note    :
import torch
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
import copy
import random
import numpy as np


class TDrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, tddroprate):
        super(TDrumorGCN, self).__init__()
        self.tddroprate = tddroprate
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        device = data.x.device
        x, edge_index = data.x, data.edge_index

        edge_index_list = edge_index.tolist()
        if self.tddroprate > 0:
            length = len(edge_index_list[0])
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            tdrow = list(np.array(edge_index_list[0])[poslist])
            tdcol = list(np.array(edge_index_list[1])[poslist])
            edge_index = torch.LongTensor([tdrow, tdcol]).to(device)

        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[index][0]
        x = th.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[index][0]
        x = th.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)
        return x


class BUrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, budroprate):
        super(BUrumorGCN, self).__init__()
        self.budroprate = budroprate
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        device = data.x.device
        x = data.x
        edge_index = data.edge_index.clone()
        edge_index[0], edge_index[1] = data.edge_index[1], data.edge_index[0]

        edge_index_list = edge_index.tolist()
        if self.budroprate > 0:
            length = len(edge_index_list[0])
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            burow = list(np.array(edge_index_list[0])[poslist])
            bucol = list(np.array(edge_index_list[1])[poslist])
            edge_index = torch.LongTensor([burow, bucol]).to(device)

        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[index][0]
        x = th.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[index][0]
        x = th.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)
        return x


class UDrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(UDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        device = data.x.device
        x, edge_index = data.x, to_undirected(data.edge_index)
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[index][0]
        x = th.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[index][0]
        x = th.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)
        return x


class Net(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_classes, tddroprate, budroprate):
        super(Net, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats, tddroprate)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats, budroprate)
        self.fc = th.nn.Linear((out_feats + hid_feats) * 2, num_classes)

    def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = th.cat((BU_x, TD_x), 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)


class UDNet(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_classes):
        super(UDNet, self).__init__()
        self.UDrumorGCN = UDrumorGCN(in_feats, hid_feats, out_feats)
        self.fc = th.nn.Linear(out_feats + hid_feats, num_classes)

    def forward(self, data):
        UD_x = self.UDrumorGCN(data)
        x = self.fc(UD_x)
        return F.log_softmax(x, dim=-1)
