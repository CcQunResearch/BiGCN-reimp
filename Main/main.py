# -*- coding: utf-8 -*-
# @Time    : 2022/8/23 18:22
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : main.py
# @Software: PyCharm
# @Note    :

# import sys
# import os.path as osp
#
# dirname = osp.dirname(osp.abspath(__file__))
# sys.path.append(osp.join(dirname, '..'))
#
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch.optim import Adam
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import time
# from Main.pargs import pargs
# from Main.model import Net, UDNet
# from Main.dataset import WeiboDataset
# from torch_geometric.data import DataLoader
# from Main.utils import create_log_dict, write_log, write_json
# from Main.word2vec import Embedding, collect_sentences, train_word2vec
# from Main.sort import sort_weibo_dataset, sort_weibo_self_dataset, sort_weibo_2class_dataset
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#
#
# def train(train_loader, model, optimizer, device):
#     model.train()
#     total_loss = 0
#
#     for batch_data in train_loader:
#         optimizer.zero_grad()
#
#         batch_data = batch_data.to(device)
#         out = model(batch_data)
#         loss = F.binary_cross_entropy(out, batch_data.y.to(torch.float32))
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * batch_data.num_graphs
#
#     return total_loss / len(train_loader.dataset)
#
#
# def test(model, dataloader, device):
#     model.eval()
#     error = 0
#
#     y_true = []
#     y_pred = []
#     for data in dataloader:
#         data = data.to(device)
#         pred = model(data)
#         error += F.binary_cross_entropy(pred, data.y.to(torch.float32)).item() * data.num_graphs
#         pred[pred >= 0.5] = 1
#         pred[pred < 0.5] = 0
#         y_true += data.y.tolist()
#         y_pred += pred.tolist()
#     acc = accuracy_score(y_true, y_pred)
#     prec = [precision_score(y_true, y_pred, pos_label=1, average='binary'),
#             precision_score(y_true, y_pred, pos_label=0, average='binary')]
#     rec = [recall_score(y_true, y_pred, pos_label=1, average='binary'),
#            recall_score(y_true, y_pred, pos_label=0, average='binary')]
#     f1 = [f1_score(y_true, y_pred, pos_label=1, average='binary'),
#           f1_score(y_true, y_pred, pos_label=0, average='binary')]
#     return error / len(dataloader.dataset), acc, prec, rec, f1
#
#
# def test_and_log(model, val_loader, test_loader, device, epoch, lr, loss, train_acc, log_record):
#     val_error, val_acc, val_prec, val_rec, val_f1 = test(model, val_loader, device)
#     test_error, test_acc, test_prec, test_rec, test_f1 = test(model, test_loader, device)
#     log_info = 'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation BCE: {:.7f}, Test BCE: {:.7f}, Train ACC: {:.3f}, Validation ACC: {:.3f}, Test ACC: {:.3f}, Test PREC(T/F): {:.3f}/{:.3f}, Test REC(T/F): {:.3f}/{:.3f}, Test F1(T/F): {:.3f}/{:.3f}' \
#         .format(epoch, lr, loss, val_error, test_error, train_acc, val_acc, test_acc, test_prec[0], test_prec[1],
#                 test_rec[0],
#                 test_rec[1], test_f1[0], test_f1[1])
#
#     log_record['val accs'].append(round(val_acc, 3))
#     log_record['test accs'].append(round(test_acc, 3))
#     log_record['test prec T'].append(round(test_prec[0], 3))
#     log_record['test prec F'].append(round(test_prec[1], 3))
#     log_record['test rec T'].append(round(test_rec[0], 3))
#     log_record['test rec F'].append(round(test_rec[1], 3))
#     log_record['test f1 T'].append(round(test_f1[0], 3))
#     log_record['test f1 F'].append(round(test_f1[1], 3))
#     return val_error, log_info, log_record
#
#
# if __name__ == '__main__':
#     args = pargs()
#
#     dataset = args.dataset
#     vector_size = args.vector_size
#     device = args.gpu if args.cuda else 'cpu'
#     runs = args.runs
#     k = args.k
#
#     m = args.model
#     batch_size = args.batch_size
#     epochs = args.epochs
#     diff_lr = args.diff_lr
#
#     label_source_path = osp.join(dirname, '..', 'Data', dataset, 'source')
#     label_dataset_path = osp.join(dirname, '..', 'Data', dataset, 'dataset')
#     train_path = osp.join(label_dataset_path, 'train')
#     val_path = osp.join(label_dataset_path, 'val')
#     test_path = osp.join(label_dataset_path, 'test')
#     model_path = osp.join(dirname, '..', 'Model', f'w2v_{dataset}_{vector_size}.model')
#
#     log_name = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(time.time()))
#     log_path = osp.join(dirname, '..', 'Log', f'{log_name}.log')
#     log_json_path = osp.join(dirname, '..', 'Log', f'{log_name}.json')
#
#     log = open(log_path, 'w')
#     log_dict = create_log_dict(args)
#
#     if not osp.exists(model_path):
#         if dataset == 'Weibo':
#             sort_weibo_dataset(label_source_path, label_dataset_path)
#         elif 'DRWeibo' in dataset:
#             sort_weibo_2class_dataset(label_source_path, label_dataset_path)
#
#         sentences = collect_sentences(label_dataset_path)
#         w2v_model = train_word2vec(sentences, vector_size)
#         w2v_model.save(model_path)
#
#     for run in range(runs):
#         write_log(log, f'run:{run}')
#         log_record = {'run': run, 'val accs': [], 'test accs': [], 'test prec T': [], 'test prec F': [],
#                       'test rec T': [], 'test rec F': [], 'test f1 T': [], 'test f1 F': []}
#
#         word2vec = Embedding(model_path)
#
#         if dataset == 'Weibo':
#             sort_weibo_dataset(label_source_path, label_dataset_path, k_shot=k)
#         elif 'DRWeibo' in dataset:
#             sort_weibo_2class_dataset(label_source_path, label_dataset_path, k_shot=k)
#
#         train_dataset = WeiboDataset(train_path, word2vec, tddroprate=args.tddroprate, budroprate=args.budroprate)
#         val_dataset = WeiboDataset(val_path, word2vec)
#         test_dataset = WeiboDataset(test_path, word2vec)
#
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size)
#         test_loader = DataLoader(test_dataset, batch_size=batch_size)
#
#         model = Net(vector_size, args.hid_feats, args.out_feats).to(device) if m == 'bigcn' else UDNet(vector_size,
#                                                                                                        args.hid_feats,
#                                                                                                        args.out_feats).to(
#             device)
#         if diff_lr:
#             BU_params = list(map(id, model.BUrumorGCN.conv1.parameters()))
#             BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))
#             base_params = filter(lambda p: id(p) not in BU_params, model.parameters())
#             optimizer = Adam([
#                 {'params': base_params},
#                 {'params': model.BUrumorGCN.conv1.parameters(), 'lr': args.lr / 5},
#                 {'params': model.BUrumorGCN.conv2.parameters(), 'lr': args.lr / 5}
#             ], lr=args.lr, weight_decay=args.weight_decay)
#         else:
#             optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#
#         scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001)
#
#         val_error, log_info, log_record = test_and_log(model, val_loader, test_loader,
#                                                        device, 0, args.lr, 0, 0, log_record)
#         write_log(log, log_info)
#
#         for epoch in range(1, epochs + 1):
#             lr = scheduler.optimizer.param_groups[0]['lr']
#             _ = train(train_loader, model, optimizer, device)
#
#             train_error, train_acc, _, _, _ = test(model, train_loader, device)
#             val_error, log_info, log_record = test_and_log(model, val_loader, test_loader,
#                                                            device, epoch, lr, train_error, train_acc,
#                                                            log_record)
#             write_log(log, log_info)
#
#             if not diff_lr:
#                 scheduler.step(val_error)
#
#         log_record['mean acc'] = round(np.mean(log_record['test accs'][-10:]), 3)
#         write_log(log, '')
#
#         log_dict['record'].append(log_record)
#         write_json(log_dict, log_json_path)

import sys
import os
import os.path as osp
import warnings

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
dirname = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(dirname, '..'))

import numpy as np
import time
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from Main.pargs import pargs
from Main.dataset import TreeDataset
from Main.word2vec import Embedding, collect_sentences, train_word2vec
from Main.sort import sort_dataset
from Main.model import Net, UDNet
from Main.utils import create_log_dict, write_log, write_json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train(train_loader, model, optimizer, device):
    model.train()
    total_loss = 0

    for batch_data in train_loader:
        optimizer.zero_grad()

        batch_data = batch_data.to(device)
        out = model(batch_data)
        loss = F.nll_loss(out, batch_data.y.long().view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_data.num_graphs

    return total_loss / len(train_loader.dataset)


def test(model, dataloader, num_classes, device):
    model.eval()
    error = 0

    y_true = []
    y_pred = []

    for data in dataloader:
        data = data.to(device)
        pred = model(data)
        error += F.nll_loss(pred, data.y.long().view(-1)).item() * data.num_graphs
        y_true += data.y.tolist()
        y_pred += pred.max(1).indices.tolist()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = round(accuracy_score(y_true, y_pred), 4)
    precs = []
    recs = []
    f1s = []
    for label in range(num_classes):
        precs.append(round(precision_score(y_true == label, y_pred == label, labels=True), 4))
        recs.append(round(recall_score(y_true == label, y_pred == label, labels=True), 4))
        f1s.append(round(f1_score(y_true == label, y_pred == label, labels=True), 4))
    micro_p = round(precision_score(y_true, y_pred, labels=range(num_classes), average='micro'), 4)
    micro_r = round(recall_score(y_true, y_pred, labels=range(num_classes), average='micro'), 4)
    micro_f1 = round(f1_score(y_true, y_pred, labels=range(num_classes), average='micro'), 4)

    macro_p = round(precision_score(y_true, y_pred, labels=range(num_classes), average='macro'), 4)
    macro_r = round(recall_score(y_true, y_pred, labels=range(num_classes), average='macro'), 4)
    macro_f1 = round(f1_score(y_true, y_pred, labels=range(num_classes), average='macro'), 4)
    return error / len(dataloader.dataset), acc, precs, recs, f1s, \
           [micro_p, micro_r, micro_f1], [macro_p, macro_r, macro_f1]


def test_and_log(model, val_loader, test_loader, num_classes, device, epoch, lr, loss, train_acc, log_record):
    val_error, val_acc, val_precs, val_recs, val_f1s, val_micro_metric, val_macro_metric = \
        test(model, val_loader, num_classes, device)
    test_error, test_acc, test_precs, test_recs, test_f1s, test_micro_metric, test_macro_metric = \
        test(model, test_loader, num_classes, device)
    log_info = 'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Val ERROR: {:.7f}, Test ERROR: {:.7f}\n  Train ACC: {:.4f}, Validation ACC: {:.4f}, Test ACC: {:.4f}\n' \
                   .format(epoch, lr, loss, val_error, test_error, train_acc, val_acc, test_acc) \
               + f'  Test PREC: {test_precs}, Test REC: {test_recs}, Test F1: {test_f1s}\n' \
               + f'  Test Micro Metric(PREC, REC, F1):{test_micro_metric}, Test Macro Metric(PREC, REC, F1):{test_macro_metric}'

    log_record['val accs'].append(val_acc)
    log_record['test accs'].append(test_acc)
    log_record['test precs'].append(test_precs)
    log_record['test recs'].append(test_recs)
    log_record['test f1s'].append(test_f1s)
    log_record['test micro metric'].append(test_micro_metric)
    log_record['test macro metric'].append(test_macro_metric)
    return val_error, log_info, log_record


if __name__ == '__main__':
    args = pargs()

    unsup_train_size = args.unsup_train_size
    dataset = args.dataset
    unsup_dataset = args.unsup_dataset
    vector_size = args.vector_size
    device = args.gpu if args.cuda else 'cpu'
    runs = args.runs
    k = args.k

    word_embedding = 'tfidf' if 'tfidf' in dataset else 'word2vec'
    lang = 'ch' if 'Weibo' in dataset else 'en'
    tokenize_mode = args.tokenize_mode

    split = args.split
    batch_size = args.batch_size
    epochs = args.epochs
    diff_lr = args.diff_lr

    label_source_path = osp.join(dirname, '..', 'Data', dataset, 'source')
    label_dataset_path = osp.join(dirname, '..', 'Data', dataset, 'dataset')
    train_path = osp.join(label_dataset_path, 'train')
    val_path = osp.join(label_dataset_path, 'val')
    test_path = osp.join(label_dataset_path, 'test')
    unlabel_dataset_path = osp.join(dirname, '..', 'Data', unsup_dataset, 'dataset')
    model_path = osp.join(dirname, '..', 'Model',
                          f'w2v_{dataset}_{tokenize_mode}_{unsup_train_size}_{vector_size}.model')

    log_name = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(time.time()))
    log_path = osp.join(dirname, '..', 'Log', f'{log_name}.log')
    log_json_path = osp.join(dirname, '..', 'Log', f'{log_name}.json')

    log = open(log_path, 'w')
    log_dict = create_log_dict(args)

    if not osp.exists(model_path) and word_embedding == 'word2vec':
        sentences = collect_sentences(label_source_path, unlabel_dataset_path, unsup_train_size, lang, tokenize_mode)
        w2v_model = train_word2vec(sentences, vector_size)
        w2v_model.save(model_path)

    for run in range(runs):
        write_log(log, f'run:{run}')
        log_record = {'run': run, 'val accs': [], 'test accs': [], 'test precs': [], 'test recs': [], 'test f1s': [],
                      'test micro metric': [], 'test macro metric': []}

        word2vec = Embedding(model_path, lang, tokenize_mode) if word_embedding == 'word2vec' else None

        sort_dataset(label_source_path, label_dataset_path, k_shot=k, split=split)

        train_dataset = TreeDataset(train_path, word_embedding, word2vec)
        val_dataset = TreeDataset(val_path, word_embedding, word2vec)
        test_dataset = TreeDataset(test_path, word_embedding, word2vec)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        num_classes = train_dataset.num_classes
        model = Net(train_dataset.num_features, args.hid_feats, args.out_feats, train_dataset.num_classes,
                    args.tddroprate, args.budroprate).to(device) if args.model == 'bigcn' else \
            UDNet(vector_size, args.hid_feats, args.out_feats, train_dataset.num_classes).to(device)
        if diff_lr:
            BU_params = list(map(id, model.BUrumorGCN.conv1.parameters()))
            BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))
            base_params = filter(lambda p: id(p) not in BU_params, model.parameters())
            optimizer = Adam([
                {'params': base_params},
                {'params': model.BUrumorGCN.conv1.parameters(), 'lr': args.lr / 5},
                {'params': model.BUrumorGCN.conv2.parameters(), 'lr': args.lr / 5}
            ], lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001)

        val_error, log_info, log_record = test_and_log(model, val_loader, test_loader, num_classes,
                                                       device, 0, args.lr, 0, 0, log_record)
        write_log(log, log_info)

        for epoch in range(1, epochs + 1):
            lr = scheduler.optimizer.param_groups[0]['lr']
            _ = train(train_loader, model, optimizer, device)

            train_error, train_acc, _, _, _, _, _ = test(model, train_loader, num_classes, device)
            val_error, log_info, log_record = test_and_log(model, val_loader, test_loader, num_classes, device, epoch,
                                                           lr, train_error, train_acc, log_record)
            write_log(log, log_info)

            if not diff_lr and split == '622':
                scheduler.step(val_error)

        log_record['mean acc'] = round(np.mean(log_record['test accs'][-10:]), 3)
        write_log(log, '')

        log_dict['record'].append(log_record)
        write_json(log_dict, log_json_path)
