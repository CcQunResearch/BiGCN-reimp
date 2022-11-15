# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 15:41
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : utils.py
# @Software: PyCharm
# @Note    :
import json
import os
import shutil
import jieba
import nltk
from nltk.tokenize import MWETokenizer

mwe_tokenizer = MWETokenizer([('<', '@', 'user', '>'), ('<', 'url', '>')], separator='')


def word_tokenizer(sentence, lang='en', mode='naive'):
    if lang == 'en':
        if mode == 'nltk':
            return mwe_tokenizer.tokenize(nltk.word_tokenize(sentence))
        elif mode == 'naive':
            return sentence.split()
    if lang == 'ch':
        if mode == 'jieba':
            return jieba.lcut(sentence)
        elif mode == 'naive':
            return sentence


def write_json(dict, path):
    with open(path, 'w', encoding='utf-8') as file_obj:
        json.dump(dict, file_obj, indent=4, ensure_ascii=False)


def write_post(post_list, path):
    for post in post_list:
        write_json(post[1], os.path.join(path, f'{post[0]}.json'))


def write_log(log, str):
    log.write(f'{str}\n')
    log.flush()


def dataset_makedirs(dataset_path):
    train_path = os.path.join(dataset_path, 'train', 'raw')
    val_path = os.path.join(dataset_path, 'val', 'raw')
    test_path = os.path.join(dataset_path, 'test', 'raw')

    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    os.makedirs(train_path)
    os.makedirs(val_path)
    os.makedirs(test_path)
    os.makedirs(os.path.join(dataset_path, 'train', 'processed'))
    os.makedirs(os.path.join(dataset_path, 'val', 'processed'))
    os.makedirs(os.path.join(dataset_path, 'test', 'processed'))

    return train_path, val_path, test_path


def create_log_dict(args):
    log_dict = {}
    log_dict['dataset'] = args.dataset
    log_dict['unsup_dataset'] = args.unsup_dataset
    log_dict['tokenize_mode'] = args.tokenize_mode
    log_dict['unsup_train_size'] = args.unsup_train_size

    log_dict['vector_size'] = args.vector_size
    log_dict['runs'] = args.runs

    log_dict['model'] = args.model
    log_dict['batch_size'] = args.batch_size
    log_dict['tddroprate'] = args.tddroprate
    log_dict['budroprate'] = args.budroprate
    log_dict['hid_feats'] = args.hid_feats
    log_dict['out_feats'] = args.out_feats

    log_dict['diff_lr'] = args.diff_lr
    log_dict['lr'] = args.lr
    log_dict['epochs'] = args.epochs
    log_dict['weight_decay'] = args.weight_decay

    log_dict['k'] = args.k

    log_dict['record'] = []
    return log_dict
