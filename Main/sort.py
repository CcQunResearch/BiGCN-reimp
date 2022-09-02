# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 15:40
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : sort.py
# @Software: PyCharm
# @Note    :
import os
import json
import random
import time
from Main.utils import write_post, dataset_makedirs


def sort_weibo_dataset(source_path, dataset_path, k_shot=10000):
    post_id_list = []
    post_label_list = []
    all_post = []

    label_path = os.path.join(source_path, 'Weibo.txt')
    train_path, val_path, test_path = dataset_makedirs(dataset_path)

    f = open(label_path, 'r', encoding='utf-8')
    post_list = f.readlines()
    for post in post_list:
        post_id_list.append(post.split()[0].strip()[4:])
        post_label_list.append(int(post.split()[1].strip()[-1]))

    for i, post_id in enumerate(post_id_list):
        reverse_dict = {}
        comment_index = 0
        comment_list = []

        post_path = os.path.join(source_path, 'post', f'{post_id}.json')
        post = json.load(open(post_path, 'r', encoding='utf-8'))
        source = {
            'content': post[0]['text'],
            'user id': post[0]['uid'],
            'tweet id': post[0]['mid'],
            'label': post_label_list[i]
        }

        for j in range(1, len(post)):
            comment_list.append({'comment id': comment_index, 'parent': -2, 'children': []})
            reverse_dict[post[j]['mid']] = comment_index
            comment_index += 1
        for k in range(1, len(post)):
            comment_list[k - 1]['content'] = post[k]['text']
            comment_list[k - 1]['user id'] = post[k]['uid']
            comment_list[k - 1]['user name'] = post[k]['username']
            if post[k]['parent'] == source['tweet id']:
                comment_list[k - 1]['parent'] = -1
            else:
                parent_index = reverse_dict[post[k]['parent']]
                comment_list[k - 1]['parent'] = parent_index
                comment_list[parent_index]['children'].append(k - 1)
        all_post.append((post_id, {'source': source, 'comment': comment_list}))

    random.seed(time.time())
    random.shuffle(all_post)

    train_post = []
    positive_num = 0
    negative_num = 0
    for post in all_post[:int(len(all_post) * 0.6)]:
        if post[1]['source']['label'] == 1 and positive_num != k_shot:
            train_post.append(post)
            positive_num += 1
        if post[1]['source']['label'] == 0 and negative_num != k_shot:
            train_post.append(post)
            negative_num += 1
        if positive_num == k_shot and negative_num == k_shot:
            break
    val_post = all_post[int(len(all_post) * 0.6):int(len(all_post) * 0.8)]
    test_post = all_post[int(len(all_post) * 0.8):]
    write_post(train_post, train_path)
    write_post(val_post, val_path)
    write_post(test_post, test_path)


def sort_weibo_self_dataset(label_source_path, label_dataset_path, unlabel_dataset_path, k_shot=10000):
    train_path, val_path, test_path = dataset_makedirs(label_dataset_path)

    years = ['2020', '2021', '2022']
    label_file_paths = []
    for year in years:
        path = os.path.join(label_source_path, year)
        for filename in os.listdir(path):
            label_file_paths.append(os.path.join(path, filename))

    unlabel_source_path = os.path.join(unlabel_dataset_path, 'raw')
    unlabel_file_paths = []
    all_unlabel_filenames = os.listdir(unlabel_source_path)
    random.seed(1234)
    random.shuffle(all_unlabel_filenames)
    for i in range(len(label_file_paths)):
        unlabel_file_paths.append(os.path.join(unlabel_source_path, all_unlabel_filenames[i]))

    all_post = []
    for filepath in label_file_paths:
        post = json.load(open(filepath, 'r', encoding='utf-8'))
        all_post.append((post['source']['tweet id'], post))
    for filepath in unlabel_file_paths:
        post = json.load(open(filepath, 'r', encoding='utf-8'))
        post['source']['label'] = 0
        all_post.append((post['source']['tweet id'], post))

    random.seed(time.time())
    random.shuffle(all_post)
    train_post = []
    positive_num = 0
    negative_num = 0
    for post in all_post[:int(len(all_post) * 0.6)]:
        if post[1]['source']['label'] == 1 and positive_num != k_shot:
            train_post.append(post)
            positive_num += 1
        if post[1]['source']['label'] == 0 and negative_num != k_shot:
            train_post.append(post)
            negative_num += 1
        if positive_num == k_shot and negative_num == k_shot:
            break
    val_post = all_post[int(len(all_post) * 0.6):int(len(all_post) * 0.8)]
    test_post = all_post[int(len(all_post) * 0.8):]
    write_post(train_post, train_path)
    write_post(val_post, val_path)
    write_post(test_post, test_path)


def sort_weibo_2class_dataset(label_source_path, label_dataset_path, k_shot=10000):
    train_path, val_path, test_path = dataset_makedirs(label_dataset_path)

    label_file_paths = []
    for filename in os.listdir(label_source_path):
        label_file_paths.append(os.path.join(label_source_path, filename))

    all_post = []
    for filepath in label_file_paths:
        post = json.load(open(filepath, 'r', encoding='utf-8'))
        all_post.append((post['source']['tweet id'], post))

    random.seed(time.time())
    random.shuffle(all_post)
    train_post = []
    positive_num = 0
    negative_num = 0
    for post in all_post[:int(len(all_post) * 0.6)]:
        if post[1]['source']['label'] == 1 and positive_num != k_shot:
            train_post.append(post)
            positive_num += 1
        if post[1]['source']['label'] == 0 and negative_num != k_shot:
            train_post.append(post)
            negative_num += 1
        if positive_num == k_shot and negative_num == k_shot:
            break
    val_post = all_post[int(len(all_post) * 0.6):int(len(all_post) * 0.8)]
    test_post = all_post[int(len(all_post) * 0.8):]
    write_post(train_post, train_path)
    write_post(val_post, val_path)
    write_post(test_post, test_path)
