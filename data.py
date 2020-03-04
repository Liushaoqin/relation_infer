import re
from collections import defaultdict

import pkuseg
import os
import numpy as np
import pickle
import pkuseg

import param as P

def load_dataset(path):
    label2data, label2name = get_file(path)

    # sample train data
    train_support_tables = list()
    train_support_labels = list()
    train_target_tables = list()
    train_target_labels = list()
    for i in range(2000):
        p = np.array([len(label2data[k]) for k in label2data.keys()])
        p = p / sum(p)
        labels = [i for i in range(0, len(label2data.keys()))]
        choosen_labels = np.random.choice(labels, size=5, p=p)

        support_table = np.array([np.random.choice(label2data[k][:len(label2data[k]) * 0.7]) for k in choosen_labels])
        support_label = choosen_labels

        target_label = np.random.choice(choosen_labels)
        target_table = np.random.choice(label2data[target_label][:len(label2data[target_label]) * 0.7])
        train_support_labels.append(support_label)
        train_support_tables.append(support_table)
        train_target_labels.append(target_label)
        train_target_tables.append(target_table)

    # sample test data
    test_support_tables = list()
    test_support_labels = list()
    test_target_tables = list()
    test_target_labels = list()
    for i in range(1000):
        p = np.array([len(label2data[k]) for k in label2data.keys()])
        p = p / sum(p)
        labels = [i for i in range(0, len(label2data.keys()))]
        choosen_labels = np.random.choice(labels, size=5, p=p)

        support_table = np.array([np.random.choice(label2data[k][len(label2data[k]) * 0.7:]) for k in choosen_labels])
        support_label = choosen_labels

        target_label = np.random.choice(choosen_labels)
        target_table = np.random.choice(label2data[target_label][len(label2data[target_label]) * 0.7:])
        test_support_labels.append(support_label)
        test_support_tables.append(support_table)
        test_target_labels.append(target_label)
        test_target_tables.append(target_table)

    with open('dataset.pkl', 'wb') as f:
        pickle.dump(f, train_support_tables)
        pickle.dump(f, train_support_labels)
        pickle.dump(f, train_target_tables)
        pickle.dump(f, train_target_labels)
        pickle.dump(f, test_support_tables)
        pickle.dump(f, test_support_labels)
        pickle.dump(f, test_target_tables)
        pickle.dump(f, test_target_labels)
        f.close()


def get_embeddings(embedding_file_path):
    embedding_file = open(embedding_file_path, 'r', encoding='utf-8')
    lines = embedding_file.readlines()
    embedding_dict = {}
    for line in lines:
        allen = line.split(' ')
        embedding = np.zeros(200)
        if len(allen) < 201:
            continue
        for i in range(1, 201):
            embedding[i - 1] = float(allen[i])
        embedding_dict[allen[0]] = embedding

    embedding_file.close()
    return embedding_dict


def preprocess_data(table_name, embedding_dict, sequence_len):
    seg = pkuseg.pkuseg()
    # seg = pkuseg.pkuseg(model_name='news')
    text = seg.cut(table_name)
    embeddings = []
    for w in text:
        if w not in embedding_dict.keys():
            continue
        embeddings.append(embedding_dict[w])

    if sequence_len > len(text):
        padding = [np.zeros(200) for i in range(0, sequence_len - len(text))]
        embeddings.extend(padding)

    return np.array(embeddings)


def gen_dataset(pickle_path, embedding_file_path):
    embedding_dict = get_embeddings(embedding_file_path)

    with open(pickle_path, 'rb') as f:
        train_support_tables = pickle.load(f)
        train_support_labels = pickle.load(f)
        train_target_tables = pickle.load(f)
        train_target_labels = pickle.load(f)
        test_support_tables = pickle.load(f)
        test_support_labels = pickle.load(f)
        test_target_tables = pickle.load(f)
        test_target_labels = pickle.load(f)
        f.close()

    parse_train_support_tables = list()
    parse_train_target_tables = list()

    parse_test_support_tables = list()
    parse_test_target_tables = list()

    for tts in train_support_tables:
        temp = []
        for table in tts:
            embedding = preprocess_data(table, embedding_dict, P.sequence_len)
            temp.append(embedding)
        parse_train_support_tables.append(temp)

    for table in train_target_tables:
        embedding = preprocess_data(table, embedding_dict, P.sequence_len)
        parse_train_target_tables.append(embedding)

    for tts in test_support_tables:
        temp = []
        for table in tts:
            embedding = preprocess_data(table, embedding_dict, P.sequence_len)
            temp.append(embedding)
        parse_test_support_tables.append(temp)

    for table in test_target_tables:
        embedding = preprocess_data(table, embedding_dict, P.sequence_len)
        parse_test_target_tables.append(embedding)

    return parse_train_support_tables, train_support_labels, parse_train_target_tables, train_target_labels, \
           parse_test_support_tables, test_support_labels, parse_test_target_tables, test_target_labels


def get_train_batches():
    pass


def get_test_batches():
    pass


def get_file(path):
    label2name = {}
    label2data = {}
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            label = len(label2name.keys())
            label2name[label] = filename
            with open(path + "/" + filename, 'r', encoding='utf8') as f:
                label2data[label] = [filter_punc(line.strip("\n")) for line in f.readlines()]
                f.close()

    return label2data, label2name


def filter_punc(table_name):
    return re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。“”《》？、~@#￥%……&*（）]+".encode('utf-8').decode('utf-8'),
                  "".encode('utf-8').decode('utf-8'), table_name)


def get_train_data():
    label2data, label2name = get_file("../data/")
    table_names = []
    labels = []
    for k in label2data:
        for t in label2data[k]:
            table_names.append(t)
            labels.append(k)

    embedding_dict_path = '/home1/lsq/Tencent_AILab_ChineseEmbedding.txt'
    embedding_dict = get_embeddings(embedding_dict_path)

    table_name_embedding = []
    seg = pkuseg.pkuseg(model_name='news')
    for t in table_names:
        text = seg.cut(t)
        embeddings = []
        for w in text:
            if w not in embedding_dict.keys():
                continue
            embeddings.append(embedding_dict[w])

        while len(embeddings) < 25:
            embeddings.append(np.zeros(200))
        embed = np.array(embeddings[:25])
        table_name_embedding.append(embed)

    filter_index = []

    for i in range(len(table_name_embedding) - 1, 0, -1):
        e = table_name_embedding[i][0]
        if (e == np.zeros(200)).all():
            filter_index.append(i)

    for i in filter_index:
        print(table_names[i])
        del (table_names[i])
        del (table_name_embedding[i])
        del (labels[i])

    index = [i for i in range(len(table_name_embedding))]

    train_index = np.random.choice(index, int(len(index) * 0.7))

    test_index = []
    for i in index:
        if i not in train_index:
            test_index.append(i)

    train_table = []
    train_label = []
    test_table = []
    test_label = []

    for i in train_index:
        train_table.append(table_name_embedding[i])
        train_label.append(labels[i])

    for i in test_index:
        test_table.append(table_name_embedding[i])
        test_label.append(labels[i])

    return train_table, train_label, test_table, test_label