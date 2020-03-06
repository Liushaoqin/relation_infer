import os
from collections import defaultdict

import numpy as np
import yaml
import pkuseg
from sklearn.cluster import KMeans


def parse_rule(path):
    f = open(path, 'r', encoding='utf8')
    y = yaml.load(f)
    stop_word = [x['name'] for x in y['stop_words']]
    return stop_word


def parse_file_name(folder_path_list, stop_word):
    file_name_list = list()
    for folder_path in folder_path_list:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if not file.endswith(".csv"):
                    continue
                for sw in stop_word:
                    if sw in file:
                        continue
                new_file_name = file.split(".")[0]
                new_file_name = new_file_name.split("_")[0]
                file_name_list.append(new_file_name)

    return file_name_list


def file_name_parser(file_name_list):
    names = list()
    word_embeddings = list()
    word_embedding_dict = dict()
    with open("sgns.renmin.word", 'r', encoding='utf8') as embedding_file:
        embedding_file.readline()
        for line in embedding_file.readlines():
            ee = line.strip("\n").strip(" ").split(" ")
            embed = np.array(ee[1:])
            word_embedding_dict[ee[0]] = embed.astype(float)
        embedding_file.close()

    seg = pkuseg.pkuseg()
    for name in file_name_list:
        words = seg.cut(name)
        file_word_embedding_list = list()
        flag = False
        for word in words:
            if word not in word_embedding_dict.keys():
                file_word_embedding_list.append(np.zeros(300))
                continue
            file_word_embedding_list.append(word_embedding_dict[word])
            flag = True
        # if len(file_word_embedding_list) > 0:
        if flag:
            names.append(name)
            word_embeddings.append(file_word_embedding_list)
    return names, word_embeddings


def parse_word_embeddings(embeddings_list):
    embeddings = list()
    for ll in embeddings_list:
        aa = np.zeros(300)
        count = 0
        for array in ll:
            assert np.shape(array) == (300,)
            aa += array
            count += 1
        embeddings.append(aa/count)
    return embeddings


def clustering(embedding_list, names):
    Kmeans = KMeans(n_clusters=10).fit(embedding_list)
    y_pred = Kmeans.labels_
    clusters = list()
    for i in range(10):
        temp = list()
        clusters.append(temp)

    for i in range(len(names)):
        clusters[y_pred[i]].append(names[i])

    return clusters


address = {"a", "b"}
for n in address:
    n = n.replace(" ", "").replace("A", "").replace("B", "")
    if len(n) < 4:
        continue
    if "市" in n:
        if n[2] == "市":
            address.add(n[:3])
        if n[3] == "市":
            address.add(n[:4])

    if "区" in n:
        if n[2] == "区":
            address.add(n[:3])
        if n[3] == "区":
            address.add(n[:4])

    if "县" in n:
        if n[2] == "县":
            address.add(n[:3])
        if n[3] == "县":
            address.add(n[:4])

if __name__ == '__main__':
    stop_words = parse_rule("../rule.yml")
    file_names = parse_file_name("", stop_words)
    a = set()
    a.add(1)