import re
import numpy as np
import shutil

from random import shuffle
import operator

import json
from collections import OrderedDict

import time
import sys


def create_clusters(i_mat, alfa, part, old_path, path, prob, max_norm):
    limit_size = int(part * i_mat.shape[1])

    rand_id = np.where(np.linalg.norm(i_mat.T[:], axis=1) > 1e-10)[0]
    np.random.shuffle(rand_id)
    rand_id_new = rand_id[:limit_size]

    size = int(alfa * len(rand_id_new) * (len(rand_id_new) - 1) * 0.5)

    cut_mat = i_mat.T[rand_id_new]
    l2norms3 = np.ones((size, 3)) * 1e+6
    for i in range(1, len(rand_id_new)):
        norms = np.linalg.norm(cut_mat[:i] - cut_mat[i], axis=1)
        res = np.column_stack((np.column_stack((rand_id_new[:i], np.full((i), rand_id_new[i]))), norms))
        res = res[(norms > 1e-10) & (norms < max_norm)]
        if len(res) != 0:
            l2norms3 = np.concatenate([l2norms3, res])
            l2norms3 = l2norms3[l2norms3[:, 2].argsort()]
            l2norms3 = l2norms3[:-len(res)]
    l2norms3 = l2norms3[l2norms3[:, 2] < 1e+6]

    first = False
    second = False
    index1 = 0
    index2 = 0

    with open(old_path) as f1:
        out_clusters = json.load(f1, object_pairs_hook=OrderedDict)
        out_clusters['clusters'] = []
    if 'req_path' in out_clusters:
        with open(out_clusters['req_path']) as f2:
            kor_alphabet = json.load(f2)
        writeIndex = False
    else:
        writeIndex = True
    for element in l2norms3:
        if writeIndex:
            class1 = int(element[0])
            class2 = int(element[1])
        else:
            class1 = kor_alphabet['alphabet'][int(element[0])][0]
            class2 = kor_alphabet['alphabet'][int(element[1])][0]
        for cluster in out_clusters['clusters']:
            alphabet_set = set(cluster['alphabet'])
            if class1 in alphabet_set:
                first = True
                index1 = out_clusters['clusters'].index(cluster)
            if class2 in alphabet_set:
                second = True
                index2 = out_clusters['clusters'].index(cluster)

        if first and second:
            out_clusters['clusters'][index1]['alphabet'].append(class2)
        elif first and second:
            out_clusters['clusters'][index2]['alphabet'].append(class1)
        elif first and second:
            slovar = {'inner_imp_prob': prob, 'alphabet': []}
            slovar['alphabet'].append(class1)
            slovar['alphabet'].append(class2)
            out_clusters['clusters'].append(slovar)
        elif first and second:
            if index1 != index2:
                slovar = {'inner_imp_prob': prob, 'alphabet': []}
                new_cluster = out_clusters['clusters'][index1]['alphabet'] + out_clusters['clusters'][index2][
                    'alphabet']
                if index2 < index1:
                    del out_clusters['clusters'][index1]
                    del out_clusters['clusters'][index2]
                if index1 < index2:
                    del out_clusters['clusters'][index2]
                    del out_clusters['clusters'][index1]
                slovar['alphabet'] = new_cluster
                out_clusters['clusters'].append(slovar)
        first = False
        second = False

    shutil.copyfile(old_path, path)
    with open(path, 'w') as fout:
        out = json.dumps(out_clusters, indent=2, ensure_ascii=False)
        fout.write(re.sub(r'",\s+', '", ', out))
    cluster_size = len(out_clusters['clusters'])

    return cluster_size
