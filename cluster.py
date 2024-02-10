import torch
import json, time
from tqdm import tqdm


def generate_clusters(ideals, raw_clusters, classes_count):

    if raw_clusters < classes_count:
        norms_count = raw_clusters
    else:
        norms_count = min(raw_clusters, classes_count * (classes_count - 1) * 0.5)

    norms_res = torch.empty((norms_count, 3)).fill_(1e+10).cuda()

    for i in tqdm(range(classes_count - 1)):
        p0 = ideals[0:classes_count - i - 1, :]
        p1 = ideals[i + 1:classes_count, :]

        norms = torch.sqrt(torch.sum((p0 - p1) ** 2, dim=1)).cuda()
        i_tmp = torch.arange(0, classes_count - i - 1).cuda()
        j_tmp = torch.arange(i + 1, classes_count).cuda()

        norms = torch.stack((norms, i_tmp, j_tmp), dim=1)

        norms_res = torch.cat((norms_res, norms))
        _, indices = torch.topk(norms_res[:, 0], k=norms_count, largest=False)
        norms_res = norms_res[indices]


    # norms_count = None
    # if raw_clusters < classes_count:
    #     norms_count = raw_clusters
    # else:
    #     norms_count = min(raw_clusters, classes_count * (classes_count - 1) * 0.5)
    #
    # norms_res = torch.empty(norms_count).fill_(1e+10).cuda()
    # i_res = torch.empty(norms_count).fill_(1e+10).cuda()
    # j_res = torch.empty(norms_count).fill_(1e+10).cuda()
    # norms_res = torch.stack((norms_res, i_res, j_res), dim=1)
    #

    # for i in tqdm(range(classes_count - 1)):
    #     p0 = ideals[0:classes_count - i - 1, :]
    #     p1 = ideals[i + 1:classes_count, :]
    #
    #     norms = torch.sqrt(torch.sum((p0 - p1) ** 2, dim=1)).cuda()
    #     i_tmp = torch.arange(0, classes_count - i - 1).cuda()
    #     j_tmp = torch.arange(i + 1, classes_count).cuda()
    #
    #     norms = torch.stack((norms, i_tmp, j_tmp), dim=1)
    #
    #     norms_res = torch.cat((norms_res, norms))
    #     norms_res = torch.stack(sorted(norms_res, key=lambda x: x[0]))[:norms_count]



    return norms_res


def merge_clusters(norms_res, merged_clusters, cluster_max_size):
    # merged_clusters = []

    for i in tqdm(range(norms_res.size()[0])):
        class1, class2 = int(norms_res[i][1]), int(norms_res[i][2])

        found_class1, found_class2 = False, False
        idx1, idx2 = None, None

        for idx, cluster in enumerate(merged_clusters):
            if class1 in cluster:
                found_class1 = True
                idx1 = idx
            if class2 in cluster:
                found_class2 = True
                idx2 = idx

        if found_class1 and not found_class2:
            merged_clusters[idx1].append(class2)
        elif not found_class1 and found_class2:
            merged_clusters[idx2].append(class1)
        elif not found_class1 and not found_class2:
            merged_clusters.append([class1, class2])
        elif found_class1 and found_class2 and idx1 != idx2:
            if (len(merged_clusters[idx1]) + len(merged_clusters[idx2]))< cluster_max_size:
                merged_clusters[idx1].extend(merged_clusters[idx2])
                merged_clusters.pop(idx2)

    # return merged_clusters


def merge_clusters111(norms_res, clusters):
    clusters = []
    class1, class2 = None, None
    for i in tqdm(range(norms_res.size()[0])):
        class1 = norms_res[i][1]
        class2 = norms_res[i][2]

        first, second = False, False
        id1, id2 = None, None
        for j in range(len(clusters)):
            if class1 in clusters[j]:
                first = True
                id1 = j
            if class2 in clusters[j]:
                second = True
                id2 = j

        if first and not second:
            clusters[id1].append(class2)
        elif not first and second:
            clusters[id2].append(class1)
        elif not first and not second:
            clusters.append([class1, class2])
        elif first and second:
            if id1 != id2:
                new_cluster = clusters[id1] + clusters[id2]
                if id2 < id1:
                    del clusters[id1]
                    del clusters[id2]
                if id1 < id2:
                    del clusters[id2]
                    del clusters[id1]
                clusters.append(new_cluster)


def save_clusters(out_path, clusters, alphabet):
    visible_clusters = []
    for cluster in clusters:
        visible_cluster = []
        for sym in cluster:
            visible_cluster.append(alphabet[sym])
        visible_clusters.append(visible_cluster)

    with open(out_path, 'w', encoding='utf8') as json_out:
        json.dump(visible_clusters, json_out, indent=2, ensure_ascii=False)
