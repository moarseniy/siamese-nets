import os
import os.path as op
import time

import torch

from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import ujson as json

from tqdm import tqdm

from torch.utils.data import Dataset
from torchvision.utils import save_image
from torchvision.io import read_image
from torchvision.io import ImageReadMode
import torchvision
import random

from PIL import Image

from mining_methods import generate_clusters
from mining_methods import merge_clusters
from mining_methods import save_clusters

from mining_methods import generate_sym_probs
from mining_methods import save_sym_probs


def ChooseDataset(dataset_type: str, cfg: dict, transforms: torchvision.transforms) -> Dataset:
    mode = cfg['batch_settings']['type']
    if cfg[dataset_type]["name"] == "KorSynthetic":
        if mode == 'triplet':
            return KorSyntheticTriplet(dataset_type=dataset_type, cfg=cfg, transforms=transforms)
        elif mode == 'pair':
            return KorSyntheticPair(dataset_type=dataset_type, cfg=cfg, transforms=transforms)
    elif cfg[dataset_type]["name"] == "Omniglot":
        if mode == 'triplet':
            return OmniglotTriplet(dataset_type=dataset_type, cfg=cfg, transforms=transforms)
        elif mode == 'pair':
            return OmniglotPair(dataset_type=dataset_type, cfg=cfg, transforms=transforms)
    elif cfg[dataset_type]["name"] == "OmniglotOneShot":
        return OmniglotOneShot(dataset_type=dataset_type, cfg=cfg)
    elif cfg[dataset_type]["name"] == "PHD08ValidDataset":
        return PHD08ValidDataset(dataset_type=dataset_type, cfg=cfg)
    else:
        print("Dataset name is not defined!!!")
        exit(-1)


def prepare_alph_old(alph_pt: str) -> list:
    alphabet = []
    for a in json.load(open(alph_pt, "r"))["alphabet"]:
        alphabet.append(a[0])
    return alphabet


def prepare_alph(alph_pt: str) -> (list, dict):
    with open(alph_pt, "r") as alph_f:
        alph = json.load(alph_f)["alphabet"]
    alph = [i[0] for i in alph[0]]
    # alph.append("NONE")
    alph_dict = {alph[i]: i for i in range(len(alph))}
    return alph, alph_dict


class SiameseDataset:
    def __init__(self):
        self.files_per_classes = []

    def choose_positive_random(self):
        pos_c = np.random.randint(len(self.files_per_classes))
        pos_id = np.random.randint(len(self.files_per_classes[pos_c]))
        return pos_c, pos_id

    def choose_positive_symprobs(self):
        pos_c = np.random.choice(len(self.files_per_classes), 1, p=self.probs_vec)
        pos_id = np.random.randint(len(self.files_per_classes[pos_c]))
        return pos_c, pos_id

    def create_positive(self, pos_c, pos_id):
        anc_id = np.random.randint(len(self.files_per_classes[pos_c]))
        while anc_id == pos_id:
            anc_id = np.random.randint(len(self.files_per_classes[pos_c]))
        return anc_id

    def create_negative_random(self, pos_c):
        neg_c = np.random.randint(len(self.files_per_classes))
        while pos_c == neg_c:
            neg_c = np.random.randint(len(self.files_per_classes))
        neg_id = np.random.randint(len(self.files_per_classes[neg_c]))
        return neg_c, neg_id

    def create_negative_clusters(self, pos_c):
        neg_c, neg_id = None, None
        if len(self.clusters) > 0:
            for cluster in self.clusters:
                if pos_c in cluster and random.random() < self.inner_imp_prob:
                    neg_c = cluster[np.random.randint(len(cluster))]

                    while pos_c == neg_c:
                        neg_c = cluster[np.random.randint(len(cluster))]
                    neg_id = np.random.randint(len(self.files_per_classes[neg_c]))

                    break

            if neg_c is None and neg_id is None:
                neg_c, neg_id = self.create_negative_random(pos_c)
        else:
            neg_c, neg_id = self.create_negative_random(pos_c)
        return neg_c, neg_id

    def get_alph_size(self) -> int:
        return len(self.files_per_classes)

    def update_rules(self, ideals, counter, dists, ep_save_pt, config, e):

        if config['batch_settings']['negative_mode'] == 'auto_clusters' and \
                e % config["batch_settings"]["make_clust_on_ep"] == 0:
            generation_time = time.time()

            norms_res = generate_clusters(ideals, self.raw_clusters, len(self.alphabet))

            print('Generation clusters time: {:.2f} sec'.format(time.time() - generation_time))

            merge_time = time.time()

            self.clusters = []
            merge_clusters(norms_res, self.clusters, self.cluster_max_size)

            print('Merge clusters time: {:.2f} sec, Total clusters size: {}'.format(time.time() - merge_time, len(self.clusters)))

            save_clusters(os.path.join(ep_save_pt, 'clusters.json'), self.clusters, self.alphabet)

        if config['batch_settings']['positive_mode'] == 'auto_sym_probs':
            sym_probs_time = time.time()

            sym_probs_gamma = config['batch_settings']['sym_probs_gamma']
            merge_w = config['batch_settings']['merge_w']

            generate_sym_probs(dists, ideals, counter, self.probs_vec, merge_w, sym_probs_gamma)

            print('Generation sym_probs time: {:.2f} sec'.format(time.time() - sym_probs_time))

            save_sym_probs(os.path.join(ep_save_pt, 'sym_probs.json'), self.probs_vec, self.alphabet)


class PairDataset(SiameseDataset):
    def __init__(self, *args):
        super().__init__(*args)

    def generatePairs(self):
        pair_ids = None
        if random.uniform(0, 1) < self.gen_imp_ratio:
            pos_c, pos_id1 = self.choose_positive_random()

            pos_id2 = self.create_positive(pos_c, pos_id1)

            pair_ids = [[pos_c, pos_id1], [pos_c, pos_id2]]
        else:
            pos_c, pos_id = None, None
            if self.positive_mode == "auto_sym_probs":
                pos_c, pos_id = self.choose_positive_symprobs()
            else:
                pos_c, pos_id = self.choose_positive_random()

            neg_c, neg_id = None, None
            if self.negative_mode == "auto_clusters":
                neg_c, neg_id = self.create_negative_clusters(pos_c)
            else:
                neg_c, neg_id = self.create_negative_random(pos_c)

            pair_ids = [[pos_c, pos_id], [neg_c, neg_id]]

        return pair_ids


class TripletDataset(SiameseDataset):
    def __init__(self, *args):
        super().__init__(*args)

    def generateTriplets(self):
        pos_c, pos_id = None, None
        if self.positive_mode == "auto_sym_probs":
            pos_c, pos_id = self.choose_positive_symprobs()
        else:
            pos_c, pos_id = self.choose_positive_random()

        anc_id = self.create_positive(pos_c, pos_id)

        neg_c, neg_id = None, None
        if self.negative_mode == "auto_clusters":
            neg_c, neg_id = self.create_negative_clusters(pos_c)
        else:
            neg_c, neg_id = self.create_negative_random(pos_c)

        triplet_ids = [[pos_c, anc_id], [pos_c, pos_id], [neg_c, neg_id]]

        return triplet_ids


class PHD08Dataset(Dataset, SiameseDataset):
    def __init__(self, dataset_type: str, cfg: dict):
        super().__init__()
        self.type = cfg['batch_settings']['type']
        self.positive_mode = cfg['batch_settings']['positive_mode']
        self.negative_mode = cfg['batch_settings']['negative_mode']

        self.inner_imp_prob = cfg['batch_settings']['inner_imp_prob']
        self.raw_clusters = cfg['batch_settings']['raw_clusters']
        self.clusters = []

        self.data_dir = cfg[dataset_type]['path']
        self.alphabet, self.alph_dict = prepare_alph(cfg["alph_pt"])

        self.all_files = []

        print("======= LOADING DATA(PHD08Dataset) =======")
        for class_dir in os.listdir(self.data_dir):
            files = os.listdir(op.join(self.data_dir, class_dir))
            files = [op.join(self.data_dir, class_dir, fi) for fi in files]
            self.files_per_classes.append(files)
            self.all_files.extend(files)
        print('Valid_dataset_length: ', len(self.files_per_classes), len(self.all_files))


class PHD08ValidDataset(Dataset):
    def __init__(self, dataset_type: str, cfg: dict):
        self.data_dir = cfg[dataset_type]['path']
        self.alphabet, self.alph_dict = prepare_alph(cfg["alph_pt"])

        png_data_dirs = os.listdir(self.data_dir)
        self.all_files, self.all_classes = [], []
        self.files_per_classes = []
        self.data = []

        print("======= LOADING DATA(PHD08ValidDataset) =======")
        start_time = time.time()

        # trans1 = torchvision.transforms.ToTensor()
        # trans2 = torchvision.transforms.Resize((37, 37), antialias=False)

        for class_dir in tqdm(png_data_dirs):
            files = os.listdir(op.join(self.data_dir, class_dir))
            files = [op.join(self.data_dir, class_dir, fi) for fi in files]

            # for img_path in files:
            #     image = Image.open(img_path).convert('L')
            #
            #     self.data.append({'img': trans2(trans1(image)),
            #                       'lbl': torch.tensor(float(class_dir))})

            self.all_classes.extend([float(class_dir) for fi in files])
            self.all_files.extend(files)
            self.files_per_classes.append(files)

        print('Valid_dataset_length: ', len(self.all_files),
              '\nValid_dataset_alph_length: ', len(self.files_per_classes),
              '\nTime: {:.2f} sec'.format(time.time() - start_time))

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, idx: int) -> dict:
        # image = Image.open(self.all_files[idx]).convert('L')
        # trans1 = torchvision.transforms.ToTensor()
        # transform = torchvision.transforms.Resize((37, 37))
        #
        # sample = {
        #     "image": transform(trans1(image)),
        #     "label": torch.tensor(self.all_classes[idx])
        # }

        # sample = {
        #     "image": self.data[idx]['img'],
        #     "label": self.data[idx]['lbl']
        # }

        sample = {
            "image": torch.load(self.all_files[idx]),
            "label": torch.tensor(self.all_classes[idx])
        }

        return sample

    def get_alph_size(self) -> int:
        return len(self.files_per_classes)

    def get_alphabet(self) -> list:
        return self.alphabet

    def get_alph_dict(self) -> dict:
        return self.alph_dict


class OmniglotOneShot(Dataset):
    def __init__(self, dataset_type: str, cfg: dict):
        self.data_dir = cfg[dataset_type]['path']
        self.alphabet, self.alph_dict = prepare_alph(cfg["alph_pt"])

        self.all_files_train, self.all_files_test = [], []
        self.all_classes = []
        self.data = []

        print("======= LOADING DATA(OmniglotOneShot) =======")
        start_time = time.time()

        for folder in tqdm(os.listdir(self.data_dir)):

            files_training = os.listdir(op.join(self.data_dir, folder, 'training'))
            files_test = os.listdir(op.join(self.data_dir, folder, 'test'))

            classes = []
            with open(op.join(self.data_dir, folder, 'class_labels.txt'), 'r') as f_in:
                lines = f_in.readlines()
                for line in lines:
                    first, second = line.split(' ')
                    num_test = int(first[15:17])
                    num_train = int(second[20:22])
                    # print(num_test, num_train)
                    classes.append(num_train - 1)

            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((37, 37))
            ])

            files_training = [transform(Image.open(op.join(self.data_dir, folder, 'training', fi)).convert('L')) for fi
                              in files_training]
            files_test = [transform(Image.open(op.join(self.data_dir, folder, 'test', fi)).convert('L')) for fi
                          in files_test]

            self.all_files_train.append(files_training)
            self.all_files_test.append(files_test)
            self.all_classes.append(classes)

        print('OmniglotOneShot_length: ', len(self.all_files_test),
              '\nValid_dataset_alph_length: ', len(self.alph_dict),
              '\nTime: {:.2f} sec'.format(time.time() - start_time))

    def __len__(self) -> int:
        return len(self.all_files_test)

    def __getitem__(self, idx: int) -> dict:

        # print(len(self.all_files_test), type(self.all_files_test[0]), len(self.all_files_test[0]), self.all_files_test[0][0].size())
        # print(torch.stack(self.all_files_test[0]).size())

        # print(self.all_classes[0])

        sample = {
            "test": torch.stack(self.all_files_test[idx]),
            "train": torch.stack(self.all_files_train[idx]),
            "lbl": torch.tensor(self.all_classes[idx])
        }

        return sample


class OmniglotPair(Dataset, PairDataset):
    def __init__(self, dataset_type: str, cfg: dict, transforms):
        super().__init__()
        self.transforms = transforms

        self.type = cfg['batch_settings']['type']
        self.positive_mode = cfg['batch_settings']['positive_mode']
        self.negative_mode = cfg['batch_settings']['negative_mode']
        self.gen_imp_ratio = cfg['batch_settings']['gen_imp_ratio']

        self.clusters = []
        self.probs_vec = []

        self.inner_imp_prob = cfg['batch_settings']['inner_imp_prob']
        self.raw_clusters = cfg['batch_settings']['raw_clusters']
        self.cluster_max_size = cfg['batch_settings']['cluster_max_size']

        self.data_dir = cfg[dataset_type]['path']

        self.alphabet, self.alph_dict = prepare_alph(cfg["alph_pt"])

        self.all_files, self.files_per_classes = [], []
        print("======= LOADING DATA(OmniglotPair) =======")
        for sub_dir in os.listdir(self.data_dir):
            for class_dir in os.listdir(op.join(self.data_dir, sub_dir)):
                files = os.listdir(op.join(self.data_dir, sub_dir, class_dir))
                files = [op.join(self.data_dir, sub_dir, class_dir, fi) for fi in files]

                self.files_per_classes.append(files)
                self.all_files.extend(files)
        print('OmniglotPair_dataset_length: ', len(self.files_per_classes), len(self.all_files))
        # assert len(self.alphabet) == len(self.files_per_classes)

        self.probs_vec = torch.empty(self.get_alph_size()).fill_(1.0 / self.get_alph_size()).cuda()

    def __getitem__(self, idx: int) -> dict:
        pair_ids = self.generatePairs()

        convert_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((37, 37))
        ])

        pair_imgs, pair_lbls = [], []
        for pair_id in pair_ids:
            c, i = pair_id[0], pair_id[1]
            file = self.files_per_classes[c][i]

            image = Image.open(file).convert('L')

            # if random.uniform(0, 1) < 0.7:
            #     image = self.transforms(image)

            pair_imgs.append(convert_transform(image))
            pair_lbls.append(torch.tensor(c))

        sample = {
            "image": pair_imgs,
            "label": pair_lbls
        }

        return sample

    def get_alph(self) -> list:
        return self.alphabet

    def __len__(self) -> int:
        return len(self.all_files)


class OmniglotTriplet(Dataset, TripletDataset):
    def __init__(self, dataset_type: str, cfg: dict, transforms):
        super().__init__()
        self.transforms = transforms

        self.type = cfg['batch_settings']['type']
        self.positive_mode = cfg['batch_settings']['positive_mode']
        self.negative_mode = cfg['batch_settings']['negative_mode']
        self.gen_imp_ratio = cfg['batch_settings']['gen_imp_ratio']

        self.clusters = []
        self.probs_vec = None

        self.inner_imp_prob = cfg['batch_settings']['inner_imp_prob']
        self.raw_clusters = cfg['batch_settings']['raw_clusters']
        self.cluster_max_size = cfg['batch_settings']['cluster_max_size']

        self.data_dir = cfg[dataset_type]['path']

        self.alphabet, self.alph_dict = prepare_alph(cfg["alph_pt"])

        self.all_files, self.files_per_classes = [], []
        print("======= LOADING DATA(OmniglotTriplet) =======")
        for sub_dir in os.listdir(self.data_dir):
            for class_dir in os.listdir(op.join(self.data_dir, sub_dir)):
                files = os.listdir(op.join(self.data_dir, sub_dir, class_dir))
                files = [op.join(self.data_dir, sub_dir, class_dir, fi) for fi in files]

                self.files_per_classes.append(files)
                self.all_files.extend(files)
        print('OmniglotTriplet_dataset_length: ', len(self.files_per_classes), len(self.all_files))
        # assert len(self.alphabet) == len(self.files_per_classes)

        self.probs_vec = torch.empty(self.get_alph_size()).fill_(1.0 / self.get_alph_size()).cuda()

    def __getitem__(self, idx: int) -> dict:
        triplet_ids = self.generateTriplets()

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((37, 37))
        ])

        triplet_imgs, triplet_lbls = [], []
        for triplet_id in triplet_ids:
            c, i = triplet_id[0], triplet_id[1]
            file = self.files_per_classes[c][i]

            # image = read_image(file, ImageReadMode.GRAY)
            image = Image.open(file).convert('L')

            triplet_imgs.append(transform(image))
            triplet_lbls.append(torch.tensor(c))

        sample = {
            "image": triplet_imgs,
            "label": triplet_lbls
        }

        return sample

    def get_alph(self) -> list:
        return self.alphabet

    def __len__(self) -> int:
        return len(self.all_files)


class KorSyntheticPair(Dataset, PairDataset):
    def __init__(self, dataset_type: str, cfg: dict, transforms):
        super().__init__()
        self.transforms = transforms

        self.type = cfg['batch_settings']['type']
        self.positive_mode = cfg['batch_settings']['positive_mode']
        self.negative_mode = cfg['batch_settings']['negative_mode']
        self.gen_imp_ratio = cfg['batch_settings']['gen_imp_ratio']
        self.clusters = []

        self.inner_imp_prob = cfg['batch_settings']['inner_imp_prob']
        self.raw_clusters = cfg['batch_settings']['raw_clusters']
        self.cluster_max_size = cfg['batch_settings']['cluster_max_size']

        self.data_dir = cfg[dataset_type]['path']

        self.alphabet, self.alph_dict = prepare_alph(cfg["alph_pt"])

        self.all_files, self.files_per_classes = [], []
        print("======= LOADING DATA(KorSyntheticPair) =======")
        for class_dir in os.listdir(self.data_dir):
            files = os.listdir(op.join(self.data_dir, class_dir))
            files = [op.join(self.data_dir, class_dir, fi) for fi in files]
            self.files_per_classes.append(files)
            self.all_files.extend(files)
        print('Train_dataset_length: ', len(self.files_per_classes), len(self.all_files))
        assert len(self.alphabet) == len(self.files_per_classes)

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, idx: int) -> dict:
        pair_ids = self.generatePairs()

        pair_imgs, pair_lbls = [], []
        for pair_id in pair_ids:
            c, i = pair_id[0], pair_id[1]
            file = self.files_per_classes[c][i]
            with open(file, "r") as data_f:
                data = json.load(data_f)
                mask = torch.tensor(data[1]["data"]).reshape(1, 37, 37)

            bgr_idx = np.random.randint(len(self.all_files))
            bgr_file = self.all_files[bgr_idx]
            with open(bgr_file, "r") as data_f:
                bgr_data = json.load(data_f)
                bgr = torch.tensor(bgr_data[0]["data"]).reshape(1, 37, 37)

            image = bgr * mask
            lbl = torch.tensor(int(data[2]["data"][0]))  # .type(torch.LongTensor)

            if random.uniform(0, 1) < 0.7:
                image = self.transforms(image)

            pair_imgs.append(image)
            pair_lbls.append(lbl)

        sample = {
            "image": pair_imgs,
            "label": pair_lbls
        }

        return sample

    def get_alph(self) -> list:
        return self.alphabet


class KorSyntheticTriplet(Dataset, TripletDataset):
    def __init__(self, dataset_type: str, cfg: dict, transforms):
        super().__init__()
        self.transforms = transforms

        self.type = cfg['batch_settings']['type']
        self.positive_mode = cfg['batch_settings']['positive_mode']
        self.negative_mode = cfg['batch_settings']['negative_mode']
        self.clusters = []

        self.inner_imp_prob = cfg['batch_settings']['inner_imp_prob']
        self.raw_clusters = cfg['batch_settings']['raw_clusters']
        self.cluster_max_size = cfg['batch_settings']['cluster_max_size']

        self.data_dir = cfg[dataset_type]["path"]

        self.alphabet, self.alph_dict = prepare_alph(cfg["alph_pt"])

        self.all_files, self.files_per_classes = [], []
        print("======= LOADING DATA(KorSyntheticTriplet) =======")
        for class_dir in os.listdir(self.data_dir):
            files = os.listdir(op.join(self.data_dir, class_dir))
            files = [op.join(self.data_dir, class_dir, fi) for fi in files]
            self.files_per_classes.append(files)
            self.all_files.extend(files)
        print('Train_dataset_length: ', len(self.files_per_classes), len(self.all_files))
        assert len(self.alphabet) == len(self.files_per_classes)

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, idx: int) -> dict:

        triplet_ids = self.generateTriplets()

        triplet_imgs, triplet_lbls = [], []
        for triplet_id in triplet_ids:
            c, i = triplet_id[0], triplet_id[1]
            file = self.files_per_classes[c][i]
            with open(file, "r") as data_f:
                data = json.load(data_f)
                mask = torch.tensor(data[1]["data"]).reshape(1, 37, 37)

            bgr_idx = np.random.randint(len(self.all_files))
            bgr_file = self.all_files[bgr_idx]
            with open(bgr_file, "r") as data_f:
                bgr_data = json.load(data_f)
                bgr = torch.tensor(bgr_data[0]["data"]).reshape(1, 37, 37)

            image = bgr * mask
            lbl = torch.tensor(int(data[2]["data"][0]))  # .type(torch.LongTensor)

            if random.uniform(0, 1) < 0.7:
                image = self.transforms(image)

            triplet_imgs.append(image)
            triplet_lbls.append(lbl)

        sample = {
            "image": triplet_imgs,
            "label": triplet_lbls
        }

        return sample

    def get_alph(self) -> list:
        return self.alphabet
