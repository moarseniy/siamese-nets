import os
import os.path as op
from os.path import join
from os import listdir as ls

import time

import torch

from multiprocessing import Pool, cpu_count
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

# np.random.seed(42) TODO: make it possible to set rand!

def ChooseDataset(dataset_type: str, cfg: dict, augmentation: dict) -> Dataset:
    if not dataset_type in cfg:
        print("No " + dataset_type + " in config!")
        return None

    mode = cfg['loss_settings']['type']
    pair_mode = mode in ('pair', 'contrastive')
    if cfg[dataset_type]["name"] == "KorSynthetic":
        if not pair_mode:
            return KorSyntheticTriplet(dataset_type=dataset_type, cfg=cfg, augmentation=augmentation)
        else:
            return KorSyntheticPair(dataset_type=dataset_type, cfg=cfg, augmentation=augmentation)
    elif cfg[dataset_type]["name"] == "Omniglot":
        if not pair_mode:
            return OmniglotTriplet(dataset_type=dataset_type, cfg=cfg, augmentation=augmentation)
        else:
            return OmniglotPair(dataset_type=dataset_type, cfg=cfg, augmentation=augmentation)
    elif cfg[dataset_type]["name"] == "OmniglotOneShot":
        return OmniglotOneShot(dataset_type=dataset_type, cfg=cfg)
    elif cfg[dataset_type]["name"] in ("PHD08Valid", "PHD08ValidDataset"):
        return PHD08ValidDataset(dataset_type=dataset_type, cfg=cfg)
    elif cfg[dataset_type]["name"] == "Common":
        if pair_mode:
            return CommonPair(dataset_type=dataset_type, cfg=cfg, augmentation=augmentation)
        else:
            return CommonTriplet(dataset_type=dataset_type, cfg=cfg, augmentation=augmentation)
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
        self.samples_per_class = []
        self.meta_data = {}
        self.meta_coeff = None

    def configure_sampling(self, cfg, dataset_type):
        settings = cfg['batch_settings'][dataset_type]
        self.positive_mode = settings.get('positive_mode', 'random')
        self.negative_mode = settings.get('negative_mode', 'random')
        if self.positive_mode == 'auto_sym_probs1':
            self.positive_mode = 'auto_sym_probs'
        if self.negative_mode == 'auto_clusters1':
            self.negative_mode = 'auto_clusters'
        if self.negative_mode == 'batch-hard':
            self.negative_mode = 'batch_hard'
        if self.negative_mode == 'semi-hard':
            self.negative_mode = 'semi_hard'

        cluster_settings = settings.get('auto_clusters', {})
        self.inner_imp_prob = settings.get('inner_imp_prob', cluster_settings.get('inner_imp_prob', 0.2))
        self.raw_clusters = settings.get('raw_clusters', cluster_settings.get('raw_clusters', 1000))
        self.cluster_max_size = settings.get('cluster_max_size', cluster_settings.get('cluster_max_size', 200))
        self.make_clust_on_ep = settings.get('make_clust_on_ep', cluster_settings.get('make_clust_on_ep', 1))

        sym_settings = settings.get('auto_sym_probs', {})
        self.sym_probs_gamma = settings.get('sym_probs_gamma', sym_settings.get('sym_probs_gamma', 2.0))
        self.merge_w = settings.get('merge_w', sym_settings.get('merge_w', 0.0))

        if self.negative_mode not in ('random', 'auto_clusters', 'hard', 'batch_hard', 'semi_hard'):
            raise ValueError(f"Unknown negative_mode {self.negative_mode!r}; expected random, auto_clusters, hard, or semi_hard")

    def initialize_sampling_state(self):
        self.clusters = []
        eligible = np.array([len(samples) >= 2 for samples in self.samples_per_class], dtype=bool)
        if not eligible.any():
            raise ValueError("Metric-learning datasets need at least one class with two or more samples")
        self.probs_vec = eligible.astype(np.float64) / eligible.sum()

    def choose_positive_random(self):
        eligible = [i for i, samples in enumerate(self.samples_per_class) if len(samples) >= 2]
        if not eligible:
            raise ValueError("Positive sampling requires at least one class with two or more images")
        pos_c = int(np.random.choice(eligible))

        if self.meta_data:
            valid_classes = [c for c in eligible if list(self.meta_data.get(c, {}).values()).count(0) != 1]
            if valid_classes:
                pos_c = int(np.random.choice(valid_classes))

        pos_id = np.random.randint(len(self.samples_per_class[pos_c]))

        if self.meta_data and pos_c in self.meta_data:
            valid_ids = [i for i in range(len(self.samples_per_class[pos_c]))
                         if self.meta_data[pos_c].get(str(i), 0) == 0]
            if valid_ids:
                pos_id = int(np.random.choice(valid_ids))

        return pos_c, pos_id

    def choose_positive_symprobs(self):
        pos_c = int(np.random.choice(len(self.samples_per_class), p=self.probs_vec))
        pos_id = np.random.randint(len(self.samples_per_class[pos_c]))
        return pos_c, pos_id

    def create_positive(self, pos_c, pos_id):
        num_samples = len(self.samples_per_class[pos_c])
        if num_samples < 2:
            raise ValueError(f"Class {pos_c} needs at least two samples to form a positive pair")

        candidates = [i for i in range(num_samples) if i != pos_id]

        if self.meta_data and pos_c in self.meta_data:
            meta_candidates = [i for i in candidates if self.meta_data[pos_c].get(str(i), 0) == 0]
            if meta_candidates:
                candidates = meta_candidates
        return int(np.random.choice(candidates))

    def create_negative_random(self, pos_c):
        candidates = [i for i, samples in enumerate(self.samples_per_class) if i != pos_c and samples]
        if not candidates:
            raise ValueError("At least two non-empty classes are required to sample a negative")
        neg_c = int(np.random.choice(candidates))
        neg_id = np.random.randint(len(self.samples_per_class[neg_c]))
        return neg_c, neg_id

    def create_negative_clusters(self, pos_c):
        if self.clusters and random.random() < self.inner_imp_prob:
            candidates = next((cluster for cluster in self.clusters if pos_c in cluster), [])
            candidates = [c for c in candidates if c != pos_c and self.samples_per_class[c]]
            if candidates:
                neg_c = int(np.random.choice(candidates))
                return neg_c, int(np.random.randint(len(self.samples_per_class[neg_c])))
        return self.create_negative_random(pos_c)

    def get_alph_size(self) -> int:
        return len(self.samples_per_class)

    def update_rules(self, dataset_type, save_ideals, ideals, counter, dists, ep_save_pt, config, e):
        settings = config['batch_settings'][dataset_type]
        cluster_settings = settings.get('auto_clusters', {})
        make_clusters_every = settings.get('make_clust_on_ep', cluster_settings.get('make_clust_on_ep', 1))
        if (save_ideals and self.negative_mode == 'auto_clusters' and make_clusters_every > 0 and
                e % make_clusters_every == 0):
            generation_time = time.time()

            class_count = len(self.samples_per_class)
            active_ids = torch.nonzero(counter > 0, as_tuple=False).flatten()
            norms_res = generate_clusters(ideals, self.raw_clusters, class_count, active_ids)

            print('Generation clusters time: {:.2f} sec'.format(time.time() - generation_time))

            merge_time = time.time()

            self.clusters = merge_clusters(norms_res, [], self.cluster_max_size)

            print('Merge clusters time: {:.2f} sec, Total clusters size: {}'.format(time.time() - merge_time,
                                                                                    len(self.clusters)))

            save_clusters(os.path.join(ep_save_pt, 'clusters.json'), self.clusters)

        if self.positive_mode == 'auto_sym_probs' and dists is not None and ideals is not None:
            sym_probs_time = time.time()
            updated_probs = generate_sym_probs(dists, ideals, counter, self.probs_vec,
                                               self.merge_w, self.sym_probs_gamma)
            eligible = torch.tensor([len(samples) >= 2 for samples in self.samples_per_class],
                                    dtype=torch.bool)
            updated_probs[~eligible] = 0
            if updated_probs.sum() > 0:
                updated_probs /= updated_probs.sum()
            self.probs_vec = updated_probs.numpy()

            print('Generation sym_probs time: {:.2f} sec'.format(time.time() - sym_probs_time))

            save_sym_probs(os.path.join(ep_save_pt, 'sym_probs.json'), self.probs_vec,
                           getattr(self, 'alphabet', None))


class PairDataset(SiameseDataset):
    def __init__(self, *args):
        super().__init__(*args)

    def generatePairs(self):
        pair_ids = None
        if random.uniform(0, 1) < self.gen_imp_ratio:
            pos_c, pos_id1 = self.choose_positive_random()

            pos_id2 = self.create_positive(pos_c, pos_id1)

            pair_ids = [[pos_c, pos_id1], [pos_c, pos_id2]]
            pair_label = 0
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

            pair_label = 1

        return pair_ids, pair_label


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

    def generateItem(self, idx: int) -> dict:
        triplet_ids = self.generateTriplets()

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(size=(self.image_size['height'],
                                                self.image_size['width']),
                                          antialias=False),
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        triplet_imgs, triplet_lbls = [], []
        for triplet_id in triplet_ids:
            c, i = triplet_id[0], triplet_id[1]
            file = self.samples_per_class[c][i]

            # image = read_image(file, ImageReadMode.GRAY)
            image = None
            if self.image_size['channels'] == 1:
                image = Image.open(file).convert('L')
            else:
                image = Image.open(file)

            # if self.dataset_type == "train":
            #     image = self.transforms(image)

            triplet_imgs.append(transform(image))
            triplet_lbls.append(torch.tensor(c))

        return {"image": torch.stack(triplet_imgs),
                "label": torch.stack(triplet_lbls)}


class PHD08ValidDataset(Dataset):
    def __init__(self, dataset_type: str, cfg: dict):
        self.data_dir = cfg[dataset_type]['path']
        self.alphabet, self.alph_dict = prepare_alph(cfg["alph_pt"])

        self.image_size = cfg['image_size']

        self.type = cfg['loss_settings']['type']
        self.data_dir = cfg[dataset_type]['path']

        self.all_files, self.all_classes = [], []
        self.samples_per_class = []
        self.data = []

        print("======= LOADING DATA(PHD08ValidDataset) =======")
        start_time = time.time()

        for sub_dir in tqdm(os.listdir(self.data_dir)):
            for class_dir in os.listdir(op.join(self.data_dir, sub_dir)):
                files = os.listdir(op.join(self.data_dir, sub_dir, class_dir))
                files = [op.join(self.data_dir, sub_dir, class_dir, fi) for fi in files]

                self.all_classes.extend([float(class_dir) for fi in files])
                self.all_files.extend(files)
                self.samples_per_class.append(files)

        print('Number of files: ', len(self.all_files),
              '\nNumber of classes: ', len(self.samples_per_class),
              '\nTime: {:.2f} sec'.format(time.time() - start_time))

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, idx: int) -> dict:
        image = Image.open(self.all_files[idx]).convert('L')
        trans1 = torchvision.transforms.ToTensor()
        transform = torchvision.transforms.Resize((37, 37))

        sample = {
            "image": transform(trans1(image)),
            "label": torch.tensor(self.all_classes[idx])
        }

        # sample = {
        #     "image": self.data[idx]['img'],
        #     "label": self.data[idx]['lbl']
        # }

        # sample = {"image": torch.load(self.all_files[idx]),
                  # "label": torch.tensor(self.all_classes[idx])}

        return sample

    def get_alph_size(self) -> int:
        return len(self.samples_per_class)

    def get_alphabet(self) -> list:
        return self.alphabet

    def get_alph_dict(self) -> dict:
        return self.alph_dict


class CommonTriplet(Dataset, TripletDataset):
    def __init__(self, dataset_type: str, cfg: dict, augmentation: dict):
        SiameseDataset.__init__(self)
        self.augmentation = augmentation
        self.dataset_type = dataset_type
        self.image_size = cfg['image_size']

        self.type = cfg['loss_settings']['type']
        self.data_dir = cfg[dataset_type]['path']
        self.gen_imp_ratio = cfg['batch_settings'][dataset_type]['gen_imp_ratio']
        self.elements_per_batch = cfg['batch_settings'][dataset_type]['elements_per_batch']
        self.batch_count = cfg['batch_settings'][dataset_type]['iterations']

        self.configure_sampling(cfg, dataset_type)

        meta_path = op.join(self.data_dir, 'meta.json')
        if op.exists(meta_path):
            self.meta_coeff = cfg['batch_settings'][dataset_type].get('meta_coeff', 0.7)
            with open(meta_path, 'r') as meta:
                meta = json.load(meta)

            print(f'Meta file: {meta_path} loaded!')

            index_map = {}
            index_counter = 0
            for key, value in meta.items():
                folder, index = key.split("/")
                if folder not in index_map:
                    index_map[folder] = index_counter
                    index_counter += 1

                mapped_folder = index_map[folder]

                if mapped_folder not in self.meta_data:
                    self.meta_data[mapped_folder] = {}
                self.meta_data[mapped_folder][index] = value

        self.all_files, self.samples_per_class = [], []
        self.alphabet = []
        print(f'======= LOADING DATA(CommonTriplet) =======({self.data_dir})')

        for sub_dir in sorted(os.listdir(self.data_dir)):
            sub_dir_path = op.join(self.data_dir, sub_dir)
            if not op.isdir(sub_dir_path):
                continue

            for class_dir in sorted(os.listdir(sub_dir_path)):
                class_path = op.join(self.data_dir, sub_dir, class_dir)
                if not op.isdir(class_path):
                    continue
                files = sorted(os.listdir(class_path))
                files = [op.join(self.data_dir, sub_dir, class_dir, fi) for fi in files]

                self.samples_per_class.append(files)
                self.all_files.extend(files)
                self.alphabet.append(f"{sub_dir}/{class_dir}")

        self.initialize_sampling_state()

        print(f'CommonTriplet classes:{len(self.samples_per_class)}, files:{len(self.all_files)}')
        # assert len(self.alphabet) == len(self.samples_per_class)

    def __getitem__(self, idx: int) -> dict:
        return self.generateItem(idx)

    def __len__(self) -> int:
        return len(self.all_files) #self.batch_count * self.elements_per_batch  # len(self.all_files)

    def get_alph(self) -> list:
        return self.alphabet


class CommonPair(Dataset, PairDataset):
    def __init__(self, dataset_type: str, cfg: dict, augmentation: dict):
        SiameseDataset.__init__(self)
        self.augmentation = augmentation
        self.dataset_type = dataset_type
        self.image_size = cfg['image_size']
        self.data_dir = cfg[dataset_type]['path']
        self.gen_imp_ratio = cfg['batch_settings'][dataset_type]['gen_imp_ratio']
        self.elements_per_batch = cfg['batch_settings'][dataset_type]['elements_per_batch']
        self.batch_count = cfg['batch_settings'][dataset_type]['iterations']
        self.configure_sampling(cfg, dataset_type)

        self.all_files, self.samples_per_class, self.alphabet = [], [], []
        print(f'======= LOADING DATA(CommonPair) =======({self.data_dir})')
        for sub_dir in sorted(os.listdir(self.data_dir)):
            sub_dir_path = op.join(self.data_dir, sub_dir)
            if not op.isdir(sub_dir_path):
                continue
            for class_dir in sorted(os.listdir(sub_dir_path)):
                class_path = op.join(sub_dir_path, class_dir)
                if not op.isdir(class_path):
                    continue
                files = [op.join(class_path, name) for name in sorted(os.listdir(class_path))
                         if op.isfile(op.join(class_path, name))]
                if not files:
                    continue
                self.samples_per_class.append(files)
                self.all_files.extend(files)
                self.alphabet.append(f'{sub_dir}/{class_dir}')
        self.initialize_sampling_state()
        print(f'CommonPair classes:{len(self.samples_per_class)}, files:{len(self.all_files)}')

    def __len__(self):
        return self.batch_count * self.elements_per_batch

    def __getitem__(self, idx):
        pair_ids, pair_label = self.generatePairs()
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((self.image_size['height'], self.image_size['width']),
                                          antialias=False),
        ])
        images = []
        for class_id, sample_id in pair_ids:
            path = self.samples_per_class[class_id][sample_id]
            image = Image.open(path)
            if self.image_size['channels'] == 1:
                image = image.convert('L')
            else:
                image = image.convert('RGB')
            images.append(transform(image))
        return {
            'image': torch.stack(images),
            'label': torch.tensor(pair_label, dtype=torch.float32),
            'class_id': torch.tensor([class_id for class_id, _ in pair_ids], dtype=torch.long),
        }


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
                torchvision.transforms.Resize((37, 37), antialias=None)
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
        return self.batch_count * self.elements_per_batch

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
    def __init__(self, dataset_type: str, cfg: dict, augmentation: dict):
        SiameseDataset.__init__(self)
        self.augmentation = augmentation

        self.image_size = cfg['image_size']

        self.type = cfg['loss_settings']['type']
        self.data_dir = cfg[dataset_type]['path']
        self.gen_imp_ratio = cfg['batch_settings'][dataset_type]['gen_imp_ratio']
        self.elements_per_batch = cfg['batch_settings'][dataset_type]['elements_per_batch']
        self.batch_count = cfg['batch_settings'][dataset_type]['iterations']

        self.configure_sampling(cfg, dataset_type)

        if "alph_pt" in cfg:
            self.alphabet, self.alph_dict = prepare_alph(cfg["alph_pt"])

        self.all_files, self.samples_per_class = [], []
        print("======= LOADING DATA(OmniglotPair) =======")
        for sub_dir in os.listdir(self.data_dir):
            for class_dir in os.listdir(op.join(self.data_dir, sub_dir)):
                files = os.listdir(op.join(self.data_dir, sub_dir, class_dir))
                files = [op.join(self.data_dir, sub_dir, class_dir, fi) for fi in files]

                self.samples_per_class.append(files)
                self.all_files.extend(files)
        print('OmniglotPair_dataset_length: ', len(self.samples_per_class), len(self.all_files))
        # assert len(self.alphabet) == len(self.samples_per_class)

        self.initialize_sampling_state()

    def __getitem__(self, idx: int) -> dict:
        pair_ids, pair_label = self.generatePairs()

        convert_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((37, 37), antialias=None)
        ])

        pair_imgs = []
        for pair_id in pair_ids:
            c, i = pair_id[0], pair_id[1]
            file = self.samples_per_class[c][i]

            image = Image.open(file).convert('L')

            # if random.uniform(0, 1) < 0.7:
            #     image = self.transforms(image)

            pair_imgs.append(convert_transform(image))

        sample = {
            "image": torch.stack(pair_imgs),
            "label": torch.tensor(pair_label, dtype=torch.float32),
            "class_id": torch.tensor([c for c, _ in pair_ids], dtype=torch.long),
        }

        return sample

    def __len__(self) -> int:
        return self.batch_count * self.elements_per_batch

    def get_alph(self) -> list:
        return self.alphabet


class OmniglotTriplet(Dataset, TripletDataset):
    def __init__(self, dataset_type: str, cfg: dict, augmentation: dict):
        SiameseDataset.__init__(self)
        self.augmentation = augmentation

        self.image_size = cfg['image_size']

        self.type = cfg['loss_settings']['type']
        self.data_dir = cfg[dataset_type]['path']
        self.gen_imp_ratio = cfg['batch_settings'][dataset_type]['gen_imp_ratio']
        self.elements_per_batch = cfg['batch_settings'][dataset_type]['elements_per_batch']
        self.batch_count = cfg['batch_settings'][dataset_type]['iterations']

        self.configure_sampling(cfg, dataset_type)

        if "alph_pt" in cfg:
            self.alphabet, self.alph_dict = prepare_alph(cfg["alph_pt"])

        self.all_files, self.samples_per_class = [], []
        print("======= LOADING DATA(OmniglotTriplet) =======")
        for sub_dir in os.listdir(self.data_dir):
            for class_dir in os.listdir(op.join(self.data_dir, sub_dir)):
                files = os.listdir(op.join(self.data_dir, sub_dir, class_dir))
                files = [op.join(self.data_dir, sub_dir, class_dir, fi) for fi in files]

                self.samples_per_class.append(files)
                self.all_files.extend(files)
        print('OmniglotTriplet_dataset_length: ', len(self.samples_per_class), len(self.all_files))
        # assert len(self.alphabet) == len(self.samples_per_class)

        self.initialize_sampling_state()

    def __getitem__(self, idx: int) -> dict:
        triplet_ids = self.generateTriplets()

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((37, 37), antialias=None)
        ])

        triplet_imgs, triplet_lbls = [], []
        for triplet_id in triplet_ids:
            c, i = triplet_id[0], triplet_id[1]
            file = self.samples_per_class[c][i]

            # image = read_image(file, ImageReadMode.GRAY)
            image = Image.open(file).convert('L')

            triplet_imgs.append(transform(image))
            triplet_lbls.append(torch.tensor(c))

        sample = {
            "image": triplet_imgs,
            "label": triplet_lbls
        }

        return sample

    def __len__(self) -> int:
        return self.batch_count * self.elements_per_batch

    def get_alph(self) -> list:
        return self.alphabet


class KorSyntheticPair(Dataset, PairDataset):
    def __init__(self, dataset_type: str, cfg: dict, augmentation: dict):
        SiameseDataset.__init__(self)
        self.augmentation = augmentation

        self.image_size = cfg['image_size']

        self.type = cfg['loss_settings']['type']
        self.data_dir = cfg[dataset_type]['path']
        self.gen_imp_ratio = cfg['batch_settings'][dataset_type]['gen_imp_ratio']
        self.elements_per_batch = cfg['batch_settings'][dataset_type]['elements_per_batch']
        self.batch_count = cfg['batch_settings'][dataset_type]['iterations']

        self.configure_sampling(cfg, dataset_type)

        if "alph_pt" in cfg:
            self.alphabet, self.alph_dict = prepare_alph(cfg["alph_pt"])

        self.all_files, self.samples_per_class = [], []
        print("======= LOADING DATA(KorSyntheticPair) =======")
        for class_dir in os.listdir(self.data_dir):
            if not op.isdir(op.join(self.data_dir, class_dir)):
                continue
            files = os.listdir(op.join(self.data_dir, class_dir))
            files = [op.join(self.data_dir, class_dir, fi) for fi in files]
            self.samples_per_class.append(files)
            self.all_files.extend(files)
        self.initialize_sampling_state()
        print('Train_dataset_length: ', len(self.samples_per_class), len(self.all_files))
        assert len(self.alphabet) == len(self.samples_per_class)

    def __len__(self) -> int:
        return self.batch_count * self.elements_per_batch

    def __getitem__(self, idx: int) -> dict:
        pair_ids, pair_label = self.generatePairs()

        pair_imgs, pair_lbls = [], []
        for pair_id in pair_ids:
            c, i = pair_id[0], pair_id[1]
            file = self.samples_per_class[c][i]
            with open(file, "r") as data_f:
                data = json.load(data_f)
                mask = torch.tensor(data[1]["data"]).reshape(1, 37, 37)

            bgr_idx = np.random.randint(len(self.all_files))
            bgr_file = self.all_files[bgr_idx]
            with open(bgr_file, "r") as data_f:
                bgr_data = json.load(data_f)
                bgr = torch.tensor(bgr_data[0]["data"]).reshape(1, 37, 37)

            image = bgr * mask
            lbl = torch.tensor(c)

            if random.uniform(0, 1) < self.augmentation["aug_prob"]:
                image = self.augmentation["transform_list"](image)

            pair_imgs.append(image)
            pair_lbls.append(lbl)

        return {
            "image": torch.stack(pair_imgs),
            "label": torch.tensor(pair_label, dtype=torch.float32),
            "class_id": torch.tensor([c for c, _ in pair_ids], dtype=torch.long),
        }

    def get_alph(self) -> list:
        return self.alphabet


class KorSyntheticTriplet(Dataset, TripletDataset):
    def __init__(self, dataset_type: str, cfg: dict, augmentation: dict):
        SiameseDataset.__init__(self)
        self.augmentation = augmentation

        self.image_size = cfg['image_size']

        self.type = cfg['loss_settings']['type']
        self.data_dir = cfg[dataset_type]['path']
        self.gen_imp_ratio = cfg['batch_settings'][dataset_type]['gen_imp_ratio']
        self.elements_per_batch = cfg['batch_settings'][dataset_type]['elements_per_batch']
        self.batch_count = cfg['batch_settings'][dataset_type]['iterations']

        self.configure_sampling(cfg, dataset_type)

        if "alph_pt" in cfg:
            self.alphabet, self.alph_dict = prepare_alph(cfg["alph_pt"])

        self.all_files, self.samples_per_class = [], []
        print("======= LOADING DATA(KorSyntheticTriplet) =======")
        for class_dir in os.listdir(self.data_dir):
            if not op.isdir(op.join(self.data_dir, class_dir)):
                continue
            files = os.listdir(op.join(self.data_dir, class_dir))
            files = [op.join(self.data_dir, class_dir, fi) for fi in files]
            self.samples_per_class.append(files)
            self.all_files.extend(files)

        print('Train_dataset_length: ', len(self.samples_per_class), len(self.all_files))
        # assert len(self.alphabet) == len(self.samples_per_class)

        self.initialize_sampling_state()

    def __getitem__(self, idx: int) -> dict:
        # print(idx)
        triplet_ids = self.generateTriplets()

        triplet_imgs, triplet_lbls = [], []
        for triplet_id in triplet_ids:
            c, i = triplet_id[0], triplet_id[1]
            file = self.samples_per_class[c][i]
            with open(file, "r") as data_f:
                data = json.load(data_f)
                mask = torch.tensor(data[1]["data"]).reshape(1, 37, 37)

            bgr_idx = np.random.randint(len(self.all_files))
            bgr_file = self.all_files[bgr_idx]
            with open(bgr_file, "r") as data_f:
                bgr_data = json.load(data_f)
                bgr = torch.tensor(bgr_data[0]["data"]).reshape(1, 37, 37)

            image = bgr * mask
            lbl = torch.tensor(c)

            # if random.uniform(0, 1) < 0.7:
            #     image = self.transforms(image)

            triplet_imgs.append(image)
            triplet_lbls.append(lbl)

        return {"image": torch.stack(triplet_imgs),
                "label": torch.stack(triplet_lbls)}

    def __len__(self) -> int:
        return len(self.all_files) #self.batch_count * self.elements_per_batch

    def get_alph(self) -> list:
        return self.alphabet
