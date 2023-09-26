import os
import os.path as op
import torch

import matplotlib.pyplot as plt
import numpy as np
import ujson as json

from tqdm import tqdm

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.io import ImageReadMode
import torchvision

from PIL import Image


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
    print(alph_dict)
    return alph, alph_dict


class PHD08Dataset(Dataset):
    def __init__(self, cfg: dict):
        self.data_dir = cfg["valid_data_dir"]
        png_data_dirs = os.listdir(self.data_dir)
        self.all_files, self.files_per_classes = [], []
        print("======= LOADING DATA =======")
        for class_dir in png_data_dirs:
            files = os.listdir(op.join(self.data_dir, class_dir))
            files = [op.join(self.data_dir, class_dir, fi) for fi in files]
            self.files_per_classes.append(files)
            self.all_files.extend(files)
        print('Valid_dataset_length:', len(self.files_per_classes))

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, idx: int) -> dict:
        pos_c, pos_id = self.choose_positive_random()
        anc_id = self.create_positive(pos_c, pos_id)
        neg_c, neg_id = self.create_negative_random(pos_c)

        triplet_ids = [[pos_c, anc_id], [pos_c, pos_id], [neg_c, neg_id]]

        triplet_imgs, triplet_lbls = [], []
        for triplet_id in triplet_ids:
            c, i = triplet_id[0], triplet_id[1]
            file = self.files_per_classes[c][i]

            #image = read_image(file, ImageReadMode.GRAY)
            image = Image.open(file).convert('L')
            trans1 = torchvision.transforms.ToTensor()
            transform = torchvision.transforms.Resize((37, 37))

            # print(image.size())
            triplet_imgs.append(transform(trans1(image)))
            triplet_lbls.append(torch.tensor(c))

        sample = {
            "image": triplet_imgs,
            "label": triplet_lbls
        }

        return sample

    def choose_positive_random(self):
        pos_c = np.random.randint(len(self.files_per_classes))
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

    def get_alph_size(self) -> int:
        return len(self.files_per_classes)


class KorRecognitionDataset(Dataset):
    def __init__(self, cfg: dict, transforms):
        self.transforms = transforms
        self.data_dir = cfg["train_data_dir"]
        json_data_dirs = os.listdir(self.data_dir)
        self.alphabet, self.alph_dict = prepare_alph(cfg["alph_pt"])
        self.all_files, self.files_per_classes = [], []
        print("======= LOADING DATA =======")
        for class_dir in json_data_dirs:
            files = os.listdir(op.join(self.data_dir, class_dir))
            files = [op.join(self.data_dir, class_dir, fi) for fi in files]
            self.files_per_classes.append(files)
            self.all_files.extend(files)
        print('Train_dataset_length:', len(self.files_per_classes))
        assert len(self.alphabet) == len(self.files_per_classes)
        # for font_dir in json_data_dirs:
        #     files = os.listdir(op.join(self.data_dir, font_dir))
        #     files = [op.join(self.data_dir, font_dir, fi) for fi in files]
        #     self.all_files.extend(files)

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, idx: int) -> dict:

        pos_c, pos_id = self.choose_positive_random()
        anc_id = self.create_positive(pos_c, pos_id)
        neg_c, neg_id = self.create_negative_random(pos_c)

        triplet_ids = [[pos_c, anc_id], [pos_c, pos_id], [neg_c, neg_id]]

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
            lbl = torch.tensor(int(data[3]["data"][0]))#.type(torch.LongTensor)

            triplet_imgs.append(image)
            triplet_lbls.append(lbl)

        sample = {
            "image": triplet_imgs,
            "label": triplet_lbls
        }

        return sample

    def choose_positive_random(self):
        pos_c = np.random.randint(len(self.alphabet))
        pos_id = np.random.randint(len(self.files_per_classes[pos_c]))
        return pos_c, pos_id

    def create_positive(self, pos_c, pos_id):
        anc_id = np.random.randint(len(self.files_per_classes[pos_c]))
        while anc_id == pos_id:
            anc_id = np.random.randint(len(self.files_per_classes[pos_c]))
        return anc_id

    def create_negative_random(self, pos_c):
        neg_c = np.random.randint(len(self.alphabet))
        while pos_c == neg_c:
            neg_c = np.random.randint(len(self.alphabet))
        neg_id = np.random.randint(len(self.files_per_classes[neg_c]))
        return neg_c, neg_id

    def get_alph_size(self) -> int:
        return len(self.alphabet)

    def get_alph(self) -> list:
        return self.alphabet