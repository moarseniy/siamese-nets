import os
import os.path as op

import torch, json
from model import *
from train_utils import export_fm
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import PHD08Dataset
def validate(config, recognizer, valid_dataset):
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=config['minibatch_size'],
                              shuffle=True,
                              num_workers=10)

    pbar = tqdm(valid_loader)

    descrs = []
    with open(config['ideals'], 'r') as json_ideal:
        data = json.load(json_ideal)
        for key, value in data.items():
            descrs.append(value)

    count = torch.zeros(valid_dataset.get_alph_size()).cuda()
    for idx, mb in enumerate(pbar):
        anchor, positive, negative = mb['image'][0].cuda(), mb['image'][1].cuda(), mb['image'][2].cuda()
        a_lbl, p_lbl, n_lbl = mb['label'][0].cuda(), mb['label'][1].cuda(), mb['label'][2].cuda()

        data_out = recognizer(anchor)
        min_norm = torch.empty(data_out.size()[0]).fill_(1e+10).cuda()

        ids = torch.zeros(a_lbl.size()).cuda()

        i = 0
        for descr in descrs:
            cur_norm = torch.sqrt(torch.sum((data_out - torch.tensor(descr).cuda()) ** 2, dim=1))

            min_norm[cur_norm < min_norm] = cur_norm[cur_norm < min_norm]
            ids[cur_norm < min_norm] = torch.tensor(i, dtype=torch.float).cuda()
            i += 1

        count += a_lbl == ids
        print(torch.sum(count))

    print('Result', torch.sum(count), len(valid_loader))
    print(count)
    exit(0)


if __name__ == "__main__":

    with open("train_config.json", "r", encoding='utf8') as cfg_file:
        cfg = json.load(cfg_file)

    torch.cuda.set_device(cfg['device'])
    print(torch.cuda.is_available())

    model = KorNet().cuda()
    if cfg['file_to_start']:
        checkpoint = torch.load(cfg['file_to_start'])
        model.load_state_dict(checkpoint)
        print("Successfully loaded weights from", cfg['file_to_start'])
    else:
        print("No path for net (file_to_start)")

    valid_dataset = PHD08Dataset(cfg=cfg)

    validate(cfg, model, valid_dataset)



