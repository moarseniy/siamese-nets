import os, time
import os.path as op

import torch, json
from model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import PHD08Dataset
from dataset import PHD08ValidDataset


def validate(config, recognizer, valid_loader, descrs):
    start_time = time.time()
    pbar = tqdm(valid_loader)

    count = 0  # torch.zeros(valid_dataset.get_alph_size()).cuda()
    for idx, mb in enumerate(pbar):
        img = mb['image'].cuda()
        lbl = mb['label'].cuda()

        data_out = recognizer(img)

        min_norm = torch.empty(data_out.size()[0]).fill_(1e+10).cuda()
        ids = torch.zeros(lbl.size()).cuda()

        # print(img.size(), lbl.size(), data_out.size())

        i = 0
        for j in range(descrs.size()[0]):
            cur_norm = torch.sqrt(torch.sum((data_out - descrs[j]) ** 2, dim=1))

            temp = cur_norm < min_norm
            min_norm[temp] = cur_norm[temp]
            ids[temp] = torch.tensor(i, dtype=torch.float).cuda()
            i += 1

        # print(lbl.size(), ids.size())
        count += torch.sum(torch.tensor(lbl == ids))
        pbar.set_description("Valid [count %d]" % count)

    print('Result', count, len(valid_loader.dataset), 100 * count / len(valid_loader.dataset), str(time.time() - start_time) + 's')
    return 100 * count / len(valid_loader.dataset)


if __name__ == "__main__":

    with open("train_config.json", "r", encoding='utf8') as cfg_file:
        cfg = json.load(cfg_file)

    torch.cuda.set_device(cfg['device'])
    print(torch.cuda.is_available())

    model = KorNet().cuda()
    if cfg['file_to_start']:
        print("Loading weights from", op.join(cfg['file_to_start'], 'model.pt'))
        checkpoint = torch.load(op.join(cfg['file_to_start'], 'model.pt'))
        model.load_state_dict(checkpoint, strict=False)
        print("Successfully loaded weights from", cfg['file_to_start'])
    else:
        print("No path for net (file_to_start)")

    valid_dataset = PHD08ValidDataset(cfg=cfg)

    ideals = torch.zeros(11190, 25).cuda()
    print(valid_dataset.get_alph_size())
    with open(op.join(cfg['file_to_start'], 'ideals.json'), 'r') as json_ideal:
        data = json.load(json_ideal)
        for key, value in data.items():
            ideals[int(key)] = torch.FloatTensor(value)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=102400,  # config['minibatch_size'],
                              shuffle=False,
                              num_workers=10)

    res = validate(cfg, model, valid_loader, ideals)
