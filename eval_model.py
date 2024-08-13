import os, time
import os.path as op

import torch, json
from torchvision.utils import save_image
from model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import PHD08Dataset
from dataset import PHD08ValidDataset


def validate_with_descrs(config, recognizer, valid_loader, descrs):
    start_time = time.time()
    pbar = tqdm(valid_loader)

    count = 0  # torch.zeros(valid_dataset.get_alph_size()).cuda()
    for idx, mb in enumerate(pbar):
        img = mb['image'].cuda()
        lbl = mb['label'].cuda()

        if idx < 10 and config['save_images']:
            save_image(img[0], os.path.join('/home/arseniy/results/out/torch_out', 'out_valid_' + str(idx) + '.png'))

        data_out = recognizer(img)

        # min_norm = torch.empty(data_out.size()[0]).fill_(1e+10).cuda()
        # ids = torch.zeros(lbl.size()).cuda()
        #
        # i = 0
        # for j in range(descrs.size()[0]):
        #     cur_norm = torch.sqrt(torch.sum((data_out - descrs[j]) ** 2, dim=1))
        #
        #     temp = cur_norm < min_norm
        #     min_norm[temp] = cur_norm[temp]
        #     ids[temp] = torch.tensor(i, dtype=torch.float).cuda()
        #     i += 1

        # Compute Euclidean distances between each vector in data_out and descrs
        distances = torch.cdist(data_out, descrs)

        # Find the index of the closest vector in descrs for each vector in data_out
        ids = torch.argmin(distances, dim=1)

        # Optionally, you can also compute the minimum distances
        # min_distances = torch.min(distances, dim=1).values

        count += torch.sum((lbl == ids).clone().detach())

        pbar.set_description("Valid [count %d]" % count)

    print('\nCount: ', count,
          '\nLength: ', len(valid_loader.dataset),
          '\nAccuracy: ', 100 * count / len(valid_loader.dataset),
          '\nTime: {:.2f} sec'.format(time.time() - start_time))

    return 100 * count / len(valid_loader.dataset)


def validate_oneshot(config, recognizer, valid_loader):
    start_time = time.time()
    pbar = tqdm(valid_loader)

    count = 0  # torch.zeros(valid_dataset.get_alph_size()).cuda()
    for idx, mb in enumerate(pbar):
        img = mb['image'].cuda()
        lbl = mb['label'].cuda()

        if idx < 10 and config['save_images']:
            save_image(img[0], os.path.join('/home/arseniy/results/out/torch_out', 'out_valid_' + str(idx) + '.png'))

        data_out = recognizer(img)

        # min_norm = torch.empty(data_out.size()[0]).fill_(1e+10).cuda()
        # ids = torch.zeros(lbl.size()).cuda()
        #
        # i = 0
        # for j in range(descrs.size()[0]):
        #     cur_norm = torch.sqrt(torch.sum((data_out - descrs[j]) ** 2, dim=1))
        #
        #     temp = cur_norm < min_norm
        #     min_norm[temp] = cur_norm[temp]
        #     ids[temp] = torch.tensor(i, dtype=torch.float).cuda()
        #     i += 1

        # Compute Euclidean distances between each vector in data_out and descrs
        distances = torch.cdist(data_out, descrs)

        # Find the index of the closest vector in descrs for each vector in data_out
        ids = torch.argmin(distances, dim=1)

        # Optionally, you can also compute the minimum distances
        # min_distances = torch.min(distances, dim=1).values

        count += torch.sum((lbl == ids).clone().detach())

        pbar.set_description("Valid [count %d]" % count)

    print('\nCount: ', count,
          '\nLength: ', len(valid_loader.dataset),
          '\nAccuracy: ', 100 * count / len(valid_loader.dataset),
          '\nTime: {:.2f} sec'.format(time.time() - start_time))

    return 100 * count / len(valid_loader.dataset)


if __name__ == "__main__":

    with open("train_config.json", "r", encoding='utf8') as cfg_file:
        cfg = json.load(cfg_file)

    torch.cuda.set_device(cfg['device'])

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(cfg['device']) + ' is available!')
    else:
        print('No GPU!!!')
        exit(-1)

    model = KorNet().cuda()

    if cfg['file_to_start']:
        print("Loading weights from", op.join(cfg['file_to_start'], 'model.pt'))
        checkpoint = torch.load(op.join(cfg['file_to_start'], 'model.pt'))
        model.load_state_dict(checkpoint, strict=False)
        print("Successfully loaded weights from", cfg['file_to_start'])
    else:
        print("No path for net (file_to_start)")
        exit(0)

    valid_dataset = PHD08ValidDataset(cfg=cfg)

    ideals = torch.zeros(len(valid_dataset.get_alphabet()), 25).cuda()
    print('\nAlphabet size: ', len(valid_dataset.get_alphabet()))

    # with open(op.join(cfg['file_to_start'], 'ideals.json'), 'r') as json_ideal:
    #     data = json.load(json_ideal)
    #     for key, value in data.items():
    #         ideals[valid_dataset.get_alphabet().index(key)] = torch.FloatTensor(value)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=cfg['batch_settings']['elements_per_batch'],
                              shuffle=False,
                              num_workers=10)

    # res = validate(cfg, model, valid_loader, ideals)
