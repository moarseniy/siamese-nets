import json
import os
import os.path as op
from datetime import datetime
import cv2

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch import nn
import torchvision
from torchvision.utils import save_image


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
            cur_norm = torch.sum((data_out - torch.tensor(descr).cuda()) ** 2, dim=1)

            min_norm[cur_norm < min_norm] = cur_norm[cur_norm < min_norm]
            ids[cur_norm < min_norm] = torch.tensor(i, dtype=torch.float).cuda()
            i += 1

        count += a_lbl == ids
        print(torch.sum(count))

    print('Result', torch.sum(count), len(valid_loader))
    print(count)
    exit(0)




def train(config, recognizer, optimizer, train_dataset, valid_dataset, save_pt, save_im_pt, start_ep):
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config['minibatch_size'],
                              shuffle=True,
                              num_workers=10)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=1024,
                              shuffle=True,
                              num_workers=10)

    # print(train_dataset[40]['image'][0], valid_dataset[40]['image'][0])
    # trans1 = torchvision.transforms.ToTensor()
    # res = trans1(valid_dataset[40]['image'][0])
    # trans2 = torchvision.transforms.Normalize(mean=0., std=1 / 255.)
    # print(trans2(valid_dataset[40]['image'][0]))
    # print(trans1(train_dataset[40]['image'][0]))
    # exit(-1)

    ideals = torch.zeros(train_dataset.get_alph_size(), 25).cuda()
    counter = torch.zeros(train_dataset.get_alph_size()).cuda()

    min_valid_loss = np.inf
    triplet_loss = nn.TripletMarginLoss(margin=1.0).cuda()

    for e in range(start_ep, config['epoch_num']):
        train_loss = 0.0
        pbar = tqdm(train_loader)

        for idx, mb in enumerate(pbar):

            anchor, positive, negative = mb['image'][0].cuda(), mb['image'][1].cuda(), mb['image'][2].cuda()
            a_lbl, p_lbl, n_lbl = mb['label'][0].cuda(), mb['label'][1].cuda(), mb['label'][2].cuda()

            anchor_out, positive_out, negative_out = recognizer(anchor), recognizer(positive), recognizer(negative)

            if idx == 0:
                save_image(anchor[0], os.path.join(save_im_pt, 'out_anc_train' + str(e) + '.png'))
                save_image(positive[0], os.path.join(save_im_pt, 'out_pos_train' + str(e) + '.png'))
                save_image(negative[0], os.path.join(save_im_pt, 'out_neg_train' + str(e) + '.png'))

            ideals[a_lbl] += anchor_out.detach()
            ideals[p_lbl] += positive_out.detach()
            ideals[n_lbl] += negative_out.detach()
            counter[a_lbl] += 1
            counter[p_lbl] += 1
            counter[n_lbl] += 1

            Loss = triplet_loss(anchor_out, positive_out, negative_out)

            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

            train_loss += Loss.item()

            print("epoch %d Train [Loss %.4f]" % (e, Loss.item()))
            # pbar.set_description("epoch %d G [Loss %.4f]"
            #                      % (e, Loss.item()))

        valid_loss = 0.0
        # recognizer.eval()
        pbar = tqdm(valid_loader)
        for idx, mb in enumerate(pbar):
            anchor, positive, negative = mb['image'][0].cuda(), mb['image'][1].cuda(), mb['image'][2].cuda()

            anchor_out, positive_out, negative_out = recognizer(anchor), recognizer(positive), recognizer(negative)
            a_lbl, p_lbl, n_lbl = mb['label'][0].cuda(), mb['label'][1].cuda(), mb['label'][2].cuda()
            if idx == 0:
                save_image(anchor[0], os.path.join(save_im_pt, 'out_anc_test' + str(e) + '.png'))
                save_image(positive[0], os.path.join(save_im_pt, 'out_pos_test' + str(e) + '.png'))
                save_image(negative[0], os.path.join(save_im_pt, 'out_neg_test' + str(e) + '.png'))

            Loss = triplet_loss(anchor_out, positive_out, negative_out)

            valid_loss += Loss.item()
            print("epoch %d Valid [Loss %.4f]" % (e, Loss.item()))

        print(f'Epoch {e} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(valid_loader)}')

        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss

            torch.save({
                'epoch': e,
                'model_state_dict': recognizer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss,
            }, op.join(save_pt, str(e) + ".pt"))

            # torch.save({},  op.join(save_pt, str(e) + ".pt"))
            ideals = torch.div(ideals.T, counter).T
            ideals_dict = {}

            for i in range(train_dataset.get_alph_size()):
                ideals_dict[str(i)] = ideals[i].cpu().tolist()

            with open(op.join(save_pt, 'ideals_' + str(e) + '.json'), 'w') as out:
                json.dump(ideals_dict, out)
            with open(op.join(save_pt, 'counter_' + str(e) + '.json'), 'w') as out:
                json.dump(counter.cpu().tolist(), out)

        ideals = torch.zeros(train_dataset.get_alph_size(), 25).cuda()
        counter = torch.zeros(train_dataset.get_alph_size()).cuda()



def prepare_dirs(config):
    save_paths = {}
    files_to_start = {}
    from_file = False
    start_ep = 0
    checkpoint_pt = config["checkpoint_pt"]
    images_pt = config["images_pt"]
    if not op.exists(checkpoint_pt):
        os.mkdir(checkpoint_pt)

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M")

    # if config['file_to_start'] != "":
    #     dir = config['file_to_start'].split("/")[0]
    #     chpnt = config['file_to_start'].split("/")[1]
    #     start_ep = int(chpnt.split(".")[0]) + 1
    #
    #     # assert sum([op.exists(pt) for pt in files_to_start.values()]) == 4
    #     from_file = True
    # else:
    save_pt = op.join(checkpoint_pt, dt_string)
    save_im_pt = op.join(images_pt, dt_string)
    if not op.exists(save_pt):
        os.makedirs(save_pt)
    if not op.exists(save_im_pt):
        os.makedirs(save_im_pt)

    return save_pt, save_im_pt
