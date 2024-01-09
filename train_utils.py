import json
import os, time
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

def _g(w, h, x, y, c, n, i):
    return ((c * h + y) * w + x) * n + i

def export_fm(fm, out_pt):
    print(out_pt)
    f = open(out_pt, "w")

    if len(fm.shape) == 4:
        h = fm.shape[2]
        w = fm.shape[3]
        nImgs = fm.shape[0]
        channels = fm.shape[1]
    elif len(fm.shape) == 3:
        h = 1
        w = fm.shape[2]
        nImgs = fm.shape[0]
        channels = fm.shape[1]
    elif len(fm.shape) == 2:
        h = 1
        w = 1
        nImgs = fm.shape[0]
        channels = fm.shape[1]
    else:
        channels = fm.shape[0]
        h = 1
        w = 1
        nImgs = 1

    print("size", w, h, channels, nImgs, file=f)
    for c in range(channels):
        for y in range(h):
            for x in range(w):
                for n in range(nImgs):
                    if len(fm.shape) == 4:
                        it = fm[n, c, y, x].item()
                    elif len(fm.shape) == 3:
                        it = fm[n, c, x].item()
                    elif len(fm.shape) == 2:
                        it = fm[n, c].item()
                    else:
                        it = fm[c].item()
                    print("{:.6e}".format(it), file=f, end=" ")

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
    save_debug = False

    ideals = torch.zeros(train_dataset.get_alph_size(), 25).cuda()
    counter = torch.empty(train_dataset.get_alph_size()).fill_(1e-10).cuda()

    min_valid_loss = np.inf
    triplet_loss = nn.TripletMarginLoss(margin=1.0).cuda()

    for e in range(start_ep, config['epoch_num']):
        train_loss = 0.0
        pbar = tqdm(train_loader)

        for idx, mb in enumerate(pbar):

            anchor, positive, negative = mb['image'][0].cuda(), mb['image'][1].cuda(), mb['image'][2].cuda()
            a_lbl, p_lbl, n_lbl = mb['label'][0].cuda(), mb['label'][1].cuda(), mb['label'][2].cuda()

            anchor_out, positive_out, negative_out = recognizer(anchor), recognizer(positive), recognizer(negative)

            if idx == 0 and save_debug:
                save_image(anchor[0], os.path.join(save_im_pt, 'out_anc_train' + str(e) + '.png'))
                save_image(positive[0], os.path.join(save_im_pt, 'out_pos_train' + str(e) + '.png'))
                save_image(negative[0], os.path.join(save_im_pt, 'out_neg_train' + str(e) + '.png'))

            # save ideals
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

        ideals = torch.div(ideals.T, counter).T
        if config['batch_settings']['negative_mode'] == 'auto_clusters':
            start_time = time.time()
            print('Started generation of clusters')
            train_dataset.update_rules(ideals, save_pt, e)
            print('Finished generation of clusters', time.time() - start_time)

        valid_loss = 0.0
        # recognizer.eval()
        pbar = tqdm(valid_loader)
        for idx, mb in enumerate(pbar):
            anchor, positive, negative = mb['image'][0].cuda(), mb['image'][1].cuda(), mb['image'][2].cuda()

            anchor_out, positive_out, negative_out = recognizer(anchor), recognizer(positive), recognizer(negative)
            a_lbl, p_lbl, n_lbl = mb['label'][0].cuda(), mb['label'][1].cuda(), mb['label'][2].cuda()
            if idx == 0 and save_debug:
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

            ep_save_pt = op.join(save_pt, str(e))
            os.mkdir(ep_save_pt)

            torch.save({
                'epoch': e,
                'model_state_dict': recognizer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss,
            }, op.join(ep_save_pt, "model.pt"))

            # ideals = torch.div(ideals.T, counter).T
            ideals_dict = {}

            for i in range(train_dataset.get_alph_size()):
                ideals_dict[str(i)] = ideals[i].cpu().tolist()

            with open(op.join(ep_save_pt, 'ideals.json'), 'w') as out:
                json.dump(ideals_dict, out)
            with open(op.join(ep_save_pt, 'counter.json'), 'w') as out:
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
