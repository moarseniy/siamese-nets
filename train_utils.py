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
from eval_model import *


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


def Dataloader_by_Index(data_loader, target=0):
    for index, data in enumerate(data_loader, target):
        return data
    return None


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin  # margin or radius

    def forward(self, output1, output2, label):
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        pos = (1 - label) * torch.pow(euclidean_distance, 2)
        neg = (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)

        loss_contrastive = torch.mean(pos + neg)
        return loss_contrastive

        # euc_dist = torch.nn.functional.pairwise_distance(y1, y2)
        #
        # if d == 0:
        #     return torch.mean(torch.pow(euc_dist, 2))  # distance squared
        # else:  # d == 1
        #     delta = self.margin - euc_dist  # sort of reverse distance
        #     delta = torch.clamp(delta, min=0.0, max=None)
        #     return torch.mean(torch.pow(delta, 2))  # mean over all rows


class MetricLoss(torch.nn.Module):
    def __init__(self, margin):
        super(MetricLoss, self).__init__()
        self.margin = margin

    def forward(self, anc, pos, neg, anc_ideal, pos_ideal, neg_ideal):
        ro, tau, xi = 0.1, 1.0, 1.0

        d_AN = torch.nn.functional.pairwise_distance(anc, neg)
        d_AP = torch.nn.functional.pairwise_distance(anc, pos)

        d_AIdeal = torch.nn.functional.pairwise_distance(anc, anc_ideal)
        d_PIdeal = torch.nn.functional.pairwise_distance(pos, pos_ideal)
        d_NIdeal = torch.nn.functional.pairwise_distance(neg, neg_ideal)

        f = nn.Softplus()

        g1 = ro * d_AP
        g2 = tau * f(d_AP - d_AN + self.margin)
        g3 = xi * (d_AIdeal + d_PIdeal + d_NIdeal) / 3.0

        loss_metric = g1 * g1 + g2 * g2 + g3 * g3
        return loss_metric


def go_metric_train(train_loader, config, recognizer, optimizer, loss, train_loss, save_im_pt, e, ideals, counter,
                    loss_type):
    # pbar = tqdm(range(config["batch_settings"]["iterations"]))
    # for idx in pbar:

    pbar = tqdm(train_loader)
    for idx, mb in enumerate(pbar):

        #     mb = Dataloader_by_Index(train_loader, torch.randint(len(train_loader), size=(1,)).item())
        size = len(mb['image'])
        img_out, lbl_out = [None] * size, [None] * size

        for img_id in range(size):
            img = mb['image'][img_id].cuda()
            lbl = mb['label'][img_id].cuda()
            out = recognizer(img)

            img_out[img_id] = out
            lbl_out[img_id] = lbl

            if idx <= 10 and config["save_images"]:
                save_image(img[0], os.path.join(save_im_pt, 'out_' + str(img_id) + '_train_' +
                                                str(int(lbl[0])) + '_' + str(e) + '.png'))

            ideals[lbl] += out.detach()
            counter[lbl] += 1

        if loss_type == "triplet":
            cur_loss = loss(img_out[0], img_out[1], img_out[2])
        elif loss_type == "contrastive":
            cur_loss = loss(img_out[0], img_out[1], (lbl_out[0] == lbl_out[1]).long())
        elif loss_type == "metric":

            cur_loss = loss(img_out[0], img_out[1], img_out[2], ideals[lbl_out[0]], ideals[lbl_out[1], ideals[lbl_out[2]]])
        elif loss_type == "BCE":
            cur_loss = loss()
        else:
            print('No type!')
            exit(-1)

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        train_loss += cur_loss.item()

        # print("epoch %d Train [Loss %.6f]" % (e, cur_loss.item()))
        pbar.set_description("epoch %d Train [Loss %.6f]" % (e, cur_loss.item()))

    return train_loss


def go_metric_test(test_loader, config, recognizer, loss, test_loss, save_im_pt, e, metric_type):
    # recognizer.eval()
    pbar = tqdm(range(config["batch_settings"]["iterations"]))
    for it in pbar:
        mb = test_loader[torch.randint(len(test_loader), size=(1,)).item()]
        size = len(mb['image'])
        img_out, lbl_out = [None] * size, [None] * size

        for img_id in range(size):
            img = mb['image'][img_id].cuda()
            lbl = mb['label'][img_id].cuda()
            out = recognizer(img)

            img_out[img_id] = out
            lbl_out[img_id] = lbl

            if it == 0 and config["save_images"]:
                save_image(img[0], os.path.join(save_im_pt, 'out_' + str(img_id) + '_test_' +
                                                str(int(lbl[0])) + '_' + str(e) + '.png'))

        if metric_type == "triplet":
            cur_loss = loss(img_out[0], img_out[1], img_out[2])
        elif metric_type == "contrastive":
            cur_loss = loss(img_out[0], img_out[1], (lbl_out[0] == lbl_out[1]).long())
        else:
            print('No type!')
            exit(-1)

        test_loss += cur_loss.item()
        pbar.set_description("epoch %d Test [Loss %.6f]" % (e, cur_loss.item()))


def run_training(config, recognizer, optimizer, train_dataset, test_dataset, valid_dataset, save_pt, save_im_pt,
                 start_ep):
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config['minibatch_size'],
                              shuffle=True,
                              num_workers=10)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=config['batch_settings']['elements_per_batch'],
                             shuffle=True,
                             num_workers=10)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=1,  # config['minibatch_size'],
                              shuffle=False,
                              num_workers=10)

    # if config['validate_settings']['validate'] and config['file_to_start']:
    #     print('Validation started!')
    #     validate(config, recognizer, valid_loader)
    #     exit(0)

    # print(train_dataset[40]['image'][0], valid_dataset[40]['image'][0])
    # trans1 = torchvision.transforms.ToTensor()
    # res = trans1(valid_dataset[40]['image'][0])
    # trans2 = torchvision.transforms.Normalize(mean=0., std=1 / 255.)
    # print(trans2(valid_dataset[40]['image'][0]))
    # print(trans1(train_dataset[40]['image'][0]))
    # exit(-1)

    loss_type = config['loss']
    batch_type = config['batch_settings']['type']

    ideals = torch.zeros(train_dataset.get_alph_size(), 25).cuda()
    counter = torch.empty(train_dataset.get_alph_size()).fill_(1e-10).cuda()

    min_test_loss = np.inf
    loss = None
    if loss_type == 'triplet':
        loss = nn.TripletMarginLoss(margin=config['batch_settings']['alpha_margin']).cuda()
    elif loss_type == 'contrastive':
        loss = ContrastiveLoss(margin=config['batch_settings']['alpha_margin']).cuda()
    elif loss_type == 'BCE':
        loss = nn.BCELoss().cuda()
    elif loss_type == 'metric':
        loss = MetricLoss(margin=config['batch_settings']['alpha_margin'])
    else:
        print('No Loss found!')
        exit(-1)

    stat = {'epochs': [],
            'train_losses': [],
            'valid_losses': [],
            'acc': []}

    for e in range(start_ep, config['epoch_num']):
        # pbar = tqdm(train_loader)
        # for idx, mb in enumerate(pbar):

        train_loss = 0.0
        start_time = time.time()
        train_loss = go_metric_train(train_loader, config, recognizer, optimizer, loss, train_loss, save_im_pt, e,
                                     ideals,
                                     counter, loss_type)

        ideals = torch.div(ideals.T, counter).T

        test_loss = 0.0
        to_test = True
        # if to_test:
        #     go_metric_test(valid_loader, config, recognizer, loss, train_loss, save_im_pt, e, metric_type)
        #
        #     print(
        #         f'Epoch {e} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {test_loss / len(valid_loader)}')
        #
        #     if min_test_loss > test_loss:
        #         print(
        #             f'Validation Loss Decreased({min_test_loss:.6f}--->{test_loss / len(valid_loader):.6f}) \t Saving The Model')
        #         min_valid_loss = test_loss

        ep_save_pt = op.join(save_pt, str(e))
        if not os.path.exists(ep_save_pt):
            os.mkdir(ep_save_pt)

        if config['batch_settings']['negative_mode'] == 'auto_clusters' and \
                e % config["batch_settings"]["make_clust_on_ep"] == 0:
            start_time = time.time()
            print('Started generation of clusters')
            train_dataset.update_rules(ideals, ep_save_pt)
            print('Finished generation of clusters {:.2f} sec'.format(time.time() - start_time))

        print('Epoch {} -> Training Loss({:.2f} sec): {}'.format(e, time.time() - start_time,
                                                                 train_loss / len(train_loader)))

        to_valid = True
        if to_valid:

            torch.save({
                'epoch': e,
                'model_state_dict': recognizer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': 0.0,  # test_loss,
            }, op.join(ep_save_pt, "model.pt"))

            # ideals = torch.div(ideals.T, counter).T
            ideals_dict = {}

            for i in range(train_dataset.get_alph_size()):
                ideals_dict[str(i)] = ideals[i].cpu().tolist()

            with open(op.join(ep_save_pt, 'ideals.json'), 'w') as out:
                json.dump(ideals_dict, out)
            with open(op.join(ep_save_pt, 'counter.json'), 'w') as out:
                json.dump(counter.cpu().tolist(), out)

            stat['epochs'].append(e)
            stat['train_losses'].append(train_loss / len(train_loader))
            # stat['valid_losses'].append(test_loss / len(test_loader))

            # acc = validate_with_descrs(config, recognizer, valid_loader, ideals)
            acc = validate_oneshot(config, recognizer, valid_loader)
            stat['acc'].append(acc.item())

            best_id = stat['acc'].index(max(stat['acc']))

            plt.figure(figsize=(12, 7))
            plt.xlabel("Epoch", fontsize=18)

            # plt.plot(stat['epochs'], stat['train_losses'], 'o-', label='train loss', ms=4)  # , alpha=0.7, label='0.01', lw=5, mec='b', mew=1, ms=7)
            # plt.plot(stat['epochs'], stat['valid_losses'], 'o-.', label='valid loss', ms=4)  # , alpha=0.7, label='0.1', lw=5, mec='b', mew=1, ms=7)
            plt.plot(stat['epochs'], stat['acc'], 'o--',
                     label='Max accuracy:' + str(stat['acc'][best_id]) + '\nEpoch:' + str(stat['epochs'][best_id]),
                     ms=4)  # , alpha=0.7, label='0.3', lw=5, mec='b', mew=1, ms=7)

            plt.legend(fontsize=18,
                       ncol=2,  # количество столбцов
                       facecolor='oldlace',  # цвет области
                       edgecolor='black',  # цвет крайней линии
                       title='value',  # заголовок
                       title_fontsize='18'  # размер шрифта заголовка
                       )
            plt.grid(True)
            plt.savefig(op.join(ep_save_pt, 'graph.png'))

            with open(op.join(ep_save_pt, 'info.txt'), 'w') as info_txt:
                info_txt.write(config['description'] + '\n')
                for el in zip(stat['epochs'], stat['acc']):
                    info_txt.write(str(el[0]) + ' ' + str(el[1]) + '\n')

        if e % config["batch_settings"]["make_clust_on_ep"] == 0:
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
