import json
import os, time
import os.path as op
from datetime import datetime
import cv2
from itertools import cycle, islice
import torch
import matplotlib.pyplot as plt
from jinja2.optimizer import optimize
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


def ChooseOptimizer(cfg, model):
    if cfg["type"] == "Adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    elif cfg["type"] == "SGD":
        return None
    else:
        print("Unknown optimizer type!")
        return None


def ChooseLoss(cfg):
    loss_type = cfg["type"]
    if loss_type == 'triplet':
        loss = nn.TripletMarginLoss(margin=cfg['alpha_margin']).cuda()
    elif loss_type == 'contrastive':
        loss = ContrastiveLoss(margin=cfg['alpha_margin']).cuda()
    elif loss_type == 'BCELoss':
        loss = nn.BCELoss().cuda()
    elif loss_type == 'BCEWithLogitsLoss':
        loss = nn.BCEWithLogitsLoss().cuda()
    elif loss_type == 'metric':
        loss = MetricLoss(margin=cfg['alpha_margin'])
    else:
        print('No Loss found!')
        exit(-1)
    return loss


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

        f = nn.Softplus(threshold=20000)
        triplet_loss = nn.TripletMarginLoss()

        g1 = ro * d_AP
        g2 = tau * f(d_AP - d_AN + self.margin)  # triplet_loss(anc, pos, neg) #
        g3 = xi * (d_AIdeal + d_PIdeal + d_NIdeal) / 3.0

        loss_metric = (g1 * g1) + (g2 * g2) + (g3 * g3)
        return loss_metric.mean()


def save_plot(stat, ep_save_pt):
    plt.figure(figsize=(12, 7))
    plt.xlabel("Epoch", fontsize=18)

    plt.plot(stat['epochs'], stat['train_losses'], 'o-', label='train loss',
             ms=4)  # , alpha=0.7, label='0.01', lw=5, mec='b', mew=1, ms=7)

    if stat['test_losses']:
        plt.plot(stat['epochs'], stat['test_losses'], 'o-.', label='test loss',
                 ms=4)  # , alpha=0.7, label='0.1', lw=5, mec='b', mew=1, ms=7)

    if stat['acc']:
        best_id = stat['acc'].index(max(stat['acc']))
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


def go_metric_train(train_loader, config, recognizer, optimizer, loss, train_loss, save_im_pt, e,
                    save_ideals, ideals, counter, dists, loss_type):

    batch_count = config["batch_settings"]["train"]["iterations"]
    elements_per_batch = config["batch_settings"]["train"]["elements_per_batch"]
    minibatch_size = config["minibatch_size"]

    pbar = tqdm(train_loader)
    for idx, cur_batch in enumerate(pbar):
        batch_img = cur_batch['image']  #.to(device='cuda', non_blocking=True)
        batch_lbl = cur_batch['label']  #.to(device='cuda', non_blocking=True)

        minibatch_imgs = torch.split(batch_img, minibatch_size)
        minibatch_lbls = torch.split(batch_lbl, minibatch_size)

        for mb_idx, (mb_img, mb_lbl) in enumerate(zip(minibatch_imgs, minibatch_lbls)):
            mb_img = mb_img.to(device='cuda', non_blocking=True)
            mb_lbl = mb_lbl.to(device='cuda', non_blocking=True)

            anc_img = mb_img[:, 0]
            pos_img = mb_img[:, 1]
            neg_img = mb_img[:, 2]

            anc_lbl = mb_lbl[:, 0]
            pos_lbl = mb_lbl[:, 1]
            neg_lbl = mb_lbl[:, 2]

            assert torch.equal(anc_lbl, pos_lbl)

            anc_out = recognizer(anc_img)
            pos_out = recognizer(pos_img)
            neg_out = recognizer(neg_img)

            if save_ideals:
                for out, lbl in ((anc_out, anc_lbl), (pos_out, pos_lbl), (neg_out, neg_lbl)):

                    ideals.scatter_add_(0, lbl.unsqueeze(1).expand(-1, ideals.size(1)), out.detach())

                    counter += torch.bincount(lbl, minlength=ideals.size(0))

                    dists.scatter_add_(0, lbl,
                                       torch.nn.functional.pairwise_distance(out.detach(), ideals[lbl]))

            if config["save_images"] and idx == 0 and mb_idx == 0:
                save_image(anc_img[0], os.path.join(save_im_pt, str(e) + '_train_anc_' +
                                                    str(int(anc_lbl[0])) + '.png'))
                save_image(pos_img[0], os.path.join(save_im_pt, str(e) + '_train_pos_' +
                                                    str(int(pos_lbl[0])) + '.png'))
                save_image(neg_img[0], os.path.join(save_im_pt, str(e) + '_train_neg_' +
                                                    str(int(neg_lbl[0])) + '.png'))

            # if loss_type == "triplet":
            cur_loss = loss(anc_out, pos_out, neg_out)
            # elif loss_type == "contrastive":
            #     cur_loss = loss(img_out[0], img_out[1], (lbl_out[0] == lbl_out[1]).long())
            # elif loss_type == "metric":
            #     cur_loss = loss(img_out[0], img_out[1], img_out[2],
            #                     ideals[lbl_out[0]], ideals[lbl_out[1]], ideals[lbl_out[2]])
            # elif loss_type == "BCE":
            #     cur_loss = loss()
            # else:
            #     print('No type!')
            #     exit(-1)

            optimizer.zero_grad()
            cur_loss.backward()
            optimizer.step()

            train_loss += cur_loss.item()

            # print("epoch %d Train [Loss %.6f]" % (e, cur_loss.item()))
            pbar.set_description("Epoch %d Train [Loss %.6f]" % (e, cur_loss.item()))

    return train_loss


def go_metric_test(test_loader, config, recognizer, loss, test_loss, save_im_pt, e, loss_type):
    batch_count = config["batch_settings"]["train"]["iterations"]
    elements_per_batch = config["batch_settings"]["train"]["elements_per_batch"]
    minibatch_size = config["minibatch_size"]

    pbar = tqdm(test_loader)
    for idx, cur_batch in enumerate(pbar):
        batch_img = cur_batch['image'].to(device='cuda', non_blocking=True)
        batch_lbl = cur_batch['label'].to(device='cuda', non_blocking=True)

        minibatch_imgs = torch.split(batch_img, minibatch_size)
        minibatch_lbls = torch.split(batch_lbl, minibatch_size)

        for mb_idx, (mb_img, mb_lbl) in enumerate(zip(minibatch_imgs, minibatch_lbls)):

            anc_img = mb_img[:, 0]
            pos_img = mb_img[:, 1]
            neg_img = mb_img[:, 2]

            anc_lbl = mb_lbl[:, 0]
            pos_lbl = mb_lbl[:, 1]
            neg_lbl = mb_lbl[:, 2]

            assert torch.equal(anc_lbl, pos_lbl)

            anc_out = recognizer(anc_img)
            pos_out = recognizer(pos_img)
            neg_out = recognizer(neg_img)

            if config["save_images"] and idx == 0 and mb_idx == 0:
                save_image(anc_img[0], os.path.join(save_im_pt, str(e) + '_test_anc_' +
                                                    str(int(anc_lbl[0])) + '.png'))
                save_image(pos_img[0], os.path.join(save_im_pt, str(e) + '_test_pos_' +
                                                    str(int(pos_lbl[0])) + '.png'))
                save_image(neg_img[0], os.path.join(save_im_pt, str(e) + '_test_neg_' +
                                                    str(int(neg_lbl[0])) + '.png'))

            cur_loss = loss(anc_out, pos_out, neg_out)

            test_loss += cur_loss.item()

            # print("epoch %d Train [Loss %.6f]" % (e, cur_loss.item()))
            pbar.set_description("Epoch %d Test [Loss %.6f]" % (e, cur_loss.item()))

    return test_loss


def run_training(config, recognizer, optimizer, train_dataset, test_dataset, valid_dataset, save_pt, save_im_pt,
                 start_ep):
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config['batch_settings']['train']['elements_per_batch'],
                              shuffle=False,
                              num_workers=os.cpu_count() - 1)
    print(f"Train DataLoader is initialized with {len(train_loader)} batches")

    test_loader = None
    if test_dataset:
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=config['batch_settings']['test']['elements_per_batch'],
                                 shuffle=False,
                                 num_workers=os.cpu_count() - 1)
        print(f"Test DataLoader is initialized with {len(test_loader)} batches")

    valid_loader = None
    if valid_dataset:
        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=config['batch_settings']['valid']['elements_per_batch'],
                                  shuffle=False,
                                  num_workers=os.cpu_count() - 1)
        print(f"Valid DataLoader is initialized with {len(valid_loader)} batches")

    # if config['validate_settings']['validate'] and config['file_to_start']:
    #     print('Validation started!')
    #     validate(config, recognizer, valid_loader)
    #     exit(0)

    save_ideals = config["save_ideals"]
    loss_type = config['loss_settings']['type']

    ideals, counter, dists = None, None, None
    if save_ideals:
        ideals = torch.zeros(train_dataset.get_alph_size(), config["image_size"]["output_dim"]).cuda()
        counter = torch.empty(train_dataset.get_alph_size(), dtype=torch.int32).fill_(1e-10).cuda()
        dists = torch.zeros(train_dataset.get_alph_size()).cuda()

    min_test_loss = np.inf
    loss = ChooseLoss(config["loss_settings"])

    stat = {'epochs': [],
            'train_losses': [],
            'valid_losses': [],
            'test_losses': [],
            'acc': []}

    for e in range(start_ep, config['epoch_num']):
        stat['epochs'].append(e)

        train_loss = 0.0
        start_time = time.time()

        if save_ideals:
            ideals = (ideals.T * counter).T

        train_loss = go_metric_train(train_loader, config, recognizer, optimizer, loss, train_loss, save_im_pt, e,
                                     save_ideals, ideals, counter, dists, loss_type)

        if save_ideals:
            ideals = torch.div(ideals.T, counter).T

        stat['train_losses'].append(train_loss / len(train_loader))

        print('Epoch {} -> Training Loss({:.2f} sec): {}'.format(e, time.time() - start_time,
                                                                 train_loss / len(train_loader)))
        test_loss = 0.0
        if test_loader:
            start_time = time.time()

            test_loss = go_metric_test(test_loader, config, recognizer, loss, test_loss, save_im_pt, e, loss_type)

            stat['test_losses'].append(test_loss / len(test_loader))

            print('Epoch {} -> Test Loss({:.2f} sec): {}'.format(e, time.time() - start_time,
                                                                 test_loss / len(test_loader)))

            if min_test_loss > (test_loss / len(test_loader)):
                print(
                    f'Test Loss Decreased({min_test_loss:.6f}--->{(test_loss / len(test_loader)):.6f})')
                min_test_loss = test_loss / len(test_loader)

        if valid_loader:
            acc = validate_with_descrs(config, recognizer, valid_loader, ideals)
            # acc = validate_oneshot(config, recognizer, valid_loader)

            stat['acc'].append(acc.item())

        ep_save_pt = op.join(save_pt, str(e))
        if not os.path.exists(ep_save_pt):
            os.makedirs(ep_save_pt)

        # start_time = time.time()
        # print('Started updating rules!')
        train_dataset.update_rules("train", save_ideals, ideals, counter, dists, ep_save_pt, config, e)
        # print('Finished updating rules {:.2f} sec'.format(time.time() - start_time))

        # print('Epoch {} -> Training Loss({:.2f} sec): {}'.format(e, time.time() - start_time,
        #                                                          train_loss / len(train_loader)))

        torch.save({
            'epoch': e,
            'model_state_dict': recognizer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': 0.0,  # test_loss,
        }, op.join(ep_save_pt, "model.pt"))

        save_plot(stat, ep_save_pt)

        with open(op.join(ep_save_pt, 'info.txt'), 'w') as info_txt:
            info_txt.write(config['description'] + '\n')
            for el in zip(stat['epochs'], stat['test_losses']):
                info_txt.write(str(el[0]) + ' ' + str(el[1]) + '\n')

        if (save_ideals and config['batch_settings']['train']['negative_mode'] == "auto_clusters" and
                e % config["batch_settings"]['train']["make_clust_on_ep"] == 0):
            # ideals = torch.div(ideals.T, counter).T
            ideals_dict = {}
            counter_dict = {}

            for i in range(train_dataset.get_alph_size()):
                ideals_dict[str(i)] = ideals[i].cpu().tolist()
                counter_dict[str(i)] = counter[i].cpu().item()

            with open(op.join(ep_save_pt, 'ideals.json'), 'w') as out:
                json.dump(ideals_dict, out)
            with open(op.join(ep_save_pt, 'counter.json'), 'w') as out:
                json.dump(counter_dict, out)

            ideals = torch.zeros(train_dataset.get_alph_size(), config["image_size"]["output_dim"]).cuda()
            counter = torch.empty(train_dataset.get_alph_size(), dtype=torch.int32).fill_(1e-10).cuda()
            dists = torch.zeros(train_dataset.get_alph_size()).cuda()


def prepare_dirs(config, device_num):
    save_paths = {}
    files_to_start = {}
    from_file = False
    start_ep = 0
    checkpoint_pt = config["checkpoint_pt"]
    images_pt = config["images_pt"]
    if not op.exists(checkpoint_pt):
        os.mkdir(checkpoint_pt)

    now = datetime.now()
    dt_string = str(device_num) + '_' + now.strftime("%d-%m-%Y-%H-%M")

    # if config['file_to_start'] != "":
    #     dir = config['file_to_start'].split("/")[0]
    #     chpnt = config['file_to_start'].split("/")[1]
    #     start_ep = int(chpnt.split(".")[0]) + 1
    #
    #     # assert sum([op.exists(pt) for pt in files_to_start.values()]) == 4
    #     from_file = True
    # else:
    save_pt = op.join(checkpoint_pt, dt_string)
    print('Checkpoint path:', save_pt)
    save_im_pt = op.join(images_pt, dt_string)
    if not op.exists(save_pt):
        os.makedirs(save_pt)
    if not op.exists(save_im_pt):
        os.makedirs(save_im_pt)

    return save_pt, save_im_pt
