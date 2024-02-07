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
from eval_model import validate


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
    print(target)
    for index, data in enumerate(data_loader, target):
        print(index)
        return data
    return None


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin  # margin or radius

    def forward(self, output1, output2, label):
        # d = 0 means y1 and y2 are supposed to be same
        # d = 1 means y1 and y2 are supposed to be different

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


def go_triplet_train(train_loader, config, recognizer, optimizer, loss, train_loss, save_im_pt, e, ideals, counter):
    # pbar = tqdm(range(config["batch_settings"]["iterations"]))
    pbar = tqdm(train_loader)
    # for idx in pbar:
    for idx, mb in enumerate(pbar):
        # mb = Dataloader_by_Index(train_loader, torch.randint(len(train_loader), size=(1,)).item())
        anchor, positive, negative = mb['image'][0].cuda(), mb['image'][1].cuda(), mb['image'][2].cuda()
        a_lbl, p_lbl, n_lbl = mb['label'][0].cuda(), mb['label'][1].cuda(), mb['label'][2].cuda()

        anchor_out, positive_out, negative_out = recognizer(anchor), recognizer(positive), recognizer(negative)

        if idx == 0 and config["save_images"]:
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

        cur_loss = loss(anchor_out, positive_out, negative_out)

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        train_loss += cur_loss.item()

        # print("epoch %d Train [Loss %.6f]" % (e, cur_loss.item()))
        pbar.set_description("epoch %d Train [Loss %.6f]" % (e, cur_loss.item()))

    return train_loss


def go_contrastive_train(train_loader, config, recognizer, optimizer, loss, train_loss, save_im_pt, e, ideals, counter):
    pbar = tqdm(range(config["batch_settings"]["iterations"]))
    for it in pbar:
        mb = Dataloader_by_Index(train_loader, torch.randint(len(train_loader), size=(1,)).item())
        first, second = mb['image'][0].cuda(), mb['image'][1].cuda()
        f_lbl, s_lbl = mb['label'][0].cuda(), mb['label'][1].cuda()

        first_out, second_out = recognizer(first), recognizer(second)

        if it == 0 and config["save_images"]:
            save_image(first[0], os.path.join(save_im_pt, 'out_pos_train' + str(e) + '.png'))
            save_image(second[0], os.path.join(save_im_pt, 'out_neg_train' + str(e) + '.png'))

        # save ideals
        ideals[f_lbl] += first_out.detach()
        ideals[s_lbl] += second_out.detach()
        counter[f_lbl] += 1
        counter[s_lbl] += 1

        cur_loss = loss(first_out, second_out, (f_lbl == s_lbl).long())

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        train_loss += cur_loss.item()

        pbar.set_description("epoch %d Train [Loss %.6f]" % (e, cur_loss.item()))

    return train_loss


def go_triplet_test(test_loader, config, recognizer, loss, test_loss, save_im_pt, e):
    # recognizer.eval()
    pbar = tqdm(range(config["batch_settings"]["iterations"]))
    for it in pbar:
        mb = test_loader[torch.randint(len(test_loader), size=(1,)).item()]
        anchor, positive, negative = mb['image'][0].cuda(), mb['image'][1].cuda(), mb['image'][2].cuda()

        anchor_out, positive_out, negative_out = recognizer(anchor), recognizer(positive), recognizer(negative)
        a_lbl, p_lbl, n_lbl = mb['label'][0].cuda(), mb['label'][1].cuda(), mb['label'][2].cuda()
        if it == 0 and config["save_images"]:
            save_image(anchor[0], os.path.join(save_im_pt, 'out_anc_test' + str(e) + '.png'))
            save_image(positive[0], os.path.join(save_im_pt, 'out_pos_test' + str(e) + '.png'))
            save_image(negative[0], os.path.join(save_im_pt, 'out_neg_test' + str(e) + '.png'))

        cur_loss = loss(anchor_out, positive_out, negative_out)

        test_loss += cur_loss.item()
        pbar.set_description("epoch %d Test [Loss %.6f]" % (e, cur_loss.item()))


def go_contrastive_test(test_loader, config, recognizer, loss, test_loss, save_im_pt, e):
    # recognizer.eval()
    pbar = tqdm(range(config["batch_settings"]["iterations"]))
    for it in pbar:
        mb = test_loader[torch.randint(len(test_loader), size=(1,)).item()]
        first, second = mb['image'][0].cuda(), mb['image'][1].cuda()

        first_out, second_out = recognizer(first), recognizer(second)
        a_lbl, p_lbl, n_lbl = mb['label'][0].cuda(), mb['label'][1].cuda(), mb['label'][2].cuda()
        if it == 0 and config["save_images"]:
            save_image(first[0], os.path.join(save_im_pt, 'out_first_test' + str(e) + '.png'))
            save_image(second[0], os.path.join(save_im_pt, 'out_second_test' + str(e) + '.png'))

        cur_loss = loss(first_out, second_out)

        test_loss += cur_loss.item()
        pbar.set_description("epoch %d Test [Loss %.6f]" % (e, cur_loss.item()))


def run_training(config, recognizer, optimizer, train_dataset, test_dataset, valid_dataset, save_pt, save_im_pt,
                 start_ep):
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config['minibatch_size'],
                              shuffle=True,
                              num_workers=10)

    # test_loader = DataLoader(dataset=test_dataset,
    #                           batch_size=1024,
    #                           shuffle=True,
    #                           num_workers=10)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=config['batch_settings']['elements_per_batch'],  # config['minibatch_size'],
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

    metric_type = config['batch_settings']['type']

    ideals = torch.zeros(train_dataset.get_alph_size(), 25).cuda()
    counter = torch.empty(train_dataset.get_alph_size()).fill_(1e-10).cuda()

    min_test_loss = np.inf
    loss = None
    if metric_type == 'triplet':
        loss = nn.TripletMarginLoss(margin=config['batch_settings']['alpha_margin']).cuda()
    elif metric_type == 'contrastive':
        loss = ContrastiveLoss(margin=config['batch_settings']['alpha_margin']).cuda()

    stat = {'epochs': [],
            'train_losses': [],
            'valid_losses': [],
            'acc': []}

    for e in range(start_ep, config['epoch_num']):
        # pbar = tqdm(train_loader)
        # for idx, mb in enumerate(pbar):

        train_loss = 0.0
        start_time = time.time()
        if metric_type == 'triplet':
            train_loss = go_triplet_train(train_loader, config, recognizer, optimizer, loss, train_loss, save_im_pt, e,
                                          ideals,
                                          counter)
        elif metric_type == 'contrastive':
            train_loss = go_contrastive_train(train_loader, config, recognizer, optimizer, loss, train_loss, save_im_pt,
                                              e, ideals,
                                              counter)

        ideals = torch.div(ideals.T, counter).T

        # test_loss = 0.0
        # to_test = False
        # if to_test:
        #     go_test()
        #
        #     print(
        #         f'Epoch {e} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {test_loss / len(test_loader)}')
        #
        #     if min_test_loss > test_loss:
        #         print(
        #             f'Validation Loss Decreased({min_test_loss:.6f}--->{test_loss / len(test_loader):.6f}) \t Saving The Model')
        #         min_valid_loss = test_loss

        ep_save_pt = op.join(save_pt, str(e))
        if not os.path.exists(ep_save_pt):
            os.mkdir(ep_save_pt)

        if config['batch_settings']['negative_mode'] == 'auto_clusters':
            start_time = time.time()
            print('Started generation of clusters')
            train_dataset.update_rules(ideals, ep_save_pt)
            print('Finished generation of clusters {:.2f} sec'.format(time.time() - start_time))

        print('Epoch {} -> Training Loss({:.2f} sec): {}'.format(e, time.time() - start_time, train_loss / len(train_loader)))

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

            acc = validate(config, recognizer, valid_loader, ideals)

            stat['acc'].append(acc.item())

            plt.figure(figsize=(12, 7))
            plt.xlabel("Epoch", fontsize=18)

            # plt.plot(stat['epochs'], stat['train_losses'], 'o-', label='train loss', ms=4)  # , alpha=0.7, label='0.01', lw=5, mec='b', mew=1, ms=7)
            # plt.plot(stat['epochs'], stat['valid_losses'], 'o-.', label='valid loss', ms=4)  # , alpha=0.7, label='0.1', lw=5, mec='b', mew=1, ms=7)
            plt.plot(stat['epochs'], stat['acc'], 'o--', label='Max accuracy:' + str(max(stat['acc'])),
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
