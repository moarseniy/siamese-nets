import json
import os, time
import os.path as op
from tqdm import tqdm
import cv2
from itertools import cycle, islice
import torch

from jinja2.optimizer import optimize
from torch.utils.data import DataLoader
import onnx

import numpy as np
from torch import nn
import torchvision

from torchvision.utils import save_image
from eval_model import *
from loss import ChooseLoss
from utils import save_plot

def Dataloader_by_Index(data_loader, target=0):
    for index, data in enumerate(data_loader, target):
        return data
    return None

def ChooseOptimizer(cfg, model):
    if cfg["type"] == "Adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    elif cfg["type"] == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    elif cfg["type"] == "SGD":
        return torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"])
    else:
        print("Unknown optimizer type!")
        return None

def ChooseScheduler(cfg, optimizer):
    if cfg["type"] == "Cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["T_max"])
    else:
        print("Unknown scheduler type!")
        return None

def go_metric_train(train_loader, config, recognizer, optimizer, loss, save_im_pt, e,
                    save_ideals, ideals, counter, dists, loss_type):

    batch_count = config["batch_settings"]["train"]["iterations"]
    elements_per_batch = config["batch_settings"]["train"]["elements_per_batch"]
    minibatch_size = config["minibatch_size"]
    train_mb_count = elements_per_batch / minibatch_size

    train_loss = 0.0
    pbar = tqdm(train_loader)
    for idx, cur_batch in enumerate(pbar):
        batch_img = cur_batch['image']  #.to(device='cuda', non_blocking=True)
        batch_lbl = cur_batch['label']  #.to(device='cuda', non_blocking=True)

        minibatch_imgs = torch.split(batch_img, minibatch_size)
        minibatch_lbls = torch.split(batch_lbl, minibatch_size)

        mb_train_loss = 0
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
                    # print(ideals.size(), lbl.size(), out.size())
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

            mb_train_loss += cur_loss.item()

            # print("epoch %d Train [Loss %.6f]" % (e, cur_loss.item()))
            pbar.set_description("Epoch %d Train [Loss %.6f]" % (e, cur_loss.item()))
        train_loss += mb_train_loss / train_mb_count

    return train_loss / len(train_loader)


def go_metric_test(test_loader, config, recognizer, loss, save_im_pt, e, loss_type):
    batch_count = config["batch_settings"]["test"]["iterations"]
    elements_per_batch = config["batch_settings"]["test"]["elements_per_batch"]
    minibatch_size = config["minibatch_size"]
    test_mb_count = elements_per_batch / minibatch_size

    test_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(test_loader)
        for idx, cur_batch in enumerate(pbar):
            batch_img = cur_batch['image'].to(device='cuda', non_blocking=True)
            batch_lbl = cur_batch['label'].to(device='cuda', non_blocking=True)

            minibatch_imgs = torch.split(batch_img, minibatch_size)
            minibatch_lbls = torch.split(batch_lbl, minibatch_size)

            mb_test_loss = 0
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

                mb_test_loss += cur_loss.item()

                # print("epoch %d Train [Loss %.6f]" % (e, cur_loss.item()))
                pbar.set_description("Epoch %d Test [Loss %.6f]" % (e, cur_loss.item()))
            test_loss += mb_test_loss / test_mb_count

    return test_loss / len(test_loader)


def run_training(config, recognizer, optimizer, scheduler,
                 train_dataset, test_dataset, valid_dataset,
                 save_pt, save_im_pt, start_ep):
    num_workers = config.get('num_workers', os.cpu_count() - 1)
    if num_workers == -1:
        num_workers = os.cpu_count() - 1

    with open(op.join(save_pt, 'info.txt'), 'a') as info_txt:
        info_txt.write(config['description'] + '\n')

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config['batch_settings']['train']['elements_per_batch'],
                              shuffle=False,
                              num_workers=num_workers)
    print(f"Train DataLoader is initialized with {len(train_loader)} batches")

    test_loader = None
    if test_dataset:
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=config['batch_settings']['test']['elements_per_batch'],
                                 shuffle=False,
                                 num_workers=num_workers)
        print(f"Test DataLoader is initialized with {len(test_loader)} batches")

    valid_loader = None
    if valid_dataset:
        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=config['batch_settings']['valid']['elements_per_batch'],
                                  shuffle=False,
                                  num_workers=num_workers)
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

        start_time = time.time()

        if save_ideals:
            ideals = (ideals.T * counter).T

        train_loss = go_metric_train(train_loader, config, recognizer, optimizer, loss, save_im_pt, e,
                                     save_ideals, ideals, counter, dists, loss_type)

        if save_ideals:
            ideals = torch.div(ideals.T, counter).T

        stat['train_losses'].append(train_loss)

        print('Epoch {} -> Training Loss({:.2f} sec): {}'.format(e, time.time() - start_time, train_loss))
        test_loss = 0.0
        if test_loader:
            start_time = time.time()

            test_loss = go_metric_test(test_loader, config, recognizer, loss, save_im_pt, e, loss_type)

            stat['test_losses'].append(test_loss)

            print('Epoch {} -> Test Loss({:.2f} sec): {}'.format(e, time.time() - start_time, test_loss))

            if min_test_loss > (test_loss):
                print(f'Test Loss Decreased({min_test_loss:.6f}--->{(test_loss):.6f})')
                min_test_loss = test_loss

        if valid_loader:
            if "oneshot" in config:
                acc = validate_oneshot(config, recognizer)
            elif save_ideals:
                acc = validate_with_descrs(config, recognizer, valid_loader, ideals)
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
            'loss': test_loss,
        }, op.join(ep_save_pt, "model.pt"))

        dummy_input = torch.randn(1, config["image_size"]["channels"],
                                     config["image_size"]["height"],
                                     config["image_size"]["width"]).to(next(recognizer.parameters()).device)

        torch.onnx.export(
            recognizer,
            dummy_input,
            op.join(ep_save_pt, "model.onnx"),
            export_params=True,
            # opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )

        save_plot(stat, save_pt)

        with open(op.join(save_pt, 'info.txt'), 'a') as info_txt:
            info_txt.write(str(e) + ' ' + str(train_loss) + ' ')
            if test_loss:
                info_txt.write(str(test_loss))
            info_txt.write('\n')

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


