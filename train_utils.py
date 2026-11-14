import json
import os, time
import os.path as op
from tqdm import tqdm
import torch

from torch.utils.data import DataLoader

import numpy as np

from torchvision.utils import save_image
from eval_model import *
from loss import ChooseLoss
from metrics import compute_eer, compute_similarity
from utils import save_plot, kill_if_memory_usage_above

def next_loader_batch(data_loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(data_loader)
        return next(iterator), iterator

def ChooseOptimizer(cfg, model):
    if cfg["type"] == "Adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    elif cfg["type"] == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01))
    elif cfg["type"] == "SGD":
        return torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg.get("momentum", 0.0),
                               weight_decay=cfg.get("weight_decay", 0.0))
    else:
        raise ValueError(f"Unknown optimizer type: {cfg['type']!r}")

def ChooseScheduler(cfg, optimizer):
    if not cfg or cfg.get("type") in (None, "None", "none", "off"):
        return None
    if cfg["type"] == "Cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["T_max"])
    else:
        raise ValueError(f"Unknown scheduler type: {cfg['type']!r}")

def select_batch_negatives(anc_out, pos_out, neg_out, anc_lbl, neg_lbl, mining_mode, margin):
    """Mine negatives from this batch's sampled negative pool."""
    mining_mode = {'batch-hard': 'batch_hard', 'semi-hard': 'semi_hard'}.get(mining_mode, mining_mode)
    if mining_mode not in ('hard', 'batch_hard', 'semi_hard'):
        return neg_out

    distances = torch.cdist(anc_out, neg_out, p=2)
    valid = anc_lbl[:, None] != neg_lbl[None, :]
    distances = distances.masked_fill(~valid, float('inf'))
    closest_indices = distances.argmin(dim=1)

    if mining_mode == 'semi_hard':
        positive_distances = torch.nn.functional.pairwise_distance(anc_out, pos_out)
        semi_hard = valid & (distances > positive_distances[:, None]) & \
                    (distances < (positive_distances + margin)[:, None])
        semi_distances = distances.masked_fill(~semi_hard, float('inf'))
        has_semi_hard = semi_hard.any(dim=1)
        semi_indices = semi_distances.argmin(dim=1)
        selected_indices = torch.where(has_semi_hard, semi_indices, closest_indices)
    else:
        selected_indices = closest_indices

    return neg_out[selected_indices]


def select_pair_negatives(embeddings, labels, class_ids, mining_mode, margin):
    """Replace negative pair endpoints with closer cross-class candidates in the batch."""
    mining_mode = {'batch-hard': 'batch_hard', 'semi-hard': 'semi_hard'}.get(mining_mode, mining_mode)
    if mining_mode not in ('hard', 'batch_hard', 'semi_hard'):
        return embeddings[:, 1]

    anchors = embeddings[:, 0]
    candidates = embeddings[:, 1]
    distances = torch.cdist(anchors, candidates, p=2)
    valid = class_ids[:, :1] != class_ids[:, 1].unsqueeze(0)
    distances = distances.masked_fill(~valid, float('inf'))
    closest = distances.argmin(dim=1)

    if mining_mode == 'semi_hard':
        within_margin = valid & (distances < margin)
        semi_distances = distances.masked_fill(~within_margin, float('inf'))
        has_candidate = within_margin.any(dim=1)
        closest_within_margin = semi_distances.argmin(dim=1)
        selected = torch.where(has_candidate, closest_within_margin, closest)
    else:
        selected = closest

    mined = candidates[selected]
    return torch.where(labels[:, None] == 1, mined, embeddings[:, 1])

def go_train(train_loader, config, recognizer, optimizer, loss, save_im_pt, e,
             save_ideals, ideals, counter, dists, loss_type):
    batch_count = config["batch_settings"]["train"]["iterations"]
    mining_mode = config['batch_settings']['train'].get('negative_mode', 'random')
    device = next(recognizer.parameters()).device
    train_loss = 0.0
    data_iter = iter(train_loader)
    pbar = tqdm(range(batch_count))
    for idx in pbar:
        mb, data_iter = next_loader_batch(train_loader, data_iter)
        mb_img = mb['image'].to(device=device, non_blocking=True)
        mb_lbl = mb['label'].to(device=device, non_blocking=True)

        anc_img, pos_img, neg_img = mb_img[:, 0], mb_img[:, 1], mb_img[:, 2]
        anc_lbl, pos_lbl, neg_lbl = mb_lbl[:, 0].long(), mb_lbl[:, 1].long(), mb_lbl[:, 2].long()
        if not torch.equal(anc_lbl, pos_lbl):
            raise ValueError('Triplet dataset must return matching anchor and positive class labels')

        anc_out = recognizer(anc_img)
        pos_out = recognizer(pos_img)
        neg_out = recognizer(neg_img)

        if save_ideals:
            for out, lbl in ((anc_out, anc_lbl), (pos_out, pos_lbl), (neg_out, neg_lbl)):
                ideals.scatter_add_(0, lbl.unsqueeze(1).expand(-1, ideals.size(1)), out.detach())
                counter += torch.bincount(lbl, minlength=ideals.size(0)).to(counter.dtype)
                dists.scatter_add_(0, lbl, torch.nn.functional.pairwise_distance(out.detach(), ideals[lbl]))

        if config["save_images"] and idx == 0:
            save_image(anc_img[0], os.path.join(save_im_pt, str(e) + '_train_anc_' + str(int(anc_lbl[0])) + '.png'))
            save_image(pos_img[0], os.path.join(save_im_pt, str(e) + '_train_pos_' + str(int(pos_lbl[0])) + '.png'))
            save_image(neg_img[0], os.path.join(save_im_pt, str(e) + '_train_neg_' + str(int(neg_lbl[0])) + '.png'))

        selected_neg = select_batch_negatives(anc_out, pos_out, neg_out, anc_lbl, neg_lbl,
                                              mining_mode, getattr(loss, 'margin', 0.0))
        if loss_type == 'metric':
            safe_counts = counter.clamp_min(1).to(ideals.dtype)
            centers = ideals / safe_counts[:, None]
            cur_loss = loss(anc_out, pos_out, selected_neg,
                            centers[anc_lbl], centers[pos_lbl], centers[neg_lbl])
        else:
            cur_loss = loss(anc_out, pos_out, selected_neg)

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        train_loss += cur_loss.item()
        pbar.set_description("Epoch %d Train [Loss %.6f]" % (e, cur_loss.item()))

    torch.cuda.empty_cache()
    pbar.close()
    return train_loss / max(1, batch_count)


def go_pair_train(train_loader, config, recognizer, optimizer, loss, e,
                  save_ideals=False, ideals=None, counter=None, dists=None):
    batch_count = config['batch_settings']['train']['iterations']
    device = next(recognizer.parameters()).device
    total_loss = 0.0
    data_iter = iter(train_loader)
    pbar = tqdm(range(batch_count))
    for idx in pbar:
        batch, data_iter = next_loader_batch(train_loader, data_iter)
        images = batch['image'].to(device=device, non_blocking=True)
        labels = batch['label'].to(device=device, dtype=torch.float32, non_blocking=True).view(-1)
        batch_size = images.size(0)
        embeddings = recognizer(images.flatten(0, 1)).reshape(batch_size, 2, -1)
        class_ids = batch['class_id'].to(device=device, dtype=torch.long, non_blocking=True)
        if save_ideals:
            for side in range(2):
                side_ids = class_ids[:, side]
                side_embeddings = embeddings[:, side].detach()
                ideals.scatter_add_(0, side_ids[:, None].expand(-1, ideals.size(1)), side_embeddings)
                counter += torch.bincount(side_ids, minlength=ideals.size(0)).to(counter.dtype)
                dists.scatter_add_(0, side_ids,
                                   torch.nn.functional.pairwise_distance(side_embeddings, ideals[side_ids]))
        mining_mode = config['batch_settings']['train'].get('negative_mode', 'random')
        mined_right = select_pair_negatives(embeddings, labels, class_ids, mining_mode,
                                            getattr(loss, 'margin', 0.0))
        cur_loss = loss(embeddings[:, 0], mined_right, labels)
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        total_loss += cur_loss.item()
        pbar.set_description("Epoch %d Train [Loss %.6f]" % (e, cur_loss.item()))
    return total_loss / max(1, batch_count)


@torch.no_grad()
def go_test(test_loader, config, recognizer, loss, save_im_pt, e, loss_type, sim_list, lbl_list, ideals=None):
    batch_count = config["batch_settings"]["test"]["iterations"]
    mining_mode = config['batch_settings']['test'].get('negative_mode', 'random')
    device = next(recognizer.parameters()).device
    test_loss = 0.0
    data_iter = iter(test_loader)
    pbar = tqdm(range(batch_count))
    for idx in pbar:
        mb, data_iter = next_loader_batch(test_loader, data_iter)
        mb_img = mb['image'].to(device=device, non_blocking=True)
        mb_lbl = mb['label'].to(device=device, non_blocking=True)
        anc_img, pos_img, neg_img = mb_img[:, 0], mb_img[:, 1], mb_img[:, 2]
        anc_lbl, pos_lbl, neg_lbl = mb_lbl[:, 0].long(), mb_lbl[:, 1].long(), mb_lbl[:, 2].long()
        if not torch.equal(anc_lbl, pos_lbl):
            raise ValueError('Triplet dataset must return matching anchor and positive class labels')

        anc_out, pos_out, neg_out = recognizer(anc_img), recognizer(pos_img), recognizer(neg_img)
        sim_list.extend(compute_similarity(anc_out, pos_out))
        lbl_list += torch.ones(anc_out.size(0)).tolist()
        sim_list.extend(compute_similarity(pos_out, neg_out))
        lbl_list += torch.zeros(anc_out.size(0)).tolist()
        sim_list.extend(compute_similarity(anc_out, neg_out))
        lbl_list += torch.zeros(anc_out.size(0)).tolist()

        selected_neg = select_batch_negatives(anc_out, pos_out, neg_out, anc_lbl, neg_lbl,
                                              mining_mode, getattr(loss, 'margin', 0.0))
        if loss_type == 'metric':
            if ideals is None:
                raise ValueError('Metric loss evaluation requires class prototypes')
            cur_loss = loss(anc_out, pos_out, selected_neg,
                            ideals[anc_lbl], ideals[pos_lbl], ideals[neg_lbl])
        else:
            cur_loss = loss(anc_out, pos_out, selected_neg)
        test_loss += cur_loss.item()
        pbar.set_description("Epoch %d Test [Loss %.6f]" % (e, cur_loss.item()))
    return test_loss / max(1, batch_count)


@torch.no_grad()
def go_pair_test(test_loader, config, recognizer, loss, sim_list, lbl_list):
    batch_count = config['batch_settings']['test']['iterations']
    device = next(recognizer.parameters()).device
    total_loss = 0.0
    data_iter = iter(test_loader)
    pbar = tqdm(range(batch_count))
    for _ in pbar:
        batch, data_iter = next_loader_batch(test_loader, data_iter)
        images = batch['image'].to(device=device, non_blocking=True)
        labels = batch['label'].to(device=device, dtype=torch.float32, non_blocking=True).view(-1)
        batch_size = images.size(0)
        embeddings = recognizer(images.flatten(0, 1)).reshape(batch_size, 2, -1)
        cur_loss = loss(embeddings[:, 0], embeddings[:, 1], labels)
        similarities = compute_similarity(embeddings[:, 0], embeddings[:, 1])
        sim_list.extend(similarities)
        lbl_list.extend((1 - labels).detach().cpu().tolist())
        total_loss += cur_loss.item()
        pbar.set_description("Test [Loss %.6f]" % cur_loss.item())
    return total_loss / max(1, batch_count)

@kill_if_memory_usage_above(70)
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
                              shuffle=True,
                              num_workers=num_workers)
    print(f"Train DataLoader is initialized with {len(train_loader)} batches")

    test_loader = None
    if test_dataset:
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=config['batch_settings']['test']['elements_per_batch'],
                                 shuffle=True,
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

    loss_type = config['loss_settings']['type']
    save_ideals = config["save_ideals"]
    if loss_type not in ('triplet', 'metric', 'contrastive', 'pair'):
        raise ValueError(f"Training supports triplet, metric, and contrastive losses; got {loss_type!r}")
    mining_mode = config['batch_settings']['train'].get('negative_mode', 'random')
    if mining_mode == 'auto_clusters1':
        mining_mode = 'auto_clusters'
    mining_mode = {'batch-hard': 'batch_hard', 'semi-hard': 'semi_hard'}.get(mining_mode, mining_mode)
    if loss_type == 'metric' and not save_ideals:
        raise ValueError("Metric loss requires save_ideals=true for class prototypes")
    if mining_mode == 'auto_clusters' and not save_ideals:
        raise ValueError("negative_mode='auto_clusters' requires save_ideals=true")

    with open(op.join(save_pt, 'config.json'), 'w', encoding='utf8') as config_file:
        json.dump(config, config_file, indent=2, ensure_ascii=False)

    ideals, counter, dists = None, None, None
    if save_ideals:
        device = next(recognizer.parameters()).device
        ideals = torch.zeros(train_dataset.get_alph_size(), config["image_size"]["output_dim"], device=device)
        counter = torch.zeros(train_dataset.get_alph_size(), dtype=torch.long, device=device)
        dists = torch.zeros(train_dataset.get_alph_size(), device=device)

    min_test_loss = np.inf
    loss = ChooseLoss(config["loss_settings"])

    stat = {'epochs': [],
            'train_losses': [],
            'valid_losses': [],
            'test_losses': [],
            'metrics': [],
            'acc': []}

    for e in range(start_ep, config['epoch_num']):
        stat['epochs'].append(e)
        recognizer.train()

        start_time = time.time()

        if save_ideals:
            ideals = ideals * counter.to(ideals.dtype)[:, None]

        if loss_type in ('contrastive', 'pair'):
            train_loss = go_pair_train(train_loader, config, recognizer, optimizer, loss, e,
                                       save_ideals, ideals, counter, dists)
        else:
            train_loss = go_train(train_loader, config, recognizer, optimizer, loss, save_im_pt, e,
                                  save_ideals, ideals, counter, dists, loss_type)

        if save_ideals:
            ideals = ideals / counter.clamp_min(1).to(ideals.dtype)[:, None]

        if scheduler is not None:
            scheduler.step()

        stat['train_losses'].append(train_loss)

        print('Epoch {} -> Training Loss({:.2f} sec): {}'.format(e, time.time() - start_time, train_loss))
        test_loss, metrics = 0.0, 0.0
        if test_loader:
            start_time = time.time()

            sim_list, lbl_list = [], []
            recognizer.eval()
            if loss_type in ('contrastive', 'pair'):
                test_loss = go_pair_test(test_loader, config, recognizer, loss, sim_list, lbl_list)
            else:
                test_loss = go_test(test_loader, config, recognizer, loss, save_im_pt, e,
                                    loss_type, sim_list, lbl_list, ideals)
            recognizer.train()

            metrics = compute_eer(lbl_list, sim_list)
            print(f"EER metric: {metrics}")
            stat['metrics'].append(metrics)

            stat['test_losses'].append(test_loss)

            print('Epoch {} -> Test Loss({:.2f} sec): {}'.format(e, time.time() - start_time, test_loss))

            if min_test_loss > (test_loss):
                print(f'Test Loss Decreased({min_test_loss:.6f}--->{(test_loss):.6f})')
                min_test_loss = test_loss

        if "oneshot" in config:
            recognizer.eval()
            acc = validate_oneshot(config, recognizer)
            recognizer.train()
            stat['acc'].append(float(acc))
        elif valid_loader and save_ideals:
            recognizer.eval()
            with torch.no_grad():
                acc = validate_with_descrs(config, recognizer, valid_loader, ideals)
            recognizer.train()
            stat['acc'].append(float(acc))

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
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
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

        try:
            save_plot(stat, save_pt)
        except Exception as exc:
            print("save_plot: ", exc)

        with open(op.join(save_pt, 'info.txt'), 'a') as info_txt:
            info_txt.write(str(e) + ' ' + str(train_loss) + ' ')
            if test_loss:
                info_txt.write(str(test_loss) + ' ')
            if metrics:
                info_txt.write(str(metrics))
            info_txt.write('\n')

        cluster_settings = config['batch_settings']['train'].get('auto_clusters', {})
        cluster_period = config['batch_settings']['train'].get('make_clust_on_ep',
                                                              cluster_settings.get('make_clust_on_ep', 1))
        if save_ideals and mining_mode == "auto_clusters" and cluster_period > 0 and e % cluster_period == 0:
            ideals_dict = {}
            counter_dict = {}

            for i in range(train_dataset.get_alph_size()):
                ideals_dict[str(i)] = ideals[i].cpu().tolist()
                counter_dict[str(i)] = counter[i].cpu().item()

            with open(op.join(ep_save_pt, 'ideals.json'), 'w') as out:
                json.dump(ideals_dict, out)
            with open(op.join(ep_save_pt, 'counter.json'), 'w') as out:
                json.dump(counter_dict, out)

            device = next(recognizer.parameters()).device
            ideals = torch.zeros(train_dataset.get_alph_size(), config["image_size"]["output_dim"], device=device)
            counter = torch.zeros(train_dataset.get_alph_size(), dtype=torch.long, device=device)
            dists = torch.zeros(train_dataset.get_alph_size(), device=device)


def run_experiment_training(config_path, data_root, output_dir, resume=None, run_test=False, evaluation_data_root=None, models_dir=None):
    """Run a manifest-based protocol through the repository training API."""
    from pathlib import Path
    from experiment_train import run as run_manifest_experiment

    return run_manifest_experiment(
        Path(config_path), Path(data_root), Path(output_dir),
        Path(resume) if resume is not None else None,
        bool(run_test),
        Path(evaluation_data_root) if evaluation_data_root is not None else None,
        Path(models_dir) if models_dir is not None else None,
    )


def evaluate_pretrained_experiment(config_path, data_root, output_path, split="validation", calibrate_with_validation=False, models_dir=None):
    """Evaluate a frozen/pretrained encoder through the repository API."""
    from pathlib import Path
    from experiment_evaluate_pretrained import evaluate

    return evaluate(
        Path(config_path), Path(data_root), Path(output_path),
        split, bool(calibrate_with_validation), Path(models_dir) if models_dir is not None else None,
    )


