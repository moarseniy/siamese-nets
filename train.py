from dataset import ChooseDataset
from augmentation import build_transform_pipeline
from utils import prepare_dirs
from train_utils import run_training, ChooseOptimizer, ChooseScheduler
from eval_model import *
import ujson as json
import argparse
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_config(config):
    """Translate the repository's older flat configs to the current schema."""
    if 'common' in config:
        return config

    old_batch = config.get('batch_settings', {})
    loss_type = config.get('loss', old_batch.get('type', 'triplet'))
    if loss_type == 'pair':
        loss_type = 'contrastive'

    def normalize_mode(mode, default):
        mode = mode or default
        if mode == 'auto_clusters1':
            return 'auto_clusters'
        if mode == 'auto_sym_probs1':
            return 'auto_sym_probs'
        return mode

    def make_split(split_name, spec):
        split = {
            'elements_per_batch': old_batch.get('elements_per_batch', 256),
            'iterations': old_batch.get('iterations', 10),
            'gen_imp_ratio': old_batch.get('gen_imp_ratio', 0.5),
            'positive_mode': normalize_mode(old_batch.get('positive_mode'), 'random'),
            'negative_mode': normalize_mode(old_batch.get('negative_mode'), 'random'),
            'inner_imp_prob': old_batch.get('inner_imp_prob', 0.2),
            'raw_clusters': old_batch.get('raw_clusters', 1000),
            'cluster_max_size': old_batch.get('cluster_max_size', 200),
            'make_clust_on_ep': old_batch.get('make_clust_on_ep', 1),
            'sym_probs_gamma': old_batch.get('sym_probs_gamma', 2.0),
            'merge_w': old_batch.get('merge_w', 0.0),
        }
        return spec, split

    common = {
        'epoch_num': config.get('epoch_num', 1),
        'image_size': config.get('image_size',
                                 {'width': 37, 'height': 37, 'channels': 1, 'output_dim': 25}),
        'checkpoint_pt': config.get('checkpoint_pt', 'checkpoints'),
        'images_pt': config.get('images_pt', 'images'),
        'device': config.get('device', 0),
        'file_to_start': config.get('file_to_start', ''),
        'save_images': config.get('save_images', False),
        'save_ideals': config.get('save_ideals', loss_type == 'metric' or
                                  normalize_mode(old_batch.get('negative_mode'), 'random') == 'auto_clusters'),
        'minibatch_size': config.get('minibatch_size', old_batch.get('elements_per_batch', 256)),
        'description': config.get('description', f"legacy config: loss={loss_type}"),
        'seed': config.get('seed'),
        'loss_settings': {
            'type': loss_type,
            'alpha_margin': old_batch.get('alpha_margin', 1.0),
        },
        'optimizer_settings': config.get('optimizer_settings', {
            'type': 'Adam', 'lr': config.get('lr', 1e-3), 'weight_decay': 0.0,
        }),
        'scheduler_settings': config.get('scheduler_settings'),
    }
    if config.get('model_name'):
        common['model_name'] = config['model_name']
    if config.get('model_path'):
        common['model_path'] = config['model_path']
    if config.get('alph_pt'):
        common['alph_pt'] = config['alph_pt']

    common['batch_settings'] = {}
    for old_key, new_key in (('train_data', 'train'), ('test_data', 'test'), ('valid_data', 'valid')):
        spec = config.get(old_key)
        if not spec:
            continue
        name = spec.get('name')
        if name == 'OmniglotOneShot':
            common['oneshot'] = {'path': spec['path']}
            continue
        if name == 'PHD08ValidDataset':
            spec = dict(spec, name='PHD08Valid')
        common[new_key], common['batch_settings'][new_key] = make_split(new_key, spec)

    if 'train' not in common or 'train' not in common['batch_settings']:
        raise ValueError('Config must define a training dataset')
    if 'valid' in common and 'valid' not in common['batch_settings']:
        common['batch_settings']['valid'] = {'elements_per_batch': 10240}
    if 'valid' in common and 'OmniglotOneShot' in str(common['valid'].get('name')):
        common['oneshot'] = {'path': common['valid']['path']}
        common.pop('valid')
        common['batch_settings'].pop('valid', None)

    return {'common': common, 'augmentation': config.get('augmentation', {'to_use': False})}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Siamese models')
    parser.add_argument('-c', type=str, help='Config model path')
    args = parser.parse_args()
    config_path = args.c

    with open(config_path, 'r', encoding='utf8') as cfg_file:
        cfg = normalize_config(json.load(cfg_file))

    if cfg['common'].get('seed') is not None:
        set_seed(int(cfg['common']['seed']))

    repo_dir = os.path.dirname(os.path.dirname(config_path))
    save_pt, save_im_pt = prepare_dirs(repo_dir, cfg['common'], cfg['common']['device'])

    augmentation = build_transform_pipeline(cfg["augmentation"]) #prepare_augmentation()

    train_dataset = ChooseDataset(dataset_type='train', cfg=cfg['common'], augmentation=augmentation)
    test_dataset = ChooseDataset(dataset_type='test', cfg=cfg['common'], augmentation=augmentation)
    valid_dataset = ChooseDataset(dataset_type='valid', cfg=cfg['common'], augmentation=augmentation)

    torch.cuda.set_device(cfg['common']['device'])
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(cfg['common']['device']) + ' is available!')
    else:
        print('No GPU!!!')
        exit(-1)

    model = None
    if 'model_name' in cfg['common']:
        model = ChooseModel(cfg['common']['model_name'],
                            cfg['common']['image_size'])
    elif 'model_path' in cfg['common']:
        model = load_model_from_config(os.path.join(repo_dir, cfg['common']['model_path']),
                                       cfg['common']['image_size'])
    else:
        print("No model!")
        exit(-1)

    start_ep = 0
    checkpoint = None
    if cfg['common']['file_to_start']:
        checkpoint = torch.load(cfg['common']['file_to_start'], map_location='cpu')
        model_state = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(model_state)
        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            start_ep = int(checkpoint['epoch']) + 1
        else:
            checkpoint_parent = os.path.basename(os.path.dirname(cfg['common']['file_to_start']))
            if checkpoint_parent.isdigit():
                start_ep = int(checkpoint_parent) + 1
        print("Successfully loaded weights from", cfg['common']['file_to_start'])

    model_optimizer = ChooseOptimizer(cfg['common']['optimizer_settings'], model)
    model_scheduler = ChooseScheduler(cfg['common'].get('scheduler_settings'), model_optimizer)
    if checkpoint is not None and isinstance(checkpoint, dict):
        if checkpoint.get('optimizer_state_dict') is not None:
            model_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if model_scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            model_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    run_training(config=cfg['common'],
                 recognizer=model,
                 optimizer=model_optimizer,
                 scheduler=model_scheduler,
                 train_dataset=train_dataset,
                 test_dataset=test_dataset,
                 valid_dataset=valid_dataset,
                 save_pt=save_pt,
                 save_im_pt=save_im_pt,
                 start_ep=start_ep)
