from dataset import ChooseDataset
from augmentation import build_transform_pipeline
from utils import prepare_dirs
from train_utils import run_training, ChooseOptimizer, ChooseScheduler
from eval_model import *
import ujson as json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Siamese models')
    parser.add_argument('-c', type=str, help='Config model path')
    args = parser.parse_args()
    config_path = args.c

    with open(config_path, 'r', encoding='utf8') as cfg_file:
        cfg = json.load(cfg_file)

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
    if cfg['common']['file_to_start']:
        chpnt = cfg['common']['file_to_start'].split("/")[-2]
        start_ep = int(chpnt) + 1

        checkpoint = torch.load(cfg['common']['file_to_start'])
        model.load_state_dict(checkpoint)  # ['model_state_dict'])
        print("Successfully loaded weights from", cfg['common']['file_to_start'])

    model_optimizer = ChooseOptimizer(cfg['common']['optimizer_settings'], model)
    model_scheduler = ChooseScheduler(cfg['common']['scheduler_settings'], model_optimizer)

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
