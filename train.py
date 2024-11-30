from dataset import ChooseDataset
from model import *
from train_utils import prepare_dirs, run_training, ChooseOptimizer
from eval_model import *
import ujson as json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Siamese models')
    parser.add_argument('-c', type=str, help='Config model path')
    args = parser.parse_args()
    config_path = args.c

    with open(config_path, "r", encoding='utf8') as cfg_file:
        cfg = json.load(cfg_file)

    save_pt, save_im_pt = prepare_dirs(cfg['common'], cfg['common']['device'])

    transforms = prepare_augmentation()

    train_dataset = ChooseDataset("train", cfg['common'], transforms)
    test_dataset = ChooseDataset("test", cfg['common'], transforms)
    valid_dataset = ChooseDataset("valid", cfg['common'], transforms)

    torch.cuda.set_device(cfg['common']['device'])
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(cfg['common']['device']) + ' is available!')
    else:
        print('No GPU!!!')
        exit(-1)

    model = ChooseModel(cfg['common']["model_name"], cfg['common']['image_size'])

    # print_model_info(model, cfg['common']['image_size'], cfg['common']['minibatch_size'])

    start_ep = 0
    if cfg['common']['file_to_start']:
        chpnt = cfg['common']['file_to_start'].split("/")[-2]
        start_ep = int(chpnt) + 1

        checkpoint = torch.load(cfg['common']['file_to_start'])
        model.load_state_dict(checkpoint)  # ['model_state_dict'])
        print("Successfully loaded weights from", cfg['common']['file_to_start'])

    model_optimizer = ChooseOptimizer(cfg['common']["optimizer_settings"], model)

    run_training(config=cfg['common'],
                 recognizer=model,
                 optimizer=model_optimizer,
                 train_dataset=train_dataset,
                 test_dataset=test_dataset,
                 valid_dataset=valid_dataset,
                 save_pt=save_pt,
                 save_im_pt=save_im_pt,
                 start_ep=start_ep)
