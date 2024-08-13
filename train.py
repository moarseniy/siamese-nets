from dataset import ChooseDataset
from model import *
from train_utils import prepare_dirs, run_training
from eval_model import validate
import ujson as json

if __name__ == "__main__":

    config_path = "train_config_0.json"

    with open(config_path, "r", encoding='utf8') as cfg_file:
        cfg = json.load(cfg_file)

    save_pt, save_im_pt = prepare_dirs(cfg)

    transforms = prepare_augmentation()

    train_dataset = ChooseDataset("train_data", cfg, transforms)
    valid_dataset = ChooseDataset("valid_data", cfg, transforms)
    test_dataset = None

    # train_dataset, test_dataset = None, None
    # if cfg['batch_settings']['type'] == 'triplet':
    #     train_dataset = KorSyntheticTriplet(cfg=cfg, transforms=transforms)
    # elif cfg['batch_settings']['type'] == 'contrastive':
    #     train_dataset = KorSyntheticContrastive(cfg=cfg, transforms=transforms)

    # test_dataset = PHD08Dataset(cfg=cfg)
    # valid_dataset = PHD08ValidDataset(cfg=cfg)

    torch.cuda.set_device(cfg['device'])
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(cfg['device']) + ' is available!')
    else:
        print('No GPU!!!')
        exit(-1)

    model = ChooseModel(cfg["model_name"])
    # model = KorNet().cuda()
    # model.summary()

    start_ep = 0
    if cfg['file_to_start']:
        chpnt = cfg['file_to_start'].split("/")[-2]
        start_ep = int(chpnt) + 1

        checkpoint = torch.load(cfg['file_to_start'])
        model.load_state_dict(checkpoint)  # ['model_state_dict'])
        print("Successfully loaded weights from", cfg['file_to_start'])

    model_optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    run_training(config=cfg,
                 recognizer=model,
                 optimizer=model_optimizer,
                 train_dataset=train_dataset,
                 test_dataset=test_dataset,
                 valid_dataset=valid_dataset,
                 save_pt=save_pt,
                 save_im_pt=save_im_pt,
                 start_ep=start_ep)
