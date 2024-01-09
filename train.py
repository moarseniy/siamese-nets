from dataset import KorRecognitionDataset, PHD08Dataset
from model import *
from train_utils import prepare_dirs, train
from eval_model import validate
import ujson as json

if __name__ == "__main__":

    with open("train_config.json", "r", encoding='utf8') as cfg_file:
        cfg = json.load(cfg_file)

    torch.cuda.set_device(cfg['device'])

    save_pt, save_im_pt = prepare_dirs(cfg)

    transforms = prepare_augmentation()

    train_dataset = KorRecognitionDataset(cfg=cfg, transforms=transforms)
    valid_dataset = PHD08Dataset(cfg=cfg)
    print(torch.cuda.is_available())

    model = KorNet().cuda()

    # model.summary()


    start_ep = 0
    if cfg['file_to_start']:
        chpnt = cfg['file_to_start'].split("/")[-1]
        start_ep = int(chpnt.split(".")[0]) + 1

        checkpoint = torch.load(cfg['file_to_start'])
        model.load_state_dict(checkpoint)#['model_state_dict'])
        print("Successfully loaded weights from", cfg['file_to_start'])

    if cfg['validate_settings']['validate']:
        print('Validation started!')
        validate(cfg, model, valid_dataset)
    else:
        model_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        train(config=cfg,
              recognizer=model,
              optimizer=model_optimizer,
              train_dataset=train_dataset,
              valid_dataset=valid_dataset,
              save_pt=save_pt,
              save_im_pt=save_im_pt,
              start_ep=start_ep)
