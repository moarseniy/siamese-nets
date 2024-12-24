import time
import os.path as op

from model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import ChooseDataset
from oneshot import *

def validate_with_descrs(config, recognizer, valid_loader, descrs):
    start_time = time.time()
    pbar = tqdm(valid_loader)

    count = 0
    if descrs.max() == descrs.min():
        print("Descrs is uniform. Check initialization or update logic.")

    for idx, mb in enumerate(pbar):

        # print(type(mb['image']), mb['image'].shape)
        img = mb['image'].cuda()
        lbl = mb['label'].cuda()

        # if idx < 10 and config['save_images']:
            # save_image(img[0], os.path.join('/home/arseniy/results/out/torch_out', 'out_valid_' + str(idx) + '.png'))

        data_out = recognizer(img)

        # Compute Euclidean distances between each vector in data_out and descrs
        distances = torch.cdist(data_out, descrs)

        # Find the index of the closest vector in descrs for each vector in data_out
        ids = torch.argmin(distances, dim=1)

        # Optionally, can compute the minimum distances
        # min_distances = torch.min(distances, dim=1).values

        count += torch.sum((lbl == ids).clone().detach())

        pbar.set_description("Valid [count %d]" % count)

    print('\nCount: ', count,
          '\nLength: ', len(valid_loader.dataset),
          '\nAccuracy: ', 100 * count / len(valid_loader.dataset),
          '\nTime: {:.2f} sec'.format(time.time() - start_time))

    return 100 * count / len(valid_loader.dataset)

def validate_oneshot(config, recognizer):
    transform = transforms.Compose([
        transforms.Resize(config["image_size"]["height"], config["image_size"]["height"]),
        transforms.ToTensor()
    ])

    all_runs_path = config["oneshot"]["path"]
    accuracies = []
    print("Oneshot validation:")
    for run_folder in sorted(os.listdir(all_runs_path)):
        run_path = os.path.join(all_runs_path, run_folder)
        if os.path.isdir(run_path):
            accuracy = run_oneshot(recognizer, run_path, transform)
            print(f"Accuracy for {run_folder}: {accuracy:.2%}")
            accuracies.append(accuracy)

    acc = sum(accuracies) / len(accuracies)
    print(f"Overall Accuracy: {acc:.2%}")

    return acc

if __name__ == "__main__":

    with open("train_config.json", "r", encoding='utf8') as cfg_file:
        cfg = json.load(cfg_file)

    torch.cuda.set_device(cfg['device'])

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(cfg['device']) + ' is available!')
    else:
        print('No GPU!!!')
        exit(-1)

    model = KorNet().cuda()

    if cfg['file_to_start']:
        print("Loading weights from", op.join(cfg['file_to_start'], 'model.pt'))
        checkpoint = torch.load(op.join(cfg['file_to_start'], 'model.pt'))
        model.load_state_dict(checkpoint, strict=False)
        print("Successfully loaded weights from", cfg['file_to_start'])
    else:
        print("No path for net (file_to_start)")
        exit(0)

    valid_dataset = ChooseDataset("valid", cfg['common'], None)

    ideals = torch.zeros(len(valid_dataset.get_alphabet()), 25).cuda()
    print('\nAlphabet size: ', len(valid_dataset.get_alphabet()))

    # with open(op.join(cfg['file_to_start'], 'ideals.json'), 'r') as json_ideal:
    #     data = json.load(json_ideal)
    #     for key, value in data.items():
    #         ideals[valid_dataset.get_alphabet().index(key)] = torch.FloatTensor(value)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=cfg['batch_settings']['elements_per_batch'],
                              shuffle=False,
                              num_workers=10)

    # res = validate(cfg, model, valid_loader, ideals)
