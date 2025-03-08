import os

# hack to avoid errors with tkinter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os.path as op
from datetime import datetime

def prepare_dirs(repo_dir, config, device_num):
    save_paths = {}
    files_to_start = {}
    from_file = False
    start_ep = 0
    checkpoint_pt = op.join(repo_dir, config["checkpoint_pt"])

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
    save_im_pt = op.join(save_pt, "out_images")
    if not op.exists(save_im_pt):
        os.makedirs(save_im_pt)

    return save_pt, save_im_pt

def save_plot(stat, save_pt):
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
    plt.savefig(op.join(save_pt, 'graph.png'))
    plt.close()