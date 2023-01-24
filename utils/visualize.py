"""
Paper Name 1.0
© E. Fazzari, Institute of Biorobotics
Scuola Superiore Sant'Anna, Pisa, Italy

https://github.com/edofazza/Crickets
Licensed under GNU General Public License v3.0
"""

import os
import seaborn as sn
from matplotlib import pyplot as plt
import random
from sklearn.metrics import confusion_matrix
from tensorflow import keras as ks
import numpy as np
import pandas as pd

from sequence_handling import create_dataset


def antennae_scatter_plot(sequence, starting_frame=0, last_frame=3480, pace=58, save_png=True, path_png='plot.png'):
    colors = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
    seq = np.transpose(sequence)
    for i in range(starting_frame, last_frame, pace):
        x, y = [], []
        frame = seq[i]
        for j in range(0, len(frame), 2):
            x.append(frame[j])
            y.append(frame[j + 1])

        c = '#'
        for _ in range(6):
            c += random.choice(colors)
        plt.scatter(x, y, color=c)
    if save_png:
        plt.savefig(path_png)
    else:
        plt.plot()


def conf_matrix(model_path, stim1_dir_path, stim2_dir_path, stim3_dir_path=None): # TODO: TEST
    train_set, train_labels = create_dataset(
        os.path.join(stim1_dir_path, 'train'),
        os.path.join(stim2_dir_path, 'train'),
        stim3_dir_path if stim3_dir_path is None else os.path.join(stim2_dir_path, 'train')
    )

    val_set, val_labels = create_dataset(
        os.path.join(stim1_dir_path, 'val'),
        os.path.join(stim2_dir_path, 'val'),
        stim3_dir_path if stim3_dir_path is None else os.path.join(stim2_dir_path, 'val')
    )

    test_set, test_labels = create_dataset(
        os.path.join(stim1_dir_path, 'test'),
        os.path.join(stim2_dir_path, 'test'),
        stim3_dir_path if stim3_dir_path is None else os.path.join(stim2_dir_path, 'test')
    )

    model = ks.models.load_model(model_path)
    if stim3_dir_path is None:
        preds = model.predict(train_set)
        preds = [1 if pred > 0.5 else 0 for pred in preds]
        cf = confusion_matrix(train_labels, preds)

        df_cm = pd.DataFrame(cf, range(2), range(2))
        sn.set(font_scale=1.4)
        sn.heatmap(df_cm, annot=True, annot_kws={'size': 16})

        preds = model.predict(val_set)
        preds = [1 if pred > 0.5 else 0 for pred in preds]
        cf = confusion_matrix(val_labels, preds)

        df_cm = pd.DataFrame(cf, range(2), range(2))
        sn.set(font_scale=1.4)
        sn.heatmap(df_cm, annot=True, annot_kws={'size': 16})

        preds = model.predict(test_set)
        preds = [1 if pred > 0.5 else 0 for pred in preds]
        cf = confusion_matrix(test_labels, preds)

        df_cm = pd.DataFrame(cf, range(2), range(2))
        sn.set(font_scale=1.4)
        sn.heatmap(df_cm, annot=True, annot_kws={'size': 16})
    else:
        pass # TODO: argmax


if __name__ =='__main__':
    dir_path = "/Users/edoardo/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/PhD/reseaches/crickets/predictions/prediction_head_centered/ammonia/test"
    files = os.listdir(dir_path)
    files = [file for file in files if file.endswith('.npy')]
    for file in files:
        antennae_scatter_plot(np.load(os.path.join(dir_path, file)),
                              path_png=os.path.join("/Users/edoardo/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/PhD/reseaches/crickets/img/head_positions/ammonia/test", file[:-4]+'.png'))
