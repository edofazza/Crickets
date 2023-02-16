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
    """
    Given a sequence it plots the position of the joints in specific frames.
    :param sequence: numpy sequence
    :param starting_frame: initial frame
    :param last_frame: last frame
    :param pace: pace to select which frames-joint positions to plot
    :param save_png: if True save as png
    :param path_png: if save_png is True this parameter indicates the path
    :return:
    """
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


def conf_matrix(model_path, stim1_dir_path, stim2_dir_path, stim3_dir_path=None, ranges=['control', 'sugar'],
                save_png=True, path_png='plot.png'):
    """
    Generate the confusion matrix
    :param model_path: path to the model
    :param stim1_dir_path: path to the directory containing the npy related to the first stimulus
    :param stim2_dir_path: path to the directory containing the npy related to the second stimulus
    :param stim3_dir_path: path to the directory containing the npy related to the third stimulus (can be None)
    :param ranges: list of strings indicating the classes
    :param save_png: boolean, True to save the confusion matrix into a png
    :param path_png: if save_png is True the confusion matrix image is saved using this path
    :return:
    """
    data, labels = create_dataset(
        stim1_dir_path,
        stim2_dir_path,
        stim3_dir_path if stim3_dir_path is None else stim3_dir_path
    )

    model = ks.models.load_model(model_path)
    preds = model.predict(data)
    if stim3_dir_path is None:
        preds = [1 if pred > 0.5 else 0 for pred in preds]
    else:
        preds = np.argmax(preds, axis=-1)
    cf = confusion_matrix(labels, preds)
    df_cm = pd.DataFrame(cf, ranges, ranges)
    sn.set(font_scale=1.4)
    fig = sn.heatmap(df_cm, annot=True, annot_kws={'size': 16}).get_figure()
    if save_png:
        fig.savefig(path_png)


def boxplot_iterated_k_crossvalidation():
    iter0 = np.load(
        "/Users/edoardo/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/PhD/reseaches/crickets/"
        "results/secondphase/iterated_cross_validation/results/iter0_results/train_accuracies.npy").tolist()
    iter1 = np.load(
        "/Users/edoardo/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/PhD/reseaches/crickets/"
        "results/secondphase/iterated_cross_validation/results/iter1_results/train_accuracies.npy").tolist()
    iter2 = np.load(
        "/Users/edoardo/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/PhD/reseaches/crickets/"
        "results/secondphase/iterated_cross_validation/results/iter2_results/train_accuracies.npy").tolist()
    iter3 = np.load(
        "/Users/edoardo/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/PhD/reseaches/crickets/"
        "results/secondphase/iterated_cross_validation/results/iter3_results/train_accuracies.npy").tolist()
    iter4 = np.load(
        "/Users/edoardo/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/PhD/reseaches/crickets/"
        "results/secondphase/iterated_cross_validation/results/iter4_results/train_accuracies.npy").tolist()
    iter5 = np.load(
        "/Users/edoardo/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/PhD/reseaches/crickets/"
        "results/secondphase/iterated_cross_validation/results/iter5_results/train_accuracies.npy").tolist()
    iter6 = np.load(
        "/Users/edoardo/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/PhD/reseaches/crickets/"
        "results/secondphase/iterated_cross_validation/results/iter6_results/train_accuracies.npy").tolist()
    iter7 = np.load(
        "/Users/edoardo/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/PhD/reseaches/crickets/"
        "results/secondphase/iterated_cross_validation/results/iter7_results/train_accuracies.npy").tolist()
    iter8 = np.load(
        "/Users/edoardo/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/PhD/reseaches/crickets/"
        "results/secondphase/iterated_cross_validation/results/iter8_results/train_accuracies.npy").tolist()
    iter9 = np.load(
        "/Users/edoardo/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/PhD/reseaches/crickets/"
        "results/secondphase/iterated_cross_validation/results/iter9_results/train_accuracies.npy").tolist()
    my_dict = {'0': iter0, '1': iter1, '2': iter2,
               '3': iter3, '4': iter4, '5': iter5,
               '6': iter6, '7': iter7, '8': iter8, '9': iter9}
    fig, ax = plt.subplots()
    ax.boxplot(my_dict.values())
    ax.set_xticklabels(my_dict.keys())
    plt.title('Iterated Cross-Validation')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.savefig('boxplot.png')


if __name__ == '__main__':
    boxplot_iterated_k_crossvalidation()
