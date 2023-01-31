import numpy as np
import tensorflow as tf
import os


def divide_sequence(sequence, length):
    """
    From a sequence (joints_axis, frames) the function reduces the frame-length to
    sequences of shape (joints_axis, length) and store them in a new np.array that
    the function returns
    :param sequence: numpy sequence
    :param length: size of the window
    :return: np.array containing all subsequences obtained from sequence
    """
    if length == 3480:
        return np.array(sequence)
    _, dim = sequence.shape
    tmp = []
    for i in range(dim - length):
        tmp_seq = sequence[:, i:i + length]
        tmp.append(tmp_seq)
    return np.array(tmp)


def normalize(x):
    """
    Normalize values in a sequence x considering its last axis
    :param x: sequence
    :return: normalized sequence
    """
    return tf.keras.utils.normalize(x, axis=-1)


def create_dataset(first_stimulus_path, second_stimulus_path, third_stimulus_path=None, length=3480):
    """
    Create numpy dataset
    :param first_stimulus_path: path to the numpy directory containing the npy sequences for the first stimulus
    :param second_stimulus_path: path to the numpy directory containing the npy sequences for the second stimulus
    :param third_stimulus_path: path to the numpy directory containing the npy third for the first stimulus (if None it will not be considered, i.e., binary case)
    :param length: length of the (sub)sequences, in our case we use as default the total length (3480)
    :return: a tuple in the form (data, labels)
    """
    if length < 3480:
        data = None
    else:
        data = []
    labels = []
    tmp_list = [c for c in os.listdir(first_stimulus_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(first_stimulus_path, i))
        labels.append(0)
        if type(data) is list:
            data.append(tmp_npy)
        else:
            if data is None:
                data = divide_sequence(tmp_npy, length)
            else:
                data = np.r_[data, divide_sequence(tmp_npy, length)]

    tmp_list = [c for c in os.listdir(second_stimulus_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(second_stimulus_path, i))
        labels.append(1)
        if type(data) is list:
            data.append(tmp_npy)
        else:
            data = np.r_[data, divide_sequence(tmp_npy, length)]

    if third_stimulus_path is not None:
        tmp_list = [c for c in os.listdir(third_stimulus_path) if c.endswith('.npy')]
        for i in tmp_list:
            tmp_npy = np.load(os.path.join(third_stimulus_path, i))
            labels.append(2)
            if type(data) is list:
                data.append(tmp_npy)
            else:
                data = np.r_[data, divide_sequence(tmp_npy, length)]

    if type(data) is list:
        return normalize(np.array(data)), np.array(labels)
    else:
        return normalize(data), np.array(labels)
