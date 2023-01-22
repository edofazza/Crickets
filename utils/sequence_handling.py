import numpy as np
import tensorflow as tf
import os


def divide_sequence(sequence, length):
    if length == 3480:
        return np.array(sequence)
    _, dim = sequence.shape
    tmp = []
    for i in range(dim - length):
        tmp_seq = sequence[:, i:i + length]
        tmp.append(tmp_seq)
    return np.array(tmp)


def normalize(x):
    return tf.keras.utils.normalize(x, axis=-1)


def create_dataset(control_path, sugar_path, ammonia_path=None, length=3480):
    if length < 3480:
        data = None
    else:
        data = []
    labels = []
    tmp_list = [c for c in os.listdir(control_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(control_path, i))
        labels.append(0)
        if type(data) is list:
            data.append(tmp_npy)
        else:
            if data is None:
                data = divide_sequence(tmp_npy, length)
            else:
                data = np.r_[data, divide_sequence(tmp_npy, length)]

    tmp_list = [c for c in os.listdir(sugar_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(sugar_path, i))
        labels.append(1)
        if type(data) is list:
            data.append(tmp_npy)
        else:
            data = np.r_[data, divide_sequence(tmp_npy, length)]

    if ammonia_path is not None:
        tmp_list = [c for c in os.listdir(ammonia_path) if c.endswith('.npy')]
        for i in tmp_list:
            tmp_npy = np.load(os.path.join(ammonia_path, i))
            labels.append(2)
            if type(data) is list:
                data.append(tmp_npy)
            else:
                data = np.r_[data, divide_sequence(tmp_npy, length)]

    if type(data) is list:
        return normalize(np.array(data)), np.array(labels)
    else:
        return normalize(data), np.array(labels)


"""
def create_dataset(control_path, sugar_path, ammonia_path, length, train=True):
    tmp = None
    tmp_list = [c for c in os.listdir(control_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(control_path, i))
        if tmp is None:
            tmp = divide_sequence(tmp_npy, length)
        else:
            tmp = np.r_[tmp, divide_sequence(tmp_npy, length)]

    if train:
        labels = (2*tmp.shape[0]) * [0] + (2*tmp.shape[0]) * [1] + (2*tmp.shape[0]) * [2]
    else:
        labels = tmp.shape[0] * [0] + tmp.shape[0] * [1] + tmp.shape[0] * [2]

    tmp_list = [c for c in os.listdir(sugar_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(sugar_path, i))
        tmp = np.r_[tmp, divide_sequence(tmp_npy, length)]

    tmp_list = [c for c in os.listdir(ammonia_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(ammonia_path, i))
        tmp = np.r_[tmp, divide_sequence(tmp_npy, length)]

    tmp2 = tmp[:, :, ::-1]
    tmp = np.r_[tmp, tmp2]
    tmp = normalize(tmp)
    labels = np.array(labels)
    print(tmp.shape)
    print(labels.shape)
    print(tmp)
    return tmp, labels


def create_dataset2(control_path, sugar_path, ammonia_path):
    data, labels = [], []
    tmp_list = [c for c in os.listdir(control_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(control_path, i))
        data.append(tmp_npy)
        labels.append(0)

    tmp_list = [c for c in os.listdir(sugar_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(sugar_path, i))
        data.append(tmp_npy)
        labels.append(1)

    tmp_list = [c for c in os.listdir(ammonia_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(ammonia_path, i))
        data.append(tmp_npy)
        labels.append(2)

    return np.array(data), np.array(labels)
"""

"""
def create_dataset(control_path, sugar_path, length, aug=True):
    tmp = None
    tmp_list = [c for c in os.listdir(control_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(control_path, i))
        if tmp is None:
            tmp = divide_sequence(tmp_npy, length)
        else:
            tmp = np.r_[tmp, divide_sequence(tmp_npy, length)]

    if aug:
        labels = (2*tmp.shape[0]) * [0] + (2*tmp.shape[0]) * [1]
    else:
        labels = tmp.shape[0] * [0] + tmp.shape[0] * [1]

    tmp_list = [c for c in os.listdir(sugar_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(sugar_path, i))
        tmp = np.r_[tmp, divide_sequence(tmp_npy, length)]

    #tmp2 = tmp[:, :, ::-1]
    #tmp = np.r_[tmp, tmp2]
    tmp = normalize(tmp)
    labels = np.array(labels)
    return tmp, labels


def create_dataset2(control_path, sugar_path):
    data, labels = [], []
    tmp_list = [c for c in os.listdir(control_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(control_path, i))
        data.append(tmp_npy)
        labels.append(0)

    tmp_list = [c for c in os.listdir(sugar_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(sugar_path, i))
        data.append(tmp_npy)
        labels.append(1)

    return normalize(np.array(data)), np.array(labels)
"""

