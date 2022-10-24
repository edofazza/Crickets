from tensorflow import keras as ks
import numpy as np
from argparse import ArgumentParser


def get_vanilla(dim: int):
    """
    Create a vanilla model given in input its initial dimension (10, dim). The model is characterized
    by a Conv1D layer equals to dim//8 and kernel size equals to 3, followed by a GlobalAveragePooling1D
    and a dense layer of 64 units.
    :param dim: An integer
    :return: the keras model
    """
    data_augmentation = ks.Sequential(
        [
            ks.layers.GaussianNoise(0.5)
        ]
    )
    inputs = ks.Input(shape=(10, dim))
    x = data_augmentation(inputs)
    x = ks.layers.Conv1D(dim // 8, 3, activation='relu', padding='SAME')(x)
    x = ks.layers.GlobalAveragePooling1D()(x)
    x = ks.layers.Dense(64, activation='relu')(x)
    outputs = ks.layers.Dense(3, activation='softmax')(x)

    return ks.Model(inputs, outputs)


def reduce_dataset10652(dataset):   # TODO: remove?
    """
    Get only the part we care
    :return:
    """
    data = [dataset[0]]
    for d in dataset[4:]:
        data.append(d)
    data = np.array(data)
    return data


def parse():
    parser = ArgumentParser()
    parser.add_argument('--dim', type=int, default=87)
    return vars(parser.parse_args())


if __name__ == '__main__':
    opt = parse()
    get_vanilla(opt['dim']).summary()