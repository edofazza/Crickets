from tensorflow import keras as ks
import numpy as np

def default_model(dim: int):
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