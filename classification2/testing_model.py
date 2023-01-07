from tensorflow import keras as ks


'''def default_model(dim: int):
    data_augmentation = ks.Sequential(
        [
            ks.layers.GaussianNoise(0.5)
        ],
        name='data_augmentation'
    )
    inputs = ks.Input(shape=(10, dim))
    x = data_augmentation(inputs)
    x = ks.layers.Conv1D(70, 3, activation='relu', padding='SAME')(x)
    x = ks.layers.Conv1D(64, 3, activation='relu', padding='SAME')(x)
    x = ks.layers.Flatten()(x)
    x = ks.layers.Dense(256, activation='relu')(x)
    x = ks.layers.Dense(128, activation='relu')(x)
    x = ks.layers.Dense(64, activation='relu')(x)
    outputs = ks.layers.Dense(3, activation='softmax')(x)

    return ks.Model(inputs, outputs)'''


def default_model(dim: int):
    data_augmentation = ks.Sequential(
        [
            ks.layers.GaussianNoise(0.5)
        ],
        name='data_augmentation'
    )
    inputs = ks.Input(shape=(10, dim))
    x = data_augmentation(inputs)
    x = ks.layers.Conv1D(64, 3, activation='relu', padding='SAME')(x)
    x = ks.layers.Conv1D(128, 3, activation='relu', padding='SAME')(x)
    x = ks.layers.Conv1D(256, 3, activation='relu', padding='SAME')(x)
    x = ks.layers.Conv1D(512, 3, activation='relu', padding='SAME')(x)
    x = ks.layers.Flatten()(x)
    x = ks.layers.Dense(512, activation='relu')(x)
    x = ks.layers.Dense(256, activation='relu')(x)
    x = ks.layers.Dense(128, activation='relu')(x)
    x = ks.layers.Dense(64, activation='relu')(x)
    outputs = ks.layers.Dense(3, activation='softmax')(x)

    return ks.Model(inputs, outputs)


if __name__ == '__main__':
    default_model(29).summary()
