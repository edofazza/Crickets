import tensorflow as tf
from tensorflow import keras as ks
import numpy as np
import os
import gc


class Autoencoder(ks.models.Model):
    def __init__(self, latent_dim, encoder_dim, decoder_dim):
        super(Autoencoder, self).__init__()
        self.data_augmentation = ks.Sequential(
            [
                ks.layers.GaussianNoise(0.5)
            ]
        )
        self.latent_dim = latent_dim
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.encoder = tf.keras.Sequential([
            ks.layers.Bidirectional(ks.layers.GRU(self.encoder_dim, return_sequences=True)),
            ks.layers.Bidirectional(ks.layers.GRU(self.latent_dim, return_sequences=True), merge_mode='sum'),
        ])
        self.decoder = tf.keras.Sequential([
            ks.layers.Bidirectional(ks.layers.GRU(self.decoder_dim, return_sequences=True)),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train(shape, latent_dim, encoder_dim, decoder_dim, train_set, train_labels, val_set, val_labels, iter_i, fold):
    # call autoencoder
    # train autoencoder
    # load
    # take only encoder
    # freeze encoder
    # train MLP on frozen encoder
    # load
    # train
    # clean
    callbacks_list = [
        ks.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=500,
        ),
        ks.callbacks.ModelCheckpoint(
            filepath=f'ae_iter{iter_i}fold{fold}.keras',
            monitor='val_loss',
            save_best_only=True,
            restore_best_weights=True
        ),
        ks.callbacks.ReduceLROnPlateau(
            patience=100,
            factor=0.5,
            min_delta=1e-3,
        )
    ]

    inputs = ks.Input((8, 3480))
    encoder = ks.Sequential(
        [ks.layers.Bidirectional(ks.layers.GRU(encoder_dim, return_sequences=True)),
         ks.layers.Bidirectional(ks.layers.GRU(latent_dim, return_sequences=True), merge_mode='sum')],
        name='encoder'
    )(inputs)
    decoder = ks.layers.Bidirectional(ks.layers.GRU(decoder_dim, return_sequences=True), name='decoder')(encoder)
    outputs = ks.layers.Dense(3480)(decoder)
    autoencoder = ks.Model(inputs, outputs)
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(
        train_set,
        train_set,
        validation_data=(val_set, val_set),
        epochs=10000,
        callbacks=callbacks_list,
        verbose=0,
        batch_size=16
    )
    del autoencoder
    ks.backend.clear_session()
    gc.collect()

    data_augmentation = ks.Sequential(
        [
            ks.layers.GaussianNoise(0.5)
        ]
    )

    encoder = ks.models.load_model(f'ae_iter{iter_i}fold{fold}.keras')
    encoder = ks.Model(encoder.get_layer("encoder").inputs, encoder.get_layer("encoder").output)
    encoder.trainable = False
    inputs = ks.Input(shape)
    x = data_augmentation(inputs)
    x = encoder(x)
    x = ks.layer.GRU(128, return_sequences=False)(x)
    outputs = ks.layer.Dense(3, activation='softmax')(x)
    model = ks.Model(inputs, outputs)
    model.summary()

    callbacks_list = [
        ks.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=500,
        ),
        ks.callbacks.ModelCheckpoint(
            filepath=f'frozen_iter{iter_i}fold{fold}.keras',
            monitor='val_loss',
            save_best_only=True,
            restore_best_weights=True
        ),
        ks.callbacks.ReduceLROnPlateau(
            patience=100,
            factor=0.5,
            min_delta=1e-3,
        )
    ]

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(
        train_set,
        train_labels,
        validation_data=(val_set, val_labels),
        epochs=10000,
        callbacks=callbacks_list,
        verbose=0,
        batch_size=16
    )
    del model
    ks.backend.clear_session()
    gc.collect()

    model = ks.models.load_model(f'frozen_iter{iter_i}fold{fold}.keras')
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    for l in model.layers:
        l.trainable = True

    model.summary()

    callbacks_list = [
        ks.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=500,
        ),
        ks.callbacks.ModelCheckpoint(
            filepath=f'iter{iter_i}fold{fold}.keras',
            monitor='val_loss',
            save_best_only=True,
            restore_best_weights=True
        ),
        ks.callbacks.ReduceLROnPlateau(
            patience=100,
            factor=0.5,
            min_delta=1e-3,
        )
    ]

    model.fit(
        train_set,
        train_labels,
        validation_data=(val_set, val_labels),
        epochs=10000,
        callbacks=callbacks_list,
        verbose=0,
        batch_size=16
    )
    del model

    model = ks.models.load_model(f'iter{iter_i}fold{fold}.keras')
    train_loss, train_accuracy = model.evaluate(train_set, train_labels)
    val_loss, val_accuracy = model.evaluate(val_set, val_labels)
    ks.backend.clear_session()
    gc.collect()
    os.remove(f'iter{iter_i}fold{fold}.keras')
    os.remove(f'frozen_iter{iter_i}fold{fold}.keras')
    os.remove(f'ae_iter{iter_i}fold{fold}.keras')
    return train_loss, train_accuracy, val_loss, val_accuracy


def normalize(x):
    """
    Normalize values in a sequence x considering its last axis
    :param x: sequence
    :return: normalized sequence
    """
    return tf.keras.utils.normalize(x, axis=-1)


def create_dataset_for_k_fold(dir_path):
    file_names = [file for file in os.listdir(dir_path) if file.endswith('.npy')]
    dataset_C, dataset_S, dataset_A = [], [], []
    name_C, name_S, name_A = [], [], []
    for f in file_names:
        if f[-5] == 'C':
            dataset_C.append(np.load(os.path.join(dir_path, f)))
            name_C.append(f)
        elif f[-5] == 'S':
            dataset_S.append(np.load(os.path.join(dir_path, f)))
            name_S.append(f)
        elif f[-5] == 'A':
            dataset_A.append(np.load(os.path.join(dir_path, f)))
            name_A.append(f)

    return normalize(dataset_C), normalize(dataset_S), normalize(dataset_A), name_C, name_S, name_A


def permute(dataset):
    p = np.random.permutation(len(dataset))
    print(f'Permute data: {p}')
    return dataset[p], p


def k_fold(k, dataset_C_tmp, dataset_S_tmp, dataset_A_tmp, iter_i, model_type='rnn'):
    num_validation_samples = 5  # 22%
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for fold in range(k):
        print(f'Starting fold #{fold}')
        val_data_C = dataset_C_tmp[num_validation_samples * fold:num_validation_samples * (fold + 1)]
        val_data_S = dataset_S_tmp[num_validation_samples * fold:num_validation_samples * (fold + 1)]
        val_data_A = dataset_A_tmp[num_validation_samples * fold:num_validation_samples * (fold + 1)]
        val_data = np.concatenate((
            val_data_C,
            val_data_S,
            val_data_A
        ))
        val_labels = len(val_data_C) * [0] + len(val_data_S) * [1] + len(val_data_A) * [2]
        train_data_C = np.concatenate((
            dataset_C_tmp[:num_validation_samples * fold],
            dataset_C_tmp[num_validation_samples * (fold + 1):]
        ))
        train_data_S = np.concatenate((
            dataset_S_tmp[:num_validation_samples * fold],
            dataset_S_tmp[num_validation_samples * (fold + 1):]
        ))
        train_data_A = np.concatenate((
            dataset_A_tmp[:num_validation_samples * fold],
            dataset_A_tmp[num_validation_samples * (fold + 1):]
        ))
        train_data = np.concatenate((
            train_data_C,
            train_data_S,
            train_data_A
        ))
        train_labels = len(train_data_C) * [0] + len(train_data_S) * [1] + len(train_data_A) * [2]
        print(train_data.shape)

        train_loss, train_accuracy, val_loss, val_accuracy = train(
            (8, 3480),
            512,
            1024,
            1024,
            train_data,
            np.array(train_labels),
            val_data,
            np.array(val_labels),
            iter_i,
            fold
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    print('Final results:')
    train_losses_average = np.average(train_losses)
    print(f'\t-Average test losses:{train_losses_average}')
    train_accuracies_average = np.average(train_accuracies)
    print(f'\t-Average test accuracies:{train_accuracies_average}')
    validation_losses_average = np.average(val_losses)
    print(f'\t-Average validation losses:{validation_losses_average}')
    validation_accuracies_average = np.average(val_accuracies)
    print(f'\t-Average validation accuracies:{validation_accuracies_average}')
    # save all values
    np.save(f'iter{iter_i}_results/train_losses.npy', train_losses)
    np.save(f'iter{iter_i}_results/train_accuracies.npy', train_accuracies)
    np.save(f'iter{iter_i}_results/val_losses.npy', val_losses)
    np.save(f'iter{iter_i}_results/val_accuracies.npy', val_accuracies)

    return train_losses_average, train_accuracies_average, \
           validation_losses_average, validation_accuracies_average


def iterated_k_fold(iterations, k):
    print('Create initial dataset and labels')
    dataset_C, dataset_S, dataset_A, name_C, name_S, name_A = create_dataset_for_k_fold(
        "all/")
    tla_average = []  # train loss
    taa_average = []  # train accuracy
    vla_average = []  # validation loss
    vaa_average = []  # validation accuracy

    print('Perform k-fold:')
    for i in range(iterations):
        os.mkdir(f'iter{i}_results')
        print('Shuffle data')
        dataset_C_tmp, p = permute(dataset_C)
        np.save(f'permutation_C_{i}.npy', p)
        dataset_S_tmp, p = permute(dataset_S)
        np.save(f'permutation_S_{i}.npy', p)
        dataset_A_tmp, p = permute(dataset_A)
        np.save(f'permutation_A_{i}.npy', p)
        tla, taa, vla, vaa = k_fold(k, dataset_C_tmp, dataset_S_tmp, dataset_A_tmp, i, model_type='rnn')
        vla_average.append(vla)
        vaa_average.append(vaa)
        tla_average.append(tla)
        taa_average.append(taa)

    # save all values
    os.mkdir('k_fold_final_results')
    np.save('k_fold_final_results/tla_averages.npy', tla_average)
    np.save('k_fold_final_results/taa_averages.npy', taa_average)
    np.save('k_fold_final_results/vla_averages.npy', vla_average)
    np.save('k_fold_final_results/vaa_averages.npy', vaa_average)
    print('[END] Iterated k-fold cross validation loop')
    print(f'\t-Mean average validation losses:{np.average(vla_average)}')
    print(f'\t-Mean average validation accuracies:{np.average(vaa_average)}')
    print(f'\t-Mean average test losses:{np.average(tla_average)}')
    print(f'\t-Mean average test accuracies:{np.average(taa_average)}')


if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)
    iterated_k_fold(10, 4)
