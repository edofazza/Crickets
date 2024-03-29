import tensorflow as tf
from tensorflow import keras as ks
import numpy as np
import os
import gc
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, GlobalAveragePooling1D, Dense, Flatten, MaxPooling1D


def vgg16_1d(input_shape=(3480, 8), num_classes=3):
    inputs = Input(shape=input_shape)

    # Block 1
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    # Block 2
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    # Block 3
    x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    # Block 4
    x = Conv1D(512, kernel_size=3, activation='relu', padding='same')(x)
    x = Conv1D(512, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    # Block 5
    x = Conv1D(512, kernel_size=3, activation='relu', padding='same')(x)
    x = Conv1D(512, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    # Flatten and fully connected layers
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


# Basic 1D Residual Block
def basic_residual_block(x, filters, kernel_size, stride=1, activation='relu', batch_norm=True, conv_first=True):
    if conv_first:
        x = Conv1D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
        if batch_norm:
            x = BatchNormalization()(x)
        if activation:
            x = Activation(activation)(x)

    else:
        if batch_norm:
            x = BatchNormalization()(x)
        if activation:
            x = Activation(activation)(x)
        x = Conv1D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)

    return x


# ResNet-18
def resnet18(input_shape=(3480, 8), num_classes=3):
    inputs = Input(shape=input_shape)
    x = inputs

    x = basic_residual_block(x, filters=64, kernel_size=7, stride=2)
    x = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding="same")(x)

    x = basic_residual_block(x, filters=64, kernel_size=3)
    x = basic_residual_block(x, filters=64, kernel_size=3)

    x = basic_residual_block(x, filters=128, kernel_size=3, stride=2)
    x = basic_residual_block(x, filters=128, kernel_size=3)

    x = basic_residual_block(x, filters=256, kernel_size=3, stride=2)
    x = basic_residual_block(x, filters=256, kernel_size=3)

    x = basic_residual_block(x, filters=512, kernel_size=3, stride=2)
    x = basic_residual_block(x, filters=512, kernel_size=3)

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


# ResNet-34
def resnet34(input_shape=(3480, 8), num_classes=3):
    inputs = Input(shape=input_shape)
    x = inputs

    x = basic_residual_block(x, filters=64, kernel_size=7, stride=2)
    x = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding="same")(x)

    for _ in range(3):
        x = basic_residual_block(x, filters=64, kernel_size=3)

    x = basic_residual_block(x, filters=128, kernel_size=3, stride=2)
    for _ in range(4):
        x = basic_residual_block(x, filters=128, kernel_size=3)

    x = basic_residual_block(x, filters=256, kernel_size=3, stride=2)
    for _ in range(6):
        x = basic_residual_block(x, filters=256, kernel_size=3)

    x = basic_residual_block(x, filters=512, kernel_size=3, stride=2)
    for _ in range(3):
        x = basic_residual_block(x, filters=512, kernel_size=3)

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


# ResNet-50
def resnet50(input_shape=(3480, 8), num_classes=3):
    inputs = Input(shape=input_shape)
    x = inputs

    x = basic_residual_block(x, filters=64, kernel_size=7, stride=2)
    x = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding="same")(x)

    x = basic_residual_block(x, filters=64, kernel_size=1)
    x = basic_residual_block(x, filters=64, kernel_size=3)
    x = basic_residual_block(x, filters=256, kernel_size=1)

    x = basic_residual_block(x, filters=128, kernel_size=1)
    x = basic_residual_block(x, filters=128, kernel_size=3)
    x = basic_residual_block(x, filters=512, kernel_size=1)

    x = basic_residual_block(x, filters=256, kernel_size=1)
    x = basic_residual_block(x, filters=256, kernel_size=3)
    x = basic_residual_block(x, filters=1024, kernel_size=1)

    x = basic_residual_block(x, filters=512, kernel_size=1)
    x = basic_residual_block(x, filters=512, kernel_size=3)
    x = basic_residual_block(x, filters=2048, kernel_size=1)

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

def best_model_3classes(shape=(8, 3480)):
    inputs = ks.Input(shape=shape)
    data_augmentation = ks.Sequential(
        [
            ks.layers.GaussianNoise(0.5)
        ]
    )
    x = data_augmentation(inputs)
    x = ks.layers.Conv1D(821, 3, padding='SAME')(x)
    x = ks.layers.BatchNormalization()(x)
    x = ks.layers.Activation('tanh', name='tanh_1')(x)
    x = ks.layers.Conv1D(668, 3, padding='SAME')(x)
    x = ks.layers.Activation('elu', name='elu_1')(x)
    x = ks.layers.Dropout(0.2)(x)
    x = ks.layers.Conv1D(483, 3, padding='SAME')(x)
    x = ks.layers.Activation('tanh', name='tanh_2')(x)
    x = ks.layers.Flatten()(x)
    outputs = ks.layers.Dense(3, activation='softmax')(x)
    return ks.Model(inputs, outputs)


def best_rnn_3classes(shape=(8, 3480)):
    # bilstm707elugru660leaky_relubigru469leaky_reluDense138geluDrop0.2Dense150leaky_relu.keras
    inputs = ks.Input(shape=shape)
    data_augmentation = ks.Sequential(
        [
            ks.layers.GaussianNoise(0.5)
        ]
    )
    x = data_augmentation(inputs)
    x = ks.layers.Bidirectional(ks.layers.LSTM(707, activation='elu', return_sequences=True))(x)
    x = ks.layers.GRU(660, activation='leaky_relu', return_sequences=True)(x)
    x = ks.layers.Bidirectional(ks.layers.GRU(469, activation='leaky_relu'))(x)
    x = ks.layers.Dense(138, activation='gelu')(x)
    x = ks.layers.Dropout(0.2)(x)
    x = ks.layers.Dense(150, activation='leaky_relu')(x)
    outputs = ks.layers.Dense(3, activation='softmax')(x)
    return ks.Model(inputs, outputs)


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

        if model_type.startswith('resnet') or model_type.startswith('vgg'):
            train_data = train_data.transpose((0, 2, 1))
            print(train_data.shape)
            val_data = val_data.transpose((0, 2, 1))

        if model_type == 'cnn':
            model = best_model_3classes()
        elif model_type == 'rnn':
            model = best_rnn_3classes()
        elif model_type == 'resnet18':
            model = resnet18()
        elif model_type == 'resnet34':
            model = resnet34()
        elif model_type == 'resnet50':
            model = resnet50()
        elif model_type == 'vgg16':
            model = vgg16_1d()
        else:
            model = best_model_3classes()

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

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
            train_data,
            np.array(train_labels),
            validation_data=(val_data, np.array(val_labels)),
            epochs=10000,
            callbacks=callbacks_list,
            verbose=0,
            batch_size=16
        )
        del model
        model = ks.models.load_model(f'iter{iter_i}fold{fold}.keras')
        train_loss, train_accuracy = model.evaluate(train_data, np.array(train_labels))
        val_loss, val_accuracy = model.evaluate(val_data, np.array(val_labels))
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        del model
        ks.backend.clear_session()
        gc.collect()

    print('Final results:')
    train_losses_average = np.average(train_losses)
    print(f'\t-Average train losses:{train_losses_average}')
    train_accuracies_average = np.average(train_accuracies)
    print(f'\t-Average train accuracies:{train_accuracies_average}')
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
        tla, taa, vla, vaa = k_fold(k, dataset_C_tmp, dataset_S_tmp, dataset_A_tmp, i, model_type='vgg16')
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
    print(f'\t-Mean average train losses:{np.average(tla_average)}')
    print(f'\t-Mean average train accuracies:{np.average(taa_average)}')


if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)
    iterated_k_fold(10, 4)
