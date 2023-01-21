import os
import tensorflow as tf
from tensorflow import keras as ks
import numpy as np
import pickle
import gc

from testing_model import *


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


def training(seq_dimensions, models_dir, batch_size):
    # create directory where the models will be saved
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    for dim in seq_dimensions:
        # create directory for specific model
        if not os.path.exists(f'{models_dir}/length{dim}'):
            os.mkdir(f'{models_dir}/length{dim}')
        else:
            print(f'Directory length{dim} already exists inside {models_dir}, cannot train there')
            continue
        if dim == 3480:
            train_set, train_labels = create_dataset2(
                'predictions_filled/control/train/',
                'predictions_filled/sugar/train/'
            )
            val_set, val_labels = create_dataset2(
                'predictions_filled/control/val/',
                'predictions_filled/sugar/val/'
            )
        else:
            train_set, train_labels = create_dataset(
                'predictions_filled/control/train/',
                'predictions_filled/sugar/train/',
                dim,
                False
            )
            val_set, val_labels = create_dataset(
                'predictions_filled/control/val/',
                'predictions_filled/sugar/val/',
                dim,
                False
            )

        model = default_model7(dim)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        callbacks_list = [
            ks.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=100,
            ),
            ks.callbacks.ModelCheckpoint(
                filepath=f'{models_dir}/length{dim}/model.keras',
                monitor='val_loss',
                save_best_only=True,
                restore_best_weights=True
            ),
            ks.callbacks.ReduceLROnPlateau(
                patience=50,
                factor=0.5,
                min_delta=1e-3
            )
        ]

        history = model.fit(
            train_set,
            train_labels,
            epochs=10000,
            callbacks=callbacks_list,
            validation_data=(val_set, val_labels),
            batch_size=batch_size
        )

        with open(f'{models_dir}/length{dim}/history', 'wb') as file:
            pickle.dump(history.history, file)

        train_loss, train_accuracy = model.evaluate(train_set, train_labels, verbose=False)
        val_loss, val_accuracy = model.evaluate(val_set, val_labels, verbose=False)

        del train_set
        del val_set
        gc.collect()
        if dim == 3480:
            test_set, test_labels = create_dataset2(
                'predictions_filled/control/utils/',
                'predictions_filled/sugar/utils/'
            )
        else:
            test_set, test_labels = create_dataset(
                'predictions_filled/control/utils/',
                'predictions_filled/sugar/utils/',
                dim,
                False
            )
        test_loss, test_accuracy = model.evaluate(test_set, test_labels, verbose=False)  # utils dataset
        print(
            f'Model {dim}:\n\tTrain loss: {train_loss}\n\tTrain accuracy: {train_accuracy}\n\tVal loss: {val_loss}\n\tVal accuracy: {val_accuracy}\n\tTest loss: {test_loss}\n\tTest accuracy: {test_accuracy}')

        del model
        del test_set
        gc.collect()
        ks.backend.clear_session()


def evaluate(dim):
    model = tf.keras.models.load_model(f'models/old/length{dim}/model.keras')
    if dim == 3480:
        train_set, train_labels = create_dataset2(
            'predictions_filled/control/train/',
            'predictions_filled/sugar/train/'
        )
        val_set, val_labels = create_dataset2(
            'predictions_filled/control/val/',
            'predictions_filled/sugar/val/'
        )
    else:
        train_set, train_labels = create_dataset(
            'predictions_filled/control/train/',
            'predictions_filled/sugar/train/',
            dim,
            False
        )
        val_set, val_labels = create_dataset(
            'predictions_filled/control/val/',
            'predictions_filled/sugar/val/',
            dim,
            False
        )
    if dim == 3480:
        test_set, test_labels = create_dataset2(
            'predictions_filled/control/utils/',
            'predictions_filled/sugar/utils/'
        )
    else:
        test_set, test_labels = create_dataset(
            'predictions_filled/control/utils/',
            'predictions_filled/sugar/utils/',
            dim,
            False
        )
    train_loss, train_accuracy = model.evaluate(train_set, train_labels, verbose=False)
    val_loss, val_accuracy = model.evaluate(val_set, val_labels, verbose=False)
    test_loss, test_accuracy = model.evaluate(test_set, test_labels, verbose=False)
    print(
        f'Model {dim}:\n\tTrain loss: {train_loss}\n\tTrain accuracy: {train_accuracy}\n\tVal loss: {val_loss}\n\tVal accuracy: {val_accuracy}\n\tTest loss: {test_loss}\n\tTest accuracy: {test_accuracy}')


if __name__ == '__main__':
    """""ranges = [3480]
    training(
        ranges,
        "models",
        16
    )
    os.system('chmod -R 777 *')

     
    with open('/trainHistoryDict', "rb") as file_pi:
        history = pickle.load(file_pi)
    """
    for i in [290, 580, 870, 3480]:
        evaluate(i)
