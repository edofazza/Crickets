import os
import tensorflow as tf
from tensorflow import keras as ks
import numpy as np
import pickle
import gc

from testing_model import default_model


def divide_sequence(sequence, length):
    _, dim = sequence.shape
    tmp = []
    for i in range(dim - length):
        tmp_seq = sequence[:, i:i + length]
        tmp.append(tmp_seq)
    return np.array(tmp)


def normalize(x):
    return tf.keras.utils.normalize(x, axis=-1)


def create_dataset(control_path, sugar_path, ammonia_path, length, batch_size):
    tmp = None
    tmp_list = [c for c in os.listdir(control_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(control_path, i))
        if tmp is None:
            tmp = divide_sequence(tmp_npy, length)
        else:
            tmp = np.r_[tmp, divide_sequence(tmp_npy, length)]
        print(tmp.shape)

    labels = tmp.shape[0] * [0] + tmp.shape[0] * [1] + tmp.shape[0] * [2]

    tmp_list = [c for c in os.listdir(sugar_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(sugar_path, i))
        tmp = np.r_[tmp, divide_sequence(tmp_npy, length)]
        print(tmp.shape)

    tmp_list = [c for c in os.listdir(ammonia_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(ammonia_path, i))
        tmp = np.r_[tmp, divide_sequence(tmp_npy, length)]

    return tf.data.Dataset.from_tensor_slices((tmp, labels)).map(normalize).batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()


def training(seq_dimensions, models_dir, batch_size):
    # create directory where the models will be saved
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    # allow multiple gpus
    mirrored_strategy = tf.distribute.MirroredStrategy()

    for dim in seq_dimensions:
        # create directory for specific model
        if not os.path.exists(f'{models_dir}/length{dim}'):
            os.mkdir(f'{models_dir}/length{dim}')
        else:
            print(f'Directory length{dim} already exists inside {models_dir}, cannot train there')
            continue

        train_set = create_dataset(
            'prediction_filled/control/train/',
            'prediction_filled/sugar/train/',
            'prediction_filled/ammonia/train/',
            dim,
            batch_size
        )
        val_set = create_dataset(
            'prediction_filled/control/val/',
            'prediction_filled/sugar/val/',
            'prediction_filled/ammonia/val/',
            dim,
            batch_size
        )

        with mirrored_strategy.scope():
            model = default_model(dim)
            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

        callbacks_list = [
            ks.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=25,
            ),
            ks.callbacks.ModelCheckpoint(
                filepath=f'{models_dir}/length{dim}/model.keras',
                monitor='val_loss',
                save_best_only=True,
                restore_best_weights=True
            ),
            ks.callbacks.ReduceLROnPlateau(
                patience=10,
                factor=0.5,
                min_delta=1e-3
            )
        ]

        history = model.fit(
            train_set,
            epochs=400,
            callbacks=callbacks_list,
            validation_data=val_set,
            verbose=0,
            batch_size=batch_size
        )

        with open(f'{models_dir}/length{dim}/history', 'wb') as file:
            pickle.dump(history.history, file)

        train_loss, train_accuracy = model.evaluate(train_set, verbose=False)
        val_loss, val_accuracy = model.evaluate(val_set, verbose=False)

        del train_set
        del val_set
        gc.collect()
        test_set = create_dataset(
            'prediction_filled/control/test/',
            'prediction_filled/sugar/test/',
            'prediction_filled/ammonia/test/',
            dim,
            batch_size
        )
        test_loss, test_accuracy = model.evaluate(test_set, verbose=False) # test dataset
        print(f'Model {dim}:\n\tTrain loss: {train_loss}\n\tTrain accuracy: {train_accuracy}\n\tVal loss: {val_loss}\n\tVal accuracy: {val_accuracy}\n\tTest loss: {test_loss}\n\tTest accuracy: {test_accuracy}')

        del model
        del test_set
        gc.collect()
        ks.backend.clear_session()


if __name__ == '__main__':
    ranges = [29, 290, 580, 870, 1160, 1450]
    training(
        ranges,
        "models",
        256
    )
    os.system('chmod 777 -r *')

    """"" 
    with open('/trainHistoryDict', "rb") as file_pi:
        history = pickle.load(file_pi)
    """
