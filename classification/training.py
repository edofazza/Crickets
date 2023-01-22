import os
import tensorflow as tf
import pickle
import gc

from tests.testing_models import *
from utils.sequence_handling import create_dataset


def training(seq_dimensions,
             models_dir,
             batch_size,
             stim1_train='predictions_filled/control/train/',
             stim1_val='predictions_filled/control/val/',
             stim1_test='predictions_filled/control/test/',
             stim2_train='predictions_filled/sugar/train/',
             stim2_val='predictions_filled/sugar/val/',
             stim2_test='predictions_filled/sugar/test/',
             stim3_train=None,
             stim3_val=None,
             stim3_test=None
             ):
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

        train_set, train_labels = create_dataset(
            stim1_train,
            stim2_train,
            stim3_train,
            seq_dimensions
        )
        val_set, val_labels = create_dataset(
            stim1_val,
            stim2_val,
            stim3_val,
            seq_dimensions
        )

        model = binary_model(dim)
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
            train_labels,
            epochs=500,
            callbacks=callbacks_list,
            validation_data=(val_set, val_labels),
            batch_size=batch_size
        )

        with open(f'{models_dir}/length{dim}/history', 'wb') as file:
            pickle.dump(history.history, file)

        del model
        model = ks.models.load_model(f'{models_dir}/length{dim}/model.keras')
        train_loss, train_accuracy = model.evaluate(train_set, train_labels, verbose=False)
        val_loss, val_accuracy = model.evaluate(val_set, val_labels, verbose=False)

        del train_set
        del val_set
        gc.collect()
        test_set, test_labels = create_dataset(
            stim1_test,
            stim2_test,
            stim3_test,
            seq_dimensions
        )
        test_loss, test_accuracy = model.evaluate(test_set, test_labels, verbose=False)  # utils dataset
        print(
            f'Model {dim}:\n\tTrain loss: {train_loss}\n\tTrain accuracy: {train_accuracy}\n\tVal loss: {val_loss}\n\tVal accuracy: {val_accuracy}\n\tTest loss: {test_loss}\n\tTest accuracy: {test_accuracy}')

        del model
        del test_set
        gc.collect()
        ks.backend.clear_session()


def evaluate(dim,
             stim1_train='predictions_filled/control/train/',
             stim1_val='predictions_filled/control/val/',
             stim1_test='predictions_filled/control/test/',
             stim2_train='predictions_filled/sugar/train/',
             stim2_val='predictions_filled/sugar/val/',
             stim2_test='predictions_filled/sugar/test/',
             stim3_train=None,
             stim3_val=None,
             stim3_test=None
             ):
    model = tf.keras.models.load_model(f'models/old/length{dim}/model.keras')
    train_set, train_labels = create_dataset(
        stim1_train,
        stim2_train,
        stim3_train,
        dim
    )
    val_set, val_labels = create_dataset(
        stim1_val,
        stim2_val,
        stim3_val,
        dim
    )
    test_set, test_labels = create_dataset(
        stim1_test,
        stim2_test,
        stim3_test,
        dim
    )
    train_loss, train_accuracy = model.evaluate(train_set, train_labels, verbose=False)
    val_loss, val_accuracy = model.evaluate(val_set, val_labels, verbose=False)
    test_loss, test_accuracy = model.evaluate(test_set, test_labels, verbose=False)
    print(
        f'Model {dim}:\n\tTrain loss: {train_loss}\n\tTrain accuracy: {train_accuracy}\n\tVal loss: {val_loss}\n\t'
        f'Val accuracy: {val_accuracy}\n\tTest loss: {test_loss}\n\tTest accuracy: {test_accuracy}')



if __name__ == '__main__':
    ranges = [870]
    training(
        ranges,
        "models",
        128
    )
    os.system('chmod -R 777 *')

    """"" 
    with open('/trainHistoryDict', "rb") as file_pi:
        history = pickle.load(file_pi)
    """
