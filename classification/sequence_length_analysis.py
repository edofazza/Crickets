import numpy as np
import os
import tensorflow as tf
from tensorflow import keras as ks
import shutil
from same_size import final_dataset
from utils import load_dataset
#from iterated_k_fold_cross_validation import permute  TODO: remove k-fold
from models.vanilla import get_vanilla
from models.resnet import get_resnet18_model, get_resnet34_model, get_resnet50_model
from sequencehandler import SequenceHandler


def get_test_indexes(indexes):
    test_len = len(indexes) // 10
    p = list(np.random.permutation(len(indexes)))
    indexes = indexes[p]
    return indexes[:test_len]


def perform_final_analysis(dir_path, type_):
    mirrored_strategy = tf.distribute.MirroredStrategy()
    models_dir = f'all_models_{type_}'
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    predictions = os.listdir(dir_path)
    predictions = [file for file in predictions if file.endswith('.npy') and file.startswith('C')]

    # Create list that will contain the results
    val_loss_list = []
    val_acc_list = []
    test_loss_list = []
    test_acc_list = []

    for i in range(29, 580, 58)[::-1]: # TODO
        os.mkdir(f'{models_dir}/length{i}')

        print(f'SEQUENCE LENGTH: {i}')
        for prediction in predictions:
            SequenceHandler.divide_sequence(dir_path + prediction, i)

        final_dataset(f'{dir_path}dataset{i}/')

        # Get the dataset and its labels
        dataset, labels = load_dataset(f'{dir_path}dataset{i}/dataset.npy', f'{dir_path}dataset{i}/labels.npy')
        print(f'Dataset shape: {dataset.shape}')
        print(f'Dataset contains {len([l for l in labels if l == 0])} C, '
              f'{len([l for l in labels if l == 1])} A, '
              f'{len([l for l in labels if l == 2])} D')
        # Create test set using 10% of C, A, D
        C_indexes = np.where(labels == 0)[0]
        C_indexes = get_test_indexes(C_indexes)
        print(f'C indexes used for test set ({len(C_indexes)}): {C_indexes}')
        A_indexes = np.where(labels == 1)[0]
        A_indexes = get_test_indexes(A_indexes)
        print(f'A indexes used for test set ({len(A_indexes)}):  {A_indexes}')
        D_indexes = np.where(labels == 2)[0]
        D_indexes = get_test_indexes(D_indexes)
        print(f'D indexes used for test set ({len(D_indexes)}):  {D_indexes}')
        print(D_indexes)

        all_indexes = np.concatenate((C_indexes, A_indexes, D_indexes))
        print(f'Test set length: {len(all_indexes)}')
        test_data = dataset[all_indexes]
        test_labels = labels[all_indexes]

        # train-val set
        mask = np.full(len(dataset), True, dtype=bool)
        mask[all_indexes] = False
        dataset = dataset[mask]
        print(f'Train-val dataset shape: {dataset.shape}')
        labels = labels[mask]
        print(f'Train-val set contains {len([l for l in labels if l == 0])} C, '
              f'{len([l for l in labels if l == 1])} A, '
              f'{len([l for l in labels if l == 2])} D')
        dataset, labels = permute(dataset, labels)
        training_data = dataset[len(dataset)//10:]
        training_l = labels[len(labels)//10:]
        validation_data = dataset[:len(dataset) // 10]
        val_l = labels[:len(labels) // 10]

        print(f'Training set contains {len([l for l in training_l if l == 0])} C, '
              f'{len([l for l in training_l if l == 1])} A, '
              f'{len([l for l in training_l if l == 2])} D')
        print(f'Validation set contains {len([l for l in val_l if l == 0])} C, '
              f'{len([l for l in val_l if l == 1])} A, '
              f'{len([l for l in val_l if l == 2])} D')

        # Train models
        for model_type in ['vanilla', 'resnet18', 'resnet34', 'resnet50']:
            print('Model: ' + model_type)
            with mirrored_strategy.scope():
                if model_type == 'vanilla':
                    model = get_vanilla(i)
                elif model_type == 'resnet18':
                    model = get_resnet18_model(i)
                elif model_type == 'resnet34':
                    model = get_resnet34_model(i)
                else:
                    model = get_resnet50_model(i)

                model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

            callbacks_list = [
                ks.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                ),
                ks.callbacks.ModelCheckpoint(
                    filepath=f'{models_dir}/length{i}/{model_type}.keras',
                    monitor='val_loss',
                    save_best_only=True,
                    restore_best_weights=True
                ),
                ks.callbacks.ReduceLROnPlateau(
                    patience=5,
                    factor=0.5,
                    min_delta=1e-3,
                )
            ]

            model.fit(
                training_data,
                training_l,
                epochs=200,
                callbacks=callbacks_list,
                validation_data=(validation_data, val_l),
                verbose=0,
                batch_size=512
            )
            val_loss, val_acc = model.evaluate(validation_data, val_l, verbose=False)
            print(f'\t-Validation loss: {val_loss}')
            print(f'\t-Validation accuracy: {val_acc}')
            test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=False)
            print(f'\t-Test loss: {test_loss}')
            print(f'\t-Test accuracy: {test_acc}')

            # insert values
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)
            del model
            ks.backend.clear_session()

        # remove dataset dir
        shutil.rmtree(f'{dir_path}dataset{i}/')
        print(f'END SEQUENCE LENGTH: {i}\n\n\n\n')

    # save results
    np.save(f'val_loss_{type_}.npy', val_loss_list)
    np.save(f'val_acc_{type_}.npy', val_acc_list)
    np.save(f'test_loss_{type_}.npy', test_loss_list)
    np.save(f'test_acc_{type_}.npy', test_acc_list)


if __name__ == '__main__':
    print('NORMALIZED')
    perform_final_analysis('/Users/edoardo/Desktop/crickets/second phase/filled_normalized/', 'NORMALIZED')

    print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nWEIGHTED NORMALIZED')
    perform_final_analysis('/Users/edoardo/Desktop/crickets/second phase/filled_weighted_normalized/',
                           'WEIGHTED_NORMALIZED')
