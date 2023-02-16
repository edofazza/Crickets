import tensorflow as tf
import numpy as np
import os
import gc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


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


def k_fold(k, dataset_C_tmp, dataset_S_tmp, dataset_A_tmp, iter_i, model_type='cnn'):
    num_validation_samples = 5  # 22%
    train_accuracies = []
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
        val_data = val_data.reshape(shape=(val_data.shape[0], val_data.shape[1]*val_data.shape[2]))
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
        train_data = train_data.reshape(shape=(train_data.shape[0], train_data.shape[1]*train_data.shape[2]))
        train_labels = len(train_data_C) * [0] + len(train_data_S) * [1] + len(train_data_A) * [2]

        forest = RandomForestClassifier(n_estimators=500, random_state=42)
        forest.fit(train_data, train_labels)
        train_accuracy = forest.score(train_data, np.array(train_labels))
        val_accuracy = forest.score(val_data, np.array(val_labels))

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        del forest
        gc.collect()

    print('Final results:')
    train_accuracies_average = np.average(train_accuracies)
    print(f'\t-Average test accuracies:{train_accuracies_average}')
    validation_accuracies_average = np.average(val_accuracies)
    print(f'\t-Average validation accuracies:{validation_accuracies_average}')
    # save all values
    np.save(f'iter{iter_i}_results/train_accuracies.npy', train_accuracies)
    np.save(f'iter{iter_i}_results/val_accuracies.npy', val_accuracies)

    return train_accuracies_average, validation_accuracies_average


def iterated_k_fold(iterations, k):
    print('Create initial dataset and labels')
    dataset_C, dataset_S, dataset_A, name_C, name_S, name_A = create_dataset_for_k_fold(
        "/Users/edofazza/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/PhD/reseaches/crickets/predictions/prediction_head_centered/all")
    taa_average = []  # train accuracy
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
        taa, vaa = k_fold(k, dataset_C_tmp, dataset_S_tmp, dataset_A_tmp, i)
        vaa_average.append(vaa)
        taa_average.append(taa)

    # save all values
    os.mkdir('k_fold_final_results')
    np.save('k_fold_final_results/taa_averages.npy', taa_average)
    np.save('k_fold_final_results/vaa_averages.npy', vaa_average)
    print('[END] Iterated k-fold cross validation loop')
    print(f'\t-Mean average validation accuracies:{np.average(vaa_average)}')
    print(f'\t-Mean average test accuracies:{np.average(taa_average)}')


if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)
    iterated_k_fold(10, 4)
