import tensorflow as tf
import os
import numpy as np


def divide_sequence(sequence, length):
    """
    From a sequence (joints_axis, frames) the function reduces the frame-length to
    sequences of shape (joints_axis, length) and store them in a new np.array that
    the function returns
    :param sequence: numpy sequence
    :param length: size of the window
    :return: np.array containing all subsequences obtained from sequence
    """
    if length == 3480:
        return np.array(sequence)
    _, dim = sequence.shape
    tmp = []
    for i in range(dim - length):
        tmp_seq = sequence[:, i:i + length]
        tmp.append(tmp_seq)
    return np.array(tmp)


def normalize(x):
    """
    Normalize values in a sequence x considering its last axis
    :param x: sequence
    :return: normalized sequence
    """
    return tf.keras.utils.normalize(x, axis=-1)


def create_dataset(first_stimulus_path, second_stimulus_path, third_stimulus_path=None, length=3480):
    """
    Create numpy dataset
    :param first_stimulus_path: path to the numpy directory containing the npy sequences for the first stimulus
    :param second_stimulus_path: path to the numpy directory containing the npy sequences for the second stimulus
    :param third_stimulus_path: path to the numpy directory containing the npy third for the first stimulus (if None it will not be considered, i.e., binary case)
    :param length: length of the (sub)sequences, in our case we use as default the total length (3480)
    :return: a tuple in the form (data, labels)
    """
    if length < 3480:
        data = None
    else:
        data = []
    labels = []
    tmp_list = [c for c in os.listdir(first_stimulus_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(first_stimulus_path, i))
        if type(data) is list:
            data.append(tmp_npy)
            labels.append(0)
        else:
            if data is None:
                data = divide_sequence(tmp_npy, length)
                labels = data.shape[0]*[0]
            else:
                tmp_data = divide_sequence(tmp_npy, length)
                data = np.r_[data, tmp_data]
                labels.extend(tmp_data.shape[0] * [0])

    tmp_list = [c for c in os.listdir(second_stimulus_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(second_stimulus_path, i))
        if type(data) is list:
            data.append(tmp_npy)
            labels.append(1)
        else:
            tmp_data = divide_sequence(tmp_npy, length)
            data = np.r_[data, tmp_data]
            labels.extend(tmp_data.shape[0]*[1])

    if third_stimulus_path is not None:
        tmp_list = [c for c in os.listdir(third_stimulus_path) if c.endswith('.npy')]
        for i in tmp_list:
            tmp_npy = np.load(os.path.join(third_stimulus_path, i))
            if type(data) is list:
                data.append(tmp_npy)
                labels.append(2)
            else:
                tmp_data = divide_sequence(tmp_npy, length)
                data = np.r_[data, tmp_data]
                labels.extend(tmp_data.shape[0] * [2])

    if type(data) is list:
        return normalize(np.array(data)), np.array(labels)
    else:
        return normalize(data), np.array(labels)



def evaluate(train_set, train_labels, val_set, val_labels, test_set, test_labels, model_name): # TODO: move to test
    model = tf.keras.models.load_model(model_name)

    train_loss, train_accuracy = model.evaluate(train_set, train_labels, verbose=False)
    val_loss, val_accuracy = model.evaluate(val_set, val_labels, verbose=False)
    test_loss, test_accuracy = model.evaluate(test_set, test_labels, verbose=False)
    print(f'{model_name}:\n\tTrain loss: {train_loss}\n\tTrain accuracy: {train_accuracy}\n\tVal loss: {val_loss}\n\tVal accuracy: {val_accuracy}\n\tTest loss: {test_loss}\n\tTest accuracy: {test_accuracy}')


def evaluate2(train_set, train_labels, val_set, val_labels, test_set, test_labels, model_name): # TODO: move to test
    model = tf.keras.models.load_model(model_name)

    train_loss, train_accuracy = model.evaluate(train_set, train_labels, verbose=False)
    val_loss, val_accuracy = model.evaluate(val_set, val_labels, verbose=False)
    test_loss, test_accuracy = model.evaluate(test_set, test_labels, verbose=False)
    if test_accuracy > 0.33 and train_accuracy > 0.33 and val_accuracy > 0.33: # and train_accuracy > val_accuracy
        print(f'{model_name}:\n\tTrain loss: {train_loss}\n\tTrain accuracy: {train_accuracy}\n\tVal loss: {val_loss}\n\tVal accuracy: {val_accuracy}\n\tTest loss: {test_loss}\n\tTest accuracy: {test_accuracy}')


if __name__=='__main__':
    """evaluate('Conv558eluDrop0.4Conv782BatchtanhDrop0.25Conv984tanhDrop0.45Conv724BatcheluGlobalDense159eluDrop0.15Dense351tanhDrop0.35Dense394leaky_reluDrop0.4Dense433sigmoid.keras')
    evaluate('Conv807BatchtanhDrop0.25Conv984tanhDrop0.15Conv760BatcheluGlobalDense151eluDrop0.3Dense146tanhDrop0.1Dense117leaky_reluDrop0.3.keras')
    evaluate('Conv844BatchtanhDrop0.25Conv984tanhDrop0.15Conv778BatcheluGlobalDense151eluDrop0.3Dense146tanhDrop0.35Dense433sigmoid.keras')
    evaluate('Conv845BatchtanhConv971tanhDrop0.15Conv729eluGlobalDense150eluDrop0.3Dense354tanhDrop0.05Dense433sigmoid.keras')
    """
    train_set, train_labels = create_dataset(
        'prediction_head_centered/control/train/',
        'prediction_head_centered/sugar/train/',
        'prediction_head_centered/ammonia/train/',
        3190
    )
    val_set, val_labels = create_dataset(
        'prediction_head_centered/control/val/',
        'prediction_head_centered/sugar/val/',
        'prediction_head_centered/ammonia/val/',
        3190
    )

    test_set, test_labels = create_dataset(
        'prediction_head_centered/control/test/',
        'prediction_head_centered/sugar/test/',
        'prediction_head_centered/ammonia/test/',
        3190
    )
    models = [model for model in os.listdir('.') if model.endswith('.keras')]
    for model in models:
        evaluate2(
            train_set,
            train_labels,
            val_set,
            val_labels,
            test_set,
            test_labels,
            model)
    """for model in ['Conv821BatchtanhConv496tanhFlatten.keras', 'Conv699eluConv492eluFlatten.keras', 'Conv698eluConv745BatchtanhConv492eluFlatten.keras']:
        evaluate(
            train_set,
            train_labels,
            val_set,
            val_labels,
            test_set,
            test_labels,
            model
        )"""
