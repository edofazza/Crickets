import tensorflow as tf
import os
import numpy as np


def divide_sequence(sequence, length): # TODO: remove and use the one in utils
    if length == 3480:
        return np.array(sequence)
    _, dim = sequence.shape
    tmp = []
    for i in range(dim - length):
        tmp_seq = sequence[:, i:i + length]
        tmp.append(tmp_seq)
    return np.array(tmp)


def create_dataset(control_path, sugar_path, ammonia_path=None, length=3480): # TODO: remove and use the one in utils
    if length < 3480:
        data = None
    else:
        data = []
    labels = []
    tmp_list = [c for c in os.listdir(control_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(control_path, i))
        labels.append(0)
        if type(data) is list:
            data.append(tmp_npy)
        else:
            if data is None:
                data = divide_sequence(tmp_npy, length)
            else:
                data = np.r_[data, divide_sequence(tmp_npy, length)]

    tmp_list = [c for c in os.listdir(sugar_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(sugar_path, i))
        labels.append(1)
        if type(data) is list:
            data.append(tmp_npy)
        else:
            data = np.r_[data, divide_sequence(tmp_npy, length)]

    if ammonia_path is not None:
        tmp_list = [c for c in os.listdir(ammonia_path) if c.endswith('.npy')]
        for i in tmp_list:
            tmp_npy = np.load(os.path.join(ammonia_path, i))
            labels.append(2)
            if type(data) is list:
                data.append(tmp_npy)
            else:
                data = np.r_[data, divide_sequence(tmp_npy, length)]

    if type(data) is list:
        return normalize(np.array(data)), np.array(labels)
    else:
        return normalize(data), np.array(labels)


def normalize(x): # TODO: remove and use the one in utils
    return tf.keras.utils.normalize(x, axis=-1)


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
    if test_accuracy > 0.5 and train_accuracy > 0.5 and val_accuracy > 0.5: # and train_accuracy > val_accuracy
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
        'prediction_head_centered/ammonia/train/'
    )
    val_set, val_labels = create_dataset(
        'prediction_head_centered/control/val/',
        'prediction_head_centered/sugar/val/',
        'prediction_head_centered/ammonia/val/'
    )

    test_set, test_labels = create_dataset(
        'prediction_head_centered/control/test/',
        'prediction_head_centered/sugar/test/',
        'prediction_head_centered/ammonia/test/'
    )
    models = [model for model in os.listdir('') if model.endswith('.keras')]
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
