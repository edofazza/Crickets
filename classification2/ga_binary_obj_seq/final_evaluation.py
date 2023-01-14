import tensorflow as tf
import os
import numpy as np


def normalize(x):
    return tf.keras.utils.normalize(x, axis=-1)


def create_dataset(control_path, sugar_path):
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


def evaluate(model_name):
    model = tf.keras.models.load_model(model_name)

    train_set, train_labels = create_dataset(
        'predictions_filled/control/train/',
        'predictions_filled/sugar/train/'
    )
    val_set, val_labels = create_dataset(
        'predictions_filled/control/val/',
        'predictions_filled/sugar/val/'
    )

    test_set, test_labels = create_dataset(
        'predictions_filled/control/test/',
        'predictions_filled/sugar/test/'
    )

    train_loss, train_accuracy = model.evaluate(train_set, train_labels, verbose=False)
    val_loss, val_accuracy = model.evaluate(val_set, val_labels, verbose=False)
    test_loss, test_accuracy = model.evaluate(test_set, test_labels, verbose=False)
    print(
        f'{model_name}:\n\tTrain loss: {train_loss}\n\tTrain accuracy: {train_accuracy}\n\tVal loss: {val_loss}\n\tVal accuracy: {val_accuracy}\n\tTest loss: {test_loss}\n\tTest accuracy: {test_accuracy}')


def evaluate2(train_set, train_labels, val_set, val_labels, test_set, test_labels, model_name):
    model = tf.keras.models.load_model(model_name)

    train_loss, train_accuracy = model.evaluate(train_set, train_labels, verbose=False)
    val_loss, val_accuracy = model.evaluate(val_set, val_labels, verbose=False)
    test_loss, test_accuracy = model.evaluate(test_set, test_labels, verbose=False)
    if test_accuracy > 0.5 and train_accuracy > val_accuracy and train_accuracy > 0.5 and val_accuracy > 0.5:
        print(f'{model_name}:\n\tTrain loss: {train_loss}\n\tTrain accuracy: {train_accuracy}\n\tVal loss: {val_loss}\n\tVal accuracy: {val_accuracy}\n\tTest loss: {test_loss}\n\tTest accuracy: {test_accuracy}')


if __name__=='__main__':
    """evaluate('Conv558eluDrop0.4Conv782BatchtanhDrop0.25Conv984tanhDrop0.45Conv724BatcheluGlobalDense159eluDrop0.15Dense351tanhDrop0.35Dense394leaky_reluDrop0.4Dense433sigmoid.keras')
    evaluate('Conv807BatchtanhDrop0.25Conv984tanhDrop0.15Conv760BatcheluGlobalDense151eluDrop0.3Dense146tanhDrop0.1Dense117leaky_reluDrop0.3.keras')
    evaluate('Conv844BatchtanhDrop0.25Conv984tanhDrop0.15Conv778BatcheluGlobalDense151eluDrop0.3Dense146tanhDrop0.35Dense433sigmoid.keras')
    evaluate('Conv845BatchtanhConv971tanhDrop0.15Conv729eluGlobalDense150eluDrop0.3Dense354tanhDrop0.05Dense433sigmoid.keras')
    """
    train_set, train_labels = create_dataset(
        'predictions_filled/control/train/',
        'predictions_filled/sugar/train/'
    )
    val_set, val_labels = create_dataset(
        'predictions_filled/control/val/',
        'predictions_filled/sugar/val/'
    )

    test_set, test_labels = create_dataset(
        'predictions_filled/control/test/',
        'predictions_filled/sugar/test/'
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
