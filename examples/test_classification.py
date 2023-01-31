import tensorflow as tf
from tensorflow import keras as ks
from sklearn.metrics import confusion_matrix
import os
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

from utils.sequence_handling import create_dataset
from utils.visualize import conf_matrix

# The following function performs the `model.evaluate `for all the sets printing the results.
def evaluate(train_set, train_labels, val_set, val_labels, test_set, test_labels, model_name):
    model = tf.keras.models.load_model(model_name)
    train_loss, train_accuracy = model.evaluate(train_set, train_labels, verbose=False)
    val_loss, val_accuracy = model.evaluate(val_set, val_labels, verbose=False)
    test_loss, test_accuracy = model.evaluate(test_set, test_labels, verbose=False)
    print(f'{model_name}:\n\tTrain loss: {train_loss}\n\tTrain accuracy: {train_accuracy}\n\tVal loss: {val_loss}\n\tVal accuracy: {val_accuracy}\n\tTest loss: {test_loss}\n\tTest accuracy: {test_accuracy}')

if __name__ == '__main__':
    # Change the following variables in order to match you path for prediction_head_centered
    # and the path related to the model. Remember that if you are using a binary classification
    # model (e.g., control-sugar) it is important to change also the `n_classes` variable along
    # with the `classes` list.
    path = 'prediction_head_centered'
    model_path = 'best3classes.keras'
    n_classes = 3
    classes = ['control', 'sugar', 'ammonia']

    # Let's create the numpy sets
    train_set, train_labels = create_dataset(
        os.path.join(path, classes[0] + '/train/'),
        os.path.join(path, classes[1] + '/train/'),
        os.path.join(path, classes[2] + '/train/') if n_classes == 3 else None
    )
    val_set, val_labels = create_dataset(
        os.path.join(path, classes[0] + '/val/'),
        os.path.join(path, classes[1] + '/val/'),
        os.path.join(path, classes[2] + '/val/') if n_classes == 3 else None
    )
    test_set, test_labels = create_dataset(
        os.path.join(path, classes[0] + '/test/'),
        os.path.join(path, classes[1] + '/test/'),
        os.path.join(path, classes[2] + '/test/') if n_classes == 3 else None
    )

    # Let's evaluate the model
    evaluate(train_set, train_labels, val_set, val_labels, test_set, test_labels, model_path)

    # Print confusion matrix for the training set
    conf_matrix(
        model_path,
        os.path.join(path, classes[0] + '/train/'),
        os.path.join(path, classes[1] + '/train/'),
        os.path.join(path, classes[2] + '/train/') if n_classes == 3 else None,
        classes
    )

    # Print confusion matrix for the val set
    conf_matrix(
        model_path,
        os.path.join(path, classes[0] + '/val/'),
        os.path.join(path, classes[1] + '/val/'),
        os.path.join(path, classes[2] + '/val/') if n_classes == 3 else None,
        classes
    )

    # Print confusion matrix for the test set
    conf_matrix(
        model_path,
        os.path.join(path, classes[0] + '/test/'),
        os.path.join(path, classes[1] + '/test/'),
        os.path.join(path, classes[2] + '/test/') if n_classes == 3 else None,
        classes
    )
