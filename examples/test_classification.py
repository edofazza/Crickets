import tensorflow as tf
from sklearn.metrics import f1_score
import os
import numpy as np

from utils.sequence_handling import create_dataset
from utils.visualize import conf_matrix

# Print the names of the wrongly classified sequences
def check_wrongly_classified_videos(pred, labels, names, n_classes):
    if n_classes == 2:
        pred = [1 if p > 0.5 else 0 for p in pred]
    else:
        pred = np.argmax(pred, axis=-1)
    for i, name in enumerate(names):
        if labels[i] != pred[i]:
            print(name)
    return pred


# The following function performs the `model.evaluate `for all the sets printing the results.
def evaluate(train_set, train_labels, val_set, val_labels, test_set, test_labels, model_name, train_names, val_names,
             test_names, n_classes, average='macro'):
    model = tf.keras.models.load_model(model_name)
    train_loss, train_accuracy = model.evaluate(train_set, train_labels, verbose=False)
    pred = model.predict(train_set)
    print('Wrongly classified video in training set:')
    pred = check_wrongly_classified_videos(pred, train_labels, train_names, n_classes)
    if n_classes == 2:
        print(f'Train f1-score {f1_score(train_labels, pred)}')
    else:
        print(f'Train f1-score {f1_score(train_labels, pred, average=average)}')
    val_loss, val_accuracy = model.evaluate(val_set, val_labels, verbose=False)
    pred = model.predict(val_set)
    print('Wrongly classified video in validation set:')
    pred = check_wrongly_classified_videos(pred, val_labels, val_names, n_classes)
    if n_classes == 2:
        print(f'Val f1-score {f1_score(val_labels, pred)}')
    else:
        print(f'Val f1-score {f1_score(val_labels, pred, average=average)}')
    test_loss, test_accuracy = model.evaluate(test_set, test_labels, verbose=False)
    pred = model.predict(test_set)
    print('Wrongly classified video in test set:')
    pred = check_wrongly_classified_videos(pred, test_labels, test_names, n_classes)
    if n_classes == 2:
        print(f'Test f1-score {f1_score(test_labels, pred)}')
    else:
        print(f'Test f1-score {f1_score(test_labels, pred, average=average)}')
    print(f'Test f1-score {f1_score(test_labels, pred, average="weighted")}')
    print(
        f'{model_name}:\n\tTrain loss: {train_loss}\n\tTrain accuracy: {train_accuracy}\n\tVal loss: {val_loss}\n\tVal accuracy: {val_accuracy}\n\tTest loss: {test_loss}\n\tTest accuracy: {test_accuracy}')


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
    train_set, train_labels, train_names = create_dataset(
        os.path.join(path, classes[0] + '/train/'),
        os.path.join(path, classes[1] + '/train/'),
        os.path.join(path, classes[2] + '/train/') if n_classes == 3 else None
    )
    val_set, val_labels, val_names = create_dataset(
        os.path.join(path, classes[0] + '/val/'),
        os.path.join(path, classes[1] + '/val/'),
        os.path.join(path, classes[2] + '/val/') if n_classes == 3 else None
    )
    test_set, test_labels, test_names = create_dataset(
        os.path.join(path, classes[0] + '/test/'),
        os.path.join(path, classes[1] + '/test/'),
        os.path.join(path, classes[2] + '/test/') if n_classes == 3 else None
    )

    # Let's evaluate the model. Add averages='mirco' or 'weighted' if you want to change the f1_score otherwise the default
    # value is 'macro' for 3 classes, and, of course, in the case of binary it is set automatically to 'binary'
    evaluate(train_set, train_labels, val_set, val_labels, test_set, test_labels, model_path, train_names, val_names, test_names, 3)

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
