import os
import numpy as np


def centralized_head_in_sequence(sequence_path, output_path):
    """
    From a numpy sequence with filled values this function centralizes the position
    of the head and moves all the other joints accordingly. Then removes the head sequences (positions 0 and 1)
    since now they are all 0, thus meaningless.
    :param sequence_path: path to where the numpy sequence is located
    :param output_path: output where the new sequence will be saved
    :return:
    """
    pred = np.load(sequence_path).T

    for frame in pred:
        x_head, y_head = frame[0], frame[1]
        for i in range(0, len(frame), 2):
            frame[i] = frame[i] - x_head
            frame[i + 1] = frame[i + 1] - y_head
    pred = pred.T[2:, :]
    print(pred.shape)
    np.save(output_path, pred)


def centralized_head_sequence_from_dir(dir_path, output_dir_path):
    """

    :param dir_path: path to directory containing sequences
    :param output_dir_path: path to new directory where the sequences will be stored
    :return:
    """
    predictions = [prediction for prediction in os.listdir(dir_path) if prediction.endswith('.npy')]

    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    for prediction in predictions:
        centralized_head_in_sequence(os.path.join(dir_path, prediction), os.path.join(output_dir_path, prediction))


def centralized_head_sequence_from_project(project_path, output_path):
    """
    Perform the transformation starting from a directory (project_path) organized as
            - control
                - train
                - val
                - test
            - sugar
                - train
                - val
                - test
            - ammonia
                - train
                - val
                - test
    And a new directory organized (output_path) in the same way where the new sequences will be stored.
    :param project_path: path to the original directory
    :param output_path: path to the new directory containing the results
    :return:
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    classes = os.listdir(project_path)
    for c in classes:  # control sugar ammonia
        if c not in ['control', 'sugar', 'ammonia']:
            continue
        class_path = os.path.join(project_path, c)  # predictions_npy/control
        sets = os.listdir(class_path)

        filled_class_path = os.path.join(output_path, c)
        os.mkdir(filled_class_path)

        for s in sets:
            if s not in ['train', 'test', 'val']:
                continue
            centralized_head_sequence_from_dir(os.path.join(class_path, s), os.path.join(filled_class_path, s))
