import os
import numpy as np


def centralized_head_in_sequence(sequence_path, output_path):
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
    predictions = [prediction for prediction in os.listdir(dir_path) if prediction.endswith('.npy')]

    if os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    for prediction in predictions:
        centralized_head_in_sequence(os.path.join(dir_path, prediction), os.path.join(output_dir_path, prediction))


def centralized_head_sequence_from_project(project_path, output_path):
    if os.path.exists(output_path):
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
            if s not in ['train', 'utils', 'val']:
                continue
            centralized_head_sequence_from_dir(os.path.join(class_path, s), os.path.join(filled_class_path, s))
            set_path = os.path.join(class_path, s)  # predictions_npy/control/train
            predictions = [prediction for prediction in os.listdir(set_path) if prediction.endswith('.npy')]

            filled_set_path = os.path.join(filled_class_path, s)
            os.mkdir(filled_set_path)

            for prediction in predictions:
                centralized_head_in_sequence(os.path.join(set_path, prediction),
                                             os.path.join(filled_set_path, prediction))


if __name__ == '__main__': # TODO: remove
    prediction_npy_path = "/Users/edofazza/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/PhD/reseaches/crickets/predictions/predictions_filled"
    centered_prediction_path = 'prediction_head_centered'

    centralized_head_sequence_from_project(prediction_npy_path, centered_prediction_path)