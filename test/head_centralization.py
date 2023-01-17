import os
import numpy as np


if __name__ == '__main__':
    prediction_npy_path = "/Users/edofazza/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/PhD/reseaches/crickets/predictions/predictions_filled"
    filled_prediction_path = 'prediction_head_centered'

    os.mkdir(filled_prediction_path)
    classes = os.listdir(prediction_npy_path)
    for c in classes:   # control sugar ammonia
        if c not in ['control', 'sugar', 'ammonia']:
            continue
        class_path = os.path.join(prediction_npy_path, c)   # predictions_npy/control
        sets = os.listdir(class_path)

        filled_class_path = os.path.join(filled_prediction_path, c)
        os.mkdir(filled_class_path)

        for s in sets:
            if s not in ['train', 'test', 'val']:
                continue
            set_path = os.path.join(class_path, s)  # predictions_npy/control/train
            predictions = [prediction for prediction in os.listdir(set_path) if prediction.endswith('.npy')]

            filled_set_path = os.path.join(filled_class_path, s)
            os.mkdir(filled_set_path)

            for prediction in predictions:
                pred = np.load(os.path.join(set_path, prediction)).T

                for frame in pred:
                    x, y = [], []
                    x_head, y_head = frame[0], frame[1]
                    for i in range(0, len(frame), 2):
                        frame[i] = frame[i] - x_head
                        frame[i + 1] = frame[i + 1] - y_head
                pred = pred.T[2:, :]
                print(pred.shape)
                np.save(os.path.join(filled_set_path, prediction), pred)
