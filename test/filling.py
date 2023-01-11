import os
import numpy as np

from posestimation.missingvalues import MissingValuesHandler

if __name__ == '__main__':
    prediction_npy_path = 'predictions_npy'
    filled_prediction_path = 'predictions_filled'

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
                print(prediction)
                print(f'\t-Shape: {pred.shape}')

                for i, row in enumerate(pred):
                    mvh = MissingValuesHandler(row)
                    missing = mvh.find_missing_values()
                    print(f'\t-Missing values for entry {i}: {len(missing)}')
                    if len(missing) == 0:
                        print(f'\tNo missing values for entry {i}, filling operation skipped')
                    else:
                        pred[i] = mvh.fill_missing_values_weighted()
                np.save(os.path.join(filled_set_path, prediction))
