import os

from posestimation.predictionhandling import get_predictions, obtain_np_seq_from_slp_pred

if __name__ == '__main__': # TODO: remove or make it a demo
    path = 'videos'
    prediction_slp_path = 'predictions_slp'
    prediction_npy_path = 'predictions_npy'

    os.mkdir(prediction_slp_path)
    os.mkdir(prediction_npy_path)
    classes = os.listdir(path)
    for c in classes: # control sugar ammonia ..
        if c not in ['control', 'sugar', 'ammonia']:
            continue
        class_path = os.path.join(path, c) # videos/control/
        sets = os.listdir(class_path)

        prediction_class_slp_path = os.path.join(prediction_slp_path, c)    # predictions_slp/control
        os.mkdir(prediction_class_slp_path)

        prediction_class_npy_path = os.path.join(prediction_npy_path, c)  # predictions_npy/control
        os.mkdir(prediction_class_npy_path)

        for s in sets: # train # utils # val
            if s not in ['train', 'utils', 'val']:
                continue

            set_path = os.path.join(class_path, s)  # predictions/control/train
            videos = [video for video in os.listdir(set_path) if video.endswith('.mp4')]

            prediction_set_slp_path = os.path.join(prediction_class_slp_path, s)
            os.mkdir(prediction_set_slp_path)

            prediction_set_npy_path = os.path.join(prediction_class_npy_path, s)
            os.mkdir(prediction_set_npy_path)

            for video in videos:
                get_predictions(
                    os.path.join(set_path, video),
                    'm64_64_1.0/221226_195308.single_instance',
                    os.path.join(prediction_set_slp_path, video[:-4] + '.pkg.slp'),
                    True
                )

                obtain_np_seq_from_slp_pred(
                    os.path.join(prediction_set_slp_path, video[:-4] + '.pkg.slp'),
                    os.path.join(prediction_set_npy_path, video[:-4] + '.npy')
                )
