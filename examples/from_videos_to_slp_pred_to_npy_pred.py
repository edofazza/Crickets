import os

from posestimation.predictionhandling import get_predictions, obtain_np_seq_from_slp_pred

if __name__ == '__main__':
    # Here you need to set the path to the appropriate locations and to tell the code if
    # you want to save also the images when predicting each video frame. If you do that
    # set `save_slp_with_images=True`, in this way you can open the slp prediction in
    # your SLEAP GUI and analyze how the results are. Be aware that this operation costs
    # a lot of additional memory, thus we recommend to set the value to `True` only if
    # you want to see the results on SLEAP otherwise is better to have it at `False`
    # (i.e., only the predictions without the images/frames will be saves in the slp files).
    video_path = 'videos 2 minutes'
    prediction_slp_path = 'predictions_slp'     # directory to where the slp files will be saved
    save_slp_with_images = True
    prediction_npy_path = 'predictions_npy'     # directory to where the npy files will be saved

    # If the directories referred by prediction_slp_path and prediction_npy_path do not exist,
    # there is the need to create them.
    if not os.path.exists(prediction_slp_path):
        os.mkdir(prediction_slp_path)
    if not os.path.exists(prediction_npy_path):
        os.mkdir(prediction_npy_path)

    # Now we need to navigate the directories performing first the prediction and then the transformation to numpy.
    classes = os.listdir(video_path)
    for c in classes: # control sugar ammonia ..
        if c not in ['control', 'sugar', 'ammonia']:
            continue
        class_path = os.path.join(video_path, c)    # e.g., videos/control/
        sets = os.listdir(class_path)

        prediction_class_slp_path = os.path.join(prediction_slp_path, c)    # e.g., predictions_slp/control
        os.mkdir(prediction_class_slp_path)

        prediction_class_npy_path = os.path.join(prediction_npy_path, c)  # e.g., predictions_npy/control
        os.mkdir(prediction_class_npy_path)

        for s in sets: # train # test # val
            if s not in ['train', 'test', 'val']:
                continue

            set_path = os.path.join(class_path, s)  # e.g., videos/control/train
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
                    save_slp_with_images
                )

                obtain_np_seq_from_slp_pred(
                    os.path.join(prediction_set_slp_path, video[:-4] + '.pkg.slp'),
                    os.path.join(prediction_set_npy_path, video[:-4] + '.npy')
                )
