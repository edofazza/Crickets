"""
Paper Name 1.0
© E. Fazzari, Department of Biorobotics
Scuola Superiore Sant'Anna, Pisa, Italy

https://github.com/edofazza/Crickets
Licensed under GNU General Public License v3.0
"""

import numpy as np
import sleap
from argparse import ArgumentParser


def get_predictions(video_path: str, model_path: str, pred_output: str, with_images: bool):
    """
    Gets the prediction from a video using a specific trained model, and saves it as a slp file
    :param with_images: boolean indicating if the labels must contain also the images
    :param video_path: string indicating the path to the video
    :param model_path: string indicating the path to the model
    :param pred_output: string indicating where the predictions are saved
    :return:
    """
    sleap.disable_preallocation()

    video = sleap.load_video(video_path)
    print(video.shape)

    predictor = sleap.load_model(model_path)
    predictions = predictor.predict(video)

    predictions = sleap.Labels(predictions.labeled_frames[:])
    predictions.save(pred_output, with_images=with_images, embed_all_labeled=True)


def obtain_head_sequence(predictions_path, output_path: str):
    """
    Save in a npy file the predictions of the head from a slp file
    :param predictions_path: path to where the slp file is located
    :param output_path: path where to save the npy file containing the head predictions
    :return:
    """
    predictions = sleap.load_file(predictions_path)
    head_list = []
    for label in predictions:
        instance = label[0].numpy()
        head_list.append(instance[0])

    np.save(output_path, head_list)


def obtain_all_sequence(predictions_path, output_path: str):
    """
    Save in a npy file the predictions from a slp file
    :param predictions_path: path to where the slp file is located
    :param output_path: path where to save the npy file containing the predictions
    :return:
    """
    predictions = sleap.load_file(predictions_path)
    dataset = []
    for label in predictions:
        frame_pred = []
        for values in label[0].numpy():
            frame_pred.append(values[0])
            frame_pred.append(values[1])
        dataset.append(frame_pred)

    np.save(output_path, dataset)


def parse():
    parser = ArgumentParser()
    parser.add_argument('--all_kp', type=bool, default=False)

    parser.add_argument('--only_head', type=bool, default=True)
    parser.add_argument('--labels_path', type=str, default='C1_C_pred.pkg.slp') # NEEDED when only_get_pred is False
    parser.add_argument('--seq_output_path', type=str, default='C1_C_pred.npy')

    parser.add_argument('--only_get_pred', type=bool, default=True)
    parser.add_argument('--video_path', type=str, default='C1_C.mp4')
    parser.add_argument('--model_path', type=str, default='models/220626_100050.single_instance')
    parser.add_argument('--pred_output_path', type=str, default='C1_C_pred.pkg.slp')
    parser.add_argument('--with_images', type=bool, default=True)
    return vars(parser.parse_args())


if __name__ == '__main__':
    opt = parse()
    if opt['only_head'] is True and opt['all_kp'] is True:
        print('--only_head and --all_kp cannot be both True')
    elif (opt['only_head'] or opt['all_kp']) and opt['only_get_pred']:
        get_predictions(
            opt['video_path'],
            opt['model_path'],
            opt['pred_output_path'],
            opt['with_images']
        )
        labels = sleap.load_file(opt['pred_output_path'])
        if opt['only_head']:
            obtain_head_sequence(labels, opt['seq_output_path'])
        else:
            obtain_all_sequence(labels, opt['seq_output_path'])
    elif opt['only_head']:
        labels = sleap.load_file(opt['labels_path'])
        obtain_head_sequence(labels, opt['seq_output_path'])
    elif opt['all_kp']:
        labels = sleap.load_file(opt['labels_path'])
        obtain_head_sequence(labels, opt['seq_output_path'])
    elif opt['only_get_pred']:
        get_predictions(
            opt['video_path'],
            opt['model_path'],
            opt['pred_output_path'],
            opt['with_images']
        )
