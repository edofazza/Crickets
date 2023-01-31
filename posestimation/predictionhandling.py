import numpy as np
import sleap


def get_predictions(video_path: str, model_path: str, pred_output: str, with_images: bool):
    """
    Gets the prediction from a video using a specific trained model, and saves it as a slp file
    :param with_images: boolean indicating if the predictions must contain also the images
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


def obtain_np_seq_from_slp_pred(predictions_path, output_path: str):
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
