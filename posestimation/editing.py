from argparse import ArgumentParser
import numpy as np
import cv2


class VideoCropping:
    def __init__(self, video_path):
        self.video_path = video_path

    @staticmethod
    def crops_frames_and_borders(video: str, head_pred: str, output: str, box_dim: int):
        """

        :param video: the path to the video to be cropped
        :param head_pred: a numpy array containing the prediction of the head for each frame
        :param output: the path where the cropped video will be stored
        :param box_dim: the dimension of the box
        :return:
        """
        ############# LOAD VIDEO #############
        cap = cv2.VideoCapture(video)
        ############# LOAD HEAD PREDICTIONS #############
        head_p = np.load(head_pred)

        ############## VIDEO INFOS #############
        if not cap.isOpened():
            print("Error opening the video file")
            return

        print("Opening video " + str.split('/')[-1])
        # Get frame rate information
        fps = int(cap.get(5))
        print("Frame Rate : ", fps, "frames per second")
        # Get frame count
        frame_count = cap.get(7)
        print("Frame count : ", frame_count)

        # Starting and ending frames, length of video in frames
        starting_frame = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)

        # Obtain frame size information using get() method
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (frame_width, frame_height)

        roi_size = (800, 800)

        # Initialize video writer object
        out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                              roi_size)

        frames_progression = 0
        # For loop that iterates until the index matches the desired length of the clip
        for i in range(int(frame_count)):  # reads through each frome within the loop, and then writes that frame into the new video isolate the roi
            # read frame
            ret, frame = cap.read()
            # read head position
            x_orig, y_orig = list(head_p[i])
            if np.isnan(x_orig):
                print(i)
            x_roi, y_roi = int(x_orig), int(y_orig)
            if ret:
                if y_roi - box_dim//2 < 0:
                    y_roi = box_dim//2

                if y_roi + box_dim//2 > frame_height:
                    y_roi -= y_roi + box_dim//2 - frame_height
                # Locating the ROI within the frames that will be cropped
                roi = frame[y_roi - box_dim//2:y_roi + box_dim//2, x_roi - box_dim//2:x_roi + box_dim//2]
                #print(y_roi - box_dim//2, y_roi + box_dim//2, x_roi - box_dim//2, x_roi + box_dim//2)
                # Write the frame into the file output .avi that was read from the original video in cap.read()
                out.write(roi)
            else:
                print("Cannot retrieve frames. Breaking.")  # If a frame cannot be retrieved, this error is thrown

            if not out.isOpened():
                print("Error opening the video file")  # If the out video cannot be opened, this error is thrown
            else:
                frames_progression = frames_progression + 1  # Shows how far the frame writting process got. Compare this to the desired frame length
        print(frames_progression)

        # Release the capture
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Finished writing new video")

        # These were just for me to verify that the right frames were written
        cap = cv2.VideoCapture(output)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        print('fps = ' + str(fps))
        print('number of frames = ' + str(frame_count))
        print('duration (S) = ' + str(duration))
        cap.release()
        cv2.destroyAllWindows()


def parse():
    parser = ArgumentParser()
    parser.add_argument('--input_video', type=str, default='C1_C.mp4')
    parser.add_argument('--output_video', type=str, default='C1_C_head.npy')
    parser.add_argument('--head_pred_path', type=str, default='C1_C_cropped.mp4')
    parser.add_argument('--box_size', type=int, default=800)
    return vars(parser.parse_args())


if __name__ == '__main__':
    opt = parse()
    VideoCropping.crops_frames_and_borders(opt['input_video'],
                                           opt['head_pred_path'],
                                           opt['output_video'],
                                           opt['box_size'])
