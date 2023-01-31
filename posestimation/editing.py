from argparse import ArgumentParser
import numpy as np
import cv2


class VideoCropping:
    @staticmethod
    def cropping(input_path: str, output_path: str, y_cropping=290, starting_frame=1740, last_frame=5220):
        """
        Crop the video starting from a specific frame to a certain frame. The crop has the same dimension
        of the frame width, producing in this way a square of dimension (width, width).
        :param input_path: path to the original video
        :param output_path: path to where to save the modified video
        :param y_cropping: y-coord where the crop starts
        :param starting_frame: frame in which the cut of the video starts
        :param last_frame: frame in which the cut of the video ends
        :return:
        """
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print('Error')

        fps = int(cap.get(5))
        frame_count = cap.get(7)
        print(f'Frame rate : {fps}')
        print(f'Frame count: {frame_count}')
        fps = 29

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (frame_width, frame_height)
        print(f'Frame size: {frame_size}')

        cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)  # set PROP_POS_FRAMES to starting_frame

        roi_size = (frame_width, frame_width)

        # Initialize video writer object
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                              roi_size)

        frames_progression = 0
        for i in range(starting_frame, last_frame):
            # read frame
            ret, frame = cap.read()

            if ret:
                roi = frame[y_cropping:y_cropping + frame_width, 0:frame_width]  # y range, x range
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
        cap = cv2.VideoCapture(output_path)
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
    parser.add_argument('--y_cropping', type=int, default=290)
    parser.add_argument('--starting_frame', type=int, default=1740)
    parser.add_argument('--last_frame', type=int, default=5220)
    return vars(parser.parse_args())


if __name__ == '__main__':
    opt = parse()
    VideoCropping.cropping(opt['input_video'],
                           opt['output_video'],
                           opt['y_cropping'],
                           opt['starting_frame'],
                           opt['last_frame'])
