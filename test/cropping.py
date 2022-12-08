import cv2

absolute_path = '/Users/edoardo/Desktop/phd/researches/crickets/video mp4/ammonia/C72_A.mp4'
output = '/Users/edoardo/Desktop/phd/researches/crickets/video 2 minutes/ammonia/C72_A.mp4'

if __name__=='__main__':
    cap = cv2.VideoCapture(absolute_path)

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
    box_dim = frame_width
    print(f'Frame size: {frame_size}')

    starting_frame = 1740
    cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame) # set PROP_POS_FRAMES to starting_frame

    roi_size = (frame_width, frame_width)

    # Initialize video writer object
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                          roi_size)

    frames_progression = 0
    for i in range(starting_frame, 5220):
        # read frame
        ret, frame = cap.read()

        if ret:
            roi = frame[290:290 + frame_width, 0:frame_width] # y range, x range
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