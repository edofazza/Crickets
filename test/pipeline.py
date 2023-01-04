import sleap


class Pipeline:
    @classmethod
    def run(cls, video_path):
        # GET VIDEO
        # IS MP4
        # CLIP AND CROP
        # LOAD VIDEO SLEAP
        # IS RGB
        # LOAD SLEAP MODEL
        # OBTAIN PREDICTIONS SLP
        # CLEAR TENSORFLOW
        # OBTAIN NPY FROM SLP
        # FILL MISSING VALUES
        # DIVIDE INTO SUBSEQUENCES
        # PREDICT SUBSEQUENCES
        # GET SUBSEQUENCES PREDICTIONS + FINAL CONSIDERATION (overall class)

        if not video_path.endswith('.mp4'):
            print('The video must be in mp4 format')
            return

        video = sleap.load_video(video_path)
        _, _, _, depth = video.shape
        if depth == 1:
            print('Video is not RGB')
            return

        # LENGTH? Divided it in parts
        # LOAD SLEAP MODEL
        # OBTAIN PREDICTION
        # CLEAR TENSORFLOW
        # TRANSFORM PREDICTIONS TO SEQUENCE
        # LOAD CLASSIFICATION MODEL
        # PREDICT
        # GET FINAL CONSIDERATION: the class more selected
        pass


if __name__ == '__main__':
    Pipeline().run('something.mp4')
