from argparse import ArgumentParser


class MovieManager:
    def __init__(self, fps=30):
        assert isinstance(fps, int)
        self.fps = fps
        pass

    def reduce_single_movie(self, in_path: str, out_path: str):
        """

        :param in_path: string containing the path of the video to be reduced
        :param out_path: path where the newly reduced video will be saved
        :return:
        """
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(in_path)
        clip = clip.without_audio()
        clip.write_videofile(out_path, fps=self.fps)
        clip.reader.close()

    def reduce_from_directories(self, *dirs: str):
        """
        Reduce videos from multiple directories
        :param dirs: a list of directories
        :return:
        """
        import os
        for dir_ in dirs:
            output_dir = dir_ + '_reduced'
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            videos = os.listdir(dir_)
            videos = [video for video in videos if video.endswith('.mp4')]
            for video in videos:
                self.reduce_single_movie(os.path.join(dir_, video), os.path.join(output_dir, video))

    def set_fps(self, fps: int):
        """

        :param fps: An integer indicating the number of frames
        :return:
        """
        self.fps = fps


def parse():
    parser = ArgumentParser()
    parser.add_argument('--fps', type=int, default=2)
    parser.add_argument('--dirs', type=bool, default=True)
    parser.add_argument('--dirs_names', type=list, default=['control', 'sugar', 'ammonia'])
    parser.add_argument('--video_name', type=str, default='C1_C.mp4')
    parser.add_argument('--output_path', type=str, default='C1_C_reduced.mp4')
    return vars(parser.parse_args())


if __name__ == '__main__':
    opt = parse()
    mm = MovieManager(opt['fps'])
    if opt['dirs']:
        mm.reduce_from_directories(opt['dir_names'])
    else:
        mm.reduce_single_movie(opt['video_name'], opt['output_path'])
