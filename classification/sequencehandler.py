import os
import numpy as np


class SequenceHandler:
    @classmethod
    def divide_sequence(cls, data_path: str, length: int):
        """

        :param data_path: path to npy file
        :param length: length of the subsequences
        :return:
        """
        # retrieve sequence
        sequence = np.load(data_path)
        _, dim = sequence.shape

        dir_path = '/'.join(data_path.split('/')[:-1]) + '/'
        for i in range(dim - length):
            tmp_seq = sequence[:, i:i + length]
            if not os.path.exists(f'{dir_path}dataset{length}'):
                os.mkdir(f'{dir_path}dataset{length}')
            np.save(f'{dir_path}dataset{length}/{data_path.split("/")[-1]}_{i}-{i + length}', tmp_seq)