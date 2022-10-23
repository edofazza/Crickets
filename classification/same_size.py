import numpy as np
import os


def padding(input_: str, output: str, length: int):
    input_ = np.load(input_)
    _, input_length = input_.shape
    to_fill = length - input_length
    pad = np.zeros((10, to_fill))
    np.save(output, np.c_[input_, pad])


def loop_operation(input_dir, output_dir):
    files_ = os.listdir(input_dir)
    files_ = [file for file in files_ if file.endswith('.npy')]
    for file in files_:
        padding(input_dir + file, output_dir + file, 10652)


def final_dataset(input_dir):
    files_ = os.listdir(input_dir)
    files_ = [file for file in files_ if file.endswith('.npy') and file.startswith('C')]
    dataset = []
    labels = []
    for file in files_:
        labels.append(file[3])
        dataset.append(np.load(input_dir + file))

    dataset = np.array(dataset)
    labels = np.array(labels)
    np.save(input_dir + 'labels.npy', labels)
    np.save(input_dir + 'dataset.npy', dataset)


if __name__ == '__main__':
    # final_dataset('/Users/edoardo/Desktop/crickets/second phase/datasets/d10652/normalized/')
    final_dataset('/Users/edoardo/Desktop/crickets/second phase/datasets/d10652/weighted_normalized/')
    '''
    loop_operation(
        '/Users/edoardo/Desktop/crickets/second phase/filled_normalized/',
        '/Users/edoardo/Desktop/crickets/second phase/datasets/d10652/normalized/'
    )

    loop_operation(
        '/Users/edoardo/Desktop/crickets/second phase/filled_weighted_normalized/',
        '/Users/edoardo/Desktop/crickets/second phase/datasets/d10652/weighted_normalized/'
    )
    '''
