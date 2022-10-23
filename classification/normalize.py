import numpy as np
import os


def normalize(pred: str, output_dir: str):
    name = pred.split('/')[-1]
    pred = np.load(pred)

    # Normalize between 0 and 1
    pred = np.interp(pred, (pred.min(), pred.max()), (0, 1))

    # strictly normalization
    pred -= pred.mean()
    pred /= pred.std()

    np.save(output_dir + name, pred)


if __name__ == '__main__':
    input_dir = '/Users/edoardo/Desktop/crickets/second phase/filled/'
    files = os.listdir(input_dir)
    files = [file for file in files if file.endswith('.npy')]

    for file in files:
        normalize(input_dir + file, '/Users/edoardo/Desktop/crickets/second phase/filled_normalized/')

    input_dir = '/Users/edoardo/Desktop/crickets/second phase/filled_weighted/'
    files = os.listdir(input_dir)
    files = [file for file in files if file.endswith('.npy')]

    for file in files:
        normalize(input_dir + file, '/Users/edoardo/Desktop/crickets/second phase/filled_weighted_normalized/')
