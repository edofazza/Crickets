import numpy as np


def load_dataset(dataset, labels):
    d = np.load(dataset)
    l = np.load(labels)
    options = {
        'C': 0,
        'A': 1,
        'D': 2
    }
    labels2 = []
    for label in l:
        labels2.append(options[label])
    labels2 = np.array(labels2)
    return d, labels2
