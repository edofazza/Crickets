import numpy as np
import tensorflow as tf
import pickle
from matplotlib import pyplot as plt
from classification2.binary.training import create_dataset

def plot():
    with open('/Users/edoardo/Desktop/phd/researches/crickets/models/secondphase/binary/length1990/history', "rb") as file_pi:
        history = pickle.load(file_pi)

    loss = history['accuracy']
    print(max(loss))
    val_loss = history['val_accuracy']
    print(max(val_loss))
    plt.plot(range(1, len(loss) + 1), loss, 'r', label='Training Accuracy')
    plt.plot(range(1, len(loss) + 1), val_loss, 'g', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('length1990.png')

if __name__ == '__main__':
    plot()
