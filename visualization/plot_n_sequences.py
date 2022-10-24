import numpy as np
import os
import matplotlib.pyplot as plt


def plot_divide_sequence(data_path: str, length: int):
    predictions = os.listdir(data_path)
    predictions = [file for file in predictions if file.endswith('.npy') and file.startswith('C')]

    count = 0
    for prediction in predictions:
        sequence = np.load(data_path + prediction)
        _, dim = sequence.shape
        for i in range(dim - length):
            count += 1
    return count


def loop_div_seq():
    path = '/Users/edoardo/Desktop/crickets/second phase/filled_normalized/'
    values = [plot_divide_sequence(path, 29)]
    for length in range(110, 1111, 100):
        values.append(plot_divide_sequence(path, length))

    list_ = list(range(110, 1111, 100))
    plt.plot([29] + list_, values)
    plt.xlabel('Sequence length')
    plt.ylabel('Number of sequences')
    plt.tick_params()
    plt.xticks([29] + list_)
    plt.savefig('training.png')


def general_plot(values1110, values, values29, ylabel, filename, plot_resnet50=True):
    vanilla = [values1110[0]]
    for i in range(0, len(values), 4):
        vanilla.append(values[i])
    vanilla.append(values29[0])
    resnet18 = [values1110[1]]
    for i in range(1, len(values), 4):
        resnet18.append(values[i])
    resnet18.append(values29[1])

    resnet34 = [values1110[2]]
    for i in range(2, len(values), 4):
        resnet34.append(values[i])
    resnet34.append(values29[2])

    resnet50 = [values1110[3]]
    for i in range(3, len(values), 4):
        resnet50.append(values[i])
    resnet50.append(values29[3])

    indexes = ([29] + list(range(110, 1111, 100)))[::-1]
    plt.plot(indexes, vanilla, label='Vanilla')
    plt.plot(indexes, resnet18, label='ResNet18')
    plt.plot(indexes, resnet34, label='ResNet34')
    if plot_resnet50:
        plt.plot(indexes, resnet50, label='ResNet50')
    plt.xlabel('Sequence length')
    plt.ylabel(ylabel)
    plt.tick_params()
    plt.xticks(indexes)
    plt.legend()
    plt.savefig(filename)


def plot_loop_results():
    dir_path = '/Users/edoardo/Desktop/crickets/second phase/results/'
    # NORMALIZED
    # validation loss
    '''values1110 = [2.6296165245298653e-10, 0.37599465250968933, 2.0219531059265137, 572.0909423828125]
    values = np.load(dir_path + 'val_loss_NORMALIZED.npy')
    values29 = np.load(dir_path + 'val_loss_NORMALIZED29.npy')
    general_plot(values1110, values, values29, 'Validation loss')


    # validation accuracy
    values1110 = [1.0, 0.8066176176071167, 0.5144608020782471, 0.4360294044017792]
    values = np.load(dir_path + 'val_acc_NORMALIZED.npy')
    values29 = np.load(dir_path + 'val_acc_NORMALIZED29.npy')
    general_plot(values1110, values, values29, 'Validation accuracy', 'validation_accuracy.png')

    # test loss
    values1110 = [3.681733806715215e-10, 0.38802483677864075, 2.0733277797698975, 569.5001220703125]
    values = np.load(dir_path + 'test_loss_NORMALIZED.npy')
    values29 = np.load(dir_path + 'test_loss_NORMALIZED29.npy')
    general_plot(values1110, values, values29, 'Test loss', 'test_loss.png')
    general_plot(values1110, values, values29, 'Test loss', 'test_loss_without50.png', False)
    '''
    # test accuracy
    values1110 = [1.0, 0.8045444488525391, 0.5268034338951111, 0.43723803758621216]
    values = np.load(dir_path + 'test_acc_NORMALIZED.npy')
    values29 = np.load(dir_path + 'test_acc_NORMALIZED29.npy')
    general_plot(values1110, values, values29, 'Test Accuracy', 'test_acc.png')
    '''
    # WEIGHTED NORMALIZED
    values1110 = np.load(dir_path + 'val_loss_WEIGHTED_NORMALIZED1110WN.npy')
    values = np.load(dir_path + 'val_loss_WEIGHTED_NORMALIZED.npy')
    values29 = np.load(dir_path + 'val_loss_WEIGHTED_NORMALIZED29.npy')
    #general_plot(values1110, values, values29, 'Validation loss', 'validation_lossWN.png')
    general_plot(values1110, values, values29, 'Validation loss', 'validation_loss_without50WN.png', False)

    # validation accuracy
    values1110 = np.load(dir_path + 'val_acc_WEIGHTED_NORMALIZED1110WN.npy')
    values = np.load(dir_path + 'val_acc_WEIGHTED_NORMALIZED.npy')
    values29 = np.load(dir_path + 'val_acc_WEIGHTED_NORMALIZED29.npy')
    general_plot(values1110, values, values29, 'Validation accuracy', 'validation_accuracyWN.png')

    # test loss
    values1110 = np.load(dir_path + 'test_loss_WEIGHTED_NORMALIZED1110WN.npy')
    values = np.load(dir_path + 'test_loss_WEIGHTED_NORMALIZED.npy')
    values29 = np.load(dir_path + 'test_loss_WEIGHTED_NORMALIZED29.npy')
    general_plot(values1110, values, values29, 'Test loss', 'test_lossWN.png')
    general_plot(values1110, values, values29, 'Test loss', 'test_loss_without50WN.png', False)

    # test accuracy
    values1110 = np.load(dir_path + 'test_acc_WEIGHTED_NORMALIZED1110WN.npy')
    values = np.load(dir_path + 'test_acc_WEIGHTED_NORMALIZED.npy')
    values29 = np.load(dir_path + 'test_acc_WEIGHTED_NORMALIZED29.npy')
    general_plot(values1110, values, values29, 'Test Accuracy', 'test_accWN.png')'''


if __name__ == '__main__':
    plot_loop_results()

