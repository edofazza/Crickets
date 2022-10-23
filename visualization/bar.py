import matplotlib.pyplot as plt
import warnings
from argparse import ArgumentParser


def plot_pose_estimation_results(results: dict, output_path: str):
    """

    :param output_path: a string stating where the plot will be saved
    :param results: A dictionary having the (key: values) pairs in the form (model_name: mAP value)
    :return:
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    models = list(results.keys())
    values = list(results.values())
    plt.bar(models, values)
    plt.xlabel('Models')
    plt.ylabel('mAP')
    plt.savefig(output_path)


def create_dict_from_text(file_path: str) -> dict:
    """
    From a file in the form
            model_name1 43
            model_name2 60
            ...
    This function create a dictionary in the form {model_name: map, ...}
    :param file_path: a string containing the file to reformat into a dictionary
    :return: the dictionary obtained reading the file
    """
    output = dict()
    with open(file_path, 'r') as f:
        for line in f:
            model, mean_ap = line.split(' ')
            output[model] = float(mean_ap)

    return output


def parse():
    parser = ArgumentParser()
    parser.add_argument('--results_path', type=str, default='results.txt')
    parser.add_argument('--plot_output_path', type=str, default='map_plot.png')
    return vars(parser.parse_args())


if __name__ == '__main__':
    opt = parse()
    results_dict = create_dict_from_text(opt['results_path'])
    plot_pose_estimation_results(
        results_dict,
        opt['plot_output_path']
    )