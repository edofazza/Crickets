import os
import json


class SingleInstanceHandler:
    def __init__(self, json_path: str, max_stride=32, filters=32,
                 filters_rate=2.0, input_scaling=0.5, model_path='models',
                 old_model_path='models'):
        with open(json_path, 'r') as f:
            self.json_content = json.loads(f.read())

        self.set_max_stride(max_stride)
        self.set_filters(filters)
        self.set_filters_rate(filters_rate)
        self.set_input_scaling(input_scaling)
        self.set_model_path(model_path)

        self.change_job_yaml(model_path, old_model_path)

    def save_single_instance_json(self, output_path: str):
        """
        Save self.json_content as a json file
        :param output_path: the path where the json file will be saved.
        :return:
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.json_content, f, ensure_ascii=False, indent=4)

    def set_max_stride(self, max_stride: int):
        """
        Set the max_stride value inside self.json_content
        :param max_stride: an integer (some suggestions are 32 and 64)
        :return:
        """
        self.json_content['model']['backbone']['unet']['max_stride'] = max_stride

    def set_filters(self, filters: int):
        """
        Set the filters value inside self.json_content
        :param filters: an integer (some suggestions are 32 and 64)
        :return:
        """
        self.json_content['model']['backbone']['unet']['filters'] = filters

    def set_filters_rate(self, filters_rate: float):
        """
        Set the filters rate value inside self.json_content
        :param filters_rate: a float (some suggestions are 2.0 and 2.5, higher values can cause OMM)
        :return:
        """
        self.json_content['model']['backbone']['unet']['filters_rate'] = filters_rate

    def set_input_scaling(self, input_scaling: float):
        """
        Set the input scaling value inside self.json_content
        :param input_scaling: a float (some suggestions are 0.5, 0.6, 0.7)
        :return:
        """
        self.json_content['data']['preprocessing']['input_scaling'] = input_scaling

    def set_model_path(self, model_path: str):
        """

        :param model_path: directory where the trained will be saved
        :return:
        """
        self.json_content['outputs']['runs_folder'] = model_path

    def get_max_stride(self):
        """
        Get the max_stride value inside self.json_content
        :return: an int
        """
        return self.json_content['model']['backbone']['unet']['max_stride']

    def get_filters(self):
        """
        Get the filters value inside self.json_content
        :return: an int
        """
        return self.json_content['model']['backbone']['unet']['filters']

    def get_filters_rate(self):
        """
        Get the filters rate value inside self.json_content
        :return: a float
        """
        return self.json_content['model']['backbone']['unet']['filters_rate']

    def get_input_scaling(self):
        """
        Get the input scaling value inside self.json_content
        :return: a float
        """
        return self.json_content['data']['preprocessing']['input_scaling']

    def get_model_path(self):
        """
        Get the run_folder inside self.json_content
        :return:
        """
        return self.json_content['outputs']['runs_folder']

    def parameter_to_str(self):
        """
        Transform the parameters changed using the class in a string
        :return: a string containing the information of the parameters changed in self.json_content
        """
        return str(self.get_max_stride()) + '_' + str(self.get_filters()) + '_' \
               + str(self.get_filters_rate()) + '_' + str(self.get_input_scaling())

    def change_job_yaml(self, model_path, old_model_path):
        """
        Change the jobs.yaml file created in the sleap training package in order to match the new
        configuration of the single_instance.json file in the same directory
        :param model_path: path to the new model that will be used for training
        :param old_model_path: path to the previous model indicated in jobs.yaml and that you wish to change
        :return:
        """
        with open('tmp.yaml', 'w') as out:
            with open('jobs.yaml', 'r') as f:
                for i in f:
                    if old_model_path in i:
                        i = i.replace(old_model_path, model_path)
                    out.write(i)
        os.remove('jobs.yaml')
        os.rename('tmp.yaml', 'jobs.yaml')


def training(
        max_stride,
        filter_,
        input_scaling,
        old_model_path
):
    """
    Run the sleap-train starting from an original configuration
    :param max_stride:
    :param filter_:
    :param input_scaling:
    :param old_model_path: path to the model in the original/previous configuration (usually is "models")
    :return:
    """
    sih = SingleInstanceHandler(
        'single_instance.json',
        max_stride=max_stride,
        filters=filter_,
        filters_rate=2.0,
        input_scaling=input_scaling,
        model_path=f'm{max_stride}_{filter}_{input_scaling}',
        old_model_path=old_model_path
    )
    sih.save_single_instance_json('single_instance.json')
    os.system(
        f'sleap-train single_instance.json train.v000.pkg.slp --val_labels val.v300.pkg.slp > {sih.parameter_to_str()}.txt')
    return f'm{max_stride}_{filter}_{input_scaling}'


def gridsearch(max_strides: list, filters: list, input_scaling_ranges: list, old_model_path='models'):
    """
    Perform a grid search using all possible combinations of the input parameters
    :param max_strides: list of max strides that you wish to test
    :param filters: list of filters that you wsih to test
    :param input_scaling_ranges: list of input scaling that you wish to test
    :param old_model_path: path to the model in the original/previous configuration (usually is "models")
    :return:
    """
    for max_stride in max_strides:
        for filter_ in filters:
            for input_scaling in input_scaling_ranges:
                old_model_path = training(max_stride, filter_, input_scaling, old_model_path)


if __name__ == '__main__':
    gridsearch([32, 64], [32, 64], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])