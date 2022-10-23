from argparse import ArgumentParser
import os

from singleinstancehandler import SingleInstanceHandler


def parse():
    parser = ArgumentParser()
    parser.add_argument('--max_stride', type=int, default=32)
    parser.add_argument('--filters', type=int, default=32)
    parser.add_argument('--filters_rate', type=float, default=2.0)
    parser.add_argument('--input_scaling', type=float, default=0.5)
    parser.add_argument('--json_path', type=str, default='single_instance.json')
    parser.add_argument('--labels_path', type=str, default='labels.v000.pkg.slp')
    return vars(parser.parse_args())


if __name__ == '__main__':
    opt = parse()
    sih = SingleInstanceHandler(opt['json_path'], opt['max_stride'], opt['filters'],
                                opt['filters_rate'], opt['input_scaling'])
    sih.save_single_instance_json(opt['json_path'])
    os.system(f'sleap-train {opt["json_path"]} {opt["labels_path"]} > {sih.parameter_to_str()}.txt')
