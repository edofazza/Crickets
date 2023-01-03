import os

from posestimation.singleinstancehandler import SingleInstanceHandler

if __name__=='__main__':
    max_strides = [32, 64]
    filters = [32, 64]
    input_scalings = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    old_model_path = 'm64_64_0.7_iter8'
    for max_stride in max_strides:
        for filter in filters:
            for input_scaling in input_scalings:
                if max_stride == 64 and filters == 64 and input_scaling == 0.7:
                    continue

                sih = SingleInstanceHandler(
                    'single_instance.json',
                    max_stride=max_stride,
                    filters=filter,
                    filters_rate=2.0,
                    input_scaling=input_scaling,
                    model_path=f'm{max_stride}_{filter}_{input_scaling}',
                    old_model_path=old_model_path
                )
                sih.save_single_instance_json('single_instance.json')
                old_model_path = f'm{max_stride}_{filter}_{input_scaling}'
                os.system(f'sleap-train single_instance.json train.v000.pkg.slp --val_labels val.v300.pkg.slp > {max_stride}_{filter}_{input_scaling}.txt')

