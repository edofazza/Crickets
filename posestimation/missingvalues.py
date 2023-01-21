"""
Paper Name 1.0
© E. Fazzari, Department of Biorobotics
Scuola Superiore Sant'Anna, Pisa, Italy

https://github.com/edofazza/Crickets
Licensed under GNU General Public License v3.0
"""

from argparse import ArgumentParser
import numpy as np


class MissingValuesHandler:
    def __init__(self, pred):
        """
        Check if the first value of pred is not NaN, if so it is changed to the
        closest not NaN value.
        :param pred: Single prediction sequence
        """
        self.pred = pred
        if np.isnan(self.pred[0]):
            i = 1
            while True:
                if not np.isnan(self.pred[i]):
                    self.pred[0] = self.pred[i]
                    break
                i += 1

    def find_missing_values(self):
        """

        :return: a list of all the indexes where the keypoint is NaN
        """
        missing = []
        for i, v in enumerate(self.pred):
            if np.isnan(v):
                missing.append(i)
        return missing

    def fill_missing_values(self):
        """
        Filling the missing values using a simple average method between the antecedent
        and the first not-nan-value.
        :return: a list containing the predictions
        """
        new_pred = []
        for i, v in enumerate(self.pred):
            if np.isnan(v):
                prev = new_pred[i - 1]
                subs = None

                for j in range(i + 1, len(self.pred)):
                    subs = self.pred[j]
                    if np.isnan(subs):
                        subs = None  # generalize in the case is the last the one missing
                        continue
                    else:
                        break
                if subs is None:
                    new_pred.append(prev)
                else:
                    new_pred.append((prev + subs) / 2)
            else:
                new_pred.append(v)

        return new_pred

    def fill_missing_values_weighted(self):
        """
        Filling the missing values using a weighted average method between the antecedent
        and the first not-nan-value. This method gives more important to the value closest
        to the missing one.
        :return: a list containing the predictions
        """
        new_pred = []
        for i, v in enumerate(self.pred):
            if np.isnan(v):
                prev = new_pred[i - 1]
                subs = None
                k = 0

                for j in range(i + 1, len(self.pred)):
                    k += 1
                    subs = self.pred[j]
                    if np.isnan(subs):
                        subs = None  # generalize in the case is the last the one missing
                        continue
                    else:
                        break
                if subs is None:
                    new_pred.append(prev)
                else:
                    alpha = 1 / k
                    new_pred.append((prev + alpha * subs) / (1 + alpha))
            else:
                new_pred.append(v)

        return new_pred


def parse():
    parser = ArgumentParser()
    parser.add_argument('--weighted', type=bool, default=False)
    parser.add_argument('--pred_path', type=str, default='C1_C.npy')
    parser.add_argument('--pred_output', type=str, default='C1_C_filled.npy')
    parser.add_argument('--head', type=bool, default=True)
    return vars(parser.parse_args())


if __name__ == '__main__':
    opt = parse()
    mvh = MissingValuesHandler(np.load(opt['pred_path']))
    if opt['head']:
        if opt['weighted']:
            np.save(opt['pred_output'], mvh.fill_missing_values_weighted())
        else:
            np.save(opt['pred_output'], mvh.fill_missing_values())
    else:
        np.save(opt['pred_output'], mvh.analyze_and_fill('a', 'a', 'a', opt['weighted']))
