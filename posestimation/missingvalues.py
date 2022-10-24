from argparse import ArgumentParser
import numpy as np


class MissingValuesHandler: # TODO: static methods
    def __init__(self, pred): # TODO: function
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

    def analyze_and_fill(self, pred32: str, pred21: str, pred18: str, weighted=False): # TODO
        return []
        '''    file_name = pred32.split("/")[-1]
        print(f'Info about: {file_name}')
        # load the datasets
        pred32 = np.load(pred32)
        pred21 = np.load(pred21)
        pred18 = np.load(pred18)

        # transpose the values
        pred32 = pred32.T
        pred21 = pred21.T
        pred18 = pred18.T

        print(f'The shape is: {pred32.shape}')

        for i, (row32, row21, row18) in enumerate(zip(pred32, pred21, pred18)):
            # get missing values for each prediction
            missing32 = find_missing_values(row32)
            missing21 = find_missing_values(row21)
            missing18 = find_missing_values(row18)

            print(f'Missing values for entry {i} using pred32: {missing32}')
            print(f'Length of missing values for entry {i} using pred32: {len(missing32)}')
            print(f'Missing values for entry {i} using pred21: {missing21}')
            print(f'Length of missing values for entry {i} using pred21: {len(missing21)}')
            print(f'Missing values for entry {i} using pred18: {missing18}')
            print(f'Length of missing values for entry {i} using pred18: {len(missing18)}')

            if len(missing32) == 0:
                print(f'No missing values in pred32, filling operations were not performed for entry {i}')
            else:
                print(f'Missing values in pred32 are present, filling operations are performed for entry {i}:')
                # find mutual values in missing32 and missing 21
                both_missing_32_21 = list(set(missing32) & set(missing21))
                print(f'\t- Values missing both in pred32 and pred21: {both_missing_32_21}')
                print(f'\t- Length of values missing both in pred32 and pred21: {len(both_missing_32_21)}')

                to_fill_from_21 = list(set(missing32) - set(both_missing_32_21))

                for index in to_fill_from_21:
                    pred32[i][index] = pred21[i][index]
                print(f'\t- Values inserted from pred21 to pred32: {to_fill_from_21}')
                print(f'\t- Length of values inserted from pred21 to pred32: {len(to_fill_from_21)}')

                # update missing32
                missing32 = find_missing_values(pred32[i])
                print(f'\t- (Update) Missing values for entry {i} using pred32: {missing32}')
                print(f'\t- (Update) Length of missing values for entry {i} using pred32: {len(missing32)}')

                if missing32 == 0:
                    print(f'\t- No more missing values present for entry {i}')
                    continue
                else:
                    print(f'\t- Some values in pred32 are still missing, checking pred18 to fill more missing values')
                    both_missing_32_18 = list(set(missing32) & set(missing18))
                    print(f'\t- Values missing both in pred32 and pred18: {both_missing_32_18}')
                    print(f'\t- Length of values missing both in pred32 and pred18: {len(both_missing_32_18)}')

                    to_fill_from_18 = list(set(missing32) - set(both_missing_32_18))

                    for index in to_fill_from_18:
                        pred32[i][index] = pred18[i][index]
                    print(f'\t- Values inserted from pred18 to pred32: {to_fill_from_18}')
                    print(f'\t- Length of values inserted from pred18 to pred32: {len(to_fill_from_18)}')

                    # update missing32
                    missing32 = find_missing_values(pred32[i])
                    print(f'\t- (Update) Missing values for entry {i} using pred32: {missing32}')
                    print(f'\t- (Update) Length of missing values for entry {i} using pred32: {len(missing32)}')

                    if len(missing32) == 0:
                        print(f'\t- No more missing values present for entry {i}')
                        continue
                    else:
                        print(f'\t- Some values in pred32 are still missing, filling them using adjacent values')
                        if weighted:
                            pred32[i] = fill_missing_values_weighted(pred32[i])
                        else:
                            pred32[i] = fill_missing_values(pred32[i])
                        if len(find_missing_values(pred32[i])) != 0:
                            print('\t-SOMETHING WENT WRONG')
                
        return pred32'''


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
