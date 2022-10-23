import numpy as np


def find_missing_values(pred):
    missing = []

    for i, v in enumerate(pred):
        if np.isnan(v):
            missing.append(i)
    return missing


def fill_missing_values(pred):
    new_pred = []
    for i, v in enumerate(pred):
        if np.isnan(v):
            prev = new_pred[i - 1]
            subs = None

            for j in range(i + 1, len(pred)):
                subs = pred[j]
                if np.isnan(subs):
                    subs = None # generalize in the case is the last the one missing
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

def fill_missing_values_weighted(pred):
    new_pred = []
    for i, v in enumerate(pred):
        if np.isnan(v):
            prev = new_pred[i - 1]
            subs = None
            k = 0

            for j in range(i + 1, len(pred)):
                k += 1
                subs = pred[j]
                if np.isnan(subs):
                    subs = None # generalize in the case is the last the one missing
                    continue
                else:
                    break
            if subs is None:
                new_pred.append(prev)
            else:
                alpha = 1/k
                new_pred.append((prev + alpha * subs) / (1 + alpha))
        else:
            new_pred.append(v)

    return new_pred


def analyze_and_fill(pred32: str, pred21: str, pred18: str, weighted=False):
    file_name = pred32.split("/")[-1]
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

    np.save('/Users/edoardo/Desktop/crickets/second phase/filled_weighted/' + file_name, pred32)



if __name__=='__main__':
    analyze_and_fill(
        '/Users/edoardo/Desktop/crickets/second phase/pred32/C1_A_0137.npy',
        '/Users/edoardo/Desktop/crickets/second phase/pred21/C1_A_0137.npy',
        '/Users/edoardo/Desktop/crickets/second phase/pred18/C1_A_0137.npy',
        True
    )