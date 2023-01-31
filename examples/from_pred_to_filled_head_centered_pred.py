from utils.missingvalues import from_pred_to_filled_pred
from utils.head_centralization import centralized_head_sequence_from_project

if __name__ == '__main__':
    # Set the prediction_npy_path to the new location of the dataset.
    # You can also change the name of the other two paths to locate them where you prefer. Keep
    # in mind that their name should differ to properly execute the code.
    prediction_npy_path = "predictions_npy"  # CHANGE HERE
    filled_prediction_path = 'predictions_filled'  # CHANGE HERE
    centered_prediction_path = 'prediction_head_centered'  # CHANGE HERE

    # With the following function the prediction will be analysed in the case they contain missing
    #  values (np.nan) and a report will be printed. To better read this report we advise you to
    # run this demo redirecting its output to a txt file:
    #           python from_pred_to_filled_hed_centered_pred.py > filling_report.txt
    # Aside from the report this function creates a new directory containing the sequences filled from
    # missing values following the same directory structure of predictions_npy. (Check the function for
    # more information)
    from_pred_to_filled_pred(prediction_npy_path, filled_prediction_path)
    # From the filled predictions we need to centralized the head in order consider only the antenna
    # positions related to the head location and not where is the cricket located in the frame. In order
    # to do that we can use the following functions that takes the filled-sequence directory and perform
    # the transformation creating a new directory with the new transformed sequences. Since the head positions
    # are not useful anymore since always set to zero, this function removes them from the sequences.
    # (Check the function for more information)
    centralized_head_sequence_from_project(filled_prediction_path, centered_prediction_path)
    