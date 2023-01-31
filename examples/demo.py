import numpy as np
from tensorflow import keras as ks

from posestimation.editing import VideoCropping
from posestimation.predictionhandling import get_predictions, obtain_np_seq_from_slp_pred
from posestimation.moviemanager import MovieManager
from utils.missingvalues import MissingValuesHandler
from utils.head_centralization import centralized_head_in_sequence

# Set of all parameters you need to change with the correct path for testing the code
video_path = 'C1_C.mp4'     # video you want to obtain the prediction
pe_model_path = 'm64_64_1.0/221226_195308.single_instance'  # pose estimation model trained with sleap
class_model = 'best3classes.keras'  # model for sequence classification
num_classes = 3     # number of classes
# If you are using the uncropped videos or video with fps different from 29, then set need_cropping_and_fps_reduction
# to True and modify the y_cropping value accordingly
need_cropping_and_fps_reduction = False
y_cropping = 290

# If the video_path does not refer to a mp4 file raise an error.
if not video_path.endswith('.mp4'):
    raise Exception('The video should be in mp4 format')

# First thing to do is to standardize the video to 29 fps. Since this operation its already performed to our cropped
# videos then you can skip it when cropping is not needed. Be aware that if you are using new videos, even with shape
# (1080, 1080)*, thus cropping is not, you still need to standardize the video to 29fps and reduce it to a length equal
# to 3480
# * In this case you can set y_cropping to zero.
if need_cropping_and_fps_reduction:
    mm = MovieManager(29)
    mm.reduce_single_movie(video_path, 'tmp_video_29frames.mp4')

    VideoCropping.cropping('tmp_video_29frames.mp4',
                           "tmp_video.mp4",
                           y_cropping=290,
                           starting_frame=1740,
                           last_frame=5220)

# Obtain the slp prediction of the video using the pose estimation model
get_predictions(
    "tmp_video.mp4" if need_cropping_and_fps_reduction else video_path,
    pe_model_path,
    'tmp.pkg.slp',
    False
)

# After using a huge model is better to clean the tensorflow-keras session with the following function
ks.backend.clear_session()

# After that we need to convert the slp predictions to numpy with the following function
obtain_np_seq_from_slp_pred(
    'tmp.pkg.slp',
    'tmp.npy'
)

# Since the obtained numpy sequence might have np.nans in it, we need to perform a missing value analysis
# and fill those np.nans interpolating their values
pred = np.load('tmp.npy').T
for i, row in enumerate(pred):
    mvh = MissingValuesHandler(row)
    missing = mvh.find_missing_values()
    print(f'\t-Missing values for entry {i}: {len(missing)}')
    if len(missing) == 0:
        print(f'\tNo missing values for entry {i}, filling operation skipped')
    else:
        pred[i] = mvh.fill_missing_values_weighted()
np.save('tmp_filled.npy', pred)

# Obtained the filled sequence we can use it to create a new sequence were the head is the center of
# our Cartesian plane, removing the head x, y position from the sequence since they do not contain any information now
centralized_head_in_sequence('tmp_filled.npy', 'centralized_tmp_filled.npy')

# Now we can load the newly created sequence and predict using our classification model
# The result of this phase will produce in output a line containing a number indicating
# the final prediction result. In the case of 3 classes the score is related to:
#       0: control
#       1: sugar
#       2: ammonia
# For the two classes analysis the name of the model indicates how the classes are arranged. E.g.,
# sugar_ammonia.keras indicates that class 0 is sugar and class 1 is ammonia.
sequence = np.load('centralized_tmp_filled.npy')
model = ks.models.load_model(class_model)
pred = model.predict(sequence)

if num_classes == 2:
    pred = [1 if p > 0.5 else 0 for p in pred]
else:
    pred = np.argmax(pred, axis=-1)
print(f'Final prediction: {pred}')
