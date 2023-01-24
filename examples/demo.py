import numpy as np
from tensorflow import keras as ks

from posestimation.editing import VideoCropping
from posestimation.predictionhandling import get_predictions, obtain_np_seq_from_slp_pred
from posestimation.moviemanager import MovieManager
from utils.missingvalues import MissingValuesHandler
from utils.head_centralization import centralized_head_in_sequence

video_path = 'YOUR-VIDEO.mp4' # TODO: test
pe_model_path = 'm64_64_1.0/221226_195308.single_instance'
class_model = 'model.keras'
num_classes = 2

if not video_path.endswith('.mp4'):
    raise Exception('The video should be in mp4 format')

mm = MovieManager(29)
mm.reduce_single_movie(video_path, 'tmp_video_29frames.mp4')

VideoCropping.cropping('tmp_video_29frames.mp4',
                       "tmp_video.mp4",
                       y_cropping=290,
                       starting_frame=1740,
                       last_frame=5220)

get_predictions(
    "tmp_video.mp4",
    pe_model_path,
    'tmp.pkg.slp',
    False
)

obtain_np_seq_from_slp_pred(
    'tmp.pkg.slp',
    'tmp.npy'
)

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

centralized_head_in_sequence('tmp_filled.npy', 'centralized_tmp_filled.npy')

sequence = np.load('centralized_tmp_filled.npy')
model = ks.models.load_model(class_model)
pred = model.predict(sequence)

if num_classes == 2:
    pred = [1 if p > 0.5 else 0 for p in pred]
    print(f'Final prediction: {pred}')
else:
    pass # TODO: softmax
