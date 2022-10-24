# MAKE DIRECTORIES FOR VIDEOS
mkdir datasets
mkdir datasets/train_val_set/ datasets/test_set/
mkdir datasets/cropped/train_val_set/ datasets/cropped/test_set/


# FPS REDUCTION
python3 moviemanager.py --dirs=['train_val_set', 'test_set'] --fps=2


# FIRST PHASE: 4k FRAMES
python3 training.py --max_stride=32 --filters=32 --input_scaling=0.5
python3 training.py --max_stride=32 --filters=32 --input_scaling=0.6
python3 training.py --max_stride=32 --filters=32 --input_scaling=0.7

python3 training.py --max_stride=32 --filters=64 --input_scaling=0.5
python3 training.py --max_stride=32 --filters=64 --input_scaling=0.6
python3 training.py --max_stride=32 --filters=64 --input_scaling=0.7

python3 training.py --max_stride=64 --filters=32 --input_scaling=0.5
python3 training.py --max_stride=64 --filters=32 --input_scaling=0.6
python3 training.py --max_stride=64 --filters=32 --input_scaling=0.7

python3 training.py --max_stride=64 --filters=64 --input_scaling=0.5
python3 training.py --max_stride=64 --filters=64 --input_scaling=0.6
python3 training.py --max_stride=64 --filters=64 --input_scaling=0.7


# MAKE DIRECTORIES FOR HEAD PREDICTIONS
mkdir head_predictions
mkdir head_predictions/slp/
mkdir head_predictions/slp/train_val_set/
mkdir head_predictions/slp/test_set/
mkdir head_predictions/npy/
mkdir head_predictions/npy/train_val_set/
mkdir head_predictions/npy/test_set/


# PREDICT HEAD EVERYWHERE
python3 predictionhandling.npy --video_path="C1_C.mp4" --model_path="models/220626_100050.single_instance" --pred_output_path="head_predictions/slp/train_val_set/C1_C_pred.pkg.slp" --with_images=False seq_output_path="head_predictions/npy/train_val_set/C1_C_pred.npy"

python3 predictionhandling.npy --video_path="C1_C.mp4" --model_path="models/220626_100050.single_instance" --pred_output_path="head_predictions/slp/test_set/C1_C_pred.pkg.slp" --with_images=False seq_output_path="head_predictions/npy/test_set/C1_C_pred.npy"


# MAKE DIRECTORIES FOR HEAD PREDICTIONS WITHOUT MISSING VALUES
mkdir head_predictions/npy_nomv/
mkdir head_predictions/npy_nomv/train_val_set/
mkdir head_predictions/npy_nomv/test_set/


# FILL MISSING VALUES HEAD (NO WEIGHTED)
python3 missingvalues.py --pred_path="head_predictions/npy/train_val_set/C1_C_pred.npy" --pred_output="head_predictions/npy_nomv/train_val_set/C1_pred_filled.npy"


# CROPPING
python3 editing.py --input_video="train_val_set/C1_C.mp4" --output_video="datasets/cropped/train_val_set/C1_C.mp4" --head_pred_path="head_predictions/npy_nomv/train_val_set/C1_pred_filled.npy"


# SECOND PHASE: 800x800 FRAMES
python3 training.py --filters_rate_values=2.0 --max_stride=32 --filters=32 --input_scaling=0.5
python3 training.py --filters_rate_values=2.0 --max_stride=32 --filters=32 --input_scaling=0.6
python3 training.py --filters_rate_values=2.0 --max_stride=32 --filters=32 --input_scaling=0.7

python3 training.py --filters_rate_values=2.0 --max_stride=32 --filters=64 --input_scaling=0.5
python3 training.py --filters_rate_values=2.0 --max_stride=32 --filters=64 --input_scaling=0.6
python3 training.py --filters_rate_values=2.0 --max_stride=32 --filters=64 --input_scaling=0.7

python3 training.py --filters_rate_values=2.0 --max_stride=64 --filters=32 --input_scaling=0.5
python3 training.py --filters_rate_values=2.0 --max_stride=64 --filters=32 --input_scaling=0.6
python3 training.py --filters_rate_values=2.0 --max_stride=64 --filters=32 --input_scaling=0.7

python3 training.py --filters_rate_values=2.0 --max_stride=64 --filters=64 --input_scaling=0.5
python3 training.py --filters_rate_values=2.0 --max_stride=64 --filters=64 --input_scaling=0.6
python3 training.py --filters_rate_values=2.0 --max_stride=64 --filters=64 --input_scaling=0.7

python3 training.py --filters_rate_values=3.0 --max_stride=32 --filters=32 --input_scaling=0.5
python3 training.py --filters_rate_values=3.0 --max_stride=32 --filters=32 --input_scaling=0.6
python3 training.py --filters_rate_values=3.0 --max_stride=32 --filters=32 --input_scaling=0.7

python3 training.py --filters_rate_values=3.0 --max_stride=32 --filters=64 --input_scaling=0.5
python3 training.py --filters_rate_values=3.0 --max_stride=32 --filters=64 --input_scaling=0.6
python3 training.py --filters_rate_values=3.0 --max_stride=32 --filters=64 --input_scaling=0.7

python3 training.py --filters_rate_values=3.0 --max_stride=64 --filters=32 --input_scaling=0.5
python3 training.py --filters_rate_values=3.0 --max_stride=64 --filters=32 --input_scaling=0.6
python3 training.py --filters_rate_values=3.0 --max_stride=64 --filters=32 --input_scaling=0.7

python3 training.py --filters_rate_values=3.0 --max_stride=64 --filters=64 --input_scaling=0.5
python3 training.py --filters_rate_values=3.0 --max_stride=64 --filters=64 --input_scaling=0.6
python3 training.py --filters_rate_values=3.0 --max_stride=64 --filters=64 --input_scaling=0.7


# MAKE DIRECTORIES FOR PREDICTIONS
mkdir predictions
mkdir predictions/slp/
mkdir predictions/slp/train_val_set/
mkdir predictions/slp/test_set/
mkdir predictions/npy/
mkdir predictions/npy/train_val_set/
mkdir predictions/npy/test_set/

mkdir predictions/npy_nomv/
mkdir predictions/npy_nomv/train_val_set/
mkdir predictions/npy_nomv/test_set/

mkdir predictions/npy_nomv_weighted/
mkdir predictions/npy_nomv_weighted/train_val_set/
mkdir predictions/npy_nomv_weighted/test_set/


# PREDICT ALL
python3 predictionhandling.npy --video_path="C1_C.mp4" --model_path="models/220626_100050.single_instance" --pred_output_path="predictions/slp/train_val_set/C1_C_pred.pkg.slp" --with_images=False seq_output_path="predictions/npy/train_val_set/C1_C_pred.npy" --all_kp=True --only_head==False


# RESOLVE MISSING VALUES
python3 missingvalues.py --pred_path="head_predictions/npy/train_val_set/C1_C_pred.npy" --pred_output="head_predictions/npy_nomv/train_val_set/C1_pred_filled.npy" --head=False