# Crickets
 
This repository contains the code used in the **** paper.
The dataset can be downloaded from here:
The most performing models can be downloaded in TABLE

## Test our models
After downloaded the dataset and the models, you can test our code running test/pipeline.py

## Workflow followed
The details of the command used during our experiments are described in the sh 
files inside test directories, but to better understand what was done a list of tasks 
is here highlighted together with the file codes used in each step:
- POSE ESTIMATION TASKS:
  - Download the dataset
  - Divide the dataset into two directories: train_val_set, test_set
  - Reduce the frames per second of each video to 2 [posestimation/moviemanager.py]
  - Label using the SLEAP GUI (using the reduced video)
  - Train for detecting the head [posestimation/training.py]
  - Predict head for each video frame (using whole videos) [posestimation/predictionhandling.py]
  - Fill missing values [posestimation/missingvalues.py]
  - Crop videos in order to consider only a box 800x800 for each frame [posestimation/editing.py]
  - Reduce the frames per second of each cropped-video to 2 [posestimation/moviemanager.py]
  - Train for detecting all keypoints (head, beginning and tip of the antennae) [posestimation/training.py]
  - Predict all [posestimation/predictionhandling.py]
  - Fill missing values [posestimation/missingvalues.py]
- CLASSIFICATION TASKS:
  - Something