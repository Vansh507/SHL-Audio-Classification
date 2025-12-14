## Grammar Scoring Engine for Spoken Data
-This project implements a machine learning–based **Grammar Scoring Engine** that evaluates spoken English audio samples and predicts a continuous grammar score (0–5) based on a MOS Likert scale.

## Problem Statement
-Given 45–60 second speech recordings, the task is to predict a grammar proficiency score using acoustic features extracted from audio data.

## 🗂 Dataset Structure

The dataset (not included due to size and licensing constraints) is expected in the following format:

'''
dataset/
├── audios/
│ ├── train/
│ │ ├── audio_1.wav
│ │ ├── audio_2.wav
│ │ └── ...
│ └── test/
│ ├── audio_101.wav
│ ├── audio_102.wav
│ └── ...
└── csvs/
├── train.csv
└── test.csv
'''

## Approach
- Extracted **MFCC features** from audio using Librosa
- Removed corrupted or extremely short audio files
- Trained a **Random Forest Regressor**
- Evaluated model using **RMSE**
- Generated predictions for unseen test data

## Model Performance
- **Training RMSE:** 0.6963

## Tech Stack
- Python
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## How to Run
1. Place the dataset in the required structure
2. Open and run:
3. Generate `submission.csv`

## Future Improvements
- Add delta & delta-delta MFCCs
- Use deep learning (CNN on spectrograms)
- Incorporate speech rate and pause features

Author: Vansh Chaudhary
Platform: Kaggle Competition Project 
