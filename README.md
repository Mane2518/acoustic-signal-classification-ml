# Acoustic Signal Classification ML

## Project Overview
This project implements a professional deep learning pipeline for classifying acoustic signals using PyTorch and signal processing libraries. It leverages Mel-spectrograms to convert raw audio into image-like representations, allowing Convolutional Neural Networks (CNNs) to extract meaningful features for classification.

## Signal Processing Concepts
### Fast Fourier Transform (FFT)
FFT converts a signal from its original time domain to a frequency domain representation. In this project, we use it to analyze spectral characteristics of audio signals.
### Mel-Spectrograms
The Mel-spectrogram is a spectrogram where frequencies are converted to the Mel scale, which more closely mimics human auditory perception. This transformation captures signal information in a way that is highly effective for machine learning models.

## Structure
- `src/preprocessing.py`: Signal processing utilities for audio manipulation.
- `src/model.py`: PyTorch-based CNN model for classification.
- `src/dataset.py`: Custom PyTorch Dataset for loading audio files and labels.
- `src/train.py`: Training script with evaluation and metric tracking.
- `src/predict.py`: Script for performing inference on new audio files.
- `notebooks/exploration.ipynb`: Jupyter notebook for data visualization and exploration.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
### Training
```bash
python src/train.py --data_dir ./data --epochs 10
```
### Inference
```bash
python src/predict.py --path sample_audio.wav --model model_weights.pth
```
