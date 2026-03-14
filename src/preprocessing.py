import librosa
import numpy as np
import torch
import torchaudio.transforms as T

def load_audio(file_path, target_sr=22050):
    """
    Load an audio file and resample it to a target sample rate.
    """
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio, sr

def get_mel_spectrogram(audio, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    """
    Convert an audio waveform into a Mel-spectrogram and normalize the intensity in dB.
    """
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
        
    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_spec = mel_transform(audio)
    
    # Use AmplitudeToDB for log-scale mel-spectrogram
    mel_spec_db = T.AmplitudeToDB()(mel_spec)
    return mel_spec_db

def get_fft(audio):
    """
    Compute the Fast Fourier Transform of an audio signal.
    """
    return np.abs(np.fft.rfft(audio))
