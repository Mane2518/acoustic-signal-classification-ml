import torch
from torch.utils.data import Dataset
from .preprocessing import load_audio, get_mel_spectrogram

class AcousticDataset(Dataset):
    """
    A custom PyTorch Dataset for loading audio files, extracting
    Mel-spectrograms, and preparing them for a CNN classifier.
    """
    def __init__(self, file_paths, labels, target_sr=22050, fixed_length=128):
        self.file_paths = file_paths
        self.labels = labels
        self.target_sr = target_sr
        self.fixed_length = fixed_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load audio data
        audio, _ = load_audio(self.file_paths[idx], self.target_sr)
        
        # Transform waveform to mel-spectrogram
        mel_spec = get_mel_spectrogram(audio, self.target_sr)
        
        # Pad or truncate temporal dimension to a fixed size
        if mel_spec.shape[2] > self.fixed_length:
            mel_spec = mel_spec[:, :, :self.fixed_length]
        else:
            padding = self.fixed_length - mel_spec.shape[2]
            mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            
        label = self.labels[idx]
        return mel_spec, torch.tensor(label, dtype=torch.long)
