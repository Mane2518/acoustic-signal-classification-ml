import torch
import argparse
from src.preprocessing import load_audio, get_mel_spectrogram
from src.model import AudioClassifier

def predict(audio_path, model_path, num_classes=2):
    """
    Load a trained model and predict the class for a single audio file.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and weights
    model = AudioClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Preprocess the audio input
    audio, sr = load_audio(audio_path)
    mel_spec = get_mel_spectrogram(audio, sr).unsqueeze(0).to(device)
    
    # Fix temporal dimension (must match training dimensions, e.g., 128)
    if mel_spec.shape[3] > 128:
        mel_spec = mel_spec[:, :, :, :128]
    else:
        padding = 128 - mel_spec.shape[3]
        mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
    
    # Forward pass
    with torch.no_grad():
        output = model(mel_spec)
        prediction = torch.argmax(output, dim=1)
    
    print(f"Prediction for {audio_path}: Class {prediction.item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict audio signal class.")
    parser.add_argument("--path", type=str, required=True, help="Path to the audio file.")
    parser.add_argument("--model", type=str, default="model.pth", help="Path to the model file.")
    args = parser.parse_args()
    
    predict(args.path, args.model)
