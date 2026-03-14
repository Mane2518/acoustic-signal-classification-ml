import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import AcousticDataset
from src.model import AudioClassifier

def train_loop(dataloader, model, loss_fn, optimizer, device):
    """
    Standard PyTorch training loop across one epoch.
    """
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 10 == 0:
            print(f"Batch {batch}, loss: {loss.item():>7f}")

def main():
    # Example hyperparameters
    batch_size = 8
    epochs = 10
    learning_rate = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Mock data initialization
    # In practice, these would be paths to wav files and their labels
    file_paths = ["data/sample1.wav", "data/sample2.wav"]
    labels = [0, 1]
    
    dataset = AcousticDataset(file_paths, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = AudioClassifier(num_classes=2).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(dataloader, model, loss_fn, optimizer, device)
        
    torch.save(model.state_dict(), "model.pth")
    print("Training Complete! Saved model weights to model.pth")

if __name__ == "__main__":
    main()
