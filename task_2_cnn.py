import os
# Fix for OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from utils import RobotDataset

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),  # Output: 64x64
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(), # Output: 32x32
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), # Output: 16x16
            nn.Flatten()
        )
        # Fully connected layers (Feature map flat size + 4 action classes)
        self.fc = nn.Sequential(
            nn.Linear(64 * 16 * 16 + 4, 256), 
            nn.ReLU(),
            nn.Linear(256, 2) # Output: x, y coordinates
        )

    def forward(self, img, action):
        x = self.features(img)
        # One-hot encode action and concatenate with image features
        action_one_hot = torch.nn.functional.one_hot(action, num_classes=4).float()
        x = torch.cat([x, action_one_hot], dim=1)
        return self.fc(x)

def train(model, loader, device, epochs=100):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    loss_history = []

    print("Starting CNN Training...")
    for epoch in range(epochs):
        total_loss = 0
        for img, action, target_pos, _ in loader:
            img = img.to(device)
            action = action.to(device)
            target_pos = target_pos.to(device)
            
            optimizer.zero_grad()
            pred = model(img, action)
            loss = criterion(pred, target_pos)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.5f}")
    
    # Save training loss plot
    plt.figure()
    plt.plot(loss_history)
    plt.yscale('log')
    plt.title("CNN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.savefig("loss_cnn.png")
    print("Loss plot saved to loss_cnn.png.")
    
    return model

def test(model, loader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    
    with torch.no_grad():
        for img, action, target_pos, _ in loader:
            img = img.to(device)
            action = action.to(device)
            target_pos = target_pos.to(device)
            
            pred = model(img, action)
            loss = criterion(pred, target_pos)
            total_loss += loss.item()
    
    avg_test_error = total_loss / len(loader)
    print(f"\n>>> CNN Test Error (MSE): {avg_test_error:.6f}")
    return avg_test_error

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and split dataset
    dataset = RobotDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Initialize, train and save model
    model = CNN().to(device)
    train(model, train_loader, device)
    torch.save(model.state_dict(), "cnn_model.pth")
    
    # Evaluate model
    test(model, test_loader, device)