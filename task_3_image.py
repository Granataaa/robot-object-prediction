import os
# Fix for OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torchvision
from utils import RobotDataset

# --- U-NET ARCHITECTURE ---
# Standard U-Net with Skip Connections.
# This architecture was chosen because it preserves high-frequency details 
# (like the robot arm and walls) from the input image via skip connections,
# while learning to move the dynamic object in the bottleneck/decoder.
class ImagePredictor(nn.Module):
    def __init__(self):
        super(ImagePredictor, self).__init__()
        
        # --- ENCODER (Downsampling) ---
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # Input: 128x128
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # -> 64x64
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # -> 16x16
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # --- BOTTLENECK + ACTION INJECTION ---
        # We concatenate the action vector to the latent representation
        # Input channels: 128 (image features) + 4 (one-hot action) = 132
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(132, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # --- DECODER (Upsampling + Skip Connections) ---
        
        # Up 1: 16x16 -> 32x32
        self.up1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        # Skip connection adds features from enc3 (64 channels) -> Total 128
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Up 2: 32x32 -> 64x64
        self.up2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        # Skip connection from enc2 -> Total 64
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Up 3: 64x64 -> 128x128
        self.up3 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        # Skip connection from enc1 -> Total 32
        self.dec3 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # Final Output Layer
        self.final = nn.Conv2d(16, 3, 3, padding=1)
        self.sigmoid = nn.Sigmoid() # Ensures output is in [0, 1] range

    def forward(self, img, action):
        # Encoder Pass
        x1 = self.enc1(img)       # 128x128 (Skip 1)
        x2 = self.enc2(x1)        # 64x64   (Skip 2)
        x3 = self.enc3(x2)        # 32x32   (Skip 3)
        x4 = self.enc4(x3)        # 16x16
        
        # Action Injection
        B, _, H, W = x4.shape
        action_one_hot = torch.nn.functional.one_hot(action, num_classes=4).float()
        action_map = action_one_hot.view(B, 4, 1, 1).expand(B, 4, H, W)
        
        x = torch.cat([x4, action_map], dim=1) 
        x = self.bottleneck_conv(x) 
        
        # Decoder Pass with Skip Connections
        d1 = self.up1(x)
        d1 = torch.cat([d1, x3], dim=1) # Join Skip 3
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, x2], dim=1) # Join Skip 2
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat([d3, x1], dim=1) # Join Skip 1
        d3 = self.dec3(d3)
        
        out = self.final(d3)
        return self.sigmoid(out)

# --- TRAIN ---
def train(model, loader, device, epochs=50):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # We use MSELoss as it is the standard metric for reconstruction.
    # Note: With small datasets (1000 samples), MSE tends to produce blurry results
    # (average of possible futures) rather than sharp predictions, but it is 
    # stable and reliable for learning the physics dynamics.
    criterion = nn.MSELoss() 
    loss_history = []

    print("Starting U-Net Training...")
    for epoch in range(epochs):
        total_loss = 0
        for img, action, _, target_img in loader:
            img = img.to(device)
            action = action.to(device)
            target_img = target_img.to(device)
            
            optimizer.zero_grad()
            pred = model(img, action)
            loss = criterion(pred, target_img)
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
    plt.title("Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.savefig("loss_recon.png")
    print("Loss plot saved to loss_recon.png.")

# --- TEST ---
def test(model, loader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    sample_imgs, sample_preds, sample_targets = None, None, None

    with torch.no_grad():
        for img, action, _, target_img in loader:
            img = img.to(device)
            action = action.to(device)
            target_img = target_img.to(device)
            
            pred = model(img, action)
            # Clamp ensures values stay strictly within valid image range
            pred = torch.clamp(pred, 0.0, 1.0)
            
            total_loss += criterion(pred, target_img).item()
            
            # Save first batch for visualization
            if sample_imgs is None:
                sample_imgs, sample_preds, sample_targets = img[:8], pred[:8], target_img[:8]
    
    print(f"\n>>> Reconstruction Test Error (MSE): {total_loss / len(loader):.6f}")
    
    # Save comparison image: Top row = Real, Bottom row = Generated
    comparison = torch.cat([sample_targets, sample_preds], dim=0)
    torchvision.utils.save_image(comparison, "reconstruction_results.png", nrow=8)
    print("Comparison image saved to reconstruction_results.png.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and split dataset
    dataset = RobotDataset()
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Initialize, train and save model
    model = ImagePredictor().to(device)
    train(model, train_loader, device)
    torch.save(model.state_dict(), "recon_model.pth")
    
    # Evaluate model
    test(model, test_loader, device)