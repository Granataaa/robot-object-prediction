import torch
from torch.utils.data import Dataset
import glob
import os

class RobotDataset(Dataset):
    def __init__(self, root_dir="data/"):
        # Locate data files
        pos_files = sorted(glob.glob(os.path.join(root_dir, "positions_*.pt")))
        act_files = sorted(glob.glob(os.path.join(root_dir, "actions_*.pt")))
        img_files = sorted(glob.glob(os.path.join(root_dir, "imgs_*.pt")))
        next_img_files = sorted(glob.glob(os.path.join(root_dir, "imgs_next_*.pt")))

        if len(pos_files) == 0:
            raise RuntimeError("No data found.")

        self.positions = []
        self.actions = []
        self.imgs = []
        self.next_imgs = []

        print(f"Loading data from {len(pos_files)} batches...")
        for i in range(len(pos_files)):
            self.positions.append(torch.load(pos_files[i]))
            self.actions.append(torch.load(act_files[i]))
            self.imgs.append(torch.load(img_files[i]))
            
            if i < len(next_img_files):
                self.next_imgs.append(torch.load(next_img_files[i]))

        # Concatenate lists into tensors
        self.positions = torch.cat(self.positions, dim=0)
        self.actions = torch.cat(self.actions, dim=0)
        
        # Normalize images to [0, 1] range
        self.imgs = torch.cat(self.imgs, dim=0).float() / 255.0
        
        if len(self.next_imgs) > 0:
            self.next_imgs = torch.cat(self.next_imgs, dim=0).float() / 255.0

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        # Returns: Initial Image, Action, Object Position, Next Image
        next_img = self.next_imgs[idx] if len(self.next_imgs) > 0 else torch.zeros(1)
        return self.imgs[idx], self.actions[idx], self.positions[idx], next_img