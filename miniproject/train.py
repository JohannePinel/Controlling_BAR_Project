import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os, subprocess
import glob
import argparse

def get_device():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    nvidia_index = result.stdout.strip().split("\n")[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = nvidia_index
    print(f"Forcing NVIDIA GPU (index {nvidia_index})")
    print(torch.__version__)
    print("CUDA dispo :", torch.cuda.is_available())
    print("GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "aucun")
    print("VRAM :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "Go")
    print(f"PyTorch {torch.__version__} | torchvision {torchvision.__version__}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device
        
class OmmatidiaDataset(Dataset):
    def __init__(self, folder_path):
        all_inputs = []
        all_gains  = []
        all_modes  = []

        for npz_path in glob.glob(os.path.join(folder_path, "*.npz")):
            data = np.load(npz_path, allow_pickle=False)
            all_inputs.append(data["inputs"])
            all_gains.append(data["gains"])
            all_modes.append(data["modes"])
            print(f"Chargé : {npz_path} ({len(data['inputs'])} samples)")

        if not all_inputs:
            raise FileNotFoundError(f"No .npz files found in folder: {folder_path}")

        self.inputs = torch.tensor(np.concatenate(all_inputs), dtype=torch.float32)
        self.gains  = torch.tensor(np.concatenate(all_gains),  dtype=torch.float32)
        self.modes  = np.concatenate(all_modes)
        self.danger = torch.tensor((self.modes == "manual").astype(np.float32))

        print(f"Total : {len(self.inputs)} samples")


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.gains[idx], self.danger[idx]


# ── Modèle ─────────────────────────────────────────────────────────────────────

class ObstacleNet(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = 5 * 2 * 721 * 2  # 14420

        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 64),        nn.ReLU(),
            nn.Linear(64, 32),         nn.ReLU(),
        )
        self.danger_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.gains_head  = nn.Linear(32, 2)

    def forward(self, x):
        features = self.backbone(x)
        danger   = self.danger_head(features).squeeze(-1)  # (B,)
        gains    = self.gains_head(features)               # (B, 2)
        return danger, gains


# ── Entraînement ───────────────────────────────────────────────────────────────

def train(folder_path, epochs=20, batch_size=32, lr=1e-3):
    device = get_device()  
    
    dataset    = OmmatidiaDataset(folder_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model     = ObstacleNet().to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce_loss  = nn.BCELoss()
    mse_loss  = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss        = 0
        total_danger_loss = 0
        total_gains_loss  = 0

        for inputs, gains, danger in dataloader:
            inputs = inputs.to(device)  
            gains  = gains.to(device)   
            danger = danger.to(device)  
            optimizer.zero_grad()
            pred_danger, pred_gains = model(inputs)

            # Tâche 1 — classifier danger (tous les samples)
            loss_danger = bce_loss(pred_danger, danger)

            # Tâche 2 — régresseur gains (manual uniquement)
            manual_mask = danger == 1
            if manual_mask.any():
                loss_gains = mse_loss(pred_gains[manual_mask], gains[manual_mask])
            else:
                loss_gains = torch.tensor(0.0, device=device)

            loss = loss_danger + loss_gains
            loss.backward()
            optimizer.step()

            total_loss        += loss.item()
            total_danger_loss += loss_danger.item()
            total_gains_loss  += loss_gains.item()

        n = len(dataloader)
        print(f"Epoch {epoch+1:02d} | "
              f"total_loss={total_loss/n:.4f} | "
              f"total_danger_loss={total_danger_loss/n:.4f} | "
              f"total_gains_loss={total_gains_loss/n:.4f}")

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train ObstacleNet on collected .npz datasets.")
    parser.add_argument(
        "--folder", 
        type=str, 
        default="miniproject/training_datas",
        help="Path to the folder containing .npz files (default: miniproject/training_datas)"
    )
    args = parser.parse_args()

    model = train(args.folder)
    torch.save(model.state_dict(), "obstacle_net.pth")
    print("Modèle sauvegardé : obstacle_net.pth")