import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import os

from src.model import get_resnet18
from src.data import get_dataloaders

EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3
NUM_WORKERS = 4

os.makedirs("./checkpoints", exist_ok=True)
os.makedirs("./results", exist_ok=True)

def accuracy(logits, targets):
    return (logits.argmax(1) == targets).float().mean().item()

def train():
    """
    Train the ResNet-18 model on the CIFAR-100 dataset.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = get_dataloaders(BATCH_SIZE, NUM_WORKERS)
    model = get_resnet18(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, running_acc, n = 0.0, 0.0, 0
        for x, y, sid in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward() 
            optimizer.step()

            batch_size = y.size(0)
            running_loss += loss.item() * batch_size
            running_acc  += accuracy(out, y) * batch_size
            n += batch_size

        model.eval()
        val_loss, val_acc, m = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y, _sid in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                batch_size = y.size(0)
                val_loss += loss.item() * batch_size
                val_acc  += accuracy(out, y) * batch_size
                m += batch_size

        print(f"Epoch {epoch}/{EPOCHS}, Loss: {running_loss/n:.4f}, Acc: {running_acc/n:.4f}, Val Loss: {val_loss/m:.4f}, Val Acc: {val_acc/m:.4f}")
        if epoch in (10, 20, 30):
            torch.save(model.state_dict(), f"checkpoints/resnet18_epoch{epoch}.pth")
    return model

if __name__ == "__main__":
    train()