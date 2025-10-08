from datasets import load_dataset
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

cifar_original = CIFAR100(root="./data/torchvision", train=True, download=True)
CLASSES = cifar_original.classes
NAME2ID = cifar_original.class_to_idx

TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3) # Normalize from [0,1] → [-1,1] 
])

def load_data():
    """
    Load the CIFAR-100 dataset with noisy labels.
    """
    ds = load_dataset("hirundo-io/Noisy-CIFAR-100")
    ds_labeled = ds.map(add_labels)
    return ds_labeled
    
def add_labels(row):
    """
    Add labels column to the CIFAR-100 dataset.
    """
    split, noisy_label_name, basename = row["__key__"].split("/")
    row["sample_id"] = int(basename)
    row["noisy_label_name"] = noisy_label_name
    row["noisy_label_id"] = NAME2ID[noisy_label_name]   
    return row


def get_dataloaders(batch_size=128, num_workers=4):
    ds = load_data()
    train_ds = ds["train"]
    val_ds   = ds["test"]

    def collate(batch):
        x   = torch.stack([TF(b["png"]) for b in batch])                     # (B,3,32,32)
        y   = torch.tensor([b["noisy_label_id"] for b in batch]).long()      # (B,)
        sid = torch.tensor([b["sample_id"] for b in batch]).long()           # (B,)
        return x, y, sid

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, collate_fn=collate)
    return train_loader, val_loader