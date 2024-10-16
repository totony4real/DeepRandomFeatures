import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def create_datasets_and_loaders(spatial_X_train, temporal_X_train, y_train, batch_size):
    dataset = TensorDataset(spatial_X_train, temporal_X_train, y_train)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def load_data():
    pass
