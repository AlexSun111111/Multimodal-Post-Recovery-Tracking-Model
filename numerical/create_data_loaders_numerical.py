from torch.utils.data import DataLoader, TensorDataset, random_split
import torch

def create_numerical_data_loaders(numerical_tensor, binary_tensor, multiclass_tensor, batch_size=16):
    # Create dataset
    dataset = TensorDataset(numerical_tensor, binary_tensor, multiclass_tensor)

    # Split the dataset into training and validation sets (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    return train_loader, val_loader
