"""
ddp_launch.py: Enhanced Distributed Data Parallel (DDP) Training Script

This script demonstrates advanced distributed training using PyTorch's DistributedDataParallel.
It's designed to utilize the best available hardware acceleration (CUDA, MPS, or CPU) while
maintaining compatibility with DDP. The script includes improved logging, model validation,
learning rate scheduling, and model saving functionality.

Key Features:
- Adaptive device selection (CUDA, MPS, or CPU)
- Distributed training across multiple processes
- Synthetic dataset generation for demonstration
- Train/validation split
- Learning rate scheduling
- Model checkpointing
- Detailed logging of training progress
- Final model parameter comparison across processes

Usage:
    python ddp_launch.py

Note: This script uses a simple linear model and synthetic data for demonstration.
      For real-world applications, replace these with your specific model and dataset.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR

class SimpleModel(nn.Module):
    """
    A simple linear model for demonstration purposes.
    
    This model consists of a single fully connected layer that maps
    10 input features to 1 output, essentially performing linear regression.
    """
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        """Forward pass of the model."""
        return self.fc(x)

def setup(rank, world_size):
    """
    Initialize the distributed environment.
    
    Args:
        rank (int): Unique identifier of the process
        world_size (int): Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def get_device(rank):
    """
    Determine the best available device, with fallback options.
    
    Args:
        rank (int): Process rank for GPU selection if available
    
    Returns:
        torch.device: The selected device
        bool: Whether DDP should use this device directly
    """
    if torch.cuda.is_available():
        return torch.device(f'cuda:{rank}'), True
    elif torch.backends.mps.is_available():
        return torch.device('mps'), False  # Use MPS but not for DDP
    else:
        return torch.device('cpu'), True

def create_data_loader(dataset, batch_size, rank, world_size, is_train=True):
    """
    Create a data loader for the given dataset.
    
    Args:
        dataset (Dataset): The dataset to load
        batch_size (int): Number of samples per batch
        rank (int): Process rank
        world_size (int): Total number of processes
        is_train (bool): Whether this is for training (enables shuffling)
    
    Returns:
        DataLoader: The created DataLoader object
    """
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=is_train)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

def validate(model, val_loader, loss_fn, device, use_ddp_device):
    """
    Validate the model on the validation set.
    
    Args:
        model (nn.Module): The model to validate
        val_loader (DataLoader): Validation data loader
        loss_fn (nn.Module): Loss function
        device (torch.device): Device to run validation on
        use_ddp_device (bool): Whether to use the device directly for DDP
    
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            if use_ddp_device:
                outputs = model(data)
            else:
                outputs = model(data.to('cpu')).to(device)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def save_model(model, epoch, rank):
    """
    Save the model checkpoint.
    
    Args:
        model (nn.Module): The model to save
        epoch (int): Current epoch number
        rank (int): Process rank
    """
    if rank == 0:  # Only save on the main process
        torch.save(model.state_dict(), f"pth/model_checkpoint_epoch_{epoch}.pth")

def train(rank, world_size):
    """
    Training function to be run in parallel across multiple processes.
    
    This function handles the entire training process including:
    - Setting up the distributed environment
    - Creating the model and moving it to the appropriate device
    - Preparing the data loaders
    - Training loop with validation
    - Learning rate scheduling
    - Model saving
    
    Args:
        rank (int): Unique identifier of the process
        world_size (int): Total number of processes
    """
    setup(rank, world_size)
    
    device, use_ddp_device = get_device(rank)
    print(f"Process {rank} using device: {device}, DDP on device: {use_ddp_device}")

    # Create synthetic datasets for demonstration
    full_dataset = TensorDataset(torch.randn(10000, 10), torch.randn(10000, 1))
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = create_data_loader(train_dataset, batch_size=32, rank=rank, world_size=world_size)
    val_loader = create_data_loader(val_dataset, batch_size=32, rank=rank, world_size=world_size, is_train=False)

    model = SimpleModel().to(device)
    if use_ddp_device:
        ddp_model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    else:
        ddp_model = DDP(model.to('cpu'), device_ids=None)

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    loss_fn = nn.MSELoss()

    num_epochs = 20
    for epoch in range(num_epochs):
        ddp_model.train()
        total_loss = 0.0
        for batch, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            if use_ddp_device:
                outputs = ddp_model(data)
            else:
                outputs = ddp_model(data.to('cpu')).to(device)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch % 100 == 0 and rank == 0:
                print(f"Epoch {epoch}, Batch {batch}, Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        val_loss = validate(ddp_model, val_loader, loss_fn, device, use_ddp_device)

        if rank == 0:
            print(f"Epoch {epoch}, Avg Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        scheduler.step()
        save_model(ddp_model, epoch, rank)

        # Synchronize processes to ensure all are at the same point before proceeding
        dist.barrier()

    # Final model comparison to verify consistency across processes
    if rank == 0:
        print("Final model parameters:")
        for name, param in ddp_model.named_parameters():
            print(f"{name}: {param.data.mean().item():.4f}")

    cleanup()

def main():
    """
    Main function to set up and run the distributed training.
    
    This function determines the number of processes to spawn and
    initiates the distributed training.
    """
    world_size = min(4, mp.cpu_count())
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    # Enable MPS fallback for operations not supported by MPS
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    main()