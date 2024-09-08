"""
mp_launch.py: Multiprocessing Training Script

This script demonstrates parallel training using Python's multiprocessing module.
It's designed to work on a single machine, utilizing multiple CPU cores or a single GPU.
This approach is simpler than DistributedDataParallel but limited to one machine.

Key Features:
- Parallel training across multiple processes on a single machine
- Synthetic dataset generation for demonstration
- Simple model architecture for easy understanding
- Device-agnostic implementation (works with CUDA, MPS, or CPU)
- Per-process logging of training progress

Usage:
    python mp_launch.py

Note: This script uses torch.multiprocessing, which is a wrapper around Python's
      multiprocessing module, tailored for PyTorch tensors and CUDA devices.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset

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

def get_device():
    """
    Determine the best available device.
    
    Returns:
        torch.device: The selected device (CUDA, MPS, or CPU)
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def train(rank, world_size, device):
    """
    Training function to be run in parallel across multiple processes.
    
    This function handles the entire training process for a subset of the data, including:
    - Creating a subset of the dataset
    - Preparing the data loader
    - Creating the model and optimizer
    - Training loop with logging
    
    Args:
        rank (int): Unique identifier of the process
        world_size (int): Total number of processes
        device (torch.device): The device to run computations on
    """
    print(f"Process {rank} using device: {device}")

    # Create a synthetic dataset for demonstration
    data = torch.randn(1000, 10)
    labels = torch.randn(1000, 1)
    dataset = TensorDataset(data, labels)
    
    # Divide the dataset among processes
    indices = list(range(rank, len(dataset), world_size))
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=32, shuffle=True)

    # Initialize the model, optimizer, and loss function
    model = SimpleModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch, (data, labels) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Process {rank}, Epoch {epoch}, Average Loss: {avg_loss:.4f}")

def main():
    """
    Main function to set up and run the parallel training.
    
    This function determines the number of processes to spawn,
    selects the appropriate device, and initiates the parallel training.
    """
    world_size = min(4, mp.cpu_count())
    device = get_device()
    
    # Set the multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Start the training processes
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=train, args=(rank, world_size, device))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

if __name__ == "__main__":
    # Enable MPS fallback for operations not supported by MPS
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    main()