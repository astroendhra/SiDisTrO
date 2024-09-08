"""
ddp_launch.py: Distributed Data Parallel (DDP) Training Script

This script demonstrates distributed training using PyTorch's DistributedDataParallel.
It's designed to work across multiple GPUs or CPU cores, showcasing how to set up
distributed training environments.

Usage:
    python ddp_launch.py

Note: This script uses the 'gloo' backend for distributed training, which works
      on both CPU and GPU. For GPU-specific optimizations, consider using 'nccl'.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

class SimpleModel(nn.Module):
    """A simple linear model for demonstration purposes."""
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
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
    Determine the best available device.
    
    Args:
        rank (int): Process rank for GPU selection if available
    
    Returns:
        torch.device: The selected device
    """
    if torch.cuda.is_available():
        return torch.device(f'cuda:{rank}')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def train(rank, world_size):
    """
    Training function to be run in parallel across multiple processes.
    
    Args:
        rank (int): Unique identifier of the process
        world_size (int): Total number of processes
    """
    setup(rank, world_size)
    
    # Use MPS for computation, but CPU for distributed operations
    compute_device = get_device(rank)
    dist_device = torch.device('cpu')
    print(f"Process {rank} using compute device: {compute_device}, dist device: {dist_device}")

    # Initialize the model and move it to the appropriate devices
    model = SimpleModel().to(compute_device)
    ddp_model = DDP(model.to(dist_device), device_ids=None)
    
    # Create a simple dataset for demonstration
    data = torch.randn(1000, 10)
    labels = torch.randn(1000, 1)
    dataset = TensorDataset(data, labels)
    
    # Set up the distributed sampler and data loader
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # Initialize the optimizer and loss function
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(10):
        sampler.set_epoch(epoch)  # Ensure different shuffling across epochs
        for batch, (data, labels) in enumerate(dataloader):
            data, labels = data.to(compute_device), labels.to(compute_device)
            optimizer.zero_grad()
            outputs = ddp_model(data.to(dist_device)).to(compute_device)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    cleanup()

def main():
    """Main function to set up and run the distributed training."""
    world_size = min(4, mp.cpu_count())
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    # Enable MPS fallback for operations not supported by MPS
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    main()