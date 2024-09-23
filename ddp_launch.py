import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR
from datetime import timedelta
import torchvision
import torchvision.transforms as transforms
from model import SimpleModel
import time

os.environ['NCCL_TIMEOUT'] = '3600'
os.environ['GLOO_TIMEOUT_SECONDS'] = '3600'

print("Script started")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Contents of current directory: {os.listdir('.')}")
print(f"Environment variables:")
for key, value in os.environ.items():
    print(f"{key}: {value}")

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=3600))

def cleanup():
    dist.destroy_process_group()

def save_model(model, epoch, rank):
    if rank == 0:  # Only save on the main process
        save_dir = 'pth'
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        save_path = os.path.join(save_dir, f"model_checkpoint_epoch_{epoch}.pth")
        if isinstance(model, DDP):
            torch.save(model.module.state_dict(), save_path)
        else:
            torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

def train(rank, world_size):
    print(f"Initializing process {rank}")
    setup(rank, world_size)
    
    # Synchronize processes
    dist.barrier()
    if rank == 0:
        print("All processes have been initialized. Starting training.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Process {rank} using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        full_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = DataLoader(full_dataset, batch_size=64, sampler=train_sampler)
    
    model = SimpleModel().to(device)
    if world_size > 1:
        ddp_model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    else:
        ddp_model = model
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    num_epochs = 20
    for epoch in range(num_epochs):
        ddp_model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99 and rank == 0:  # print every 100 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        scheduler.step()
        save_model(ddp_model, epoch, rank)
    
    if rank == 0:
        print('Finished Training')
    
    cleanup()

def main():
    print("Entering main function")
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    
    print(f"Starting process with rank {rank} out of {world_size} processes")
    
    train(rank, world_size)

if __name__ == "__main__":
    print("Starting script execution")
    main()