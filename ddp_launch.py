import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR
import torchvision
import torchvision.transforms as transforms
from model import SimpleModel
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup(rank, world_size):
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    logging.info(f"Initializing process group: rank={rank}, world_size={world_size}")
    logging.info(f"MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={os.environ['MASTER_PORT']}")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    logging.info("Process group initialized")

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args):
    setup(rank, world_size)
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
    elif torch.backends.mps.is_available():
        device = torch.device("cpu")
        print("Using MPS device with CPU fallback for unsupported operations.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    logging.info(f"Process {rank} using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    
    model = SimpleModel().to(device)
    ddp_model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)  # Important for proper shuffling
        ddp_model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                logging.info(f'[Rank {rank}, Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        scheduler.step()
    
        # Synchronize at the end of each epoch
        dist.barrier()
        
        # Aggregate and log the epoch loss across all processes
        epoch_loss = torch.tensor(running_loss, device=device)
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        if rank == 0:
            logging.info(f'Epoch {epoch + 1} completed. Average loss: {epoch_loss.item() / world_size:.3f}')
    
    if rank == 0:
        save_dir = 'model_output'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'cifar_net.pth')
        torch.save(ddp_model.module.state_dict(), save_path)
        logging.info(f'Finished Training and saved model to {save_path}')

    cleanup()

def main():
    parser = argparse.ArgumentParser(description="Distributed training")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=64, help='Input batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    
    logging.info(f"Starting process with rank {rank} out of {world_size} processes")
    
    train(rank, world_size, args)

if __name__ == "__main__":
    main()