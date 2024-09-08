import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_device(rank):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{rank}')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def train(rank, world_size):
    setup(rank, world_size)
    
    device = get_device(rank)
    print(f"Process {rank} using device: {device}")

    model = SimpleModel().to(device)
    ddp_model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    
    # Create a simple dataset
    data = torch.randn(1000, 10)
    labels = torch.randn(1000, 1)
    dataset = TensorDataset(data, labels)
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(10):
        for batch, (data, labels) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = ddp_model(data)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    cleanup()

def main():
    world_size = min(4, mp.cpu_count())
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()