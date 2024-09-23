# Distributed Data Parallel (DDP) Training with PyTorch

This project demonstrates how to perform distributed training using PyTorch's DistributedDataParallel (DDP) on multiple machines. It includes a simple convolutional neural network trained on the CIFAR-10 dataset.

## Project Structure

```
.
├── ddp_launch.py
├── model.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Prerequisites

- Docker installed on all machines
- Machines should be able to communicate with each other over the network
- Basic understanding of PyTorch and distributed training concepts

## Setup and Installation

1. Clone this repository to all machines that will participate in the training:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Build the Docker image on all machines:
   ```
   docker build -t ddp-training .
   ```

## Running the Distributed Training

1. On the master node (replace `<MASTER_IP>` with the actual IP address of the master node):
   ```
   docker run --rm -it \
     -e MASTER_ADDR=<MASTER_IP> \
     -e MASTER_PORT=29500 \
     -e WORLD_SIZE=2 \
     -e RANK=0 \
     -p 29500:29500 \
     ddp-training
   ```

2. On the worker node:
   ```
   docker run --rm -it \
     -e MASTER_ADDR=<MASTER_IP> \
     -e MASTER_PORT=29500 \
     -e WORLD_SIZE=2 \
     -e RANK=1 \
     ddp-training
   ```

Make sure to start the master node (RANK=0) first, then start the worker node(s).

## Code Overview

- `ddp_launch.py`: Main script that sets up and runs the distributed training.
- `model.py`: Contains the definition of the neural network model.
- `Dockerfile`: Defines the Docker image for the training environment.
- `requirements.txt`: Lists the Python dependencies.

## Key Features

- Distributed training using PyTorch's DistributedDataParallel
- CIFAR-10 dataset used for demonstration
- Simple CNN model architecture
- Proper data distribution and parallelization across nodes
- Logging of training progress for each node

## Customization

To adapt this project for your own use:

1. Modify the `SimpleModel` in `model.py` to change the neural network architecture.
2. Adjust training parameters in `ddp_launch.py` (e.g., number of epochs, learning rate, etc.).
3. Replace CIFAR-10 with your own dataset by modifying the dataset loading part in `ddp_launch.py`.

## Troubleshooting

- Ensure all machines can communicate over port 29500 (or the port you've specified).
- Verify that the `MASTER_ADDR` is correctly set to the IP address of the master node.
- Check that the `WORLD_SIZE` matches the total number of nodes participating in the training.
- If you're using GPUs, ensure CUDA is properly set up on all machines.

## Viewing Training Progress

The training progress is logged to the console. You should see output from both the master and worker nodes showing the loss for each epoch and batch.

## Saving and Using the Trained Model

After training, the model is saved as `cifar_net.pth` on the master node. To use this model for inference or further training, you can load it using PyTorch's `torch.load()` function.

## Contributing

Feel free to open issues or submit pull requests if you have suggestions for improvements or encounter any problems.

## License

AGPL-3.0 license

## Acknowledgments

- PyTorch team for their excellent distributed training documentation
- CIFAR-10 dataset creators