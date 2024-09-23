# Distributed Data Parallel (DDP) Training with PyTorch

This project demonstrates how to perform distributed training using PyTorch's DistributedDataParallel (DDP) on multiple machines. It includes a simple convolutional neural network trained on the CIFAR-10 dataset.

## Project Structure

```
.
├── ddp_launch.py
├── model.py
├── use_trained_model.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup and Installation

1. Clone this repository to all machines that will participate in the training.

2. Install Docker on all machines.

3. Build the Docker image:
   ```
   docker build -t ddp-training .
   ```

## Running the Distributed Training

1. On the master node (replace `<MASTER_IP>` with the internal IP of the master node):
   ```
   docker run -e MASTER_ADDR=<MASTER_IP> -e MASTER_PORT=29500 -e WORLD_SIZE=2 -e RANK=0 -p 29500:29500 ddp-training
   ```

2. On the worker node:
   ```
   docker run -e MASTER_ADDR=<MASTER_IP> -e MASTER_PORT=29500 -e WORLD_SIZE=2 -e RANK=1 ddp-training
   ```

The training script includes synchronization to ensure all nodes are ready before starting the training process. You should see output indicating that the nodes are initialized and waiting for each other before training begins.

## Using the Trained Model

After training, you can use the trained model for making predictions. The `use_trained_model.py` script demonstrates how to load and use the model:

1. Ensure you have the trained model file (e.g., `model_checkpoint_epoch_19.pth`) in the `pth` directory.

2. Update the `model_path` and `image_path` in `use_trained_model.py` to point to your model file and a test image, respectively.

3. Run the script:
   ```
   python use_trained_model.py
   ```

This script will load the model, use it to classify the provided image, and print the predicted class.

## Customization

- To train on a different dataset or modify the model architecture, update the `model.py` file.
- Adjust training parameters in `ddp_launch.py` as needed.
- For more complex usage of the trained model, modify `use_trained_model.py` or create new scripts based on your requirements.

## Troubleshooting

- Ensure all machines can communicate over the specified port (default is 29500).
- Check that the `MASTER_ADDR` is correctly set to the internal IP of the master node.
- Verify that the `WORLD_SIZE` matches the total number of nodes participating in the training.

## Contributing

Feel free to open issues or submit pull requests if you have suggestions for improvements or encounter any problems.

## License

[Specify your license here]