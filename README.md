# Simplified Distributed Training Example

This project demonstrates a distributed training setup using PyTorch's DistributedDataParallel (DDP) on multiple machines or a single machine with multiple processes.

## Prerequisites

- Python 3.8+
- PyTorch 1.8+

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/simplified-distro-training.git
   cd simplified-distro-training
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Distributed Training

### Single Machine

To run the distributed training simulation on a single machine:

```
python ddp_launch.py
```

This will start 4 processes on your local machine, simulating a distributed training environment.

### Multiple Machines

To run on multiple machines:

1. Choose one machine as the master node. Note its IP address.

2. On the master node, modify the `ddp_launch.py` file:
   - Set `MASTER_ADDR` to the IP address of the master node.
   - Choose an available port number for `MASTER_PORT`.

3. On each machine, run:
   ```
   python ddp_launch.py --node_rank <rank> --nnodes <total_nodes> --master_addr <master_ip> --master_port <port>
   ```
   Where:
   - `<rank>` is the unique ID for this node (0 for master, 1, 2, etc. for others)
   - `<total_nodes>` is the total number of machines
   - `<master_ip>` is the IP address of the master node
   - `<port>` is the port number chosen in step 2

   Example for a 2-node setup:
   - On master: `python ddp_launch.py --node_rank 0 --nnodes 2 --master_addr 192.168.1.100 --master_port 29500`
   - On second node: `python ddp_launch.py --node_rank 1 --nnodes 2 --master_addr 192.168.1.100 --master_port 29500`

## Project Structure

- `ddp_launch.py`: Contains the setup for distributed training, including model definition, dataset creation, and training loop.
- `requirements.txt`: Lists the Python package dependencies.
- `README.md`: This file, containing setup and running instructions.

## Customization

You can modify the following parameters in `ddp_launch.py`:

- `world_size`: Number of processes per node (default is 4)
- Number of epochs: Change the range in the `for epoch in range(10)` line
- Learning rate: Modify the `lr` parameter in the optimizer initialization
- Model architecture: Modify the `Net` class to experiment with different architectures

## Hardware Acceleration

The script automatically detects and uses the best available hardware:
- CUDA GPUs if available
- Apple Silicon (MPS) on compatible Macs
- Falls back to CPU if neither is available

## Troubleshooting

- If you encounter "address already in use" errors, try changing the 'MASTER_PORT' to a different number.
- For MPS devices, the script uses a workaround to initialize DDP on CPU before moving the model back to MPS.

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). This license is highly restrictive and requires that:

1. The complete source code must be made available to any network user of the software.
2. Any modifications or derived works must also be licensed under AGPL-3.0.
3. Any software that incorporates or links to this code must also be released under AGPL-3.0.
4. The source code must be made available even if the software is run as a service over a network.

For the full license text, see the [LICENSE](LICENSE) file in this repository or visit [GNU AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.en.html).