# Simplified Distributed Training Example

This project demonstrates a basic distributed training setup using PyTorch's DistributedDataParallel on a single machine with multiple processes.

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

To run the distributed training simulation:

```
python simplified_launch_script.py
```

This will start 4 processes on your local machine, simulating a distributed training environment.

## Project Structure

- `simplified_launch_script.py`: Contains the entire setup for distributed training, including a simple model, dataset creation, and training loop.
- `requirements.txt`: Lists the Python package dependencies.
- `README.md`: This file, containing setup and running instructions.

## Customization

You can modify the following parameters in the `simplified_launch_script.py`:

- `world_size`: Number of processes to spawn (default is 4)
- Number of epochs: Change the range in the `for epoch in range(10)` line
- Learning rate: Modify the `lr` parameter in the optimizer initialization
- Model architecture: Modify the `SimpleModel` class to experiment with different architectures

## Troubleshooting

If you encounter any "address already in use" errors, try changing the 'MASTER_PORT' in the setup function to a different number.

## Next Steps

Once this basic version is working, you can gradually add more complex features like custom optimizers or blockchain integration.

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). This license is highly restrictive and requires that:

1. The complete source code must be made available to any network user of the software.
2. Any modifications or derived works must also be licensed under AGPL-3.0.
3. Any software that incorporates or links to this code must also be released under AGPL-3.0.
4. The source code must be made available even if the software is run as a service over a network.

For the full license text, see the [LICENSE](LICENSE) file in this repository or visit [GNU AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.en.html).