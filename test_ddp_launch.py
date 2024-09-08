import unittest
from unittest.mock import patch, MagicMock
import torch
import torch.distributed as dist
from ddp_launch import train, setup, cleanup

class TestDDPLaunch(unittest.TestCase):
    @patch('ddp_launch.create_data_loader')
    @patch('ddp_launch.random_split')
    @patch('ddp_launch.TensorDataset')
    @patch('torch.nn.parallel.DistributedDataParallel', side_effect=lambda model, *args, **kwargs: model)
    @patch('torch.optim.SGD')
    @patch('torch.optim.lr_scheduler.StepLR')
    @patch('ddp_launch.validate', return_value=0.5)
    def test_train(self, mock_validate, mock_step_lr, mock_sgd, mock_ddp, mock_tensor_dataset, mock_random_split, mock_create_data_loader):
        # Mock dataset and data loader
        mock_dataset = MagicMock()
        mock_random_split.return_value = (mock_dataset, mock_dataset)
        mock_data_loader = MagicMock()
        mock_create_data_loader.return_value = mock_data_loader

        # Mock optimizer and scheduler
        mock_optimizer = MagicMock()
        mock_sgd.return_value = mock_optimizer
        mock_scheduler = MagicMock()
        mock_step_lr.return_value = mock_scheduler

        # Initialize process group for testing
        if not dist.is_initialized():
            dist.init_process_group(backend='gloo', rank=0, world_size=1, init_method='tcp://localhost:12355')

        try:
            train(0, 1)  # Use world_size=1 for testing
        finally:
            # Clean up the process group
            cleanup()

        # Add your assertions here
        mock_validate.assert_called()
        mock_sgd.assert_called()
        mock_step_lr.assert_called()
        mock_ddp.assert_called()
        mock_optimizer.step.assert_called()
        mock_scheduler.step.assert_called()

if __name__ == '__main__':
    unittest.main()