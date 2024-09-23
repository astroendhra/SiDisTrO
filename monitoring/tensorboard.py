from torch.utils.tensorboard import SummaryWriter

# In your train function, add:
writer = SummaryWriter(f'runs/experiment_rank_{rank}')

# Inside the training loop:
writer.add_scalar('Loss/train', loss.item(), global_step=epoch * len(train_loader) + batch)
writer.add_scalar('Loss/validation', val_loss, global_step=epoch)

# Don't forget to close the writer at the end of training
writer.close()
