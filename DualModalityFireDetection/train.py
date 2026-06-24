"""
Training script for Dual Modality Fire Detection Network
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from models import build_model
from utils import (
    build_loss,
    create_dataloaders,
    MetricLogger,
    DetectionMetrics
)


class Trainer:
    """Trainer class for dual modality fire detection"""

    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.device = config['train']['device']

        # Build model
        self.model = build_model(config)
        self.model = self.model.to(self.device)

        # Build optimizer
        self.optimizer = self._build_optimizer()

        # Build scheduler
        self.scheduler = self._build_scheduler()

        # Build loss function
        self.criterion = build_loss(config)
        self.criterion = self.criterion.to(self.device)

        # Build dataloaders
        self.dataloaders = create_dataloaders(config)

        # Training state
        self.epoch = 0
        self.best_map = 0.0

        # Logging
        self.log_dir = os.path.join(
            config['path']['log_dir'],
            datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        self.writer = SummaryWriter(self.log_dir)
        self.metric_logger = MetricLogger()

        # Checkpoint directory
        self.checkpoint_dir = config['path']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Resume training if specified
        if args.resume:
            self.load_checkpoint(args.resume)

    def _build_optimizer(self):
        """Build optimizer"""
        train_config = self.config['train']

        optimizer = optim.SGD(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            momentum=train_config['momentum'],
            weight_decay=train_config['weight_decay']
        )

        return optimizer

    def _build_scheduler(self):
        """Build learning rate scheduler"""
        train_config = self.config['train']
        lr_config = train_config['lr_scheduler']

        if lr_config['type'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['train']['num_epochs'],
                eta_min=lr_config['min_lr']
            )
        elif lr_config['type'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1)

        return scheduler

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        self.metric_logger.reset()

        epoch_start = time.time()

        for batch_idx, (rgb_imgs, thermal_imgs, targets) in enumerate(self.dataloaders['train']):
            batch_start = time.time()

            # Move data to device
            rgb_imgs = rgb_imgs.to(self.device)
            thermal_imgs = thermal_imgs.to(self.device)

            # Forward pass
            outputs = self.model(rgb_imgs, thermal_imgs)

            # Compute loss
            loss_dict = self.criterion(outputs, targets)
            total_loss = loss_dict['total_loss']

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Update metrics
            batch_time = time.time() - batch_start
            self.metric_logger.update(
                loss=total_loss.item(),
                rgb_loss=loss_dict['rgb_loss'].item(),
                thermal_loss=loss_dict['thermal_loss'].item(),
                fused_loss=loss_dict['fused_loss'].item(),
                lr=self.optimizer.param_groups[0]['lr'],
                batch_time=batch_time
            )

            # Log batch
            if batch_idx % 10 == 0:
                print(f"Epoch [{self.epoch}][{batch_idx}/{len(self.dataloaders['train'])}] "
                      f"Loss: {total_loss.item():.4f} "
                      f"RGB: {loss_dict['rgb_loss'].item():.4f} "
                      f"Thermal: {loss_dict['thermal_loss'].item():.4f} "
                      f"Fused: {loss_dict['fused_loss'].item():.4f}")

        epoch_time = time.time() - epoch_start
        metrics = self.metric_logger.summary()

        print(f"\nEpoch {self.epoch} completed in {epoch_time:.2f}s")
        print(f"Avg Loss: {metrics['loss']['avg']:.4f}")
        print(f"Avg LR: {metrics['lr']['avg']:.6f}")

        return metrics

    def validate(self):
        """Validate the model"""
        if 'val' not in self.dataloaders:
            return None

        self.model.eval()
        self.metric_logger.reset()

        metrics_calculator = DetectionMetrics(
            num_classes=self.config['detection_head']['num_classes']
        )

        with torch.no_grad():
            for rgb_imgs, thermal_imgs, targets in self.dataloaders['val']:
                rgb_imgs = rgb_imgs.to(self.device)
                thermal_imgs = thermal_imgs.to(self.device)

                outputs = self.model(rgb_imgs, thermal_imgs)

                loss_dict = self.criterion(outputs, targets)
                self.metric_logger.update(loss=loss_dict['total_loss'].item())

                # TODO: Convert predictions to detection format for mAP calculation

        metrics = self.metric_logger.summary()
        return metrics

    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config['train']['num_epochs']} epochs")
        print(f"Device: {self.device}")
        print(f"Log directory: {self.log_dir}")

        for epoch in range(self.epoch, self.config['train']['num_epochs']):
            self.epoch = epoch

            # Train one epoch
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update learning rate
            self.scheduler.step()

            # Log to tensorboard
            if train_metrics:
                self.writer.add_scalar('train/loss', train_metrics['loss']['avg'], epoch)
                self.writer.add_scalar('train/rgb_loss', train_metrics['rgb_loss']['avg'], epoch)
                self.writer.add_scalar('train/thermal_loss', train_metrics['thermal_loss']['avg'], epoch)
                self.writer.add_scalar('train/fused_loss', train_metrics['fused_loss']['avg'], epoch)
                self.writer.add_scalar('train/lr', train_metrics['lr']['avg'], epoch)

            if val_metrics:
                self.writer.add_scalar('val/loss', val_metrics['loss']['avg'], epoch)

            # Save checkpoint
            self.save_checkpoint(is_best=False)

            # Save best model
            if val_metrics and val_metrics['loss']['avg'] < self.best_map:
                self.best_map = val_metrics['loss']['avg']
                self.save_checkpoint(is_best=True)

        print("Training completed!")
        self.writer.close()

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_map': self.best_map,
            'config': self.config
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{self.epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch'] + 1
        self.best_map = checkpoint.get('best_map', 0.0)

        print(f"Resumed from epoch {self.epoch}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Dual Modality Fire Detection')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--config', type=str, default='configs/config.py',
                        help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')

    args = parser.parse_args()
    return args


def main():
    """Main entry point"""
    args = parse_args()

    # Load configuration
    config = Config.get_config()

    # Override with command line arguments
    if args.epochs:
        config['train']['num_epochs'] = args.epochs
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
    if args.lr:
        config['train']['learning_rate'] = args.lr

    # Create trainer
    trainer = Trainer(config, args)

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
