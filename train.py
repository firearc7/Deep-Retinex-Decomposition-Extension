"""
Training script for Deep Retinex Decomposition Network

Dataset: LOL (Low-Light) dataset from the paper
Paper: "Deep Retinex Decomposition for Low-Light Enhancement"
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import logging
from datetime import datetime

from src.model.retinexnet import RetinexNet
from config import TRAIN_CONFIG, MODEL_CONFIG, PROJECT_ROOT


class LOLDataset(Dataset):
    """
    LOL (Low-Light) Dataset
    Structure expected:
        data/train/low/  - low-light images
        data/train/high/ - normal-light (reference) images
    """
    def __init__(self, low_dir, high_dir, patch_size=48, augment=True):
        self.low_dir = Path(low_dir)
        self.high_dir = Path(high_dir)
        self.patch_size = patch_size
        self.augment = augment
        
        # Get all image files
        self.low_images = sorted(list(self.low_dir.glob('*.png')) + 
                                 list(self.low_dir.glob('*.jpg')))
        self.high_images = sorted(list(self.high_dir.glob('*.png')) + 
                                  list(self.high_dir.glob('*.jpg')))
        
        assert len(self.low_images) == len(self.high_images), \
            f"Mismatch in number of images: {len(self.low_images)} low vs {len(self.high_images)} high"
        
        print(f"Loaded {len(self.low_images)} image pairs from {low_dir}")
    
    def __len__(self):
        return len(self.low_images)
    
    def __getitem__(self, idx):
        # Load images
        low_img = Image.open(self.low_images[idx]).convert('RGB')
        high_img = Image.open(self.high_images[idx]).convert('RGB')
        
        # Convert to numpy arrays
        low_np = np.array(low_img).astype(np.float32) / 255.0
        high_np = np.array(high_img).astype(np.float32) / 255.0
        
        # Random crop to patch_size
        if self.patch_size > 0:
            h, w = low_np.shape[:2]
            if h > self.patch_size and w > self.patch_size:
                x = np.random.randint(0, w - self.patch_size)
                y = np.random.randint(0, h - self.patch_size)
                low_np = low_np[y:y+self.patch_size, x:x+self.patch_size]
                high_np = high_np[y:y+self.patch_size, x:x+self.patch_size]
        
        # Data augmentation
        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                low_np = np.fliplr(low_np)
                high_np = np.fliplr(high_np)
            
            # Random vertical flip
            if np.random.random() > 0.5:
                low_np = np.flipud(low_np)
                high_np = np.flipud(high_np)
            
            # Random rotation (90, 180, 270 degrees)
            if np.random.random() > 0.5:
                k = np.random.randint(1, 4)
                low_np = np.rot90(low_np, k)
                high_np = np.rot90(high_np, k)
        
        # Convert to torch tensors (C, H, W)
        low_tensor = torch.from_numpy(low_np.transpose(2, 0, 1).copy())
        high_tensor = torch.from_numpy(high_np.transpose(2, 0, 1).copy())
        
        return low_tensor, high_tensor


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (low_img, high_img) in enumerate(pbar):
        low_img = low_img.to(device)
        high_img = high_img.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        loss = model(low_img, high_img)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return total_loss / num_batches


def validate(model, dataloader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for low_img, high_img in tqdm(dataloader, desc="Validating"):
            low_img = low_img.to(device)
            high_img = high_img.to(device)
            
            loss = model(low_img, high_img)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {epoch}, loss: {loss:.4f}")
    return epoch, loss


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create directories
    checkpoint_dir = Path(PROJECT_ROOT) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(PROJECT_ROOT) / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = logs_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Log file: {log_file}")
    print(f"Using device: {device}")
    print(f"Training logs will be saved to: {log_file}")
    
    # Initialize model
    model = RetinexNet().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with {num_params:,} parameters")
    print(f"Model initialized with {num_params:,} parameters")
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=0.1)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            start_epoch, _ = load_checkpoint(model, optimizer, args.resume, device)
            start_epoch += 1
            logger.info(f"Resumed training from epoch {start_epoch}")
        else:
            logger.warning(f"Checkpoint {args.resume} not found. Starting from scratch.")
            print(f"Checkpoint {args.resume} not found. Starting from scratch.")
    
    # Prepare datasets
    train_dataset = LOLDataset(
        low_dir=args.train_low_dir,
        high_dir=args.train_high_dir,
        patch_size=args.patch_size,
        augment=True
    )
    
    val_dataset = LOLDataset(
        low_dir=args.val_low_dir,
        high_dir=args.val_high_dir,
        patch_size=0,  # No cropping for validation
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,  # Use same batch size as training for faster validation
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    logger.info("\nStarting training...")
    logger.info(f"Total epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Patch size: {args.patch_size}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    print("\nStarting training...")
    print(f"Total epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Patch size: {args.patch_size}")
    print("-" * 50)
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch + 1)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log results
        epoch_info = f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}"
        logger.info(epoch_info)
        
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        print("-" * 50)
        
        # Save training history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['learning_rate'].append(current_lr)
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = checkpoint_dir / f'retinexnet_epoch_{epoch + 1}.pt'
            save_checkpoint(model, optimizer, epoch + 1, train_loss, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = checkpoint_dir / 'retinexnet_best.pt'
            save_checkpoint(model, optimizer, epoch + 1, val_loss, best_model_path)
            logger.info(f"New best model saved! Val Loss: {val_loss:.4f}")
            print(f"New best model saved! Val Loss: {val_loss:.4f}")
    
    # Save final model
    final_model_path = checkpoint_dir / 'retinexnet_final.pt'
    save_checkpoint(model, optimizer, args.epochs, train_loss, final_model_path)
    
    # Save training history
    history_path = checkpoint_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=4)
    logger.info(f"Training history saved to {history_path}")
    print(f"\nTraining history saved to {history_path}")
    
    logger.info("=" * 50)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final model saved to {final_model_path}")
    logger.info("=" * 50)
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to {final_model_path}")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deep Retinex Decomposition Network')
    
    # Dataset paths
    parser.add_argument('--train_low_dir', type=str, 
                        default='data/train/low',
                        help='Directory containing training low-light images')
    parser.add_argument('--train_high_dir', type=str,
                        default='data/train/high',
                        help='Directory containing training normal-light images')
    parser.add_argument('--val_low_dir', type=str,
                        default='data/eval/low',
                        help='Directory containing validation low-light images')
    parser.add_argument('--val_high_dir', type=str,
                        default='data/eval/high',
                        help='Directory containing validation normal-light images')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--patch_size', type=int, default=48,
                        help='Patch size for training (0 for full image)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--lr_decay_step', type=int, default=50,
                        help='Learning rate decay step')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Checkpoint
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.train_low_dir):
        raise ValueError(f"Training low-light directory not found: {args.train_low_dir}")
    if not os.path.exists(args.train_high_dir):
        raise ValueError(f"Training normal-light directory not found: {args.train_high_dir}")
    
    main(args)
