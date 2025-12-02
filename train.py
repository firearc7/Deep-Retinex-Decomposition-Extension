# Training script for Deep Retinex Decomposition Network

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import logging
from datetime import datetime

from src.model.retinexnet import RetinexNet
from config import TRAIN_CONFIG, PROJECT_ROOT


class LOLDataset(Dataset):
    # LOL (Low-Light) Dataset loader
    
    def __init__(self, low_dir, high_dir, patch_size=48, augment=True):
        self.low_dir = Path(low_dir)
        self.high_dir = Path(high_dir)
        self.patch_size = patch_size
        self.augment = augment
        
        self.low_images = sorted(list(self.low_dir.glob('*.png')) + 
                                 list(self.low_dir.glob('*.jpg')))
        self.high_images = sorted(list(self.high_dir.glob('*.png')) + 
                                  list(self.high_dir.glob('*.jpg')))
        
        assert len(self.low_images) == len(self.high_images)
        print(f"Loaded {len(self.low_images)} image pairs from {low_dir}")
    
    def __len__(self):
        return len(self.low_images)
    
    def __getitem__(self, idx):
        low_img = Image.open(self.low_images[idx]).convert('RGB')
        high_img = Image.open(self.high_images[idx]).convert('RGB')
        
        low_np = np.array(low_img).astype(np.float32) / 255.0
        high_np = np.array(high_img).astype(np.float32) / 255.0
        
        # Random crop
        if self.patch_size > 0:
            h, w = low_np.shape[:2]
            if h > self.patch_size and w > self.patch_size:
                x = np.random.randint(0, w - self.patch_size)
                y = np.random.randint(0, h - self.patch_size)
                low_np = low_np[y:y+self.patch_size, x:x+self.patch_size]
                high_np = high_np[y:y+self.patch_size, x:x+self.patch_size]
        
        # Data augmentation
        if self.augment:
            if np.random.random() > 0.5:
                low_np = np.fliplr(low_np)
                high_np = np.fliplr(high_np)
            if np.random.random() > 0.5:
                low_np = np.flipud(low_np)
                high_np = np.flipud(high_np)
            if np.random.random() > 0.5:
                k = np.random.randint(1, 4)
                low_np = np.rot90(low_np, k)
                high_np = np.rot90(high_np, k)
        
        low_tensor = torch.from_numpy(low_np.transpose(2, 0, 1).copy())
        high_tensor = torch.from_numpy(high_np.transpose(2, 0, 1).copy())
        
        return low_tensor, high_tensor


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (low_img, high_img) in enumerate(pbar):
        low_img = low_img.to(device)
        high_img = high_img.to(device)
        
        optimizer.zero_grad()
        loss = model(low_img, high_img)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{total_loss / (batch_idx + 1):.4f}'})
    
    torch.cuda.empty_cache()
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for low_img, high_img in tqdm(dataloader, desc="Validating"):
            low_img = low_img.to(device)
            high_img = high_img.to(device)
            loss = model(low_img, high_img)
            total_loss += loss.item()
            torch.cuda.empty_cache()
    
    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    checkpoint_dir = Path(PROJECT_ROOT) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(PROJECT_ROOT) / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = logs_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(message)s',
                       handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    
    # Initialize model
    model = RetinexNet().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=0.1)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Prepare datasets
    train_dataset = LOLDataset(args.train_low_dir, args.train_high_dir, 
                               patch_size=args.patch_size, augment=True)
    val_dataset = LOLDataset(args.val_low_dir, args.val_high_dir, 
                             patch_size=0, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    
    # Training loop
    best_val_loss = float('inf')
    training_history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}
    
    logger.info(f"Starting training for {args.epochs} epochs")
    
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch + 1)
        val_loss = validate(model, val_loader, device)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['learning_rate'].append(current_lr)
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1, train_loss, 
                           checkpoint_dir / f'retinexnet_epoch_{epoch + 1}.pt')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch + 1, val_loss, 
                           checkpoint_dir / 'retinexnet_best.pt')
            logger.info(f"New best model saved with val loss: {val_loss:.4f}")
    
    # Save final model and history
    save_checkpoint(model, optimizer, args.epochs, train_loss, 
                   checkpoint_dir / 'retinexnet_final.pt')
    
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=4)
    
    logger.info(f"Training completed. Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deep Retinex Network')
    
    parser.add_argument('--train_low_dir', type=str, default='data/train/low')
    parser.add_argument('--train_high_dir', type=str, default='data/train/high')
    parser.add_argument('--val_low_dir', type=str, default='data/eval/low')
    parser.add_argument('--val_high_dir', type=str, default='data/eval/high')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_step', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--resume', type=str, default='')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.train_low_dir):
        raise ValueError(f"Training directory not found: {args.train_low_dir}")
    
    main(args)
