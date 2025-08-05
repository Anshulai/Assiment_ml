import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from datetime import datetime

from unet_model import ConditionalUNet, create_color_mapping
from dataset import create_data_loaders, PolygonColorDataset


class SimpleLoss(nn.Module):
    """Simple loss function with MSE and color consistency only"""
    def __init__(self, mse_weight=1.0, color_weight=0.5):
        super(SimpleLoss, self).__init__()
        self.mse_weight = mse_weight
        self.color_weight = color_weight
        
        self.mse_loss = nn.MSELoss()
    
    def color_consistency_loss(self, pred, target):
        """Ensure color consistency across the polygon"""
        # Calculate mean color in each channel
        pred_mean = torch.mean(pred, dim=[2, 3])  # (batch, channels)
        target_mean = torch.mean(target, dim=[2, 3])  # (batch, channels)
        return self.mse_loss(pred_mean, target_mean)
    
    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        color_consistency = self.color_consistency_loss(pred, target)
        
        total_loss = (self.mse_weight * mse + 
                     self.color_weight * color_consistency)
        
        return total_loss, {
            'mse': mse.item(),
            'color_consistency': color_consistency.item(),
            'total': total_loss.item()
        }


def calculate_metrics(pred, target):
    """Calculate evaluation metrics"""
    with torch.no_grad():
        # MSE
        mse = torch.mean((pred - target) ** 2).item()
        
        # PSNR
        psnr = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse))).item()
        
        return {'mse': mse, 'psnr': psnr}


def save_sample_predictions(model, val_loader, device, epoch, save_dir):
    """Save sample predictions for visualization"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        batch = next(iter(val_loader))
        inputs = batch['input'][:4].to(device)
        targets = batch['target'][:4].to(device)
        color_indices = batch['color_idx'][:4].to(device)
        color_names = batch['color_name'][:4]
        polygon_names = batch['polygon_name'][:4]
        
        predictions = model(inputs, color_indices)
        
        # Create comparison grid
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        for i in range(4):
            # Input
            input_img = inputs[i].cpu().permute(1, 2, 0).numpy()
            axes[0, i].imshow(input_img)
            axes[0, i].set_title(f'Input: {polygon_names[i]}')
            axes[0, i].axis('off')
            
            # Target
            target_img = targets[i].cpu().permute(1, 2, 0).numpy()
            axes[1, i].imshow(target_img)
            axes[1, i].set_title(f'Target: {color_names[i]}')
            axes[1, i].axis('off')
            
            # Prediction
            pred_img = predictions[i].cpu().permute(1, 2, 0).numpy()
            axes[2, i].imshow(pred_img)
            axes[2, i].set_title(f'Prediction: {color_names[i]}')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'predictions_epoch_{epoch}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    loss_components = {'mse': 0, 'color_consistency': 0}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        color_indices = batch['color_idx'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(inputs, color_indices)
        loss, loss_dict = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    # Calculate averages
    avg_loss = total_loss / len(train_loader)
    for key in loss_components:
        loss_components[key] /= len(train_loader)
    
    return avg_loss, loss_components


def validate_epoch(model, val_loader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    loss_components = {'mse': 0, 'color_consistency': 0}
    all_metrics = {'mse': 0, 'psnr': 0}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            color_indices = batch['color_idx'].to(device)
            
            # Forward pass - FIXED BUG HERE
            predictions = model(inputs, color_indices)
            loss, loss_dict = criterion(predictions, targets)
            
            # Update loss metrics
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += loss_dict[key]
            
            # Calculate evaluation metrics
            batch_metrics = calculate_metrics(predictions, targets)
            for key in all_metrics:
                all_metrics[key] += batch_metrics[key]
    
    # Calculate averages
    avg_loss = total_loss / len(val_loader)
    for key in loss_components:
        loss_components[key] /= len(val_loader)
    for key in all_metrics:
        all_metrics[key] /= len(val_loader)
    
    return avg_loss, loss_components, all_metrics


def main():
    parser = argparse.ArgumentParser(description='Train UNet for polygon coloring (simplified)')
    parser.add_argument('--dataset_root', type=str, default='dataset', 
                       help='Path to dataset root directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--save_dir', type=str, default='results', help='Save directory')
    parser.add_argument('--wandb_project', type=str, default='polygon-coloring-simple', 
                       help='Wandb project name')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        config={
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'image_size': args.image_size,
            'architecture': 'ConditionalUNet',
            'dataset': 'polygon_coloring',
            'loss': 'simple_mse_color'
        }
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data loaders
    train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(
        args.dataset_root, 
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    
    # Create model
    model = ConditionalUNet(n_channels=3, n_classes=3, num_colors=8).to(device)
    
    # Create loss function and optimizer
    criterion = SimpleLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # Train
        train_loss, train_loss_components = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_loss_components, val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_mse': train_loss_components['mse'],
            'train_color_consistency': train_loss_components['color_consistency'],
            'val_mse': val_loss_components['mse'],
            'val_color_consistency': val_loss_components['color_consistency'],
            'val_psnr': val_metrics['psnr'],
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Save sample predictions
        if epoch % 5 == 0:
            save_sample_predictions(model, val_loader, device, epoch, args.save_dir)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': vars(args)
            }, os.path.join(args.save_dir, 'best_model.pth'))
        
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Val PSNR: {val_metrics["psnr"]:.2f}')
    
    print('Training completed!')
    wandb.finish()


if __name__ == '__main__':
    main()
