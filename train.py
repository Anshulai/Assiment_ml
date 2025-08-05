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


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    def __init__(self, device=None):
        super(PerceptualLoss, self).__init__()
        import torchvision.models as models
        vgg = models.vgg16(pretrained=True).features[:16]  # Use up to relu3_3
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Move VGG to the correct device
        if device is not None:
            self.vgg = self.vgg.to(device)
        
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        return self.mse(pred_features, target_features)


class CombinedLoss(nn.Module):
    """Combined loss function with MSE, perceptual, and color consistency"""
    def __init__(self, mse_weight=1.0, perceptual_weight=0.1, color_weight=0.5, device=None):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        self.color_weight = color_weight
        
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss(device=device)
    
    def color_consistency_loss(self, pred, target):
        """Ensure color consistency across the polygon"""
        # Calculate mean color in each channel
        pred_mean = torch.mean(pred, dim=[2, 3])  # (batch, channels)
        target_mean = torch.mean(target, dim=[2, 3])  # (batch, channels)
        return self.mse_loss(pred_mean, target_mean)
    
    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        color_consistency = self.color_consistency_loss(pred, target)
        
        total_loss = (self.mse_weight * mse + 
                     self.perceptual_weight * perceptual + 
                     self.color_weight * color_consistency)
        
        return total_loss, {
            'mse': mse.item(),
            'perceptual': perceptual.item(),
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
        
        # SSIM (simplified version)
        def ssim_simple(x, y):
            mu_x = torch.mean(x)
            mu_y = torch.mean(y)
            sigma_x = torch.var(x)
            sigma_y = torch.var(y)
            sigma_xy = torch.mean((x - mu_x) * (y - mu_y))
            
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
            return ssim.item()
        
        ssim = ssim_simple(pred, target)
        
        return {'mse': mse, 'psnr': psnr, 'ssim': ssim}


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
        # Determine number of samples to visualize (min of 4 or batch size)
        num_samples = min(4, inputs.size(0))
        fig, axes = plt.subplots(3, num_samples, figsize=(4*num_samples, 12))
        
        # Handle case where num_samples is 1
        if num_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_samples):
            # Input
            input_img = inputs[i].cpu().permute(1, 2, 0).numpy()
            axes[0, i].imshow(input_img)
            axes[0, i].set_title(f'Input: {polygon_names[i] if i < len(polygon_names) else "Unknown"}')
            axes[0, i].axis('off')
            
            # Target
            target_img = targets[i].cpu().permute(1, 2, 0).numpy()
            axes[1, i].imshow(target_img)
            axes[1, i].set_title(f'Target: {color_names[i] if i < len(color_names) else "Unknown"}')
            axes[1, i].axis('off')
            
            # Prediction
            pred_img = predictions[i].cpu().permute(1, 2, 0).numpy()
            axes[2, i].imshow(pred_img)
            axes[2, i].set_title(f'Prediction: {color_names[i] if i < len(color_names) else "Unknown"}')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'predictions_epoch_{epoch}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    loss_components = {'mse': 0, 'perceptual': 0, 'color_consistency': 0}
    
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
    loss_components = {'mse': 0, 'perceptual': 0, 'color_consistency': 0}
    all_metrics = {'mse': 0, 'psnr': 0, 'ssim': 0}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            color_indices = batch['color_idx'].to(device)
            
            # Forward pass
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
    parser = argparse.ArgumentParser(description='Train UNet for polygon coloring')
    parser.add_argument('--dataset_root', type=str, default='dataset', 
                       help='Path to dataset root directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--save_dir', type=str, default='results', help='Save directory')
    parser.add_argument('--wandb_project', type=str, default='polygon-coloring-unet', 
                       help='Wandb project name')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
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
            'dataset': 'polygon_coloring'
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
    criterion = CombinedLoss(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f'Resumed from epoch {start_epoch}')
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
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
            'train_perceptual': train_loss_components['perceptual'],
            'train_color_consistency': train_loss_components['color_consistency'],
            'val_mse': val_loss_components['mse'],
            'val_perceptual': val_loss_components['perceptual'],
            'val_color_consistency': val_loss_components['color_consistency'],
            'val_psnr': val_metrics['psnr'],
            'val_ssim': val_metrics['ssim'],
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Save sample predictions
        if epoch % 10 == 0:
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
        
        # Save checkpoint
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': vars(args)
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Val PSNR: {val_metrics["psnr"]:.2f}, Val SSIM: {val_metrics["ssim"]:.4f}')
    
    print('Training completed!')
    wandb.finish()


if __name__ == '__main__':
    main()
