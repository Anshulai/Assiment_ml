import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from unet_model import create_color_mapping


class PolygonColorDataset(Dataset):
    """Dataset for polygon coloring task"""
    
    def __init__(self, data_dir, transform=None, image_size=256):
        """
        Args:
            data_dir (str): Path to dataset directory (training or validation)
            transform (callable, optional): Optional transform to be applied on images
            image_size (int): Size to resize images to
        """
        self.data_dir = data_dir
        self.image_size = image_size
        
        # Load data mapping
        with open(os.path.join(data_dir, 'data.json'), 'r') as f:
            self.data = json.load(f)
        
        # Create color mapping
        self.color_to_idx, self.idx_to_color = create_color_mapping()
        
        # Set up transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load input polygon image
        input_path = os.path.join(self.data_dir, 'inputs', item['input_polygon'])
        input_image = Image.open(input_path).convert('RGB')
        
        # Load target colored polygon image
        output_path = os.path.join(self.data_dir, 'outputs', item['output_image'])
        target_image = Image.open(output_path).convert('RGB')
        
        # Apply transforms
        input_tensor = self.transform(input_image)
        target_tensor = self.transform(target_image)
        
        # Get color index
        color_name = item['colour']
        color_idx = self.color_to_idx[color_name]
        
        return {
            'input': input_tensor,
            'target': target_tensor,
            'color_idx': torch.tensor(color_idx, dtype=torch.long),
            'color_name': color_name,
            'polygon_name': item['input_polygon']
        }


def create_data_loaders(dataset_root, batch_size=8, num_workers=4, image_size=256):
    """Create training and validation data loaders"""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
    ])
    
    # No augmentation for validation
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    # Create datasets
    train_dataset = PolygonColorDataset(
        os.path.join(dataset_root, 'training'),
        transform=train_transform,
        image_size=image_size
    )
    
    val_dataset = PolygonColorDataset(
        os.path.join(dataset_root, 'validation'),
        transform=val_transform,
        image_size=image_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset, val_dataset


def visualize_batch(batch, num_samples=4):
    """Visualize a batch of data"""
    import matplotlib.pyplot as plt
    
    inputs = batch['input'][:num_samples]
    targets = batch['target'][:num_samples]
    color_names = batch['color_name'][:num_samples]
    polygon_names = batch['polygon_name'][:num_samples]
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        # Input image
        input_img = inputs[i].permute(1, 2, 0).numpy()
        axes[0, i].imshow(input_img)
        axes[0, i].set_title(f'Input: {polygon_names[i]}')
        axes[0, i].axis('off')
        
        # Target image
        target_img = targets[i].permute(1, 2, 0).numpy()
        axes[1, i].imshow(target_img)
        axes[1, i].set_title(f'Target: {color_names[i]}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test the dataset
    dataset_root = "dataset"
    
    train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(
        dataset_root, batch_size=4, image_size=256
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Test loading a batch
    batch = next(iter(train_loader))
    print(f"Batch input shape: {batch['input'].shape}")
    print(f"Batch target shape: {batch['target'].shape}")
    print(f"Batch color indices: {batch['color_idx']}")
    print(f"Color names: {batch['color_name']}")
    
    # Visualize batch (uncomment to see)
    # visualize_batch(batch)
