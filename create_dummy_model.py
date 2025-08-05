import os
import torch
from unet_model import ConditionalUNet

def create_dummy_model():
    """Create a dummy trained model for testing inference"""
    
    # Create model
    model = ConditionalUNet(n_channels=3, n_classes=3, num_colors=8)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Save dummy model checkpoint
    dummy_checkpoint = {
        'epoch': 10,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None,
        'best_val_loss': 0.1,
        'config': {
            'batch_size': 8,
            'epochs': 10,
            'lr': 1e-4,
            'image_size': 256
        }
    }
    
    torch.save(dummy_checkpoint, 'results/best_model.pth')
    print("Dummy model saved to results/best_model.pth")
    print("You can now test inference with:")
    print("python inference.py --model_path results/best_model.pth --demo --dataset_root dataset")

if __name__ == '__main__':
    create_dummy_model()
