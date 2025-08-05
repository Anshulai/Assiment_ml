import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import argparse

from unet_model import ConditionalUNet, create_color_mapping


class PolygonColorInference:
    """Inference class for polygon coloring model"""
    
    def __init__(self, model_path, device=None):
        """
        Initialize inference class
        
        Args:
            model_path (str): Path to trained model checkpoint
            device (str): Device to run inference on
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.color_to_idx, self.idx_to_color = create_color_mapping()
        
        # Load model
        self.model = ConditionalUNet(n_channels=3, n_classes=3, num_colors=8)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Available colors: {list(self.color_to_idx.keys())}")
    
    def predict(self, image_path, color_name):
        """
        Generate colored polygon prediction
        
        Args:
            image_path (str): Path to input polygon image
            color_name (str): Name of color to apply
            
        Returns:
            tuple: (input_image, predicted_image) as numpy arrays
        """
        # Validate color
        if color_name not in self.color_to_idx:
            raise ValueError(f"Color '{color_name}' not supported. Available colors: {list(self.color_to_idx.keys())}")
        
        # Load and preprocess image
        input_image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(input_image).unsqueeze(0).to(self.device)
        
        # Get color index
        color_idx = torch.tensor([self.color_to_idx[color_name]], dtype=torch.long).to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            prediction = self.model(input_tensor, color_idx)
        
        # Convert to numpy arrays
        input_np = input_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        prediction_np = prediction.squeeze().cpu().permute(1, 2, 0).numpy()
        
        return input_np, prediction_np
    
    def predict_batch(self, image_paths, color_names):
        """
        Generate predictions for multiple images
        
        Args:
            image_paths (list): List of paths to input images
            color_names (list): List of color names
            
        Returns:
            tuple: (input_images, predicted_images) as lists of numpy arrays
        """
        assert len(image_paths) == len(color_names), "Number of images and colors must match"
        
        input_images = []
        predicted_images = []
        
        for img_path, color in zip(image_paths, color_names):
            input_img, pred_img = self.predict(img_path, color)
            input_images.append(input_img)
            predicted_images.append(pred_img)
        
        return input_images, predicted_images
    
    def visualize_prediction(self, image_path, color_name, save_path=None):
        """
        Visualize a single prediction
        
        Args:
            image_path (str): Path to input image
            color_name (str): Color name
            save_path (str, optional): Path to save visualization
        """
        input_img, pred_img = self.predict(image_path, color_name)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Input image
        axes[0].imshow(input_img)
        axes[0].set_title(f'Input: {os.path.basename(image_path)}')
        axes[0].axis('off')
        
        # Predicted image
        axes[1].imshow(pred_img)
        axes[1].set_title(f'Predicted: {color_name}')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_color_variations(self, image_path, save_path=None):
        """
        Visualize all color variations for a single polygon
        
        Args:
            image_path (str): Path to input image
            save_path (str, optional): Path to save visualization
        """
        colors = list(self.color_to_idx.keys())
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, color in enumerate(colors):
            input_img, pred_img = self.predict(image_path, color)
            
            axes[i].imshow(pred_img)
            axes[i].set_title(f'{color.capitalize()}')
            axes[i].axis('off')
        
        plt.suptitle(f'Color Variations: {os.path.basename(image_path)}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Color variations saved to {save_path}")
        
        plt.show()
    
    def create_comparison_grid(self, image_paths, color_names, save_path=None):
        """
        Create a comparison grid for multiple predictions
        
        Args:
            image_paths (list): List of image paths
            color_names (list): List of color names
            save_path (str, optional): Path to save grid
        """
        input_images, predicted_images = self.predict_batch(image_paths, color_names)
        
        n_samples = len(image_paths)
        fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
        
        if n_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(n_samples):
            # Input
            axes[0, i].imshow(input_images[i])
            axes[0, i].set_title(f'Input: {os.path.basename(image_paths[i])}')
            axes[0, i].axis('off')
            
            # Prediction
            axes[1, i].imshow(predicted_images[i])
            axes[1, i].set_title(f'Predicted: {color_names[i]}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison grid saved to {save_path}")
        
        plt.show()


def demo_inference(model_path, dataset_root):
    """Run demonstration of inference capabilities"""
    
    # Initialize inference
    inferencer = PolygonColorInference(model_path)
    
    # Get all 8 polygon shapes from validation set
    val_inputs_dir = os.path.join(dataset_root, 'validation', 'inputs')
    sample_images = [
        os.path.join(val_inputs_dir, 'triangle.png'),
        os.path.join(val_inputs_dir, 'square.png'),
        os.path.join(val_inputs_dir, 'circle.png'),
        os.path.join(val_inputs_dir, 'diamond.png'),
        os.path.join(val_inputs_dir, 'hexagon.png'),
        os.path.join(val_inputs_dir, 'pentagon.png'),
        os.path.join(val_inputs_dir, 'octagon.png'),
        os.path.join(val_inputs_dir, 'star.png')
    ]
    
    # Filter existing images
    sample_images = [img for img in sample_images if os.path.exists(img)]
    
    if not sample_images:
        print("No sample images found in validation set")
        return
    
    print("=== Polygon Coloring Inference Demo ===\n")
    
    # Demo 1: Single prediction
    print("1. Single Prediction Demo")
    if sample_images:
        inferencer.visualize_prediction(sample_images[0], 'red', 'demo_single_prediction.png')
    
    # Demo 2: Color variations
    print("\n2. Color Variations Demo")
    if sample_images:
        inferencer.visualize_color_variations(sample_images[0], 'demo_color_variations.png')
    
    # Demo 3: Multiple predictions
    print("\n3. Multiple Predictions Demo")
    if len(sample_images) >= 4:
        colors = ['blue', 'green', 'purple', 'orange']
        inferencer.create_comparison_grid(
            sample_images[:4], 
            colors, 
            'demo_comparison_grid.png'
        )
    
    print("\nDemo completed! Check the generated visualization files.")


def main():
    parser = argparse.ArgumentParser(description='Inference for polygon coloring model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image_path', type=str, 
                       help='Path to input image for single prediction')
    parser.add_argument('--color', type=str, default='blue',
                       help='Color name for prediction')
    parser.add_argument('--dataset_root', type=str, default='dataset',
                       help='Path to dataset root for demo')
    parser.add_argument('--demo', action='store_true',
                       help='Run inference demo')
    parser.add_argument('--save_path', type=str,
                       help='Path to save visualization')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_inference(args.model_path, args.dataset_root)
    elif args.image_path:
        inferencer = PolygonColorInference(args.model_path)
        inferencer.visualize_prediction(args.image_path, args.color, args.save_path)
    else:
        print("Please specify either --demo or --image_path for inference")


if __name__ == '__main__':
    main()
