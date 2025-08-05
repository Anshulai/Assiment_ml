import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import argparse

from unet_model import ConditionalUNet, create_color_mapping


class PolygonColorInference:
    
    
    def __init__(self, model_path, device=None):
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.color_to_idx, self.idx_to_color = create_color_mapping()
        
        # Load model
        self.model = ConditionalUNet(n_channels=3, n_classes=3, num_colors=8)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Available colors: {list(self.color_to_idx.keys())}")
    
    def predict(self, image_path, color_name):
        
        if color_name not in self.color_to_idx:
            raise ValueError(f"Color '{color_name}' not supported. Available colors: {list(self.color_to_idx.keys())}")
        
        
        input_image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(input_image).unsqueeze(0).to(self.device)
        
        
        color_idx = torch.tensor([self.color_to_idx[color_name]], dtype=torch.long).to(self.device)
        
        
        with torch.no_grad():
            prediction = self.model(input_tensor, color_idx)
        
        
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
    
        input_img, pred_img = self.predict(image_path, color_name)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        
        axes[0].imshow(input_img)
        axes[0].set_title(f'Input: {os.path.basename(image_path)}')
        axes[0].axis('off')
        
        
        axes[1].imshow(pred_img)
        axes[1].set_title(f'Predicted: {color_name}')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_color_variations(self, image_path, save_path=None):
        
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
       
        input_images, predicted_images = self.predict_batch(image_paths, color_names)
        
        n_samples = len(image_paths)
        fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
        
        if n_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(n_samples):
            
            axes[0, i].imshow(input_images[i])
            axes[0, i].set_title(f'Input: {os.path.basename(image_paths[i])}')
            axes[0, i].axis('off')
            
            
            axes[1, i].imshow(predicted_images[i])
            axes[1, i].set_title(f'Predicted: {color_names[i]}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison grid saved to {save_path}")
        
        plt.show()


def demo_inference(model_path, dataset_root):
   
    inferencer = PolygonColorInference(model_path)
    
   
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
    
   
    sample_images = [img for img in sample_images if os.path.exists(img)]
    
    if not sample_images:
        print("No sample images found in validation set")
        return
    
    print("=== Polygon Coloring Inference Demo ===\n")
    
    
    print("1. Single Prediction Demo")
    if sample_images:
        inferencer.visualize_prediction(sample_images[0], 'red', 'demo_single_prediction.png')
    
   
    print("\n2. Color Variations Demo")
    if sample_images:
        inferencer.visualize_color_variations(sample_images[0], 'demo_color_variations.png')
    
  
    print("\n3. Multiple Predictions Demo")
    if len(sample_images) >= 4:
        colors = ['blue', 'green', 'purple', 'orange']
        inferencer.create_comparison_grid(
            sample_images[:4], 
            colors, 
            'demo_comparison_grid.png'
        )
    
    print("\nDemo completed! Check the generated visualization files.")


def interactive_inference(model_path, dataset_root):
    
    
    print("\n" + "="*60)
    print(" INTERACTIVE POLYGON COLORING AI")
    print("="*60)
    
    
    try:
        inferencer = PolygonColorInference(model_path)
        print(" Model loaded successfully!")
    except Exception as e:
        print(f" Error loading model: {e}")
        return
    
    
    val_inputs_dir = os.path.join(dataset_root, 'validation', 'inputs')
    if not os.path.exists(val_inputs_dir):
        val_inputs_dir = os.path.join(dataset_root, 'training', 'inputs')
    
    if not os.path.exists(val_inputs_dir):
        print(f" Dataset directory not found: {dataset_root}")
        return
    
    
    available_shapes = [f for f in os.listdir(val_inputs_dir) if f.endswith('.png')]
    available_shapes.sort()
    
    if not available_shapes:
        print(" No polygon shapes found in dataset!")
        return
    
    
    available_colors = ['blue', 'cyan', 'green', 'magenta', 'orange', 'purple', 'red', 'yellow']
    
    print(f"\n Found {len(available_shapes)} polygon shapes and {len(available_colors)} colors")
    
    while True:
        print("\n" + "-"*50)
        print(" AVAILABLE POLYGON SHAPES:")
        print("-"*50)
        
        for i, shape in enumerate(available_shapes, 1):
            shape_name = shape.replace('.png', '').title()
            print(f"{i:2d}. {shape_name}")
        
        print(f"{len(available_shapes)+1:2d}. Show all color variations for a shape")
        print(f"{len(available_shapes)+2:2d}. Exit")
        
        try:
            shape_choice = input("\n Choose a polygon shape (number): ").strip()
            shape_idx = int(shape_choice) - 1
            
            if shape_idx == len(available_shapes) + 1:  
                print("\n Thank you for using Polygon Coloring AI!")
                break
            elif shape_idx == len(available_shapes):  
                
                print("\n Choose shape for color variations:")
                for i, shape in enumerate(available_shapes, 1):
                    shape_name = shape.replace('.png', '').title()
                    print(f"{i:2d}. {shape_name}")
                
                var_choice = input("\nShape number: ").strip()
                var_idx = int(var_choice) - 1
                
                if 0 <= var_idx < len(available_shapes):
                    selected_shape = available_shapes[var_idx]
                    shape_path = os.path.join(val_inputs_dir, selected_shape)
                    shape_name = selected_shape.replace('.png', '').title()
                    
                    print(f"\n Generating all color variations for {shape_name}...")
                    save_path = f"color_variations_{selected_shape.replace('.png', '')}.png"
                    inferencer.visualize_color_variations(shape_path, save_path)
                    print(f" Color variations saved as: {save_path}")
                continue
            elif 0 <= shape_idx < len(available_shapes):
                selected_shape = available_shapes[shape_idx]
                shape_path = os.path.join(val_inputs_dir, selected_shape)
                shape_name = selected_shape.replace('.png', '').title()
                
                print(f"\n Selected: {shape_name}")
            else:
                print(" Invalid choice! Please try again.")
                continue
                
        except (ValueError, IndexError):
            print(" Invalid input! Please enter a number.")
            continue
        
        
        print("\n" + "-"*50)
        print(" AVAILABLE COLORS:")
        print("-"*50)
        
        for i, color in enumerate(available_colors, 1):
            print(f"{i:2d}. {color.title()}")
        
       
        try:
            color_choice = input("\n Choose a color (number): ").strip()
            color_idx = int(color_choice) - 1
            
            if 0 <= color_idx < len(available_colors):
                selected_color = available_colors[color_idx]
                print(f" Selected: {selected_color.title()}")
            else:
                print(" Invalid choice! Please try again.")
                continue
                
        except (ValueError, IndexError):
            print(" Invalid input! Please enter a number.")
            continue
        
        
        print(f"\n Generating {selected_color} {shape_name}...")
        
        try:

            output_filename = f"{selected_color}_{selected_shape.replace('.png', '')}_prediction.png"
            
           
            inferencer.visualize_prediction(shape_path, selected_color, output_filename)
            
            print(f"\n SUCCESS!")
            print(f" Output saved as: {output_filename}")
            print(f" Generated: {selected_color.title()} {shape_name}")
            
        except Exception as e:
            print(f" Error generating prediction: {e}")
        

        print("\n" + "-"*50)
        continue_choice = input(" Generate another polygon? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            print("\n Thank you for using Polygon Coloring AI!")
            break

def main():
    parser = argparse.ArgumentParser(description='Polygon Coloring Inference')
    parser.add_argument('--model_path', type=str, default='results/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--dataset_root', type=str, default='dataset',
                       help='Root directory of dataset')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo with predefined examples')
    parser.add_argument('--interactive', action='store_true',
                       help='Run interactive mode to choose shapes and colors')
    parser.add_argument('--image_path', type=str,
                       help='Path to input image for single prediction')
    parser.add_argument('--color', type=str,
                       help='Color name for single prediction')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Directory to save output images')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_inference(args.model_path, args.dataset_root)
    elif args.demo:
        demo_inference(args.model_path, args.dataset_root)
    elif args.image_path and args.color:
       
        inferencer = PolygonColorInference(args.model_path)
        inferencer.visualize_prediction(args.image_path, args.color, 
                                       os.path.join(args.output_dir, 'prediction.png'))
    else:
        print("\n Polygon Coloring AI - Usage Options:")
        print("\n1. Interactive Mode (Recommended):")
        print("   python inference.py --interactive")
        print("\n2. Demo Mode:")
        print("   python inference.py --demo")
        print("\n3. Single Prediction:")
        print("   python inference.py --image_path path/to/image.png --color blue")
        print("\n4. Help:")
        print("   python inference.py --help")


if __name__ == '__main__':
    main()
