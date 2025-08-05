#!/usr/bin/env python3
"""
Generate a comprehensive visualization showing all 8 polygon shapes in all 8 colors
"""

import os
import torch
import matplotlib.pyplot as plt
from inference import PolygonColorInference
import numpy as np

def generate_all_shapes_visualization():
    """Generate a comprehensive grid showing all 8 shapes in all 8 colors"""
    
    # Initialize inference
    model_path = "results/best_model.pth"
    dataset_root = "dataset"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    inferencer = PolygonColorInference(model_path)
    
    # All 8 shapes and 8 colors
    shapes = ['circle', 'diamond', 'hexagon', 'octagon', 'pentagon', 'square', 'star', 'triangle']
    colors = ['blue', 'cyan', 'green', 'magenta', 'orange', 'purple', 'red', 'yellow']
    
    # Get input directory
    input_dir = os.path.join(dataset_root, 'validation', 'inputs')
    if not os.path.exists(input_dir):
        input_dir = os.path.join(dataset_root, 'training', 'inputs')
    
    # Create a large grid: 8 shapes x 8 colors
    fig, axes = plt.subplots(len(shapes), len(colors), figsize=(24, 24))
    fig.suptitle('All 8 Polygon Shapes in All 8 Colors', fontsize=20, fontweight='bold')
    
    for i, shape in enumerate(shapes):
        shape_path = os.path.join(input_dir, f'{shape}.png')
        
        if not os.path.exists(shape_path):
            print(f"Warning: {shape}.png not found, skipping...")
            continue
            
        for j, color in enumerate(colors):
            try:
                # Generate colored polygon
                input_img, pred_img = inferencer.predict(shape_path, color)
                
                # Display the prediction
                axes[i, j].imshow(pred_img)
                axes[i, j].axis('off')
                
                # Add title for first row (colors) and first column (shapes)
                if i == 0:
                    axes[i, j].set_title(color.title(), fontsize=12, fontweight='bold')
                if j == 0:
                    axes[i, j].set_ylabel(shape.title(), fontsize=12, fontweight='bold', rotation=90, labelpad=20)
                    
            except Exception as e:
                print(f"Error generating {shape} in {color}: {e}")
                axes[i, j].text(0.5, 0.5, f'Error\n{shape}\n{color}', 
                               ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('all_shapes_all_colors_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Comprehensive visualization saved to: all_shapes_all_colors_grid.png")
    print("📊 Grid shows: 8 shapes × 8 colors = 64 colored polygons!")

if __name__ == "__main__":
    generate_all_shapes_visualization()
