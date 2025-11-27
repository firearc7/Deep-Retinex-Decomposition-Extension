"""
Generate visual comparison images showing DIP enhancements

Creates side-by-side comparisons of:
- Input (low-light)
- Baseline (model output)
- Minimal preset
- Balanced preset
- Aggressive preset (optional)
"""

import os
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from src.model.retinexnet import RetinexNet
from src.enhancements import EnhancementPipeline, EnhancementFactory


def load_model(checkpoint_path, device):
    """Load trained model"""
    model = RetinexNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model


def load_image(image_path):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img).astype(np.float32) / 255.0
    return img_array


def tensor_to_numpy(tensor):
    """Convert PyTorch tensor to numpy array"""
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    img = tensor.cpu().detach().numpy()
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    elif img.shape[0] == 1:
        img = np.transpose(img, (1, 2, 0))
        img = np.repeat(img, 3, axis=2)
    return np.clip(img, 0, 1)


def numpy_to_pil(img_array):
    """Convert numpy array to PIL Image"""
    img_uint8 = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_uint8)


def add_label(img_pil, label, font_size=40):
    """Add label to image"""
    draw = ImageDraw.Draw(img_pil)
    
    # Try to load a nice font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Get text size
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Draw background rectangle
    padding = 10
    rect_coords = [
        (img_pil.width - text_width - 2*padding, 0),
        (img_pil.width, text_height + 2*padding)
    ]
    draw.rectangle(rect_coords, fill=(0, 0, 0, 180))
    
    # Draw text
    text_pos = (img_pil.width - text_width - padding, padding)
    draw.text(text_pos, label, fill=(255, 255, 255), font=font)
    
    return img_pil


def create_comparison_grid(images_dict, output_path, title="DIP Enhancement Comparison"):
    """Create a grid comparison image"""
    # Resize all images to same height
    target_height = 400
    resized_images = []
    labels = []
    
    for label, img in images_dict.items():
        pil_img = numpy_to_pil(img) if isinstance(img, np.ndarray) else img
        aspect_ratio = pil_img.width / pil_img.height
        new_width = int(target_height * aspect_ratio)
        pil_img = pil_img.resize((new_width, target_height), Image.Resampling.LANCZOS)
        pil_img = add_label(pil_img, label)
        resized_images.append(pil_img)
        labels.append(label)
    
    # Calculate grid dimensions
    num_images = len(resized_images)
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols
    
    # Create canvas
    max_width = max(img.width for img in resized_images)
    canvas_width = max_width * cols + 20 * (cols - 1)
    canvas_height = target_height * rows + 20 * (rows - 1) + 60  # Extra for title
    
    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Add title
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 30)
    except:
        title_font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = bbox[2] - bbox[0]
    title_pos = ((canvas_width - title_width) // 2, 10)
    draw.text(title_pos, title, fill=(0, 0, 0), font=title_font)
    
    # Place images
    y_offset = 60
    for i, img in enumerate(resized_images):
        row = i // cols
        col = i % cols
        x = col * (max_width + 20)
        y = y_offset + row * (target_height + 20)
        canvas.paste(img, (x, y))
    
    # Save
    canvas.save(output_path, quality=95)
    print(f"Comparison grid saved to: {output_path}")
    return canvas


def process_image(model, image_path, presets, device):
    """Process single image with all presets"""
    # Load image
    input_img = load_image(image_path)
    
    # Convert to tensor
    img_tensor = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Run model inference
    with torch.no_grad():
        R, I, I_delta, output = model.inference(img_tensor)
    
    # Convert to numpy
    R_np = tensor_to_numpy(R)
    I_np = tensor_to_numpy(I)
    I_delta_np = tensor_to_numpy(I_delta)
    output_np = tensor_to_numpy(output)
    
    results = {
        'Input (Low-light)': input_img,
        'Baseline (Model Only)': output_np
    }
    
    # Apply each preset
    for preset_name in presets:
        if preset_name == 'baseline':
            continue
        
        config = EnhancementFactory.create_config(preset_name)
        pipeline = EnhancementPipeline(config)
        
        enhanced_output, _ = pipeline.process_full_pipeline(R_np, I_np, I_delta_np)
        results[f'{preset_name.capitalize()} Preset'] = enhanced_output
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Generate visual DIP enhancement comparisons')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with test images')
    parser.add_argument('--output_dir', type=str, default='results/visual_comparison', help='Output directory')
    parser.add_argument('--num_images', type=int, default=5, help='Number of images to process')
    parser.add_argument('--presets', nargs='+', default=['minimal', 'balanced', 'aggressive'], 
                       help='Presets to test')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Get image files
    input_dir = Path(args.input_dir)
    image_files = sorted(list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg')))[:args.num_images]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Processing {len(image_files)} images...")
    print(f"Presets: {', '.join(args.presets)}\n")
    print("=" * 80)
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
        
        # Process image
        results = process_image(model, image_path, ['baseline'] + args.presets, device)
        
        # Create comparison grid
        output_path = output_dir / f'comparison_{image_path.stem}.png'
        create_comparison_grid(results, output_path, title=f"Enhancement Comparison - {image_path.name}")
        
        # Also save individual outputs
        img_output_dir = output_dir / image_path.stem
        img_output_dir.mkdir(exist_ok=True)
        
        for label, img in results.items():
            safe_label = label.replace(' ', '_').replace('(', '').replace(')', '').lower()
            save_path = img_output_dir / f'{safe_label}.png'
            pil_img = numpy_to_pil(img) if isinstance(img, np.ndarray) else img
            pil_img.save(save_path)
        
        print(f"  ✓ Grid saved: {output_path}")
        print(f"  ✓ Individual images saved to: {img_output_dir}")
    
    print("\n" + "=" * 80)
    print("✅ All comparison images generated successfully!")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nView comparison grids:")
    for image_path in image_files:
        print(f"  - {output_dir / f'comparison_{image_path.stem}.png'}")


if __name__ == '__main__':
    main()
