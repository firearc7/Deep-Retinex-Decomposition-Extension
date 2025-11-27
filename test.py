"""
Inference script for Deep Retinex Decomposition Network

This script:
1. Loads a trained model
2. Processes test images
3. Saves decomposition results (R, I, I_delta)
4. Optionally applies DIP enhancements

Usage:
    # Basic inference (no DIP)
    python test.py --checkpoint checkpoints/retinexnet_best.pt --input_dir data/test/low --output_dir results/inference
    
    # With DIP enhancement
    python test.py --checkpoint checkpoints/retinexnet_best.pt --input_dir data/test/low --output_dir results/inference --enhance balanced
"""

import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
import json
from datetime import datetime

from src.model.retinexnet import RetinexNet
from src.enhancements import EnhancementPipeline, EnhancementFactory
from config import PROJECT_ROOT


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    model = RetinexNet().to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")
        print(f"Trained for {checkpoint.get('epoch', 'unknown')} epochs")
        print(f"Training loss: {checkpoint.get('loss', 'unknown'):.4f}")
    else:
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    
    model.eval()
    return model


def load_image(image_path):
    """Load image and convert to tensor"""
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0)
    return img_tensor, img_np


def tensor_to_numpy(tensor):
    """Convert tensor to numpy array (H, W, C)"""
    if len(tensor.shape) == 4:  # Batch
        tensor = tensor[0]
    img_np = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0, 1)
    return img_np


def save_image(img_np, save_path):
    """Save numpy array as image"""
    img_np = (img_np * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    img_pil.save(save_path)


def process_single_image(model, image_path, output_dir, device, enhance_config=None):
    """Process a single image through the model"""
    # Load image
    img_tensor, img_np = load_image(image_path)
    img_tensor = img_tensor.to(device)
    
    # Get image name
    img_name = Path(image_path).stem
    
    # Create output directory for this image
    img_output_dir = Path(output_dir) / img_name
    img_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save input image
    save_image(img_np, img_output_dir / 'input.png')
    
    with torch.no_grad():
        # Decomposition
        R_low, I_low = model.DecomNet(img_tensor)
        
        # Illumination adjustment
        I_delta = model.RelightNet(I_low, img_tensor)
        
        # Final output (without enhancement)
        S = R_low * I_delta
    
    # Convert to numpy
    R_np = tensor_to_numpy(R_low)
    I_np = tensor_to_numpy(I_low)
    I_delta_np = tensor_to_numpy(I_delta)
    S_np = tensor_to_numpy(S)
    
    # Save decomposition results
    save_image(R_np, img_output_dir / 'reflectance.png')
    save_image(I_np, img_output_dir / 'illumination_low.png')
    save_image(I_delta_np, img_output_dir / 'illumination_enhanced.png')
    save_image(S_np, img_output_dir / 'output_no_dip.png')
    
    # Apply DIP enhancement if requested
    if enhance_config:
        pipeline = EnhancementPipeline(enhance_config)
        enhanced_output, enhancement_results = pipeline.process_full_pipeline(
            R_np, I_np, I_delta_np
        )
        
        # Save enhanced results
        save_image(enhanced_output, img_output_dir / 'output_with_dip.png')
        
        # Save intermediate DIP results
        if 'illumination_enhanced' in enhancement_results:
            save_image(enhancement_results['illumination_enhanced'], 
                      img_output_dir / 'illumination_dip_enhanced.png')
        
        if 'reflectance_enhanced' in enhancement_results:
            save_image(enhancement_results['reflectance_enhanced'],
                      img_output_dir / 'reflectance_dip_enhanced.png')
        
        return {
            'image_name': img_name,
            'input': img_np,
            'reflectance': R_np,
            'illumination_low': I_np,
            'illumination_enhanced': I_delta_np,
            'output_no_dip': S_np,
            'output_with_dip': enhanced_output,
            'enhancement_results': enhancement_results
        }
    else:
        return {
            'image_name': img_name,
            'input': img_np,
            'reflectance': R_np,
            'illumination_low': I_np,
            'illumination_enhanced': I_delta_np,
            'output_no_dip': S_np
        }


def compute_metrics(img1, img2):
    """Compute image quality metrics"""
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    
    # Convert to uint8 for some metrics
    img1_uint8 = (img1 * 255).astype(np.uint8)
    img2_uint8 = (img2 * 255).astype(np.uint8)
    
    # PSNR
    psnr_value = psnr(img1, img2, data_range=1.0)
    
    # SSIM
    ssim_value = ssim(img1, img2, multichannel=True, channel_axis=2, data_range=1.0)
    
    # Entropy
    def calculate_entropy(img):
        hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 1))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    entropy1 = calculate_entropy(img1)
    entropy2 = calculate_entropy(img2)
    
    # Contrast (RMS contrast)
    def calculate_contrast(img):
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        return gray.std()
    
    contrast1 = calculate_contrast(img1)
    contrast2 = calculate_contrast(img2)
    
    return {
        'psnr': psnr_value,
        'ssim': ssim_value,
        'entropy_input': entropy1,
        'entropy_output': entropy2,
        'contrast_input': contrast1,
        'contrast_output': contrast2
    }


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    # Setup enhancement if requested
    enhance_config = None
    if args.enhance:
        print(f"\nEnhancement preset: {args.enhance}")
        enhance_config = EnhancementFactory.create_config(args.enhance)
    else:
        print("\nNo DIP enhancement applied")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get input images
    input_dir = Path(args.input_dir)
    image_files = list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg'))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {input_dir}")
    
    print(f"\nProcessing {len(image_files)} images from {input_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    # Process images
    all_results = []
    all_metrics = []
    
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            result = process_single_image(
                model, image_path, output_dir, device, enhance_config
            )
            
            # Compute metrics if reference is available
            if args.compute_metrics and 'output_with_dip' in result:
                metrics = compute_metrics(result['input'], result['output_with_dip'])
                metrics['image_name'] = result['image_name']
                all_metrics.append(metrics)
            
            all_results.append({
                'image_name': result['image_name'],
                'input_path': str(image_path),
                'output_path': str(output_dir / result['image_name'])
            })
            
        except Exception as e:
            print(f"\nError processing {image_path}: {str(e)}")
            continue
    
    # Save summary
    summary = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'checkpoint': args.checkpoint,
        'input_dir': args.input_dir,
        'output_dir': args.output_dir,
        'enhancement_preset': args.enhance,
        'num_images': len(all_results),
        'results': all_results
    }
    
    if all_metrics:
        # Compute average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if key != 'image_name':
                values = [m[key] for m in all_metrics]
                avg_metrics[f'avg_{key}'] = np.mean(values)
        
        summary['metrics'] = {
            'per_image': all_metrics,
            'average': avg_metrics
        }
    
    summary_path = output_dir / 'inference_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("\n" + "=" * 50)
    print("Inference completed!")
    print(f"Processed {len(all_results)} images")
    print(f"Results saved to {output_dir}")
    print(f"Summary saved to {summary_path}")
    
    if all_metrics:
        print("\nAverage Metrics:")
        for key, value in avg_metrics.items():
            print(f"  {key}: {value:.4f}")
    
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Deep Retinex Decomposition Network')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    
    # Enhancement options
    parser.add_argument('--enhance', type=str, default=None,
                        choices=['none', 'minimal', 'balanced', 'aggressive', 
                                'illumination_only', 'output_only'],
                        help='Enhancement preset to apply (default: no enhancement)')
    
    # Metrics
    parser.add_argument('--compute_metrics', action='store_true',
                        help='Compute quality metrics')
    
    args = parser.parse_args()
    
    main(args)
