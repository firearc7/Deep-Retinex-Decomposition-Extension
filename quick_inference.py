"""
Quick Inference Script for Custom Images

This script allows you to quickly test the trained model on custom images
with different DIP enhancement configurations.

Usage Examples:
    # Basic inference with balanced preset
    python quick_inference.py --input image.jpg --preset balanced
    
    # Try different presets
    python quick_inference.py --input image.jpg --preset aggressive
    
    # Process entire directory
    python quick_inference.py --input /path/to/images/ --preset balanced --output results/custom
    
    # Compare multiple presets side-by-side
    python quick_inference.py --input image.jpg --compare minimal balanced aggressive
    
    # Use custom checkpoint
    python quick_inference.py --input image.jpg --checkpoint my_model.pt --preset balanced
    
    # Save intermediate outputs (R, I, I_delta)
    python quick_inference.py --input image.jpg --preset balanced --save_intermediates
"""

import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from tqdm import tqdm
import json
import time

from src.model.retinexnet import RetinexNet
from src.enhancements.pipeline import EnhancementPipeline, EnhancementFactory
from src.enhancements.advanced_pipeline import AdvancedEnhancementPipeline, AdvancedEnhancementFactory
from src.enhancements.preprocessing import ImagePreprocessor
from experiments.experiment_framework import QualityMetrics


def parse_args():
    parser = argparse.ArgumentParser(description='Quick inference on custom images')
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path or directory')
    parser.add_argument('--output', type=str, default='results/quick_inference',
                        help='Output directory')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/retinexnet_best.pt',
                        help='Model checkpoint path')
    
    # Enhancement options
    parser.add_argument('--preset', type=str, default='balanced',
                       choices=['baseline', 'minimal', 'balanced', 'aggressive', 
                               'illumination_only', 'output_only', 'custom',
                               'minimal_plus', 'balanced_plus', 'aggressive_plus',
                               'quality_focused', 'speed_focused', 
                               'outdoor_optimized', 'indoor_optimized'],
                       help='Enhancement preset (add _plus for advanced pipeline with preprocessing)')
    
    # Comparison mode
    parser.add_argument('--compare', nargs='+', 
                        choices=['baseline', 'minimal', 'balanced', 'aggressive', 
                                'illumination_only', 'output_only',
                                'minimal_plus', 'balanced_plus', 'aggressive_plus',
                                'quality_focused', 'speed_focused',
                                'outdoor_optimized', 'indoor_optimized'],
                        help='Compare multiple presets (creates comparison grid)')
    
    # Additional options
    parser.add_argument('--save_intermediates', action='store_true',
                        help='Save R, I, I_delta intermediate outputs')
    parser.add_argument('--compute_metrics', action='store_true',
                        help='Compute and save quality metrics')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--max_size', type=int, default=1024,
                        help='Max dimension (resize if larger to save GPU memory, 0=no resize)')
    
    # Custom enhancement parameters (for preset='custom')
    parser.add_argument('--clahe', action='store_true', help='Enable CLAHE')
    parser.add_argument('--bilateral', action='store_true', help='Enable Bilateral Filter')
    parser.add_argument('--gamma', action='store_true', help='Enable Adaptive Gamma')
    parser.add_argument('--unsharp', action='store_true', help='Enable Unsharp Masking')
    parser.add_argument('--color_balance', action='store_true', help='Enable Color Balance')
    parser.add_argument('--guided_filter', action='store_true', help='Enable Guided Filter')
    
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load trained RetinexNet model"""
    print(f"Loading model from {checkpoint_path}...")
    model = RetinexNet().to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded (trained for {checkpoint.get('epoch', 'unknown')} epochs)")
    return model


def load_image(image_path, max_size=None):
    """Load image and convert to tensor
    
    Args:
        image_path: Path to image file
        max_size: If set, resize image so largest dimension is max_size (preserves aspect ratio)
    """
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    
    # Resize if needed
    if max_size and max_size > 0:
        w, h = img.size
        if max(w, h) > max_size:
            if w > h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            print(f"  → Resized from {original_size[0]}×{original_size[1]} to {new_w}×{new_h} (saves GPU memory)")
    
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0)
    return img_tensor, img_np, img


def save_image(img_np, save_path):
    """Save numpy array as image"""
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    img_pil.save(save_path, quality=95)


def tensor_to_numpy(tensor):
    """Convert tensor to numpy array"""
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    
    np_array = tensor.detach().cpu().numpy()
    
    # Handle single channel (grayscale) images
    if np_array.shape[0] == 1:
        # (1, H, W) -> (H, W, 1) -> (H, W, 3) by repeating
        np_array = np.repeat(np_array.transpose(1, 2, 0), 3, axis=2)
    else:
        # (C, H, W) -> (H, W, C)
        np_array = np_array.transpose(1, 2, 0)
    
    return np_array


def create_comparison_grid(images_dict, labels, output_path):
    """Create side-by-side comparison grid"""
    num_images = len(images_dict)
    
    # Resize all images to same height
    target_height = 400
    resized_images = []
    
    for label in labels:
        img = images_dict[label]
        h, w = img.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Add label
        labeled = resized.copy()
        cv2.putText(labeled, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(labeled, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (0, 0, 0), 1, cv2.LINE_AA)
        
        resized_images.append(labeled)
    
    # Concatenate horizontally
    grid = np.hstack(resized_images)
    
    # Add title
    title_height = 60
    title_img = np.ones((title_height, grid.shape[1], 3), dtype=np.uint8) * 240
    cv2.putText(title_img, f"Comparison: {' vs '.join(labels)}", 
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
    
    final_grid = np.vstack([title_img, grid])
    
    # Save
    final_grid_rgb = cv2.cvtColor(final_grid, cv2.COLOR_BGR2RGB)
    save_image(final_grid_rgb.astype(np.float32) / 255.0, output_path)
    print(f"  ✓ Comparison grid saved: {output_path}")


def process_single_image(model, image_path, output_dir, args):
    """Process a single image"""
    # Clear GPU cache before processing
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    
    # Load image
    img_tensor, img_np, img_pil = load_image(image_path, max_size=args.max_size)
    img_tensor = img_tensor.to(args.device)
    
    # Get image name
    img_name = Path(image_path).stem
    img_output_dir = Path(output_dir) / img_name
    img_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing: {Path(image_path).name}")
    
    # Save input
    save_image(img_np, img_output_dir / 'input.png')
    
    # Model inference
    start_time = time.time()
    with torch.no_grad():
        R_low, I_low = model.DecomNet(img_tensor)
        I_delta = model.RelightNet(I_low, img_tensor)
        S_baseline = R_low * I_delta
    
    inference_time = time.time() - start_time
    
    # Convert to numpy
    R_np = tensor_to_numpy(R_low)
    I_np = tensor_to_numpy(I_low)
    I_delta_np = tensor_to_numpy(I_delta)
    S_baseline_np = tensor_to_numpy(S_baseline)
    
    # Free GPU memory after inference
    del R_low, I_low, I_delta, S_baseline, img_tensor
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    
    # Always save baseline (model-only output, no DIP)
    save_image(S_baseline_np, img_output_dir / 'output_baseline.png')
    
    # Save intermediates if requested
    if args.save_intermediates:
        save_image(R_np, img_output_dir / 'reflectance.png')
        save_image(I_np, img_output_dir / 'illumination_original.png')
        save_image(I_delta_np, img_output_dir / 'illumination_enhanced.png')
        print(f"  ✓ Intermediates saved")
    
    # Apply enhancements
    results = {'Input': img_np, 'Baseline': S_baseline_np}
    
    if args.compare:
        # Multiple presets comparison
        presets = args.compare
        for preset_name in presets:
            # Check if advanced preset (ends with _plus or is optimized)
            is_advanced = (preset_name.endswith('_plus') or 
                         'focused' in preset_name or 
                         'optimized' in preset_name)
            
            if is_advanced:
                config = AdvancedEnhancementFactory.create_config(preset_name)
                pipeline = AdvancedEnhancementPipeline(config)
            else:
                config = EnhancementFactory.create_config(preset_name)
                pipeline = EnhancementPipeline(config)
            
            start_time = time.time()
            enhanced, _ = pipeline.process_full_pipeline(R_np, I_np, I_delta_np)
            enhance_time = time.time() - start_time
            
            results[preset_name.capitalize()] = enhanced
            save_image(enhanced, img_output_dir / f'output_{preset_name}.png')
            print(f"  ✓ {preset_name}: {enhance_time:.3f}s")
        
        # Create comparison grid
        labels = ['Input', 'Baseline'] + [p.capitalize() for p in presets]
        grid_path = img_output_dir / 'comparison_grid.png'
        create_comparison_grid(results, labels, grid_path)
        
    elif args.preset == 'custom':
        # Custom configuration
        custom_config = {
            'clahe': {'enabled': args.clahe},
            'bilateral_filter': {'enabled': args.bilateral},
            'adaptive_gamma': {'enabled': args.gamma},
            'unsharp_mask': {'enabled': args.unsharp},
            'color_balance': {'enabled': args.color_balance},
            'guided_filter': {'enabled': args.guided_filter}
        }
        
        pipeline = EnhancementPipeline(custom_config)
        start_time = time.time()
        enhanced, _ = pipeline.process_full_pipeline(R_np, I_np, I_delta_np)
        enhance_time = time.time() - start_time
        
        save_image(enhanced, img_output_dir / 'output_custom.png')
        print(f"  ✓ Custom enhancement: {enhance_time:.3f}s")
        results['Custom'] = enhanced
        
    else:
        # Single preset
        # Check if advanced preset
        is_advanced = (args.preset.endswith('_plus') or 
                      'focused' in args.preset or 
                      'optimized' in args.preset)
        
        if is_advanced:
            config = AdvancedEnhancementFactory.create_config(args.preset)
            pipeline = AdvancedEnhancementPipeline(config)
            print(f"  Using ADVANCED pipeline for '{args.preset}'")
        else:
            config = EnhancementFactory.create_config(args.preset)
            pipeline = EnhancementPipeline(config)
        
        start_time = time.time()
        enhanced, debug_results = pipeline.process_full_pipeline(R_np, I_np, I_delta_np)
        enhance_time = time.time() - start_time
        
        save_image(enhanced, img_output_dir / f'output_{args.preset}.png')
        print(f"  ✓ {args.preset}: {enhance_time:.3f}s")
        results[args.preset.capitalize()] = enhanced
    
    # Compute metrics if requested
    if args.compute_metrics:
        metrics_calculator = QualityMetrics()
        metrics_results = {}
        
        for name, img in results.items():
            if name != 'Input':
                metrics = metrics_calculator.calculate_all_metrics(img, img_np)
                metrics_results[name] = metrics
        
        # Save metrics
        metrics_file = img_output_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics_results, f, indent=2)
        print(f"  ✓ Metrics saved: {metrics_file}")
    
    total_time = time.time() - start_time + inference_time
    print(f"  ✓ Total time: {total_time:.3f}s (inference: {inference_time:.3f}s)")
    
    return results


def main():
    args = parse_args()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get input files
    input_path = Path(args.input)
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = sorted(list(input_path.glob('*.jpg')) + 
                           list(input_path.glob('*.png')) + 
                           list(input_path.glob('*.jpeg')))
    else:
        raise ValueError(f"Input path not found: {input_path}")
    
    print(f"\nFound {len(image_files)} image(s) to process")
    
    # Process images
    all_results = []
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            results = process_single_image(model, img_path, output_dir, args)
            all_results.append((img_path.name, results))
        except Exception as e:
            print(f"  ✗ Error processing {img_path.name}: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ Processing complete!")
    print(f"{'='*80}")
    print(f"Processed: {len(all_results)}/{len(image_files)} images")
    print(f"Output directory: {output_dir}")
    print(f"\nView results:")
    for img_name, _ in all_results:
        img_stem = Path(img_name).stem
        print(f"  - {output_dir}/{img_stem}/")


if __name__ == '__main__':
    main()
