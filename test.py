# Inference script for Deep Retinex Decomposition Network

import os
import argparse
import torch
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
    model = RetinexNet().to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")
        print(f"Trained for {checkpoint.get('epoch', 'unknown')} epochs")
    else:
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    
    model.eval()
    return model


def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0)
    return img_tensor, img_np


def tensor_to_numpy(tensor):
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    img_np = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(img_np, 0, 1)


def save_image(img_np, save_path):
    img_np = (img_np * 255).astype(np.uint8)
    Image.fromarray(img_np).save(save_path)


def process_single_image(model, image_path, output_dir, device, enhance_config=None):
    img_tensor, img_np = load_image(image_path)
    img_tensor = img_tensor.to(device)
    img_name = Path(image_path).stem
    
    img_output_dir = Path(output_dir) / img_name
    img_output_dir.mkdir(parents=True, exist_ok=True)
    
    save_image(img_np, img_output_dir / 'input.png')
    
    with torch.no_grad():
        R_low, I_low = model.DecomNet(img_tensor)
        I_delta = model.RelightNet(I_low, img_tensor)
        S = R_low * I_delta
    
    R_np = tensor_to_numpy(R_low)
    I_np = tensor_to_numpy(I_low)
    I_delta_np = tensor_to_numpy(I_delta)
    S_np = tensor_to_numpy(S)
    
    # Save decomposition results
    save_image(R_np, img_output_dir / 'reflectance.png')
    save_image(I_np, img_output_dir / 'illumination_low.png')
    save_image(I_delta_np, img_output_dir / 'illumination_enhanced.png')
    save_image(S_np, img_output_dir / 'output_baseline.png')
    
    # Apply DIP enhancement if requested
    if enhance_config:
        pipeline = EnhancementPipeline(enhance_config)
        enhanced_output, _ = pipeline.process_full_pipeline(R_np, I_np, I_delta_np)
        save_image(enhanced_output, img_output_dir / 'output_enhanced.png')
        
        return {
            'image_name': img_name,
            'input': img_np,
            'output_baseline': S_np,
            'output_enhanced': enhanced_output
        }
    
    return {
        'image_name': img_name,
        'input': img_np,
        'output_baseline': S_np
    }


def compute_metrics(img1, img2):
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    
    psnr_value = psnr(img1, img2, data_range=1.0)
    ssim_value = ssim(img1, img2, multichannel=True, channel_axis=2, data_range=1.0)
    
    # Entropy
    hist, _ = np.histogram(img2.flatten(), bins=256, range=(0, 1))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    
    # Contrast
    gray = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    contrast = gray.std()
    
    return {
        'psnr': psnr_value,
        'ssim': ssim_value,
        'entropy': entropy,
        'contrast': contrast
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = load_model(args.checkpoint, device)
    
    enhance_config = None
    if args.enhance:
        print(f"Enhancement preset: {args.enhance}")
        enhance_config = EnhancementFactory.create_config(args.enhance)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_dir = Path(args.input_dir)
    image_files = list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg'))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {input_dir}")
    
    print(f"Processing {len(image_files)} images")
    
    all_results = []
    all_metrics = []
    
    for image_path in tqdm(image_files, desc="Processing"):
        try:
            result = process_single_image(model, image_path, output_dir, device, enhance_config)
            
            if args.compute_metrics and 'output_enhanced' in result:
                metrics = compute_metrics(result['input'], result['output_enhanced'])
                metrics['image_name'] = result['image_name']
                all_metrics.append(metrics)
            
            all_results.append({
                'image_name': result['image_name'],
                'input_path': str(image_path)
            })
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'checkpoint': args.checkpoint,
        'enhancement_preset': args.enhance,
        'num_images': len(all_results),
    }
    
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if key != 'image_name':
                avg_metrics[f'avg_{key}'] = np.mean([m[key] for m in all_metrics])
        summary['metrics'] = avg_metrics
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nResults saved to {output_dir}")
    if all_metrics:
        print("Average Metrics:")
        for key, value in summary.get('metrics', {}).items():
            print(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Deep Retinex Network')
    
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--enhance', type=str, default=None,
                        choices=['none', 'minimal', 'balanced', 'aggressive', 
                                'illumination_only', 'output_only'])
    parser.add_argument('--compute_metrics', action='store_true')
    
    args = parser.parse_args()
    main(args)
