"""
Compare different DIP enhancement presets

This script:
1. Loads a trained model
2. Runs inference on test images
3. Applies different DIP enhancement presets
4. Computes quality metrics for each preset
5. Generates comparison report (CSV + HTML)

Usage:
    python compare_enhancements.py \
        --checkpoint checkpoints/retinexnet_best.pt \
        --input_dir data/real/test/low \
        --output_dir results/comparison \
        --presets balanced aggressive minimal
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv
import json
from datetime import datetime

from src.model.retinexnet import RetinexNet
from src.enhancements import EnhancementPipeline, EnhancementFactory
from test import load_image, tensor_to_numpy, save_image, compute_metrics


def compare_presets_on_image(model, image_path, presets, device):
    """
    Apply different enhancement presets to a single image and compare
    
    Returns:
        dict: Results for each preset including metrics and images
    """
    # Load image
    img_tensor, img_np = load_image(image_path)
    img_tensor = img_tensor.to(device)
    img_name = Path(image_path).stem
    
    results = {
        'image_name': img_name,
        'input': img_np,
        'presets': {}
    }
    
    with torch.no_grad():
        # Run model inference once
        R_low, I_low = model.DecomNet(img_tensor)
        I_delta = model.RelightNet(I_low, img_tensor)
        S_baseline = R_low * I_delta
    
    # Convert to numpy
    R_np = tensor_to_numpy(R_low)
    I_np = tensor_to_numpy(I_low)
    I_delta_np = tensor_to_numpy(I_delta)
    S_baseline_np = tensor_to_numpy(S_baseline)
    
    # Store baseline (no enhancement)
    results['baseline'] = {
        'output': S_baseline_np,
        'R': R_np,
        'I': I_np,
        'I_delta': I_delta_np
    }
    
    # Apply each preset
    for preset_name in presets:
        print(f"  Applying preset: {preset_name}")
        
        if preset_name == 'baseline' or preset_name == 'none':
            # Already have baseline
            results['presets'][preset_name] = {
                'output': S_baseline_np,
                'metrics': {}
            }
        else:
            # Apply DIP enhancement
            config = EnhancementFactory.create_config(preset_name)
            pipeline = EnhancementPipeline(config)
            
            enhanced_output, enhancement_results = pipeline.process_full_pipeline(
                R_np, I_np, I_delta_np
            )
            
            results['presets'][preset_name] = {
                'output': enhanced_output,
                'enhancement_results': enhancement_results,
                'metrics': {}
            }
    
    return results


def compute_comparison_metrics(results):
    """
    Compute quality metrics for each preset
    
    Metrics computed:
    - Entropy (information content)
    - Contrast (RMS contrast)
    - Sharpness (Laplacian variance)
    - Colorfulness
    - Brightness (mean luminance)
    - Processing time (from enhancement results)
    """
    from experiments.experiment_framework import QualityMetrics
    
    input_img = results['input']
    
    # Compute metrics for baseline
    baseline_metrics = QualityMetrics.calculate_all_metrics(
        results['baseline']['output'],
        reference=input_img
    )
    results['baseline']['metrics'] = baseline_metrics
    
    # Compute metrics for each preset
    for preset_name, preset_data in results['presets'].items():
        output_img = preset_data['output']
        metrics = QualityMetrics.calculate_all_metrics(output_img, reference=input_img)
        preset_data['metrics'] = metrics
    
    return results


def generate_comparison_report(all_results, output_dir, presets):
    """
    Generate CSV and HTML comparison reports
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare CSV data
    csv_data = []
    
    for result in all_results:
        img_name = result['image_name']
        
        # Baseline metrics
        baseline_metrics = result['baseline']['metrics']
        
        for preset_name in presets:
            if preset_name in result['presets']:
                preset_metrics = result['presets'][preset_name]['metrics']
                
                # Calculate improvements over baseline
                improvements = {}
                for metric_name in ['entropy', 'contrast', 'sharpness', 'colorfulness']:
                    baseline_val = baseline_metrics.get(metric_name, 0)
                    preset_val = preset_metrics.get(metric_name, 0)
                    if baseline_val > 0:
                        improvement = ((preset_val - baseline_val) / baseline_val) * 100
                        improvements[f'{metric_name}_improvement'] = improvement
                
                csv_data.append({
                    'image': img_name,
                    'preset': preset_name,
                    'entropy': preset_metrics.get('entropy', 0),
                    'contrast': preset_metrics.get('contrast', 0),
                    'sharpness': preset_metrics.get('sharpness', 0),
                    'colorfulness': preset_metrics.get('colorfulness', 0),
                    'brightness': preset_metrics.get('brightness', 0),
                    **improvements
                })
    
    # Save CSV
    csv_path = output_dir / 'comparison_results.csv'
    if csv_data:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"\nCSV report saved to: {csv_path}")
    
    # Calculate average metrics per preset
    preset_averages = {}
    for preset_name in presets:
        preset_metrics_list = []
        for result in all_results:
            if preset_name in result['presets']:
                preset_metrics_list.append(result['presets'][preset_name]['metrics'])
        
        if preset_metrics_list:
            avg_metrics = {}
            for metric_name in ['entropy', 'contrast', 'sharpness', 'colorfulness', 'brightness']:
                values = [m.get(metric_name, 0) for m in preset_metrics_list]
                avg_metrics[metric_name] = np.mean(values)
            preset_averages[preset_name] = avg_metrics
    
    # Generate HTML report
    html_path = output_dir / 'comparison_report.html'
    generate_html_report(html_path, preset_averages, presets, all_results)
    print(f"HTML report saved to: {html_path}")
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY - Average Metrics Across All Images")
    print("=" * 80)
    print(f"\n{'Preset':<20} {'Entropy':<12} {'Contrast':<12} {'Sharpness':<12} {'Colorfulness':<15}")
    print("-" * 80)
    
    for preset_name in presets:
        if preset_name in preset_averages:
            metrics = preset_averages[preset_name]
            print(f"{preset_name:<20} "
                  f"{metrics['entropy']:<12.4f} "
                  f"{metrics['contrast']:<12.4f} "
                  f"{metrics['sharpness']:<12.4f} "
                  f"{metrics['colorfulness']:<15.4f}")
    
    print("=" * 80)
    
    # Save summary JSON (convert numpy types to Python types)
    def convert_to_python_types(obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    summary = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'num_images': len(all_results),
        'presets_tested': presets,
        'average_metrics': convert_to_python_types(preset_averages)
    }
    
    summary_path = output_dir / 'comparison_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"\nSummary JSON saved to: {summary_path}")


def generate_html_report(html_path, preset_averages, presets, all_results):
    """Generate HTML comparison report with charts"""
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>DIP Enhancement Comparison Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric-card {{
            display: inline-block;
            width: 200px;
            margin: 10px;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .metric-label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .best {{
            background-color: #4CAF50 !important;
            font-weight: bold;
        }}
        .summary {{
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .preset-section {{
            margin: 30px 0;
            padding: 20px;
            border-left: 4px solid #4CAF50;
            background-color: #f9f9f9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 DIP Enhancement Comparison Report</h1>
        <div class="summary">
            <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
            <strong>Images Tested:</strong> {len(all_results)}<br>
            <strong>Presets Compared:</strong> {', '.join(presets)}
        </div>
        
        <h2>📊 Average Metrics Comparison</h2>
        <table>
            <tr>
                <th>Preset</th>
                <th>Entropy</th>
                <th>Contrast</th>
                <th>Sharpness</th>
                <th>Colorfulness</th>
                <th>Brightness</th>
            </tr>
"""
    
    # Find best values for each metric
    best_values = {}
    for metric in ['entropy', 'contrast', 'sharpness', 'colorfulness']:
        best_values[metric] = max(preset_averages.values(), 
                                   key=lambda x: x.get(metric, 0))
    
    for preset_name in presets:
        if preset_name in preset_averages:
            metrics = preset_averages[preset_name]
            html_content += f"""
            <tr>
                <td><strong>{preset_name}</strong></td>
                <td class="{'best' if metrics == best_values['entropy'] else ''}">{metrics['entropy']:.4f}</td>
                <td class="{'best' if metrics == best_values['contrast'] else ''}">{metrics['contrast']:.4f}</td>
                <td class="{'best' if metrics == best_values['sharpness'] else ''}">{metrics['sharpness']:.4f}</td>
                <td class="{'best' if metrics == best_values['colorfulness'] else ''}">{metrics['colorfulness']:.4f}</td>
                <td>{metrics['brightness']:.4f}</td>
            </tr>
"""
    
    html_content += """
        </table>
        
        <h2>🎯 Recommendations</h2>
        <div class="preset-section">
"""
    
    # Find best preset for each metric
    for metric_name, display_name in [('entropy', 'Information Content'), 
                                       ('contrast', 'Contrast'), 
                                       ('sharpness', 'Sharpness')]:
        best_preset = max(preset_averages.items(), 
                         key=lambda x: x[1].get(metric_name, 0))
        html_content += f"""
            <p><strong>Best for {display_name}:</strong> {best_preset[0]} 
            ({best_preset[1][metric_name]:.4f})</p>
"""
    
    html_content += """
        </div>
        
        <h2>📋 Per-Image Results</h2>
"""
    
    for result in all_results:
        html_content += f"""
        <h3>Image: {result['image_name']}</h3>
        <table>
            <tr>
                <th>Preset</th>
                <th>Entropy</th>
                <th>Contrast</th>
                <th>Sharpness</th>
                <th>Colorfulness</th>
            </tr>
"""
        for preset_name in presets:
            if preset_name in result['presets']:
                metrics = result['presets'][preset_name]['metrics']
                html_content += f"""
            <tr>
                <td>{preset_name}</td>
                <td>{metrics.get('entropy', 0):.4f}</td>
                <td>{metrics.get('contrast', 0):.4f}</td>
                <td>{metrics.get('sharpness', 0):.4f}</td>
                <td>{metrics.get('colorfulness', 0):.4f}</td>
            </tr>
"""
        html_content += """
        </table>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    with open(html_path, 'w') as f:
        f.write(html_content)


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    model = RetinexNet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded (trained for {checkpoint.get('epoch', 'unknown')} epochs)")
    
    # Get presets to compare
    presets = args.presets if args.presets else ['baseline', 'minimal', 'balanced', 'aggressive']
    print(f"\nComparing presets: {', '.join(presets)}")
    
    # Get test images
    input_dir = Path(args.input_dir)
    image_files = list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg'))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {input_dir}")
    
    print(f"\nProcessing {len(image_files)} images from {input_dir}")
    print("-" * 80)
    
    # Process each image
    all_results = []
    
    for image_path in tqdm(image_files, desc="Processing images"):
        print(f"\nProcessing: {image_path.name}")
        
        try:
            result = compare_presets_on_image(model, image_path, presets, device)
            result = compute_comparison_metrics(result)
            all_results.append(result)
            
            # Save individual image results
            if args.save_images:
                img_output_dir = Path(args.output_dir) / result['image_name']
                img_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save input
                save_image(result['input'], img_output_dir / 'input.png')
                
                # Save baseline
                save_image(result['baseline']['output'], img_output_dir / 'baseline.png')
                
                # Save each preset output
                for preset_name, preset_data in result['presets'].items():
                    save_image(preset_data['output'], 
                             img_output_dir / f'{preset_name}.png')
                
                print(f"  Saved outputs to: {img_output_dir}")
        
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    # Generate comparison report
    print("\n" + "=" * 80)
    print("Generating comparison report...")
    print("=" * 80)
    generate_comparison_report(all_results, args.output_dir, presets)
    
    print("\n✅ Comparison complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare different DIP enhancement presets',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save comparison results')
    
    # Optional arguments
    parser.add_argument('--presets', nargs='+', 
                        default=['baseline', 'minimal', 'balanced', 'aggressive'],
                        help='Presets to compare (default: baseline minimal balanced aggressive)')
    parser.add_argument('--save_images', action='store_true',
                        help='Save output images for each preset')
    
    args = parser.parse_args()
    
    main(args)
