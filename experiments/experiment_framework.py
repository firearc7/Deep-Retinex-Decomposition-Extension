"""
Comprehensive Experimental Testing Framework for Deep Retinex + Traditional DIP
This module provides tools to systematically test different enhancement combinations
"""
import os
import json
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import itertools


class ExperimentConfig:
    """Configuration for experiments"""
    
    # Enhancement method combinations to test
    ILLUMINATION_METHODS = {
        'none': [],
        'clahe_only': ['clahe'],
        'bilateral_only': ['bilateral_filter'],
        'adaptive_gamma_only': ['adaptive_gamma'],
        'clahe_bilateral': ['clahe', 'bilateral_filter'],
        'clahe_gamma': ['clahe', 'adaptive_gamma'],
        'bilateral_gamma': ['bilateral_filter', 'adaptive_gamma'],
        'all_illumination': ['clahe', 'bilateral_filter', 'adaptive_gamma'],
    }
    
    OUTPUT_METHODS = {
        'none': [],
        'unsharp_only': ['unsharp_mask'],
        'color_balance_only': ['color_balance'],
        'tone_mapping_only': ['tone_mapping'],
        'unsharp_color': ['unsharp_mask', 'color_balance'],
        'all_output': ['unsharp_mask', 'color_balance', 'tone_mapping'],
    }
    
    # Parameter variations
    CLAHE_PARAMS = [
        {'clip_limit': 1.0, 'tile_size': (8, 8)},
        {'clip_limit': 2.0, 'tile_size': (8, 8)},
        {'clip_limit': 3.0, 'tile_size': (8, 8)},
        {'clip_limit': 2.0, 'tile_size': (16, 16)},
    ]
    
    BILATERAL_PARAMS = [
        {'d': 7, 'sigma_color': 50, 'sigma_space': 50},
        {'d': 9, 'sigma_color': 75, 'sigma_space': 75},
        {'d': 11, 'sigma_color': 100, 'sigma_space': 100},
    ]
    
    UNSHARP_PARAMS = [
        {'kernel_size': 5, 'sigma': 1.0, 'amount': 1.0},
        {'kernel_size': 5, 'sigma': 1.0, 'amount': 1.5},
        {'kernel_size': 5, 'sigma': 1.0, 'amount': 2.0},
    ]


class QualityMetrics:
    """Image quality metrics for evaluation"""
    
    @staticmethod
    def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(max_val / np.sqrt(mse))
    
    @staticmethod
    def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Structural Similarity Index"""
        from skimage.metrics import structural_similarity
        
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = (img1 * 255).astype(np.uint8)
            img2_gray = (img2 * 255).astype(np.uint8)
        
        return structural_similarity(img1_gray, img2_gray)
    
    @staticmethod
    def calculate_entropy(img: np.ndarray) -> float:
        """Calculate image entropy (information content)"""
        if len(img.shape) == 3:
            img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            img = (img * 255).astype(np.uint8)
        
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        hist = hist[hist > 0]
        
        return -np.sum(hist * np.log2(hist))
    
    @staticmethod
    def calculate_contrast(img: np.ndarray) -> float:
        """Calculate RMS contrast"""
        if len(img.shape) == 3:
            img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            img = (img * 255).astype(np.uint8)
        
        return np.std(img)
    
    @staticmethod
    def calculate_brightness(img: np.ndarray) -> float:
        """Calculate average brightness"""
        if len(img.shape) == 3:
            img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            img = (img * 255).astype(np.uint8)
        
        return np.mean(img)
    
    @staticmethod
    def calculate_colorfulness(img: np.ndarray) -> float:
        """Calculate colorfulness metric (Hasler and Süsstrunk)"""
        if len(img.shape) != 3:
            return 0.0
        
        img = (img * 255).astype(np.uint8)
        (B, G, R) = cv2.split(img.astype(float))
        
        rg = R - G
        yb = 0.5 * (R + G) - B
        
        std_rg = np.std(rg)
        std_yb = np.std(yb)
        
        mean_rg = np.mean(rg)
        mean_yb = np.mean(yb)
        
        std_root = np.sqrt(std_rg ** 2 + std_yb ** 2)
        mean_root = np.sqrt(mean_rg ** 2 + mean_yb ** 2)
        
        return std_root + 0.3 * mean_root
    
    @staticmethod
    def calculate_sharpness(img: np.ndarray) -> float:
        """Calculate sharpness using Laplacian variance"""
        if len(img.shape) == 3:
            img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            img = (img * 255).astype(np.uint8)
        
        return cv2.Laplacian(img, cv2.CV_64F).var()
    
    @staticmethod
    def calculate_all_metrics(img: np.ndarray, reference: Optional[np.ndarray] = None) -> Dict:
        """Calculate all quality metrics"""
        metrics = {
            'entropy': QualityMetrics.calculate_entropy(img),
            'contrast': QualityMetrics.calculate_contrast(img),
            'brightness': QualityMetrics.calculate_brightness(img),
            'colorfulness': QualityMetrics.calculate_colorfulness(img),
            'sharpness': QualityMetrics.calculate_sharpness(img),
        }
        
        if reference is not None:
            metrics['psnr'] = QualityMetrics.calculate_psnr(img, reference)
            metrics['ssim'] = QualityMetrics.calculate_ssim(img, reference)
        
        return metrics


class ExperimentRunner:
    """Main experiment runner"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = self.output_dir / 'results'
        self.images_dir = self.output_dir / 'images'
        self.logs_dir = self.output_dir / 'logs'
        
        for d in [self.results_dir, self.images_dir, self.logs_dir]:
            d.mkdir(exist_ok=True)
        
        self.metrics = QualityMetrics()
        self.experiment_log = []
    
    def generate_experiment_configs(self, mode: str = 'systematic') -> List[Dict]:
        """
        Generate experiment configurations
        
        Modes:
        - 'systematic': Test all combinations systematically
        - 'preset': Test only preset configurations
        - 'ablation': Ablation study (add one method at a time)
        - 'parameter_sweep': Sweep through parameter variations
        """
        configs = []
        
        if mode == 'systematic':
            # Test all combinations of illumination and output methods
            for illum_name, illum_methods in ExperimentConfig.ILLUMINATION_METHODS.items():
                for out_name, out_methods in ExperimentConfig.OUTPUT_METHODS.items():
                    config = {
                        'name': f'{illum_name}_{out_name}',
                        'apply_to_illumination': len(illum_methods) > 0,
                        'apply_to_reflectance': False,
                        'apply_to_output': len(out_methods) > 0,
                        'illumination_methods': illum_methods,
                        'reflectance_methods': [],
                        'output_methods': out_methods,
                        'clahe_params': {'clip_limit': 2.0, 'tile_size': (8, 8)},
                        'bilateral_params': {'d': 9, 'sigma_color': 75, 'sigma_space': 75},
                        'unsharp_params': {'kernel_size': 5, 'sigma': 1.0, 'amount': 1.5},
                        'color_balance_percent': 1,
                    }
                    configs.append(config)
        
        elif mode == 'ablation':
            # Start with baseline (no enhancements)
            base_config = {
                'name': 'baseline_none',
                'apply_to_illumination': False,
                'apply_to_reflectance': False,
                'apply_to_output': False,
                'illumination_methods': [],
                'reflectance_methods': [],
                'output_methods': [],
            }
            configs.append(base_config)
            
            # Add illumination methods one by one
            illum_methods = ['clahe', 'bilateral_filter', 'adaptive_gamma']
            for i in range(1, len(illum_methods) + 1):
                for combo in itertools.combinations(illum_methods, i):
                    config = base_config.copy()
                    config['name'] = f'ablation_illum_{"_".join(combo)}'
                    config['apply_to_illumination'] = True
                    config['illumination_methods'] = list(combo)
                    config['clahe_params'] = {'clip_limit': 2.0, 'tile_size': (8, 8)}
                    config['bilateral_params'] = {'d': 9, 'sigma_color': 75, 'sigma_space': 75}
                    configs.append(config)
        
        elif mode == 'parameter_sweep':
            # Sweep through CLAHE parameters
            for clahe_params in ExperimentConfig.CLAHE_PARAMS:
                config = {
                    'name': f'clahe_clip{clahe_params["clip_limit"]}_tile{clahe_params["tile_size"][0]}',
                    'apply_to_illumination': True,
                    'apply_to_reflectance': False,
                    'apply_to_output': False,
                    'illumination_methods': ['clahe'],
                    'reflectance_methods': [],
                    'output_methods': [],
                    'clahe_params': clahe_params,
                }
                configs.append(config)
            
            # Sweep through bilateral filter parameters
            for bilateral_params in ExperimentConfig.BILATERAL_PARAMS:
                config = {
                    'name': f'bilateral_d{bilateral_params["d"]}_sc{bilateral_params["sigma_color"]}',
                    'apply_to_illumination': True,
                    'apply_to_reflectance': False,
                    'apply_to_output': False,
                    'illumination_methods': ['bilateral_filter'],
                    'reflectance_methods': [],
                    'output_methods': [],
                    'bilateral_params': bilateral_params,
                }
                configs.append(config)
        
        elif mode == 'preset':
            # Use predefined presets
            from ..enhancements.pipeline import EnhancementFactory
            for preset in ['none', 'minimal', 'balanced', 'aggressive', 
                          'illumination_only', 'output_only']:
                config = EnhancementFactory.create_config(preset)
                config['name'] = f'preset_{preset}'
                configs.append(config)
        
        return configs
    
    def run_single_experiment(self, config: Dict, R: np.ndarray, I: np.ndarray, 
                             I_delta: Optional[np.ndarray], 
                             input_image: np.ndarray,
                             reference: Optional[np.ndarray] = None,
                             image_name: str = 'test') -> Dict:
        """Run a single experiment configuration"""
        from ..enhancements.pipeline import EnhancementPipeline
        
        start_time = time.time()
        
        # Create pipeline with config
        pipeline = EnhancementPipeline(config)
        
        # Process
        enhanced_output, intermediate = pipeline.process_full_pipeline(R, I, I_delta)
        
        processing_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self.metrics.calculate_all_metrics(enhanced_output, reference)
        
        # Save output image
        output_filename = f'{image_name}_{config["name"]}.jpg'
        output_path = self.images_dir / output_filename
        cv2.imwrite(str(output_path), (enhanced_output * 255).astype(np.uint8))
        
        # Compile results
        result = {
            'config_name': config['name'],
            'config': config,
            'metrics': metrics,
            'processing_time': processing_time,
            'output_path': str(output_path),
            'timestamp': datetime.now().isoformat(),
        }
        
        return result
    
    def run_experiments(self, experiments_config: List[Dict], 
                       test_images: List[Dict]) -> Dict:
        """
        Run all experiments on all test images
        
        Args:
            experiments_config: List of enhancement configurations
            test_images: List of dicts with keys: 'R', 'I', 'I_delta', 'input', 'reference' (optional), 'name'
        """
        all_results = []
        
        for img_data in test_images:
            image_name = img_data['name']
            print(f"\nProcessing image: {image_name}")
            
            for idx, config in enumerate(experiments_config):
                print(f"  Experiment {idx+1}/{len(experiments_config)}: {config['name']}")
                
                result = self.run_single_experiment(
                    config=config,
                    R=img_data['R'],
                    I=img_data['I'],
                    I_delta=img_data.get('I_delta'),
                    input_image=img_data['input'],
                    reference=img_data.get('reference'),
                    image_name=image_name
                )
                
                result['image_name'] = image_name
                all_results.append(result)
        
        # Save results
        results_summary = self.analyze_results(all_results)
        
        # Save to JSON
        results_file = self.results_dir / f'experiment_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w') as f:
            json.dump({
                'all_results': all_results,
                'summary': results_summary,
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        return results_summary
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze experiment results and find best configurations"""
        # Group by configuration
        config_metrics = {}
        
        for result in results:
            config_name = result['config_name']
            if config_name not in config_metrics:
                config_metrics[config_name] = {
                    'entropy': [],
                    'contrast': [],
                    'brightness': [],
                    'colorfulness': [],
                    'sharpness': [],
                    'psnr': [],
                    'ssim': [],
                    'processing_time': [],
                }
            
            for metric, value in result['metrics'].items():
                if metric in config_metrics[config_name]:
                    config_metrics[config_name][metric].append(value)
            
            config_metrics[config_name]['processing_time'].append(result['processing_time'])
        
        # Calculate average metrics
        summary = {}
        for config_name, metrics in config_metrics.items():
            summary[config_name] = {
                metric: {
                    'mean': np.mean(values) if values else 0,
                    'std': np.std(values) if values else 0,
                    'min': np.min(values) if values else 0,
                    'max': np.max(values) if values else 0,
                }
                for metric, values in metrics.items() if values
            }
        
        # Find best configurations
        best_configs = {
            'best_entropy': max(summary.items(), key=lambda x: x[1].get('entropy', {}).get('mean', 0)),
            'best_contrast': max(summary.items(), key=lambda x: x[1].get('contrast', {}).get('mean', 0)),
            'best_colorfulness': max(summary.items(), key=lambda x: x[1].get('colorfulness', {}).get('mean', 0)),
            'best_sharpness': max(summary.items(), key=lambda x: x[1].get('sharpness', {}).get('mean', 0)),
            'fastest': min(summary.items(), key=lambda x: x[1].get('processing_time', {}).get('mean', float('inf'))),
        }
        
        return {
            'config_summary': summary,
            'best_configs': {k: v[0] for k, v in best_configs.items()},
        }
    
    def generate_comparison_report(self, results_file: str):
        """Generate visual comparison report"""
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        all_results = data['all_results']
        summary = data['summary']
        
        # Create comparison HTML report
        html_content = self._create_html_report(all_results, summary)
        
        report_file = self.output_dir / 'comparison_report.html'
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        print(f"Comparison report saved to: {report_file}")
    
    def _create_html_report(self, results: List[Dict], summary: Dict) -> str:
        """Create HTML comparison report"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Deep Retinex Enhancement Experiments</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                .image-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .image-card { border: 1px solid #ddd; padding: 10px; }
                .image-card img { width: 100%; height: auto; }
                .metrics { font-size: 12px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
                .best { background-color: #90EE90; }
            </style>
        </head>
        <body>
            <h1>Deep Retinex + Traditional DIP Enhancement Experiments</h1>
        """
        
        # Add best configurations
        html += "<h2>Best Configurations</h2><ul>"
        for metric, config_name in summary['best_configs'].items():
            html += f"<li><strong>{metric}</strong>: {config_name}</li>"
        html += "</ul>"
        
        # Add metrics table
        html += "<h2>Metrics Summary</h2><table><tr><th>Configuration</th><th>Entropy</th><th>Contrast</th><th>Colorfulness</th><th>Sharpness</th><th>Time (s)</th></tr>"
        
        for config_name, metrics in summary['config_summary'].items():
            html += f"<tr><td>{config_name}</td>"
            for metric in ['entropy', 'contrast', 'colorfulness', 'sharpness', 'processing_time']:
                if metric in metrics:
                    mean_val = metrics[metric]['mean']
                    html += f"<td>{mean_val:.4f}</td>"
                else:
                    html += "<td>N/A</td>"
            html += "</tr>"
        
        html += "</table>"
        
        # Add images
        html += "<h2>Visual Comparison</h2><div class='image-grid'>"
        
        for result in results:
            html += f"""
            <div class='image-card'>
                <h3>{result['image_name']} - {result['config_name']}</h3>
                <img src='{result['output_path']}' alt='{result['config_name']}'>
                <div class='metrics'>
                    <strong>Metrics:</strong><br>
                    Entropy: {result['metrics'].get('entropy', 'N/A'):.4f}<br>
                    Contrast: {result['metrics'].get('contrast', 'N/A'):.4f}<br>
                    Sharpness: {result['metrics'].get('sharpness', 'N/A'):.4f}<br>
                    Time: {result['processing_time']:.4f}s
                </div>
            </div>
            """
        
        html += "</div></body></html>"
        
        return html
