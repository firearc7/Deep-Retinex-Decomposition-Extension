"""
Configuration file for Deep Retinex + Traditional DIP
"""

# Project paths
PROJECT_ROOT = '/home/yajat/Documents/DIP/Deep-Retinex-Decomposition-Extension'
DATA_DIR = f'{PROJECT_ROOT}/data'
CHECKPOINT_DIR = f'{PROJECT_ROOT}/checkpoints'
RESULTS_DIR = f'{PROJECT_ROOT}/results'
EXPERIMENT_DIR = f'{PROJECT_ROOT}/experiments'

# Model parameters
MODEL_CONFIG = {
    'decom_channel': 64,
    'decom_kernel_size': 3,
    'relight_channel': 64,
    'relight_kernel_size': 3,
}

# Training parameters
TRAIN_CONFIG = {
    'batch_size': 16,
    'patch_size': 96,
    'epochs': 100,
    'learning_rate': 0.001,
    'lr_decay_epoch': 20,
    'eval_every_epoch': 10,
}

# Enhancement presets
ENHANCEMENT_PRESETS = {
    'none': {
        'apply_to_illumination': False,
        'apply_to_reflectance': False,
        'apply_to_output': False,
        'illumination_methods': [],
        'reflectance_methods': [],
        'output_methods': [],
    },
    
    'minimal': {
        'apply_to_illumination': True,
        'apply_to_reflectance': False,
        'apply_to_output': False,
        'illumination_methods': ['adaptive_gamma'],
        'reflectance_methods': [],
        'output_methods': [],
    },
    
    'balanced': {
        'apply_to_illumination': True,
        'apply_to_reflectance': False,
        'apply_to_output': True,
        'illumination_methods': ['clahe', 'bilateral_filter'],
        'clahe_params': {'clip_limit': 2.0, 'tile_size': (8, 8)},
        'bilateral_params': {'d': 9, 'sigma_color': 75, 'sigma_space': 75},
        'reflectance_methods': [],
        'output_methods': ['unsharp_mask', 'color_balance'],
        'unsharp_params': {'kernel_size': 5, 'sigma': 1.0, 'amount': 1.5},
        'color_balance_percent': 1,
    },
    
    'aggressive': {
        'apply_to_illumination': True,
        'apply_to_reflectance': False,
        'apply_to_output': True,
        'illumination_methods': ['clahe', 'bilateral_filter', 'adaptive_gamma'],
        'clahe_params': {'clip_limit': 3.0, 'tile_size': (8, 8)},
        'bilateral_params': {'d': 9, 'sigma_color': 75, 'sigma_space': 75},
        'reflectance_methods': [],
        'output_methods': ['unsharp_mask', 'color_balance', 'tone_mapping'],
        'unsharp_params': {'kernel_size': 5, 'sigma': 1.0, 'amount': 2.0},
        'color_balance_percent': 2,
    },
    
    'noise_reduction': {
        'apply_to_illumination': True,
        'apply_to_reflectance': False,
        'apply_to_output': False,
        'illumination_methods': ['bilateral_filter'],
        'bilateral_params': {'d': 11, 'sigma_color': 100, 'sigma_space': 100},
        'reflectance_methods': [],
        'output_methods': [],
    },
    
    'detail_enhancement': {
        'apply_to_illumination': False,
        'apply_to_reflectance': False,
        'apply_to_output': True,
        'illumination_methods': [],
        'reflectance_methods': [],
        'output_methods': ['unsharp_mask'],
        'unsharp_params': {'kernel_size': 5, 'sigma': 1.0, 'amount': 2.0},
    },
}

# Experiment configurations
EXPERIMENT_CONFIG = {
    'test_modes': ['systematic', 'ablation', 'parameter_sweep', 'preset'],
    'metrics': ['entropy', 'contrast', 'brightness', 'colorfulness', 'sharpness', 'psnr', 'ssim'],
    'output_formats': ['jpg', 'png'],
    'generate_report': True,
}

# Dataset paths
DATASET_PATHS = {
    'train_low': f'{DATA_DIR}/train/low',
    'train_high': f'{DATA_DIR}/train/high',
    'test_low': f'{DATA_DIR}/test/low',
    'test_high': f'{DATA_DIR}/test/high',
    'eval_low': f'{DATA_DIR}/eval/low',
}
