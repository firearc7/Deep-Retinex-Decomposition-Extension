# configuration for deep retinex and traditional dip enhancement

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

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
    'patch_size': 48,
    'epochs': 100,
    'learning_rate': 0.001,
    'lr_decay_epoch': 50,
}

# Enhancement presets
ENHANCEMENT_PRESETS = {
    'none': {
        'apply_to_illumination': False,
        'apply_to_reflectance': False,
        'apply_to_output': False,
    },
    'minimal': {
        'apply_to_illumination': True,
        'illumination_methods': ['adaptive_gamma'],
    },
    'balanced': {
        'apply_to_illumination': True,
        'apply_to_output': True,
        'illumination_methods': ['clahe', 'bilateral_filter'],
        'output_methods': ['unsharp_mask', 'color_balance'],
    },
    'aggressive': {
        'apply_to_illumination': True,
        'apply_to_output': True,
        'illumination_methods': ['clahe', 'bilateral_filter', 'adaptive_gamma'],
        'output_methods': ['unsharp_mask', 'color_balance', 'tone_mapping'],
    },
}

# Dataset paths
DATASET_PATHS = {
    'train_low': os.path.join(DATA_DIR, 'train', 'low'),
    'train_high': os.path.join(DATA_DIR, 'train', 'high'),
    'test_low': os.path.join(DATA_DIR, 'test', 'low'),
    'test_high': os.path.join(DATA_DIR, 'test', 'high'),
    'eval_low': os.path.join(DATA_DIR, 'eval', 'low'),
    'eval_high': os.path.join(DATA_DIR, 'eval', 'high'),
}
