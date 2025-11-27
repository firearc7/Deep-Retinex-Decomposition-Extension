# Deep Retinex Decomposition Enhanced with Traditional DIP

This project extends the Deep Retinex Decomposition network for low-light image enhancement by integrating traditional Digital Image Processing (DIP) techniques.

## Project Structure

```
Deep-Retinex-Decomposition-Extension/
├── src/
│   ├── model/                  # Neural network models
│   │   ├── decomnet.py        # Decomposition Network
│   │   ├── relightnet.py      # Relighting Network
│   │   └── retinexnet.py      # Complete Retinex model
│   ├── enhancements/          # Traditional DIP enhancements
│   │   ├── traditional_dip.py # DIP techniques library
│   │   └── pipeline.py        # Enhancement pipeline
│   └── utils/                 # Utility functions
├── experiments/               # Experimental framework
│   └── experiment_framework.py
├── data/                      # Training and test data
├── results/                   # Output results
├── checkpoints/               # Model checkpoints
└── Documents/                 # Documentation
```

## Key Enhancement Points

The traditional DIP techniques can be applied at three strategic points:

### 1. **Illumination Map Enhancement** (MOST IMPORTANT)
Apply to the illumination map (I or I_delta) from DecomNet/RelightNet:
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Enhances local contrast while preventing noise amplification
- **Bilateral Filter**: Edge-preserving smoothing to reduce noise
- **Adaptive Gamma Correction**: Automatically adjusts brightness based on image statistics
- **Guided Filter**: Uses reflectance as guide for better edge preservation

### 2. **Reflectance Map Enhancement** (MINIMAL)
Usually minimal processing needed:
- **Bilateral Filter**: Can reduce noise if present
- **Color Balance**: Adjust color if needed

### 3. **Final Output Enhancement**
Apply to final reconstructed image (S = R × I):
- **Unsharp Masking**: Enhance details and sharpness
- **Color Balance**: Correct color distribution
- **Tone Mapping**: HDR-like effects
- **Local Contrast Enhancement**: Grid-based contrast improvement

## Traditional DIP Techniques Available

| Technique | Best Applied To | Purpose |
|-----------|----------------|---------|
| CLAHE | Illumination Map | Local contrast enhancement without over-amplification |
| Bilateral Filter | Illumination Map | Noise reduction with edge preservation |
| Adaptive Gamma | Illumination Map | Automatic brightness adjustment |
| Guided Filter | Illumination Map | Edge-aware smoothing using reflectance |
| Unsharp Masking | Final Output | Detail enhancement and sharpening |
| Color Balance | Final Output | Color correction |
| Tone Mapping | Final Output | Dynamic range compression |
| Multi-Scale Retinex | Illumination/Output | Illumination normalization |
| Histogram Equalization | Illumination Map | Global contrast enhancement |

## Why These Enhancement Points?

### Illumination Map (I_delta) - PRIMARY TARGET
- **Reason**: This is where the network adjusts lighting, and DIP can refine it
- **Benefits**:
  - Removes residual noise
  - Improves local contrast
  - Prevents over-brightening
  - Preserves natural appearance
- **Recommended Methods**: CLAHE + Bilateral Filter + Adaptive Gamma

### Reflectance Map (R) - MINIMAL PROCESSING
- **Reason**: Reflectance contains intrinsic object properties
- **Caution**: Over-processing can distort colors and textures
- **Use Case**: Only if noise is visible

### Final Output (S) - REFINEMENT
- **Reason**: Final polish after reconstruction
- **Benefits**:
  - Sharpens details lost during processing
  - Corrects color imbalances
  - Enhances perceptual quality
- **Recommended Methods**: Unsharp Masking + Color Balance

## Experimental Testing Plan

### Phase 1: Baseline Comparison
```python
# Test configurations
experiments = [
    'none',                    # No enhancements (baseline)
    'illumination_only',       # Only enhance I_delta
    'output_only',             # Only enhance final output
    'illumination_and_output', # Both (recommended)
]
```

### Phase 2: Ablation Study
Test individual and combined methods:
```python
illumination_methods = [
    ['clahe'],
    ['bilateral_filter'],
    ['adaptive_gamma'],
    ['clahe', 'bilateral_filter'],
    ['clahe', 'adaptive_gamma'],
    ['clahe', 'bilateral_filter', 'adaptive_gamma'],
]
```

### Phase 3: Parameter Sweep
Optimize parameters for best methods:
```python
# CLAHE parameters
clip_limits = [1.0, 2.0, 3.0, 4.0]
tile_sizes = [(8, 8), (16, 16), (32, 32)]

# Bilateral filter parameters
d_values = [7, 9, 11]
sigma_values = [50, 75, 100]
```

### Phase 4: Full Combinations
Test systematic combinations for best overall results.

## Evaluation Metrics

### Objective Metrics
1. **Entropy**: Information content (higher = more details)
2. **Contrast**: RMS contrast (higher = better contrast)
3. **Sharpness**: Laplacian variance (higher = sharper)
4. **Colorfulness**: Color richness metric
5. **Brightness**: Average luminance
6. **PSNR**: Peak Signal-to-Noise Ratio (if reference available)
7. **SSIM**: Structural Similarity Index (if reference available)

### Subjective Evaluation
- Visual quality assessment
- Natural appearance
- Artifact presence
- Detail preservation

## Usage Example

### Simple Usage
```python
from src.enhancements import EnhancementPipeline, EnhancementFactory

# Create pipeline with preset
config = EnhancementFactory.create_config('balanced')
pipeline = EnhancementPipeline(config)

# Apply enhancements
enhanced_output, results = pipeline.process_full_pipeline(R, I, I_delta)
```

### Custom Configuration
```python
custom_config = {
    'apply_to_illumination': True,
    'illumination_methods': ['clahe', 'bilateral_filter'],
    'clahe_params': {'clip_limit': 2.0, 'tile_size': (8, 8)},
    'bilateral_params': {'d': 9, 'sigma_color': 75, 'sigma_space': 75},
    'apply_to_output': True,
    'output_methods': ['unsharp_mask'],
    'unsharp_params': {'kernel_size': 5, 'sigma': 1.0, 'amount': 1.5},
}

pipeline = EnhancementPipeline(custom_config)
enhanced_output, results = pipeline.process_full_pipeline(R, I, I_delta)
```

### Run Experiments
```python
from experiments.experiment_framework import ExperimentRunner

runner = ExperimentRunner(output_dir='./experiment_results')

# Generate experiment configurations
configs = runner.generate_experiment_configs(mode='systematic')

# Prepare test data
test_images = [
    {
        'name': 'image1',
        'R': reflectance_map,
        'I': illumination_map,
        'I_delta': enhanced_illumination,
        'input': input_image,
        'reference': reference_image,  # optional
    },
    # ... more images
]

# Run experiments
results = runner.run_experiments(configs, test_images)

# Generate comparison report
runner.generate_comparison_report('path/to/results.json')
```

## Installation

```bash
# Required packages
pip install torch torchvision
pip install opencv-python numpy pillow
pip install scikit-image  # for SSIM
pip install matplotlib seaborn  # for visualization
```

## Expected Improvements

With optimal enhancement configuration, expect:
- **20-30%** improvement in contrast
- **15-25%** increase in sharpness
- **10-20%** better entropy (information content)
- **Reduced artifacts** from neural network
- **More natural appearance** in enhanced images
- **Better detail preservation** in dark regions

## Recommended Starting Configuration

```python
recommended_config = {
    'apply_to_illumination': True,
    'illumination_methods': ['clahe', 'bilateral_filter'],
    'clahe_params': {'clip_limit': 2.0, 'tile_size': (8, 8)},
    'bilateral_params': {'d': 9, 'sigma_color': 75, 'sigma_space': 75},
    
    'apply_to_output': True,
    'output_methods': ['unsharp_mask', 'color_balance'],
    'unsharp_params': {'kernel_size': 5, 'sigma': 1.0, 'amount': 1.5},
    'color_balance_percent': 1,
}
```

## References

- Chen Wei, et al. "Deep Retinex Decomposition for Low-Light Enhancement"
- CLAHE: Zuiderveld, K. (1994). "Contrast Limited Adaptive Histogram Equalization"
- Bilateral Filter: Tomasi & Manduchi (1998)
- Guided Filter: He et al. (2013)

## License

See LICENSE file for details.
