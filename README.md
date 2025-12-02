# Deep Retinex Decomposition with Traditional DIP Enhancement

A low-light image enhancement system combining deep learning (RetinexNet) with traditional digital image processing techniques.

## Project Overview

This project implements the Deep Retinex Decomposition approach for low-light image enhancement, extended with traditional DIP post-processing techniques including CLAHE, bilateral filtering, gamma correction, and unsharp masking.

### Key Components

1. **DecomNet** - Decomposes images into reflectance and illumination components
2. **RelightNet** - Enhances the illumination map using a U-Net architecture  
3. **Traditional DIP Pipeline** - Applies classical enhancement techniques to the decomposed components

## Project Structure

```
Deep-Retinex-Decomposition-Extension/
├── src/
│   ├── model/
│   │   ├── decomnet.py          # Decomposition network
│   │   ├── relightnet.py        # Illumination enhancement network
│   │   └── retinexnet.py        # Combined model with loss functions
│   └── enhancements/
│       ├── traditional_dip.py   # Traditional DIP techniques
│       └── pipeline.py          # Enhancement pipeline
├── checkpoints/                  # Trained model weights
├── report_figures/               # Figures for report/presentation
├── Documents/                    # Project documentation
├── train.py                      # Training script
├── test.py                       # Inference script
├── download_dataset.py           # Dataset downloader
├── config.py                     # Configuration settings
└── requirements.txt              # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Download Dataset

```bash
python download_dataset.py --output_dir data
```

### 2. Train Model

```bash
python train.py --train_low_dir data/train/low --train_high_dir data/train/high --epochs 100
```

### 3. Test with Enhancement

```bash
python test.py --checkpoint checkpoints/retinexnet_best.pt --input_dir data/test/low --output_dir results --enhance balanced --compute_metrics
```

## Enhancement Presets

| Preset | Description |
|--------|-------------|
| none | No enhancement (baseline) |
| minimal | Adaptive gamma correction only |
| balanced | CLAHE + bilateral filter + unsharp mask + color balance |
| aggressive | All techniques with stronger parameters |

## Traditional DIP Techniques

The following techniques are implemented in `src/enhancements/traditional_dip.py`:

- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Bilateral Filter (edge-preserving smoothing)
- Gamma Correction (brightness adjustment)
- Unsharp Masking (detail enhancement)
- Color Balance (automatic white balance)
- Multi-scale Retinex
- Guided Filter
- Tone Mapping
- Shadow Enhancement

## Results

Results and comparison figures are available in the `report_figures/` directory.

## References

- Wei, C., et al. "Deep Retinex Decomposition for Low-Light Enhancement." BMVC 2018.
- Zuiderveld, K. "Contrast Limited Adaptive Histogram Equalization." Graphics Gems IV, 1994.
- Tomasi, C., Manduchi, R. "Bilateral Filtering for Gray and Color Images." ICCV 1998.
