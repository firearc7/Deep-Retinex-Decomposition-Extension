# ICIP Report Results Summary

## Deep Retinex Decomposition Enhanced with Traditional DIP Techniques

**Conference Target:** IEEE International Conference on Image Processing (ICIP)

---

## Abstract Summary

This work extends the Deep Retinex Decomposition framework for low-light image enhancement by integrating traditional Digital Image Processing (DIP) techniques at strategic points in the pipeline. The proposed hybrid approach achieves significant improvements in visual quality metrics while maintaining computational efficiency.

---

## I. Model Architecture

### A. Network Components

| Component | Description | Parameters |
|-----------|-------------|------------|
| **DecomNet** | Encoder-decoder network for Retinex decomposition | ~250K |
| **RelightNet** | U-Net style network for illumination adjustment | ~305K |
| **Total** | Complete RetinexNet | **555,205** |

### B. DecomNet Architecture
- **Input:** 4-channel (RGB + max channel)
- **Encoder:** 64-channel convolutional layers with 9×9 initial kernel
- **Feature Extraction:** 5 conv layers with ReLU activation (3×3 kernels)
- **Output:** 4-channel (3 for Reflectance R, 1 for Illumination L)
- **Activation:** Sigmoid for both outputs

### C. RelightNet Architecture
- **Input:** 4-channel (Reflectance R + Illumination L)
- **Encoder:** Multi-scale with stride-2 convolutions (3 levels)
- **Decoder:** Skip connections with upsampling
- **Feature Fusion:** 1×1 convolution for multi-scale features
- **Output:** Single-channel enhanced illumination I_δ

---

## II. Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | LOL (Low-Light) Dataset |
| Training Samples | 689 pairs |
| Validation Samples | 100 pairs |
| Batch Size | 8 |
| Patch Size | 48×48 |
| Optimizer | Adam |
| Initial Learning Rate | 0.001 |
| LR Schedule | Step decay (×0.1 at epoch 50) |
| Total Epochs | 100 |
| Hardware | NVIDIA GPU (CUDA) |
| Training Time | ~18 minutes |

### Loss Function
The total loss combines decomposition and relighting losses:

$$\mathcal{L}_{total} = \mathcal{L}_{decom} + \mathcal{L}_{relight}$$

Where:
$$\mathcal{L}_{decom} = \mathcal{L}_{recon}^{low} + \mathcal{L}_{recon}^{high} + 0.001 \cdot \mathcal{L}_{mutual} + 0.1 \cdot \mathcal{L}_{smooth} + 0.01 \cdot \mathcal{L}_{equal}$$

$$\mathcal{L}_{relight} = \mathcal{L}_{recon}^{relight} + 3 \cdot \mathcal{L}_{smooth}^{\delta}$$

---

## III. Training Results

### A. Convergence Analysis

| Metric | Value |
|--------|-------|
| Initial Train Loss | 0.4188 |
| Final Train Loss | 0.1681 |
| **Loss Reduction** | **59.8%** |
| Best Val Loss | 0.1640 (Epoch 84) |
| Final Val Loss | 0.1711 |

### B. Training Stability (Last 10 Epochs)

| Metric | Mean ± Std |
|--------|------------|
| Train Loss | 0.1743 ± 0.0029 |
| Val Loss | 0.1762 ± 0.0081 |
| Train-Val Gap | -0.0019 |

**Observation:** Minimal overfitting with excellent generalization (train-val gap < 0.002)

### C. Training Curve Summary

```
Phase 1 (Epochs 1-50, LR=0.001):
  - Rapid convergence from 0.42 to ~0.20
  - Best validation achieved: 0.1650 (Epoch 26)

Phase 2 (Epochs 51-100, LR=0.0001):
  - Fine-tuning with stable convergence
  - Best validation achieved: 0.1640 (Epoch 84)
```

---

## IV. Traditional DIP Enhancement Pipeline

### A. Enhancement Presets

| Preset | Description | Primary Use Case |
|--------|-------------|------------------|
| **Baseline** | Model output only (no DIP) | Reference comparison |
| **Minimal** | Light CLAHE + bilateral filter | Subtle enhancement |
| **Balanced** | CLAHE + gamma + unsharp mask + color balance | General purpose |
| **Aggressive** | Strong CLAHE + adaptive gamma + heavy sharpening | Maximum enhancement |

### B. DIP Techniques Implemented

1. **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
   - Applied to: Illumination map
   - Parameters: clip_limit=2.0, tile_size=(8,8)

2. **Bilateral Filter**
   - Applied to: Illumination map
   - Parameters: d=9, σ_color=75, σ_space=75

3. **Adaptive Gamma Correction**
   - Applied to: Illumination map
   - Formula: γ = -0.3 / log₁₀(μ_L)

4. **Unsharp Masking**
   - Applied to: Final output
   - Parameters: kernel_size=5, σ=1.0, amount=1.5

5. **Color Balance**
   - Applied to: Final output
   - Method: Percentile stretching (1%)

### C. Advanced Techniques (Available)

- Anisotropic Diffusion (Perona-Malik)
- Multi-Scale Detail Enhancement (Laplacian Pyramid)
- Shadow Enhancement
- Haze Removal (Dark Channel Prior)
- Guided Filtering
- Multi-Scale Retinex

---

## V. Quantitative Results

### A. Quality Metrics (100 Test Images)

| Preset | Entropy | Contrast | Sharpness | Colorfulness | Brightness |
|--------|---------|----------|-----------|--------------|------------|
| Baseline | 5.91±0.61 | 23.37±5.40 | 150.84±100.97 | 25.05±10.73 | 112.70±12.54 |
| Minimal | 6.04±0.61 | 25.53±5.94 | 180.79±121.22 | 27.47±11.76 | 123.71±14.20 |
| **Balanced** | **7.32±0.42** | **57.22±8.59** | 2069.40±1311.10 | **51.20±20.71** | 165.40±20.53 |
| Aggressive | 4.60±0.71 | 63.82±7.65 | 5103.83±3042.68 | 49.36±20.16 | 217.08±10.54 |

### B. Improvement Over Baseline (%)

| Preset | Entropy | Contrast | Sharpness | Colorfulness |
|--------|---------|----------|-----------|--------------|
| Minimal | +2.2% | +9.3% | +19.9% | +9.6% |
| **Balanced** | **+23.9%** | **+144.8%** | **+1271.9%** | **+104.4%** |
| Aggressive | -22.2% | +173.1% | +3283.5% | +97.0% |

### C. Key Observations

1. **Balanced preset achieves optimal trade-off:**
   - Highest entropy (7.32) indicates better information preservation
   - 144.8% contrast improvement without over-saturation
   - 104.4% colorfulness improvement maintains natural appearance

2. **Aggressive preset limitations:**
   - Decreased entropy (-22.2%) indicates information loss
   - Over-sharpening (3283.5%) may introduce artifacts
   - Excessive brightness (217.08) risks over-exposure

3. **Minimal preset:**
   - Conservative improvements suitable for subtle enhancement
   - Maintains natural appearance with minimal artifacts

---

## VI. Visual Results

### A. Available Comparison Images

Located in: `results/visual_examples_100_epochs/`

- Side-by-side comparisons for 10 representative images
- Individual preset outputs per image
- Includes: input, baseline, minimal, balanced, aggressive

### B. Recommended Figures for Paper

1. **Figure 1:** Architecture diagram (DecomNet + RelightNet + DIP Pipeline)
2. **Figure 2:** Training curves (loss vs. epochs)
3. **Figure 3:** Visual comparison grid (3-4 images × 4 presets)
4. **Figure 4:** Illumination and reflectance decomposition visualization

---

## VII. Computational Efficiency

| Stage | Time (per image) |
|-------|------------------|
| Model Inference | ~50 ms |
| Minimal DIP | ~20 ms |
| Balanced DIP | ~80 ms |
| Aggressive DIP | ~150 ms |
| **Total (Balanced)** | **~130 ms** |

*Measured on NVIDIA GPU with 1024×768 images*

---

## VIII. Ablation Study Summary

### Enhancement Stage Impact

| Enhancement Point | Effect |
|-------------------|--------|
| Illumination only | +14.4% entropy, +36.1% contrast |
| Output only | +21.4% entropy, +142.1% contrast |
| **Both (Balanced)** | **+23.9% entropy, +144.8% contrast** |

### Conclusion
Applying DIP techniques to both illumination map and final output yields optimal results.

---

## IX. Files for Report

### Required Images
```
results/visual_examples_100_epochs/
├── comparison_low00690.png  # Side-by-side comparison
├── comparison_low00692.png  # Different scene type
├── comparison_low00696.png  # Indoor scene
└── comparison_low00698.png  # Challenging low-light

results/training_analysis/
└── training_analysis.png    # Training curves
```

### Data Files
```
results/final_comparison_100_epochs/
├── comparison_results.csv   # All metrics (600 rows)
└── comparison_summary.json  # Aggregated statistics

checkpoints/
├── retinexnet_best.pt       # Best model weights
└── training_history.json    # Training logs
```

---

## X. Citation Information

```bibtex
@inproceedings{author2025retinex,
  title={Deep Retinex Decomposition Enhanced with Traditional Image Processing for Low-Light Image Enhancement},
  author={Author Name},
  booktitle={IEEE International Conference on Image Processing (ICIP)},
  year={2025}
}
```

---

## XI. Reproducibility

### Environment
- Python 3.10+
- PyTorch 2.0+
- OpenCV 4.x
- NumPy, SciPy

### Commands
```bash
# Training
python train.py --epochs 100 --batch_size 8

# Evaluation
python test.py --checkpoint checkpoints/retinexnet_best.pt

# Quick inference
python quick_inference.py --image path/to/image.jpg --preset balanced
```


