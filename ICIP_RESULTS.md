# ICIP Report Results Summary

## Deep Retinex Decomposition Enhanced with Traditional DIP Techniques

## Results

The link to the results folder can be found in [here](https://drive.google.com/drive/folders/19FJj0dF5fJPFPah-pQ1SKeHtHnN5Co6Y?usp=sharing).

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

### D. E1 and E3 Enhancement Techniques (NEW)

**E1: Illumination-Aware Edge-Preserving Denoising on Reflectance**

| Technique | Description |
|-----------|-------------|
| `illumination_aware_bilateral` | Bilateral filter with adaptive strength based on illumination |
| `illumination_aware_guided_filter` | Guided filter with illumination-adaptive epsilon |
| `illumination_aware_wavelet` | Wavelet shrinkage with illumination-aware threshold |

**E3: Photometric Post-Adjustments**

| Technique | Description |
|-----------|-------------|
| `illumination_gamma_curve` | Gamma correction applied to illumination map |
| `reflectance_micro_contrast_unsharp` | Unsharp masking for local contrast on reflectance |
| `reflectance_micro_contrast_dog` | Difference of Gaussians for micro-contrast on reflectance |

---

## V. Quantitative Results

### A. Quality Metrics (100 Test Images) - Including E1/E3 Presets

| Preset | PSNR↑ | SSIM↑ | Entropy↑ | Contrast | Sharpness | Colorfulness | Brightness |
|--------|-------|-------|----------|----------|-----------|--------------|------------|
| Baseline | **17.25** | **0.691** | 6.09 | 9.13 | 5254.87 | 9.87 | 112.52 |
| Minimal | 16.03 | 0.670 | 6.18 | 9.98 | 6309.64 | 10.82 | 123.52 |
| Balanced | 9.92 | 0.455 | 7.42 | 22.25 | 46500.12 | 20.14 | 165.27 |
| Aggressive | 5.58 | 0.362 | 3.87 | 24.81 | 73715.53 | 19.71 | 217.02 |
| **E1: Bilateral** | 9.91 | 0.516 | 7.39 | 22.01 | 38668.83 | 19.51 | 165.97 |
| **E1: Guided Filter** | 9.88 | 0.556 | **7.47** | 22.38 | 25643.69 | **22.16** | 166.92 |
| **E3: Unsharp** | 9.76 | 0.478 | 7.42 | 23.06 | 35280.72 | 21.52 | 167.12 |
| **E3: DoG** | 9.54 | 0.458 | 7.36 | 23.00 | 41509.27 | 21.17 | 169.37 |
| **E1+E3 Combined** | 9.79 | 0.526 | 7.41 | 22.89 | 31757.34 | 20.96 | 167.11 |

### B. Improvement Over Baseline (%)

| Preset | PSNR | SSIM | Entropy | Contrast | Sharpness | Colorfulness |
|--------|------|------|---------|----------|-----------|--------------|
| Minimal | -7.1% | -3.0% | +1.5% | +9.3% | +20.1% | +9.6% |
| Balanced | -42.5% | -34.2% | +22.0% | +143.7% | +784.9% | +104.0% |
| Aggressive | -67.6% | -47.6% | -36.4% | +171.7% | +1302.8% | +99.7% |
| **E1: Bilateral** | -42.6% | **-25.4%** | +21.4% | +141.0% | +635.9% | +97.6% |
| **E1: Guided Filter** | -42.7% | **-19.6%** | **+22.7%** | +145.1% | +388.0% | **+124.5%** |
| **E3: Unsharp** | -43.4% | -30.8% | +21.8% | +152.6% | +571.4% | +118.0% |
| **E3: DoG** | -44.7% | -33.8% | +20.9% | +151.9% | +689.9% | +114.4% |
| **E1+E3 Combined** | -43.3% | -24.0% | +21.7% | +150.7% | +504.3% | +112.3% |

### C. Best Preset Per Metric

| Metric | Best Preset | Value |
|--------|-------------|-------|
| **PSNR** | Baseline | 17.25 dB |
| **SSIM** | Baseline | 0.691 |
| **Entropy** | E1: Guided Filter | 7.47 |
| **Contrast** | Aggressive | 24.81 |
| **Colorfulness** | E1: Guided Filter | 22.16 |

### D. Key Observations

1. **E1 Guided Filter achieves best visual quality trade-off:**
   - Highest entropy (7.47) indicates best information preservation
   - Best colorfulness improvement (+124.5%)
   - Best SSIM among enhanced presets (0.556, only -19.6% vs baseline)

2. **E1 Techniques outperform standard balanced preset in SSIM:**
   - E1 Guided: 0.556 SSIM (-19.6% from baseline)
   - E1 Bilateral: 0.516 SSIM (-25.4% from baseline)
   - E1+E3 Combined: 0.526 SSIM (-24.0% from baseline)
   - Standard Balanced: 0.455 SSIM (-34.2% from baseline)

3. **Illumination-aware denoising preserves structure better:**
   - Adaptive strength based on local illumination
   - Less aggressive in already well-lit regions
   - More aggressive noise reduction in dark areas

4. **PSNR/SSIM vs Perceptual Quality Trade-off:**
   - Baseline has best PSNR/SSIM (closest to ground truth)
   - Enhanced presets sacrifice PSNR/SSIM for perceptual improvements
   - E1 techniques minimize this trade-off compared to standard enhancement

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

### E1/E3 Technique Comparison

| Technique | SSIM Retention | Entropy Improvement | Colorfulness Improvement |
|-----------|----------------|---------------------|--------------------------|
| E1: Guided Filter | **80.4%** | **+22.7%** | **+124.5%** |
| E1: Bilateral | 74.6% | +21.4% | +97.6% |
| E1+E3 Combined | 76.0% | +21.7% | +112.3% |
| Standard Balanced | 65.8% | +22.0% | +104.0% |

**Key Finding:** E1 illumination-aware guided filter provides the best balance between perceptual improvement and structural fidelity (SSIM retention).

---
