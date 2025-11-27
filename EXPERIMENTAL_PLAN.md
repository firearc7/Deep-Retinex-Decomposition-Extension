# Comprehensive Experimental Testing Plan
## Deep Retinex + Traditional DIP Enhancement

## Overview
This document outlines a systematic approach to testing and comparing different traditional DIP enhancement techniques applied to Deep Retinex decomposition.

---

## Phase 1: Baseline Establishment (Week 1)

### Objective
Establish baseline performance without enhancements

### Experiments
1. **Baseline-None**: No enhancements applied
   - Purpose: Reference point for all comparisons
   - Metrics: Record all quality metrics

2. **Model-Only**: Original Deep Retinex output
   - Test on multiple image categories (indoor, outdoor, night, etc.)
   - Document limitations and artifacts

### Deliverables
- Baseline metrics table
- Visual quality assessment
- Common artifacts documentation

---

## Phase 2: Single Method Evaluation (Week 2-3)

### Objective
Test each enhancement method individually

### 2.1 Illumination Map Enhancements

#### Test 1: CLAHE Only
```python
configs = [
    {'illumination_methods': ['clahe'], 'clahe_params': {'clip_limit': 1.0, 'tile_size': (8, 8)}},
    {'illumination_methods': ['clahe'], 'clahe_params': {'clip_limit': 2.0, 'tile_size': (8, 8)}},
    {'illumination_methods': ['clahe'], 'clahe_params': {'clip_limit': 3.0, 'tile_size': (8, 8)}},
    {'illumination_methods': ['clahe'], 'clahe_params': {'clip_limit': 2.0, 'tile_size': (16, 16)}},
]
```
**Expected**: Improved local contrast, potential noise amplification at high clip_limit

#### Test 2: Bilateral Filter Only
```python
configs = [
    {'illumination_methods': ['bilateral_filter'], 'bilateral_params': {'d': 7, 'sigma_color': 50, 'sigma_space': 50}},
    {'illumination_methods': ['bilateral_filter'], 'bilateral_params': {'d': 9, 'sigma_color': 75, 'sigma_space': 75}},
    {'illumination_methods': ['bilateral_filter'], 'bilateral_params': {'d': 11, 'sigma_color': 100, 'sigma_space': 100}},
]
```
**Expected**: Noise reduction, smoother illumination, preserved edges

#### Test 3: Adaptive Gamma Only
```python
configs = [
    {'illumination_methods': ['adaptive_gamma']},
]
```
**Expected**: Automatic brightness adjustment, better exposure

#### Test 4: Guided Filter Only
```python
configs = [
    {'illumination_methods': ['guided_filter'], 'radius': 8, 'eps': 0.01},
    {'illumination_methods': ['guided_filter'], 'radius': 16, 'eps': 0.01},
]
```
**Expected**: Edge-aware smoothing, texture preservation

### 2.2 Output Enhancements

#### Test 5: Unsharp Masking Only
```python
configs = [
    {'output_methods': ['unsharp_mask'], 'unsharp_params': {'amount': 1.0}},
    {'output_methods': ['unsharp_mask'], 'unsharp_params': {'amount': 1.5}},
    {'output_methods': ['unsharp_mask'], 'unsharp_params': {'amount': 2.0}},
]
```
**Expected**: Enhanced details, sharpness, potential halos at high amount

#### Test 6: Color Balance Only
```python
configs = [
    {'output_methods': ['color_balance'], 'color_balance_percent': 1},
    {'output_methods': ['color_balance'], 'color_balance_percent': 2},
]
```
**Expected**: Corrected colors, stretched dynamic range

#### Test 7: Tone Mapping Only
```python
configs = [
    {'output_methods': ['tone_mapping']},
]
```
**Expected**: HDR-like appearance, compressed dynamic range

### Deliverables
- Individual method performance comparison
- Optimal parameter identification for each method
- Side-by-side visual comparisons

---

## Phase 3: Method Combination Testing (Week 4-5)

### Objective
Test combinations of complementary methods

### 3.1 Illumination Combinations

#### Test 8: CLAHE + Bilateral Filter
```python
configs = [
    {
        'illumination_methods': ['clahe', 'bilateral_filter'],
        'clahe_params': {'clip_limit': 2.0, 'tile_size': (8, 8)},
        'bilateral_params': {'d': 9, 'sigma_color': 75, 'sigma_space': 75},
    }
]
```
**Hypothesis**: CLAHE enhances contrast, bilateral reduces noise

#### Test 9: CLAHE + Adaptive Gamma
```python
configs = [
    {
        'illumination_methods': ['clahe', 'adaptive_gamma'],
        'clahe_params': {'clip_limit': 2.0, 'tile_size': (8, 8)},
    }
]
```
**Hypothesis**: Combined contrast and brightness improvement

#### Test 10: Bilateral + Adaptive Gamma
```python
configs = [
    {
        'illumination_methods': ['bilateral_filter', 'adaptive_gamma'],
        'bilateral_params': {'d': 9, 'sigma_color': 75, 'sigma_space': 75},
    }
]
```
**Hypothesis**: Smooth noise-free illumination with optimal brightness

#### Test 11: All Illumination Methods
```python
configs = [
    {
        'illumination_methods': ['clahe', 'bilateral_filter', 'adaptive_gamma'],
        'clahe_params': {'clip_limit': 2.0, 'tile_size': (8, 8)},
        'bilateral_params': {'d': 9, 'sigma_color': 75, 'sigma_space': 75},
    }
]
```
**Hypothesis**: Maximal illumination enhancement

### 3.2 Output Combinations

#### Test 12: Unsharp + Color Balance
```python
configs = [
    {
        'output_methods': ['unsharp_mask', 'color_balance'],
        'unsharp_params': {'amount': 1.5},
        'color_balance_percent': 1,
    }
]
```
**Hypothesis**: Sharp and color-corrected output

### 3.3 Full Pipeline Combinations

#### Test 13: Best Illumination + Best Output
```python
configs = [
    {
        'illumination_methods': ['clahe', 'bilateral_filter'],
        'output_methods': ['unsharp_mask', 'color_balance'],
        # Use best parameters from previous tests
    }
]
```
**Hypothesis**: Optimal end-to-end enhancement

### Deliverables
- Combination effectiveness analysis
- Interaction effects between methods
- Recommended combinations for different scenarios

---

## Phase 4: Ablation Study (Week 6)

### Objective
Understand contribution of each component

### Test Series 14: Sequential Addition
```python
ablation_configs = [
    {'name': 'baseline', 'illumination_methods': []},
    {'name': '+clahe', 'illumination_methods': ['clahe']},
    {'name': '+clahe+bilateral', 'illumination_methods': ['clahe', 'bilateral_filter']},
    {'name': '+clahe+bilateral+gamma', 'illumination_methods': ['clahe', 'bilateral_filter', 'adaptive_gamma']},
]
```

### Test Series 15: Sequential Removal
Start with all methods, remove one at a time to see impact

### Deliverables
- Component importance ranking
- Minimal effective configuration
- Diminishing returns analysis

---

## Phase 5: Application-Specific Testing (Week 7)

### Objective
Optimize for different image types

### 5.1 Indoor Low-Light Images
- Focus on noise reduction
- Preserve texture details
- Natural color appearance

### 5.2 Outdoor Night Images
- Handle street lights and light sources
- Sky region processing
- Global illumination adjustment

### 5.3 Extremely Dark Images
- Maximum enhancement without artifacts
- Detail recovery in shadows
- Noise suppression

### 5.4 Mixed Lighting
- Handle multiple light sources
- Preserve color temperature
- Local adaptation

### Deliverables
- Application-specific optimal configurations
- Dataset-specific parameter recommendations

---

## Phase 6: Comparison with State-of-the-Art (Week 8)

### Objective
Compare with other enhancement methods

### Comparisons
1. **Deep Retinex (Original)** vs **Deep Retinex + Our Enhancement**
2. **Traditional Retinex (MSR)** vs **Deep Retinex + Enhancement**
3. **Histogram Equalization** vs **Our Method**
4. **Other deep learning methods** (if available)

### Deliverables
- Quantitative comparison tables
- Visual quality comparison
- Processing time analysis
- Publication-ready figures

---

## Metrics and Evaluation

### Quantitative Metrics (Computed for Every Test)

1. **Entropy** (bits)
   - Measures information content
   - Higher is generally better
   - Target: > 7.0

2. **Contrast** (std deviation)
   - RMS contrast measure
   - Higher indicates better contrast
   - Target: > 50

3. **Sharpness** (Laplacian variance)
   - Edge strength measure
   - Higher is sharper
   - Target: > 100

4. **Colorfulness** (custom metric)
   - Color richness
   - Higher is more colorful
   - Target: > 40

5. **Brightness** (mean luminance)
   - Average brightness
   - Target: 100-150 (for 0-255 range)

6. **PSNR** (dB) - if reference available
   - Higher is better
   - Target: > 20 dB

7. **SSIM** (0-1) - if reference available
   - Structural similarity
   - Target: > 0.8

8. **Processing Time** (seconds)
   - Computational efficiency
   - Lower is better

### Qualitative Assessment

For each configuration, evaluate:
- **Naturalness**: Does the image look realistic?
- **Artifacts**: Are there halos, over-enhancement, or color shifts?
- **Detail Preservation**: Are fine details maintained?
- **Overall Quality**: Subjective quality rating (1-10)

---

## Experimental Workflow

### For Each Configuration:

1. **Setup**
   ```python
   config = create_config(...)
   pipeline = EnhancementPipeline(config)
   ```

2. **Process Images**
   ```python
   for image in test_set:
       enhanced, results = pipeline.process_full_pipeline(R, I, I_delta)
       save_output(enhanced, config_name, image_name)
   ```

3. **Compute Metrics**
   ```python
   metrics = compute_all_metrics(enhanced, reference)
   log_metrics(config_name, image_name, metrics)
   ```

4. **Visual Inspection**
   - Save images with config name
   - Create side-by-side comparisons
   - Note any artifacts or issues

5. **Analysis**
   - Compare metrics across configurations
   - Identify trends and patterns
   - Document findings

---

## Expected Timeline

| Week | Phase | Activities |
|------|-------|------------|
| 1 | Baseline | Establish baselines, collect initial metrics |
| 2-3 | Single Methods | Test each method individually |
| 4-5 | Combinations | Test method combinations |
| 6 | Ablation | Ablation studies |
| 7 | Application | Application-specific tuning |
| 8 | Comparison | Compare with state-of-the-art |

---

## Data Organization

```
experiment_results/
├── baseline/
│   ├── images/
│   ├── metrics.json
│   └── analysis.md
├── single_methods/
│   ├── clahe/
│   ├── bilateral/
│   ├── gamma/
│   └── ...
├── combinations/
│   ├── clahe_bilateral/
│   └── ...
├── ablation/
├── application_specific/
└── final_comparison/
    ├── comparison_report.html
    ├── metrics_table.csv
    └── visual_comparisons.pdf
```

---

## Success Criteria

### Minimum Acceptable Improvement
- **Entropy**: +10% over baseline
- **Contrast**: +15% over baseline
- **Sharpness**: +20% over baseline
- **Subjective Quality**: Clear visual improvement

### Optimal Goals
- **Entropy**: +25% over baseline
- **Contrast**: +30% over baseline
- **Sharpness**: +40% over baseline
- **Artifacts**: Minimal to none
- **Naturalness**: Maintained or improved

---

## Tools and Scripts

### Quick Test Script
```python
from experiments.experiment_framework import ExperimentRunner

runner = ExperimentRunner(output_dir='./quick_test')
configs = runner.generate_experiment_configs(mode='preset')
results = runner.run_experiments(configs, test_images)
runner.generate_comparison_report('results.json')
```

### Full Systematic Test
```python
runner = ExperimentRunner(output_dir='./full_systematic')
configs = runner.generate_experiment_configs(mode='systematic')
results = runner.run_experiments(configs, all_test_images)
```

### Parameter Sweep
```python
configs = runner.generate_experiment_configs(mode='parameter_sweep')
```

---

## Notes and Best Practices

1. **Always test on diverse images**: Indoor, outdoor, night, day, different lighting conditions
2. **Save intermediate results**: Keep R, I, I_delta for analysis
3. **Document parameter choices**: Record reasoning for parameter selections
4. **Version control**: Track config changes and results
5. **Reproducibility**: Set random seeds, document environment
6. **Visual inspection is crucial**: Metrics don't tell the whole story
7. **Compare fairly**: Use same test set for all methods
8. **Processing time matters**: Consider real-world deployment

---

## Report Format

### Final Report Should Include:

1. **Executive Summary**
   - Best configuration found
   - Key improvements over baseline
   - Recommended use cases

2. **Methodology**
   - Experimental design
   - Test images description
   - Evaluation metrics

3. **Results**
   - Quantitative results tables
   - Visual comparisons
   - Statistical analysis

4. **Discussion**
   - Why certain methods work better
   - Trade-offs and limitations
   - Application recommendations

5. **Conclusion**
   - Summary of findings
   - Future work suggestions

6. **Appendix**
   - Complete metrics tables
   - All configuration details
   - Additional visual examples
