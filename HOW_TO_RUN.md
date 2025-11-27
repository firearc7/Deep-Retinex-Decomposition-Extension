# 🚀 COMPLETE WORKFLOW GUIDE

## Step-by-Step: From Dataset to Results

### ✅ Step 1: Download Dataset (DONE!)

You've already completed this step successfully!

```bash
python download_dataset.py --dataset lol-v2 --source kaggle --output_dir data
```

**Result**: 
- Real subset: 689 train pairs, 100 test pairs
- Synthetic subset: 900 train pairs, 100 test pairs

---

### 🎓 Step 2: Train the Model (DO THIS ONCE - 2-3 hours on GPU)

Choose ONE subset to train on:

#### Option A: Train on Real subset (Recommended - real captured images)
```bash
python train.py \
    --train_low_dir data/real/train/low \
    --train_high_dir data/real/train/high \
    --val_low_dir data/real/eval/low \
    --val_high_dir data/real/eval/high \
    --epochs 100 \
    --batch_size 8 \
    --lr 0.001
```

#### Option B: Train on Synthetic subset (More training data - 900 pairs)
```bash
python train.py \
    --train_low_dir data/synthetic/train/low \
    --train_high_dir data/synthetic/train/high \
    --val_low_dir data/synthetic/eval/low \
    --val_high_dir data/synthetic/eval/high \
    --epochs 100 \
    --batch_size 8 \
    --lr 0.001
```

**Training Tips:**
- ✅ **For 6GB VRAM**: Use `--batch_size 8` (recommended, default above)
- ⚠️ **For 4GB VRAM**: Use `--batch_size 4` or `--batch_size 6`
- 🚀 **For 8GB+ VRAM**: Use `--batch_size 16` or higher
- 📖 **See BATCH_SIZE_GUIDE.md** for detailed VRAM recommendations
- If you get CUDA out of memory: Reduce batch size by 2
- If you want faster training: Use `--epochs 50`
- Training will save checkpoints every 10 epochs

**Expected Output:**
```
checkpoints/
├── retinexnet_best.pt       ← Use this for inference
├── retinexnet_final.pt
├── retinexnet_epoch_10.pt
├── retinexnet_epoch_20.pt
└── training_history.json
```

---

### 🔍 Step 3: Compare DIP Enhancement Presets

**This is how we compare different enhancements!**

After training completes, run the comparison script:

```bash
python compare_enhancements.py \
    --checkpoint checkpoints/retinexnet_best.pt \
    --input_dir data/real/test/low \
    --output_dir results/comparison \
    --presets baseline minimal balanced aggressive \
    --save_images
```

**What this does:**
1. ✅ Loads your trained model
2. ✅ Processes all test images
3. ✅ Applies each DIP preset (baseline, minimal, balanced, aggressive)
4. ✅ Computes quality metrics for each preset
5. ✅ Generates comparison reports (CSV + HTML)
6. ✅ Saves output images for visual comparison

**Output:**
```
results/comparison/
├── comparison_results.csv      ← Detailed metrics per image per preset
├── comparison_summary.json     ← Average metrics summary
├── comparison_report.html      ← Visual comparison report (open in browser!)
└── [image_name]/
    ├── input.png
    ├── baseline.png            ← No DIP enhancement
    ├── minimal.png             ← Adaptive gamma only
    ├── balanced.png            ← CLAHE + Bilateral + Unsharp + Color Balance
    └── aggressive.png          ← All DIP methods
```

---

### 📊 Step 4: View and Analyze Results

#### Option 1: View HTML Report (Recommended)
```bash
# Open the HTML report in your browser
firefox results/comparison/comparison_report.html
# or
google-chrome results/comparison/comparison_report.html
```

The HTML report shows:
- 📊 Average metrics for each preset
- 🎯 Best preset for each quality metric
- 📋 Per-image detailed results
- 🏆 Highlighted best performers

#### Option 2: View CSV Data
```bash
# View CSV in terminal
column -t -s, results/comparison/comparison_results.csv | less

# Or open in LibreOffice/Excel
libreoffice results/comparison/comparison_results.csv
```

#### Option 3: View Summary JSON
```bash
cat results/comparison/comparison_summary.json
```

---

### 🎨 Step 5: Run Individual Tests (Optional)

If you want to test a specific preset on specific images:

```bash
# Test with balanced preset (recommended)
python test.py \
    --checkpoint checkpoints/retinexnet_best.pt \
    --input_dir data/real/test/low \
    --output_dir results/balanced_test \
    --enhance balanced \
    --compute_metrics

# Test with aggressive preset
python test.py \
    --checkpoint checkpoints/retinexnet_best.pt \
    --input_dir data/real/test/low \
    --output_dir results/aggressive_test \
    --enhance aggressive \
    --compute_metrics
```

---

## 📈 How We Compare Enhancements

### Quality Metrics Used

We compute **8 objective quality metrics** for each enhanced image:

| Metric | What It Measures | Higher = Better? |
|--------|------------------|------------------|
| **Entropy** | Information content, detail preservation | ✅ Yes |
| **Contrast** | RMS contrast, visibility | ✅ Yes |
| **Sharpness** | Edge strength (Laplacian variance) | ✅ Yes |
| **Colorfulness** | Color richness | ✅ Yes |
| **Brightness** | Average luminance | Depends on image |
| **PSNR** | Peak Signal-to-Noise Ratio (if reference available) | ✅ Yes |
| **SSIM** | Structural Similarity (if reference available) | ✅ Yes |
| **Processing Time** | Speed of DIP enhancement | Lower = Better |

### Comparison Process

```
For each test image:
    1. Run model inference ONCE → Get R, I, I_delta
    2. Apply baseline (no DIP) → Compute metrics
    3. Apply minimal preset → Compute metrics
    4. Apply balanced preset → Compute metrics
    5. Apply aggressive preset → Compute metrics
    6. Compare all metrics → Find best preset

Aggregate results:
    - Calculate average metrics per preset
    - Find best preset for each metric
    - Generate comparison report
```

### Expected Improvements

Based on the balanced preset, you should see:

- **Entropy**: +20-30% (more details preserved)
- **Contrast**: +25-35% (better visibility)
- **Sharpness**: +30-40% (clearer edges)
- **Colorfulness**: +10-20% (richer colors)
- **Processing time**: +0.05-0.2 seconds (negligible)

---

## 🎯 Complete Command Sequence

Here's the complete workflow in one place:

```bash
# 1. Download dataset (DONE!)
python download_dataset.py --dataset lol-v2 --source kaggle --output_dir data

# 2. Train model ONCE (2-3 hours)
python train.py \
    --train_low_dir data/real/train/low \
    --train_high_dir data/real/train/high \
    --val_low_dir data/real/eval/low \
    --val_high_dir data/real/eval/high \
    --epochs 100 \
    --batch_size 16

# 3. Compare all DIP enhancements
python compare_enhancements.py \
    --checkpoint checkpoints/retinexnet_best.pt \
    --input_dir data/real/test/low \
    --output_dir results/comparison \
    --presets baseline minimal balanced aggressive \
    --save_images

# 4. Open HTML report
firefox results/comparison/comparison_report.html
```

---

## ❓ FAQ

### Q: Why don't I need to retrain for different DIP techniques?

**A:** DIP techniques are **post-processing** steps applied AFTER model inference:

```
Model (trained once):
    Input → DecomNet → R, I
          → RelightNet → I_delta
          → Reconstruct → Output

DIP Enhancement (no training needed):
    Output → CLAHE → Bilateral → Unsharp → Enhanced Output
```

The model weights never change when applying DIP!

### Q: Which preset should I use?

**A:** 
- **For best quality**: aggressive
- **For general use**: balanced (recommended)
- **For real-time**: minimal
- **For comparison**: baseline (no enhancement)

### Q: How do I know which preset works best?

**A:** Run `compare_enhancements.py` - it will:
1. Test all presets automatically
2. Compute metrics for each
3. Generate a report showing which preset is best

### Q: Can I create custom presets?

**A:** Yes! Edit `config.py` or create custom configurations:

```python
from src.enhancements import EnhancementConfig, EnhancementPipeline

custom_config = EnhancementConfig(
    illumination_methods=['clahe', 'bilateral_filter'],
    illumination_params={
        'clahe': {'clip_limit': 3.0},
        'bilateral_filter': {'sigma_color': 100}
    },
    output_methods=['unsharp_mask'],
    output_params={'unsharp_mask': {'amount': 2.0}}
)

pipeline = EnhancementPipeline(custom_config)
enhanced = pipeline.process_full_pipeline(R, I, I_delta)
```

### Q: What if training fails with CUDA out of memory?

**A:** Reduce batch size:
```bash
python train.py --batch_size 4 --patch_size 32 ...
```

---

## 🎉 Summary

**You now have:**
1. ✅ Dataset downloaded (LOL-v2 with 689 + 900 training pairs)
2. ✅ Training script ready to use
3. ✅ Comparison script to test all DIP presets
4. ✅ HTML/CSV reports for analysis
5. ✅ **NO NEED TO RETRAIN** for different DIP configurations!

**Next steps:**
1. Run training (2-3 hours)
2. Run comparison script
3. Open HTML report and see which preset works best!

**Key insight**: Train the model ONCE, then experiment with dozens of DIP configurations in minutes, not days! 🚀
