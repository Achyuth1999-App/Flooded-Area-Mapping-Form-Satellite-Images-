# Flooded Area Mapping from Sentinel-1 Satellite Images

Pixel-level **water / flood-inundation mapping** from Sentinel-1 SAR imagery, using the [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) hand-labelled dataset. The project compares three approaches under identical evaluation conditions:

1. **SAR VH threshold baseline** – a single low-backscatter threshold, tuned on the validation set only.
2. **Compact U-Net (from scratch)** – ResNet18-style encoder, random initialisation.
3. **Transfer U-Net** – same architecture, ResNet18 encoder pretrained on ImageNet and fine-tuned on 2-channel SAR.

Individual project · MSc Data Science, University of Europe for Applied Sciences · Notebook: `Flood_mapping_satellite_images.ipynb` (Google Colab)

---

## Results (held-out test set)

All metrics are pixel-level, computed on the official test split only, with invalid (`-1`) pixels excluded.

| Model | IoU (water) | Dice / F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| **Compact U-Net (scratch)** | **0.622** | **0.767** | 0.944 | 0.804 | **0.734** |
| Transfer U-Net (ImageNet) | 0.615 | 0.761 | **0.946** | **0.852** | 0.688 |
| VH threshold baseline | 0.561 | 0.719 | 0.932 | 0.750 | 0.691 |

Key observations:

- Both U-Nets beat the threshold baseline by roughly 5–6 IoU points.
- The two U-Nets are close (IoU difference ≈ 0.007). With one run and one seed, this should **not** be read as a meaningful win for training from scratch.
- Transfer learning gave the highest precision (fewer false water pixels) but lower recall; the compact model was more balanced.
- Accuracy is high for all models because water is a minority class (~9% of training pixels); **water IoU and Dice are the primary metrics**.
- Validation IoU fluctuated noticeably between epochs (e.g. 0.42–0.59 for the compact model), so best-epoch selection on a small validation set is somewhat noisy.

Selected VH threshold (normalised): **0.525** (validation IoU 0.511).

## Dataset

- **Sen1Floods11 v1.1**, hand-labelled subset: **446** chips of 512×512 pixels.
- Input: **2-band Sentinel-1 SAR (VV, VH)** – not RGB.
- Labels (`LabelHand`): `1` = water, `0` = non-water, `-1` = invalid/ignored.
- Official chip-level splits are used (`flood_train_data.csv`, `flood_valid_data.csv`, `flood_test_data.csv`); the notebook asserts that the splits do not overlap. Training uses 252 chips.
- Training pixel distribution: 12,963,114 non-water · 1,362,418 water · 2,189,540 invalid.
- Data is read anonymously from the public bucket `gs://sen1floods11/v1.1/...`. No API key is required. Only SAR imagery and hand labels are downloaded (~700 MB); weak labels and Sentinel-2 bands are skipped.

## Method

**Preprocessing**
- Backscatter (dB) is clipped to [-50, 1], NaNs replaced, and scaled to [0, 1].
- Images and masks are resized to 256×256 (bilinear for images, nearest-neighbour for masks to preserve class IDs).
- Training-only augmentation: random horizontal/vertical flips and 90° rotations applied identically to image and mask.

**Models** (`segmentation-models-pytorch`)
- `smp.Unet`, ResNet18 encoder, `encoder_depth=4`, decoder channels `(128, 64, 32, 16)`, 2 input channels, 1 output logit. ~11.96 M parameters each.
- `compact_unet`: `encoder_weights=None`. `transfer_unet`: `encoder_weights='imagenet'`.

**Training**
- Loss: binary cross-entropy with `-1` pixels masked out.
- Optimiser: AdamW, lr `3e-4`, batch size 8, 12 epochs, seed 42.
- Best checkpoint chosen by validation water IoU.

**Baseline**
- Pixel predicted as water if normalised VH < threshold; threshold searched over 37 values in [0.05, 0.95] on the validation set, then applied once to the test set.

## Repository / output contents

```
Flood_mapping_satellite_images.ipynb   # full pipeline, runs end to end on Colab
flood_mapping/
├── data/                              # downloaded S1Hand, LabelHand, split CSVs
└── outputs/
    ├── model_comparison.csv           # test metrics for all three models
    ├── *_classification_report.txt    # pixel-level reports
    ├── *_confusion_matrix.png
    ├── *_history.csv, *_curves.png    # per-epoch loss / accuracy / IoU
    ├── *_best.pth                     # best-validation weights
    ├── baseline_threshold_tuning.png
    ├── class_distribution.csv / .png
    ├── sample_dataset.png
    ├── test_overlay_1..5.png          # prediction overlays
    ├── split_counts.csv
    └── run_metadata.json              # seed, hyperparameters, threshold, splits
```

The notebook also zips `outputs/` into `flood_mapping_results.zip`.

## How to run

1. Open the notebook in **Google Colab**.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. **Run all.** The first run downloads the data, trains both networks and evaluates everything.

Dependencies (installed in the first cell):

```
rasterio scikit-learn segmentation-models-pytorch pandas matplotlib tqdm google-cloud-storage
```

plus PyTorch (preinstalled on Colab). `NUM_WORKERS=0` is set deliberately to avoid DataLoader worker-shutdown errors in hosted notebooks. To change training length or image size, edit `EPOCHS` / `SIZE` in the configuration cell.

## Limitations

- **Water ≠ flood.** Labels mark water vs non-water and cannot separate *new floodwater* from *permanent water* (rivers, lakes). Results should be described as water/flood-inundation mapping, not flood detection.
- Official splits are chip-level; spatially nearby chips can still leak across splits and may inflate generalisation estimates.
- Results come from a single seed and single run; no confidence intervals. Differences between the two U-Nets are within plausible run-to-run noise.
- Validation IoU is unstable across epochs, so the saved "best" checkpoint partly reflects selection noise.
- Test data is evaluated once, after all model and threshold selection on train/validation.
- Chips were resized from 512×512 to 256×256, which can lose fine detail such as narrow water bodies.
- The Bolivia chips (`flood_bolivia_data.csv`) are downloaded but not used for evaluation here; they would be a natural out-of-region generalisation test.

## Possible extensions

- Multiple seeds with mean ± std; per-event/region breakdown.
- Dice or focal loss to address class imbalance.
- Evaluate on the Bolivia hold-out set.
- Add Sentinel-2 bands or permanent-water masks to separate flood from permanent water.

## Reference

Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). *Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1.* CVPR Workshops. Dataset: https://github.com/cloudtostreet/Sen1Floods11
