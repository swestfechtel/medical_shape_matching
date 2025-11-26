# Complete FUSS Workflow

This document provides a step-by-step guide for the complete workflow from raw segmentation data to trained statistical shape models.

## Overview

```
Segmentation Masks → .OFF Meshes → Preprocessing → Training → Testing/SSM Building
```

## Step-by-Step Workflow

### Step 1: Prepare Your Data

Organize your 3D segmentation masks in a directory:
```
raw_data/
  └── segmentations/
      ├── subject_001.nii.gz
      ├── subject_002.nii.gz
      ├── subject_003.nii.gz
      └── ...
```

Supported formats: NIfTI (`.nii`, `.nii.gz`), NumPy (`.npy`, `.npz`), DICOM (directory)

### Step 2: Convert Segmentations to .OFF Format

Use the `segmentation_to_off.py` utility to convert your segmentation masks:

```bash
# Basic conversion
python segmentation_to_off.py \
    --input_dir raw_data/segmentations/ \
    --output_dir data/my_organ/off \
    --pattern "*.nii.gz"

# With mesh processing (recommended)
python segmentation_to_off.py \
    --input_dir raw_data/segmentations/ \
    --output_dir data/my_organ/off \
    --pattern "*.nii.gz" \
    --smooth \
    --simplify \
    --target_faces 5000
```

**Important Options:**
- `--smooth`: Apply Gaussian smoothing for smoother surfaces
- `--simplify --target_faces 5000`: Reduce mesh complexity (recommended: 3000-10000 faces)
- `--labels 1 2 3`: Extract specific labels only (for multi-organ segmentations)

After this step, you should have:
```
data/my_organ/
  └── off/
      ├── shape_001.off
      ├── shape_002.off
      └── ...
```

### Step 3: Preprocess Meshes

Compute spectral operators (Laplacian eigenvectors/eigenvalues) and mesh statistics:

```bash
python preprocess.py \
    --data_root data/my_organ/ \
    --n_eig 200
```

**Parameters:**
- `--n_eig 200`: Number of eigenvalues to compute (default: 200-300)
- `--no_normalize`: Skip mesh normalization (not recommended)

After this step, you should have:
```
data/my_organ/
  ├── off/
  ├── diffusion/           # Spectral operators
  │   ├── shape_001_000.mat
  │   ├── shape_002_000.mat
  │   └── ...
  └── mesh_info.csv        # Mesh statistics
```

### Step 4: Configure Training

Create a YAML configuration file in `options/train/my_organ.yaml`:

```yaml
name: fuss_my_organ
backend: dp
type: FussModel
num_gpu: auto
manual_seed: 1234
visualize: true

# Dataset configuration
datasets:
  train_dataset:
    name: Train
    type: PairPancreasDataset  # Or create custom dataset
    data_root: data/my_organ
    phase: train
    return_evecs: true
    return_faces: true
    num_evecs: 40
    start_index: 0
    end_index: 200  # Adjust based on your dataset size
    n_combination: 20

  test_dataset:
    name: Test
    type: PairPancreasDataset
    data_root: data/my_organ
    phase: test
    return_evecs: true
    return_faces: true
    num_evecs: 40
    start_index: 200
    end_index: 250

  batch_size: 1
  num_worker: 16

# Network configuration (use defaults or customize)
networks:
  feature_extractor:
    type: DiffusionNet
    in_channels: 128
    out_channels: 384
    cache_dir: data/my_organ/diffusion
    input_type: wks
  fmap_net:
    type: RegularizedFMNet
    bidirectional: true
  permutation:
    type: Similarity
    tau: 0.07
  interpolator:
    type: ResnetECPos
    c_dim: 3
    dim: 7
    hidden_dim: 128

# Training settings
train:
  total_epochs: 15
  optims:
    feature_extractor:
      type: Adam
      lr: 1.0e-3
    interpolator:
      type: Adam
      lr: 1.0e-3
  schedulers:
    feature_extractor:
      type: CosineAnnealingLR
      eta_min: 1.0e-4
      T_max: 10
    interpolator:
      type: CosineAnnealingLR
      eta_min: 1.0e-4
      T_max: 10
  losses:
    surfmnet_loss:
      type: SURFMNetLoss
      w_bij: 1.0
      w_orth: 1.0
    couple_loss:
      type: SquaredFrobeniusLoss
      loss_weight: 1.0
    align_loss:
      type: SquaredFrobeniusLoss
      loss_weight: 10.0
    smoothness_loss:
      type: DirichletLoss
      loss_weight: 2.0
    dirichlet_shape_loss:
      type: DirichletLoss
      loss_weight: 1.0e+2
    chamfer_shape_loss:
      type: ChamferLoss
      loss_weight: 1.0e+4
    edge_shape_loss:
      type: EdgeLoss
      loss_weight: 1.0e+5

# Validation
val:
  val_freq: 4000
  metrics:
    specificity:
      type: calculate_specificity
    generalization:
      type: calculate_generalization

# Logging
logger:
  print_freq: 20
  save_checkpoint_freq: 5000
```

### Step 5: Train the Model

```bash
python train.py --opt options/train/my_organ.yaml
```

Monitor training progress with TensorBoard:
```bash
tensorboard --logdir experiments/
```

Training outputs:
```
experiments/
  └── fuss_my_organ/
      ├── models/          # Checkpoints (.pth files)
      ├── visualization/   # Validation visualizations
      └── log/            # Training logs
```

**Training Tips:**
- Training typically takes several hours to days depending on dataset size
- Monitor loss curves in TensorBoard
- Validation visualizations show correspondence quality
- Press Ctrl+C to gracefully stop training (saves checkpoint)

### Step 6: Configure Testing

Create a test configuration in `options/test/my_organ.yaml`:

```yaml
name: fuss_my_organ
backend: dp
type: FussModel
num_gpu: auto
visualize: true
pose_timestep: 6

# Path to trained model
path:
  resume_state: experiments/fuss_my_organ/models/final.pth
  resume: false

# Datasets for testing
datasets:
  0_reference_dataset:
    name: Reference
    type: PairPancreasDataset
    data_root: data/my_organ
    phase: train
    return_evecs: true
    return_faces: true
    num_evecs: 40
    start_index: 0
    end_index: 200

  1_train_dataset:
    name: Train
    type: SinglePancreasDataset
    data_root: data/my_organ
    phase: train
    return_evecs: true
    return_faces: true
    num_evecs: 40
    start_index: 0
    end_index: 200

  2_test_dataset:
    name: Test
    type: SinglePancreasDataset
    data_root: data/my_organ
    phase: test
    return_evecs: true
    return_faces: true
    num_evecs: 40
    start_index: 200
    end_index: 250

# Networks (same as training)
networks:
  # ... (copy from train config)

# Validation metrics
val:
  metrics:
    specificity:
      type: calculate_specificity
    generalization:
      type: calculate_generalization
```

### Step 7: Test and Build SSM

```bash
python test.py --opt options/test/my_organ.yaml
```

This will:
1. Select a reference shape from the training set
2. Deform the reference to all training shapes
3. Build SSM using PCA
4. Evaluate on test set:
   - **Generalization**: How well SSM fits unseen shapes
   - **Specificity**: How anatomically plausible generated samples are

Results are saved to:
```
results/
  └── fuss_my_organ/
      ├── generalization_metrics.txt
      ├── specificity_metrics.txt
      └── visualizations/
```

## Quick Start Example

Here's a complete example from scratch:

```bash
# 1. Convert segmentations
python segmentation_to_off.py \
    --input_dir ~/data/pancreas_segmentations/ \
    --output_dir data/pancreas/off \
    --smooth --simplify --target_faces 5000

# 2. Preprocess
python preprocess.py --data_root data/pancreas/ --n_eig 200

# 3. Train (after creating config)
python train.py --opt options/train/pancreas.yaml

# 4. Monitor training
tensorboard --logdir experiments/

# 5. Test (after creating test config)
python test.py --opt options/test/pancreas.yaml
```

## Common Issues and Solutions

### Issue: "No label found in mask"
**Solution:** Check your segmentation labels with:
```bash
python -c "import nibabel as nib; import numpy as np; print(np.unique(nib.load('seg.nii.gz').get_fdata()))"
```

### Issue: Meshes have very different sizes
**Solution:** Use `--simplify --target_faces N` to standardize mesh resolution

### Issue: Training loss not decreasing
**Solution:**
- Check if meshes are properly normalized (preprocessing should handle this)
- Verify dataset paths in config
- Try reducing learning rate or adjusting loss weights

### Issue: Out of memory during training
**Solution:**
- Reduce `num_evecs` in config (try 20-30)
- Use fewer eigenvalues for spectral decomposition
- Reduce batch size (though it's typically 1)

### Issue: Marching cubes fails during conversion
**Solution:**
- Add `--smooth` to smooth the mask
- Check if segmentation is too small or has issues
- Try adjusting `--level` parameter

## Dataset Size Recommendations

- **Minimum**: 50 shapes (not recommended, limited generalization)
- **Good**: 100-200 shapes
- **Optimal**: 200+ shapes

Split recommendation: 80-90% training, 10-20% testing

## Performance Expectations

- **Conversion**: ~1-10 seconds per segmentation (depends on size)
- **Preprocessing**: ~1-30 seconds per mesh (depends on `--n_eig`)
- **Training**: 1-24 hours (depends on dataset size, GPU)
- **Testing**: 10 minutes - 2 hours (depends on dataset size)

## Next Steps

After building your SSM, you can:
1. Generate new plausible shape samples
2. Use for shape analysis and comparison
3. Fit SSM to new test data
4. Extract statistical shape features
5. Perform shape-based classification or prediction

Refer to the original FUSS paper for applications and analysis techniques.