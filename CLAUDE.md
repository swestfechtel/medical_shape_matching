# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the official implementation of **FUSS** (A Universal and Flexible Framework for Unsupervised Statistical Shape Model Learning) from MICCAI 2024. The framework performs unsupervised learning of statistical shape models (SSM) on medical imaging data, specifically focused on 3D mesh shapes like pancreas organs.

## Environment Setup

**Python Environment:**
```bash
conda create -n fuss python=3.9
conda activate fuss
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
conda install pyg -c pyg
pip install -r requirements.txt
```

**Note:** The `pyproject.toml` requires Python >=3.13 but README specifies Python 3.9. The conda environment with Python 3.9 and the specific library versions in README should be used for compatibility with PyTorch3D and other dependencies.

## Common Commands

**Data Conversion (Segmentation to .OFF):**
```bash
# Convert segmentation masks to .off format
python segmentation_to_off.py --input seg.nii.gz --output_dir data/pancreas/off

# Batch convert with simplification
python segmentation_to_off.py --input_dir segmentations/ --output_dir data/pancreas/off --smooth --simplify --target_faces 5000

# See SEGMENTATION_CONVERSION.md for detailed usage
```

**Data Preprocessing:**
```bash
# Compute spectral operators and normalize shapes
python preprocess.py --data_root ../data/pancreas/ --n_eig 200
```

**Training:**
```bash
# Train FUSS model
python train.py --opt options/train/pancreas.yaml

# Monitor training with TensorBoard
tensorboard --logdir experiments/
```

**Testing:**
```bash
# Build SSM and evaluate metrics (generalization, specificity)
python test.py --opt options/test/pancreas.yaml
```

**Computing Correspondences:**
```bash
# Compute point-wise correspondences between shapes
python compute_correspondence.py \
    --model experiments/fuss_pancreas/models/final.pth \
    --config options/test/pancreas.yaml \
    --reference data/pancreas/off/shape_001.off \
    --target data/pancreas/off/shape_050.off \
    --output results/correspondences/ \
    --visualize

# See CORRESPONDENCE.md for detailed usage and Python API
```

## Architecture Overview

### Registry Pattern
The codebase uses a **registry pattern** (`utils/registry.py`) for modular component registration:
- `DATASET_REGISTRY`: Dataset types (e.g., `PairPancreasDataset`, `SinglePancreasDataset`)
- `NETWORK_REGISTRY`: Neural networks (e.g., `DiffusionNet`, `RegularizedFMNet`)
- `MODEL_REGISTRY`: Training models (e.g., `FussModel`)
- `LOSS_REGISTRY`: Loss functions
- `METRIC_REGISTRY`: Evaluation metrics

Components are automatically discovered via naming conventions (`*_dataset.py`, `*_network.py`, `*_model.py`) and imported in `__init__.py` files.

### Core Pipeline

**Training Flow (train.py):**
1. Parse YAML config → `utils/options.py:parse_options()`
2. Build datasets → `datasets/__init__.py:build_dataset()` using registry
3. Build model → `models/__init__.py:build_model()` (creates `FussModel`)
4. Training loop:
   - `model.feed_data()`: Process data pairs, extract features, compute permutation matrices
   - `model.optimize_parameters()`: Compute losses, backprop, update weights
   - `model.validation()`: Validate on test set periodically
5. Save checkpoints at regular intervals

**Testing Flow (test.py):**
1. Load trained model from checkpoint
2. Build SSM via PCA:
   - Select reference shape from training set (lowest pairwise loss)
   - Deform reference template to all training shapes
   - Compute PCA on deformed shapes
3. Evaluate metrics:
   - **Generalization**: How well SSM fits unseen test shapes
   - **Specificity**: How anatomically plausible generated samples are

### Model Architecture (FussModel)

Located in `models/fuss_model.py`, inherits from `BaseModel`:

**Key Components:**
- `feature_extractor`: DiffusionNet extracts geometric features from meshes
- `fmap_net`: Computes functional maps between shape pairs
- `permutation`: Converts features to permutation matrices (correspondences)
- `interpolator`: Neural network for shape interpolation/deformation

**Loss Functions:**
- `surfmnet_loss`: Functional map regularization (bijectivity, orthogonality)
- `couple_loss`, `align_loss`, `symmetry_loss`: Correspondence consistency
- `smoothness_loss`: Dirichlet energy on point-to-point maps
- `dirichlet_shape_loss`, `chamfer_shape_loss`, `edge_shape_loss`: Shape deformation regularization

### Data Structure

**Input Format:**
- 3D meshes in `.off` format under `data_root/off/`
- Preprocessing generates:
  - Laplacian eigenvectors/values in `data_root/diffusion/`
  - Mesh statistics in `data_root/mesh_info.csv`

**Dataset Types:**
- `PairPancreasDataset`: Returns pairs of shapes for correspondence learning (training)
- `SinglePancreasDataset`: Returns individual shapes for SSM building (testing)

**Data Dictionary Keys:**
- `verts`: Vertex coordinates [N, 3]
- `faces`: Triangle faces [F, 3]
- `evecs`, `evals`: Laplacian eigenvectors/eigenvalues
- `L`: Cotangent Laplacian matrix
- `mass`: Vertex mass matrix
- `name`: Shape identifier

## Configuration Files

YAML config files in `options/{train,test}/` specify:
- Dataset paths, splits (start_index, end_index)
- Network architectures and hyperparameters
- Training settings (epochs, optimizers, schedulers, losses)
- Validation metrics
- Paths for experiments, logs, visualizations

**Important:** Test configs require `resume_state` path to trained model checkpoint.

## Output Structure

```
experiments/
  └── {experiment_name}/
      ├── models/          # Saved checkpoints (.pth files)
      ├── visualization/   # .obj meshes and .mat correspondence files
      └── log/            # Training logs

results/
  └── {experiment_name}/  # Test results, metrics
```

## Key Utilities

- `utils/geometry_util.py`: Laplacian operations, spectral decomposition
- `utils/shape_util.py`: Mesh I/O, geodesic distances
- `utils/fmap_util.py`: Functional map ↔ point-to-point map conversions
- `utils/tensor_util.py`: Device management, numpy/torch conversions
- `utils/logger.py`: Logging, TensorBoard integration
- `utils/options.py`: YAML config parsing

## Notes on BaseModel

`models/base_model.py` is the abstract base class providing:
- Automatic network setup from config via registry
- Optimizer/scheduler creation (Adam, CosineAnnealingLR, etc.)
- Training state save/resume (checkpoints include networks, optimizers, schedulers, epoch/iter)
- Distributed training support (DataParallel, DistributedDataParallel)
- Validation loop with metrics computation

Subclasses must implement:
- `feed_data()`: Process input batch
- `validate_single()`: Validation logic for single sample
- `deform_template()`: Shape deformation for SSM building
- `get_loss_between_shapes()`: For reference shape selection

## Data Expectations

The pancreas dataset should be organized as:
```
../data/pancreas/
  └── off/
      ├── shape_001.off
      ├── shape_002.off
      └── ...
```

After preprocessing, additional files are created:
```
../data/pancreas/
  ├── off/
  ├── diffusion/           # Spectral operators cache
  └── mesh_info.csv        # Shape statistics
```