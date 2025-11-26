# Computing Shape Correspondences with FUSS

This document describes how to compute dense point-wise correspondences between 3D shapes using a trained FUSS model.

## Overview

After training a FUSS model, you can use it to compute correspondences between any pair of shapes in your dataset. The framework provides:

1. **Point-to-point maps**: Dense vertex-to-vertex correspondences
2. **Soft permutation matrices**: Probabilistic correspondences with confidence scores
3. **Functional maps**: Spectral domain correspondences

## Quick Start

### Command-Line Interface

Compute correspondence between two shapes:

```bash
python compute_correspondence.py \
    --model experiments/fuss_pancreas/models/final.pth \
    --config options/test/pancreas.yaml \
    --reference data/pancreas/off/shape_001.off \
    --target data/pancreas/off/shape_050.off \
    --output results/correspondences/
```

With visualization:

```bash
python compute_correspondence.py \
    --model experiments/fuss_pancreas/models/final.pth \
    --config options/test/pancreas.yaml \
    --reference data/pancreas/off/shape_001.off \
    --target data/pancreas/off/shape_050.off \
    --output results/correspondences/ \
    --visualize \
    --save_deformed
```

### Python API

```python
from correspondence_api import CorrespondenceComputer

# Initialize
computer = CorrespondenceComputer(
    model_path='experiments/fuss_pancreas/models/final.pth',
    config_path='options/test/pancreas.yaml',
    num_evecs=40
)

# Compute correspondence
results = computer.compute(
    reference_path='data/pancreas/off/shape_001.off',
    target_path='data/pancreas/off/shape_050.off'
)

# Access results
p2p = results['p2p_ref_to_target']  # Point-to-point map [N_ref]
P = results['P_ref_to_target']      # Soft permutation [N_ref, N_tgt]
C = results['C_ref_to_target']      # Functional map [K, K]
```

## Understanding the Results

### Point-to-Point Maps

The point-to-point map is an array of indices:

```python
p2p_ref_to_target[i] = j
```

This means vertex `i` in the reference shape corresponds to vertex `j` in the target shape.

**Usage:**
```python
# Transfer vertex positions
deformed_ref = target_verts[p2p_ref_to_target]

# Transfer per-vertex labels
target_labels = reference_labels[p2p_target_to_ref]
```

### Soft Permutation Matrices

The soft permutation matrix `P[i, j]` gives the "probability" that reference vertex `i` corresponds to target vertex `j`.

**Properties:**
- Shape: `[N_ref, N_tgt]`
- Values: `[0, 1]`
- Each row sums to approximately 1 (after Sinkhorn normalization)

**Usage:**
```python
# Soft transfer of features
soft_transferred = P_ref_to_target @ target_features

# Get correspondence confidence
confidence = P_ref_to_target.max(axis=1)
```

### Functional Maps

Functional maps represent correspondences in the spectral (frequency) domain using Laplacian eigenfunctions.

**Properties:**
- Shape: `[K, K]` where K is number of eigenvectors
- Compact representation of correspondences
- Useful for spectral analysis

**Usage:**
```python
# Transfer functions in spectral domain
target_coeffs = C_ref_to_target @ reference_coeffs
```

## Batch Processing

Compute correspondences from one reference to multiple targets:

```bash
python compute_correspondence.py \
    --model experiments/fuss_pancreas/models/final.pth \
    --config options/test/pancreas.yaml \
    --reference data/pancreas/off/shape_001.off \
    --target_dir data/pancreas/off/ \
    --output results/correspondences/
```

Using Python API:

```python
import os
from glob import glob

computer = CorrespondenceComputer(...)

reference = computer.load_shape('data/pancreas/off/shape_001.off')

for target_path in glob('data/pancreas/off/*.off'):
    target = computer.load_shape(target_path)

    results = computer.compute(
        reference_data=reference,
        target_data=target
    )

    # Process results...
```

## Applications

### 1. Label Transfer

Transfer segmentation labels from reference to target:

```python
# Reference has per-vertex labels (e.g., anatomical regions)
reference_labels = np.array([...])  # [N_ref]

# Transfer to target
target_labels = computer.transfer_labels(reference_labels, results)
```

### 2. Feature Transfer

Transfer any per-vertex features:

```python
# Reference has per-vertex features (e.g., curvature, thickness)
reference_features = np.array([...])  # [N_ref, D]

# Transfer to target
target_features = computer.transfer_features(reference_features, results)
```

### 3. Shape Deformation

Deform reference mesh to match target:

```python
deformed_verts = computer.deform_reference_to_target(results)

# Save deformed mesh
from utils.shape_util import write_off
write_off('deformed.off', deformed_verts, results['reference_faces'])
```

### 4. Statistical Analysis

Build statistical models by putting shapes in correspondence:

```python
# Compute correspondences for all shapes to a reference
reference = computer.load_shape('reference.off')

all_deformed = []
for target_path in target_paths:
    target = computer.load_shape(target_path)
    results = computer.compute(reference_data=reference, target_data=target)
    deformed = computer.deform_reference_to_target(results)
    all_deformed.append(deformed)

# Stack and compute PCA
all_deformed = np.stack(all_deformed)  # [N_shapes, N_verts, 3]
mean_shape = all_deformed.mean(axis=0)
# ... compute PCA modes ...
```

## Advanced Usage

### Custom Spectral Operators

Precompute spectral operators for faster processing:

```python
from datasets.shape_dataset import get_spectral_ops
import torch

# Load shape
verts, faces = read_shape('shape.off')
item = {
    'name': 'shape',
    'verts': torch.from_numpy(verts).float(),
    'faces': torch.from_numpy(faces).long()
}

# Compute and cache spectral operators
item = get_spectral_ops(
    item,
    num_evecs=40,
    cache_dir='cache/spectral_ops'
)

# Use cached data
results = computer.compute(reference_data=item, target_data=...)
```

### Correspondence Quality Assessment

Evaluate correspondence quality:

```python
# If you have ground truth correspondences
ground_truth = np.array([...])  # [N_ref]

metrics = computer.compute_correspondence_error(
    results,
    ground_truth_p2p=ground_truth
)

print(f"Accuracy: {metrics['accuracy']:.2%}")
```

### Visualization

The `--visualize` flag creates textured .obj files:

```bash
python compute_correspondence.py \
    --model experiments/fuss_pancreas/models/final.pth \
    --config options/test/pancreas.yaml \
    --reference shape_001.off \
    --target shape_050.off \
    --output results/ \
    --visualize
```

Output files:
```
results/visualizations/
  ├── shape_001.obj                      # Original reference
  ├── shape_001_to_shape_050.obj         # Target warped to reference topology
  ├── shape_050.obj                      # Original target
  └── shape_050_to_shape_001.obj         # Reference warped to target topology
```

Load these files in MeshLab or Blender to visualize correspondences through texture mapping.

## Output Format

### .mat Files

Correspondence results are saved as MATLAB .mat files:

```matlab
% Load in MATLAB
data = load('shape_001_to_shape_050.mat');

% Access results
p2p_ref_to_tgt = data.p2p_ref_to_tgt;  % [N_ref x 1], 1-indexed
p2p_tgt_to_ref = data.p2p_tgt_to_ref;  % [N_tgt x 1], 1-indexed
P_ref_to_tgt = data.P_ref_to_tgt;      % [N_ref x N_tgt]
C_ref_to_tgt = data.C_ref_to_tgt;      % [K x K]
```

```python
# Load in Python
import scipy.io as sio

data = sio.loadmat('shape_001_to_shape_050.mat')

p2p = data['p2p_ref_to_tgt'].squeeze() - 1  # Convert to 0-indexed
P = data['P_ref_to_tgt']
C = data['C_ref_to_tgt']
```

## Tips and Best Practices

### 1. Number of Eigenvectors

- **Default (40)**: Good balance of speed and accuracy
- **Fewer (20-30)**: Faster but less accurate for fine details
- **More (50-100)**: Better for high-frequency details but slower

```bash
python compute_correspondence.py ... --num_evecs 60
```

### 2. Caching Spectral Operators

Spectral decomposition is expensive. Cache results:

```bash
python compute_correspondence.py \
    ... \
    --cache_dir cache/spectral_ops
```

The cache directory structure:
```
cache/spectral_ops/
  ├── shape_001_000.mat
  ├── shape_002_000.mat
  └── ...
```

### 3. GPU Usage

Correspondences are computed on GPU by default (if available). For CPU:

```python
computer = CorrespondenceComputer(..., device='cpu')
```

### 4. Memory Considerations

For large meshes or batch processing:
- Use fewer eigenvectors (`--num_evecs 20`)
- Process shapes in batches
- Clear GPU memory between batches:

```python
import torch

for batch in batches:
    results = computer.compute(...)
    # Process results...
    torch.cuda.empty_cache()
```

## Troubleshooting

### "CUDA out of memory"

**Solution:**
- Reduce `--num_evecs`
- Use CPU: `device='cpu'`
- Process shapes sequentially instead of in batches

### Poor correspondence quality

**Possible causes:**
1. Shapes are too different from training data
2. Not enough eigenvectors
3. Model needs more training

**Solutions:**
- Increase `--num_evecs`
- Retrain model with more data
- Check if shapes are properly normalized

### Slow computation

**Solutions:**
- Use cached spectral operators: `--cache_dir`
- Reduce number of eigenvectors
- Use GPU if available

## Complete Example

Here's a complete workflow for transferring labels:

```python
from correspondence_api import CorrespondenceComputer
import numpy as np
from utils.shape_util import write_off

# 1. Initialize computer
computer = CorrespondenceComputer(
    model_path='experiments/fuss_pancreas/models/final.pth',
    config_path='options/test/pancreas.yaml',
    num_evecs=40,
    cache_dir='cache/spectral_ops'
)

# 2. Load reference shape with labels
reference = computer.load_shape('reference_with_labels.off')
reference_labels = np.load('reference_labels.npy')  # [N_ref]

# 3. Process all target shapes
target_paths = glob('data/test_set/*.off')

for target_path in target_paths:
    # Compute correspondence
    target = computer.load_shape(target_path)
    results = computer.compute(
        reference_data=reference,
        target_data=target
    )

    # Transfer labels
    target_labels = computer.transfer_labels(reference_labels, results)

    # Save results
    output_name = Path(target_path).stem
    np.save(f'results/{output_name}_labels.npy', target_labels)

    print(f"Processed {output_name}: {len(target_labels)} vertices labeled")
```

## References

- FUSS paper: "A Universal and Flexible Framework for Unsupervised Statistical Shape Model Learning" (MICCAI 2024)
- For functional maps: Ovsjanikov et al., "Functional Maps: A Flexible Representation of Maps Between Shapes" (2012)