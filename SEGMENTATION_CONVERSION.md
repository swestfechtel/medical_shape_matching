# Segmentation to .OFF Conversion Utility

This document describes how to use the `segmentation_to_off.py` utility to convert 3D medical imaging segmentation masks into .off format meshes compatible with the FUSS framework.

## Overview

The conversion script uses the **marching cubes algorithm** to extract surface meshes from volumetric segmentation masks. It supports multiple input formats and provides various preprocessing options.

## Requirements

### Required Dependencies
```bash
pip install scikit-image
```

### Optional Dependencies
```bash
# For NIfTI support (.nii, .nii.gz)
pip install nibabel

# For DICOM support
pip install pydicom

# For mesh processing (simplification, cleaning)
pip install trimesh

# For smoothing
pip install scipy
```

## Supported Input Formats

- **NIfTI**: `.nii`, `.nii.gz` (requires nibabel)
- **NumPy**: `.npy`, `.npz`
- **DICOM**: Directory containing DICOM series (requires pydicom)

## Basic Usage

### Single File Conversion

Convert a single segmentation mask:
```bash
python segmentation_to_off.py \
    --input path/to/segmentation.nii.gz \
    --output_dir data/pancreas/off
```

### Batch Conversion

Process an entire directory of segmentation files:
```bash
python segmentation_to_off.py \
    --input_dir path/to/segmentations/ \
    --output_dir data/pancreas/off \
    --pattern "*.nii.gz"
```

### Multi-Label Segmentation

If your segmentation contains multiple anatomical structures (e.g., label 1 = pancreas, label 2 = liver):
```bash
# Extract all labels (creates separate .off file for each)
python segmentation_to_off.py \
    --input multi_organ_seg.nii.gz \
    --output_dir data/output

# Extract specific labels only
python segmentation_to_off.py \
    --input multi_organ_seg.nii.gz \
    --output_dir data/pancreas/off \
    --labels 1  # Extract only pancreas (label 1)
```

## Advanced Options

### Mesh Simplification

Reduce the number of faces for faster processing:
```bash
python segmentation_to_off.py \
    --input seg.nii.gz \
    --output_dir data/output \
    --simplify \
    --target_faces 5000
```

### Smoothing

Apply Gaussian smoothing before mesh extraction for smoother surfaces:
```bash
python segmentation_to_off.py \
    --input seg.nii.gz \
    --output_dir data/output \
    --smooth
```

### Disable Hole Filling

By default, morphological closing is applied to fill small holes. To disable:
```bash
python segmentation_to_off.py \
    --input seg.nii.gz \
    --output_dir data/output \
    --no_fill_holes
```

### Custom Output Naming

```bash
python segmentation_to_off.py \
    --input seg.nii.gz \
    --output_dir data/output \
    --prefix pancreas
# Output: pancreas_seg.off
```

## Complete Workflow Example

Here's a complete workflow from segmentation to running FUSS:

```bash
# Step 1: Convert segmentations to .off format
python segmentation_to_off.py \
    --input_dir raw_data/segmentations/ \
    --output_dir data/pancreas/off \
    --pattern "*.nii.gz" \
    --smooth \
    --simplify \
    --target_faces 5000

# Step 2: Preprocess shapes (compute spectral operators)
python preprocess.py \
    --data_root data/pancreas/ \
    --n_eig 200

# Step 3: Train FUSS model
python train.py --opt options/train/pancreas.yaml

# Step 4: Test and build SSM
python test.py --opt options/test/pancreas.yaml
```

## Python API Usage

You can also use the conversion functions directly in your Python code:

```python
from segmentation_to_off import convert_segmentation_to_off, batch_convert

# Single file
output_files = convert_segmentation_to_off(
    input_path='seg.nii.gz',
    output_dir='data/output',
    labels=[1],  # Extract only label 1
    smooth=True,
    simplify=True,
    target_faces=5000
)

# Batch processing
all_files = batch_convert(
    input_dir='segmentations/',
    output_dir='data/output',
    pattern='*.nii.gz',
    smooth=True,
    simplify=True,
    target_faces=5000
)
```

## Output Format

The script generates .off (Object File Format) files with the following structure:

```
OFF
<num_vertices> <num_faces> 0
<x> <y> <z>  # vertex 0
<x> <y> <z>  # vertex 1
...
3 <v0> <v1> <v2>  # face 0 (triangle)
3 <v0> <v1> <v2>  # face 1
...
```

## Troubleshooting

### "No label found in mask"
- Check that your segmentation contains the specified label
- Verify the label values with: `python -c "import nibabel as nib; import numpy as np; print(np.unique(nib.load('seg.nii.gz').get_fdata()))"`

### "Marching cubes failed"
- Try adding `--smooth` to smooth the binary mask
- The segmentation might be too small or have issues
- Check if the label exists in the volume

### Mesh has too many faces
- Use `--simplify --target_faces 5000` to reduce mesh complexity
- Consider that FUSS works best with meshes of similar resolution

### Disconnected components
- The script automatically removes small disconnected components
- Only the largest connected component is kept
- Use trimesh for manual inspection: `python -c "import trimesh; mesh = trimesh.load('output.off'); print(mesh.split())"`

## Tips for Best Results

1. **Consistent Resolution**: Ensure all segmentations have similar voxel spacing for best results
2. **Label Values**: Use consistent label values across your dataset
3. **Mesh Quality**: For FUSS, aim for 3000-10000 faces per mesh
4. **Smoothing**: Apply smoothing if your segmentations are noisy or have jagged edges
5. **Validation**: Visually inspect a few converted meshes in MeshLab or similar before processing the entire dataset

## Visualization

You can visualize the generated .off files using:

- **MeshLab**: Open source mesh viewer (https://www.meshlab.net/)
- **Open3D**: `python -c "import open3d as o3d; mesh = o3d.io.read_triangle_mesh('shape.off'); o3d.visualization.draw_geometries([mesh])"`
- **PyVista**: `python -c "import pyvista as pv; mesh = pv.read('shape.off'); mesh.plot()"`

## Data Expectations

After conversion, your data directory should look like:

```
data/pancreas/
  └── off/
      ├── shape_001.off
      ├── shape_002.off
      ├── shape_003.off
      └── ...
```

Then run preprocessing:
```bash
python preprocess.py --data_root data/pancreas/ --n_eig 200
```

Which creates:
```
data/pancreas/
  ├── off/
  ├── diffusion/           # Spectral operators (created by preprocessing)
  └── mesh_info.csv        # Shape statistics (created by preprocessing)
```