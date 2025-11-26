"""
Convert 3D segmentation masks to .off mesh format for FUSS framework.

This script processes medical imaging segmentation masks and converts them to .off format
meshes using marching cubes algorithm. If a segmentation contains multiple labels,
separate mesh files are created for each label.

Supported input formats:
- NIfTI (.nii, .nii.gz)
- NumPy arrays (.npy, .npz)
- DICOM series (directory)

Usage:
    # Single file conversion
    python segmentation_to_off.py --input seg_mask.nii.gz --output_dir data/pancreas/off

    # Batch conversion of directory
    python segmentation_to_off.py --input_dir segmentations/ --output_dir data/pancreas/off

    # Specify labels to extract
    python segmentation_to_off.py --input seg.nii.gz --output_dir data/output --labels 1 2 3

    # Apply smoothing and simplification
    python segmentation_to_off.py --input seg.nii.gz --output_dir data/output --smooth --simplify --target_faces 5000
"""

import os
import argparse
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm
import warnings

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    warnings.warn("nibabel not installed. NIfTI support disabled. Install with: pip install nibabel")

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False
    warnings.warn("pydicom not installed. DICOM support disabled. Install with: pip install pydicom")

try:
    from skimage import measure
    from skimage.morphology import binary_closing, ball
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    raise ImportError("scikit-image is required. Install with: pip install scikit-image")

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    warnings.warn("trimesh not installed. Mesh processing features disabled. Install with: pip install trimesh")

from utils.shape_util import write_off


def load_segmentation(file_path):
    """
    Load segmentation mask from file.

    Args:
        file_path (str): Path to segmentation file

    Returns:
        mask (np.ndarray): 3D segmentation mask
        affine (np.ndarray): Affine transformation matrix (if available)
        spacing (tuple): Voxel spacing (if available)
    """
    file_path = Path(file_path)
    affine = None
    spacing = (1.0, 1.0, 1.0)

    if file_path.suffix in ['.nii', '.gz']:
        if not HAS_NIBABEL:
            raise ImportError("nibabel required for NIfTI files. Install with: pip install nibabel")

        nii = nib.load(str(file_path))
        mask = nii.get_fdata()
        affine = nii.affine
        # Extract voxel spacing from affine
        # spacing = tuple(np.abs(np.diag(affine[:3, :3])))
        spacing = nii.header.get_zooms()

    elif file_path.suffix == '.npy':
        mask = np.load(str(file_path))

    elif file_path.suffix == '.npz':
        data = np.load(str(file_path))
        # Try common keys
        if 'segmentation' in data:
            mask = data['segmentation']
        elif 'mask' in data:
            mask = data['mask']
        elif 'data' in data:
            mask = data['data']
        else:
            # Use first array
            mask = data[list(data.keys())[0]]

        # Check for spacing/affine
        if 'spacing' in data:
            spacing = tuple(data['spacing'])
        if 'affine' in data:
            affine = data['affine']

    elif file_path.is_dir():
        if not HAS_PYDICOM:
            raise ImportError("pydicom required for DICOM files. Install with: pip install pydicom")

        # Load DICOM series
        dicom_files = sorted(glob(str(file_path / '*.dcm')))
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {file_path}")

        slices = [pydicom.dcmread(f) for f in dicom_files]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        mask = np.stack([s.pixel_array for s in slices], axis=-1)

        # Get spacing
        pixel_spacing = slices[0].PixelSpacing
        slice_thickness = slices[0].SliceThickness if hasattr(slices[0], 'SliceThickness') else 1.0
        spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]), float(slice_thickness))

    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    return mask.astype(np.int32), affine, spacing


def extract_unique_labels(mask, exclude_background=True):
    """
    Extract unique labels from segmentation mask.

    Args:
        mask (np.ndarray): Segmentation mask
        exclude_background (bool): Exclude label 0 (background)

    Returns:
        labels (list): List of unique labels
    """
    labels = np.unique(mask)
    if exclude_background:
        labels = labels[labels != 0]
    return labels.tolist()


def mask_to_mesh(mask, label=1, spacing=(1.0, 1.0, 1.0), smooth=False,
                 fill_holes=True, level=0.5):
    """
    Convert binary mask to mesh using marching cubes.

    Args:
        mask (np.ndarray): 3D binary mask
        label (int): Label value to extract
        spacing (tuple): Voxel spacing (x, y, z)
        smooth (bool): Apply Gaussian smoothing before marching cubes
        fill_holes (bool): Fill holes in binary mask using morphological closing
        level (float): Isosurface level for marching cubes

    Returns:
        verts (np.ndarray): Vertices [V, 3]
        faces (np.ndarray): Faces [F, 3]
    """
    # Extract binary mask for specific label
    binary_mask = (mask == label).astype(np.uint8)

    if binary_mask.sum() == 0:
        raise ValueError(f"Label {label} not found in mask")

    # Optional: fill holes using morphological closing
    if fill_holes:
        try:
            binary_mask = binary_closing(binary_mask, footprint=ball(1))
        except Exception as e:
            warnings.warn(f"Morphological closing failed: {e}")

    # Optional: smooth mask
    if smooth:
        from scipy.ndimage import gaussian_filter
        binary_mask = gaussian_filter(binary_mask.astype(float), sigma=0.5)

    # Run marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes(
            binary_mask,
            level=level,
            spacing=spacing,
            allow_degenerate=False
        )
    except Exception as e:
        raise RuntimeError(f"Marching cubes failed for label {label}: {e}")

    return verts, faces


def process_mesh(verts, faces, simplify=False, target_faces=None,
                 remove_small_components=True, min_component_size=100):
    """
    Post-process mesh (simplification, cleaning).

    Args:
        verts (np.ndarray): Vertices [V, 3]
        faces (np.ndarray): Faces [F, 3]
        simplify (bool): Simplify mesh
        target_faces (int): Target number of faces for simplification
        remove_small_components (bool): Remove small disconnected components
        min_component_size (int): Minimum number of faces for a component

    Returns:
        verts (np.ndarray): Processed vertices
        faces (np.ndarray): Processed faces
    """
    if not HAS_TRIMESH:
        warnings.warn("trimesh not available. Skipping mesh processing.")
        return verts, faces

    try:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        # Remove small disconnected components
        if remove_small_components:
            components = mesh.split(only_watertight=False)
            if len(components) > 1:
                # Keep only components with sufficient faces
                components = [c for c in components if len(c.faces) >= min_component_size]
                if components:
                    # Keep largest component
                    mesh = max(components, key=lambda c: len(c.faces))

        # Simplify mesh

        if simplify and target_faces is not None:
            if len(mesh.faces) > target_faces:
                faces_before = len(mesh.faces)
                mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
                faces_after = len(mesh.faces)
                print(f"    Simplified mesh from {faces_before} to {faces_after} faces")

        # More processing
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.process()
        mesh.remove_unreferenced_vertices()

        # Fix normals
        mesh.fix_normals()

        verts = np.array(mesh.vertices)
        faces = np.array(mesh.faces)

    except Exception as e:
        warnings.warn(f"Mesh processing failed: {e}. Returning original mesh.")
        """
        # print stack trace
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
        """

    return verts, faces


def convert_segmentation_to_off(input_path, output_dir, labels=None,
                                smooth=False, simplify=False, target_faces=None,
                                fill_holes=True, prefix='shape', verbose=True):
    """
    Convert segmentation mask to .off format meshes.

    Args:
        input_path (str): Path to segmentation file
        output_dir (str): Output directory for .off files
        labels (list): Specific labels to extract (None = all non-zero labels)
        smooth (bool): Apply smoothing
        simplify (bool): Simplify meshes
        target_faces (int): Target number of faces for simplification
        fill_holes (bool): Fill holes in binary masks
        prefix (str): Prefix for output filenames
        verbose (bool): Print progress

    Returns:
        output_files (list): List of generated .off files
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load segmentation
    if verbose:
        print(f"Loading segmentation from {input_path}")
    mask, affine, spacing = load_segmentation(input_path)

    # Get labels to process
    if labels is None:
        labels = extract_unique_labels(mask, exclude_background=True)

    if verbose:
        print(f"Found {len(labels)} label(s): {labels}")
        print(f"Mask shape: {mask.shape}, spacing: {spacing}")

    output_files = []

    # Process each label
    for label in tqdm(labels, desc="Processing labels", disable=not verbose):
        try:
            # Extract mesh
            verts, faces = mask_to_mesh(
                mask,
                label=label,
                spacing=spacing,
                smooth=smooth,
                fill_holes=fill_holes
            )

            if verbose:
                print(f"  Label {label}: {len(verts)} vertices, {len(faces)} faces")

            # Post-process mesh
            verts, faces = process_mesh(
                verts, faces,
                simplify=simplify,
                target_faces=target_faces
            )

            if verbose and simplify:
                print(f"  After processing: {len(verts)} vertices, {len(faces)} faces")

            # Generate output filename
            base_name = input_path.stem
            if base_name.endswith('.nii'):
                base_name = base_name[:-4]  # Remove .nii from .nii.gz

            if len(labels) > 1:
                output_file = output_dir / f"{prefix}_{base_name}_label{label}.off"
            else:
                output_file = output_dir / f"{prefix}_{base_name}.off"

            # Write .off file
            write_off(str(output_file), verts, faces)
            output_files.append(str(output_file))

            if verbose:
                print(f"  Saved to {output_file}")

        except Exception as e:
            print(f"Error processing label {label}: {e}")
            continue

    return output_files


def batch_convert(input_dir, output_dir, pattern='*.nii.gz', **kwargs):
    """
    Batch convert multiple segmentation files.

    Args:
        input_dir (str): Input directory
        output_dir (str): Output directory
        pattern (str): File pattern to match
        **kwargs: Additional arguments for convert_segmentation_to_off

    Returns:
        all_output_files (list): List of all generated .off files
    """
    input_dir = Path(input_dir)
    input_files = sorted(input_dir.glob(pattern))

    if not input_files:
        print(f"No files matching pattern '{pattern}' found in {input_dir}")
        return []

    print(f"Found {len(input_files)} file(s) to process")

    all_output_files = []

    for i, input_file in enumerate(input_files, 1):
        print(f"\n[{i}/{len(input_files)}] Processing {input_file.name}")
        try:
            output_files = convert_segmentation_to_off(
                input_file,
                output_dir,
                prefix=f"shape_{i:03d}",
                **kwargs
            )
            all_output_files.extend(output_files)
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            continue

    print(f"\n\nBatch conversion complete! Generated {len(all_output_files)} .off files")
    return all_output_files


def main():
    parser = argparse.ArgumentParser(
        description='Convert 3D segmentation masks to .off mesh format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input/output
    parser.add_argument('--input', type=str, help='Input segmentation file')
    parser.add_argument('--input_dir', type=str, help='Input directory for batch processing')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for .off files')
    parser.add_argument('--pattern', type=str, default='*.nii.gz',
                       help='File pattern for batch processing (default: *.nii.gz)')

    # Label selection
    parser.add_argument('--labels', type=int, nargs='+',
                       help='Specific labels to extract (default: all non-zero)')

    # Processing options
    parser.add_argument('--smooth', action='store_true',
                       help='Apply Gaussian smoothing before marching cubes')
    parser.add_argument('--no_fill_holes', action='store_true',
                       help='Disable hole filling with morphological closing')
    parser.add_argument('--simplify', action='store_true',
                       help='Simplify meshes using quadric decimation')
    parser.add_argument('--target_faces', type=int,
                       help='Target number of faces for simplification')
    parser.add_argument('--level', type=float, default=0.5,
                       help='Isosurface level for marching cubes (default: 0.5)')

    # Output naming
    parser.add_argument('--prefix', type=str, default='shape',
                       help='Prefix for output filenames (default: shape)')

    # Other
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')

    args = parser.parse_args()

    # Validate input
    if args.input is None and args.input_dir is None:
        parser.error("Either --input or --input_dir must be specified")

    if args.input is not None and args.input_dir is not None:
        parser.error("Cannot specify both --input and --input_dir")

    # Check dependencies
    if not HAS_SKIMAGE:
        raise ImportError("scikit-image is required. Install with: pip install scikit-image")

    # Run conversion
    kwargs = {
        'labels': args.labels,
        'smooth': args.smooth,
        'simplify': args.simplify,
        'target_faces': args.target_faces,
        'fill_holes': not args.no_fill_holes,
        'verbose': not args.quiet
    }

    if args.input_dir:
        # Batch processing
        batch_convert(
            args.input_dir,
            args.output_dir,
            pattern=args.pattern,
            **kwargs
        )
    else:
        # Single file processing
        output_files = convert_segmentation_to_off(
            args.input,
            args.output_dir,
            prefix=args.prefix,
            **kwargs
        )

        if not args.quiet:
            print(f"\nConversion complete! Generated {len(output_files)} .off file(s)")


if __name__ == '__main__':
    main()