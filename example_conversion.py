"""
Example script demonstrating segmentation to .off conversion.

This creates a synthetic 3D segmentation mask and converts it to .off format
to demonstrate the conversion pipeline.
"""

import numpy as np
from pathlib import Path
from segmentation_to_off import convert_segmentation_to_off


def create_synthetic_segmentation(shape=(128, 128, 128), num_labels=2):
    """
    Create a synthetic 3D segmentation with multiple ellipsoid structures.

    Args:
        shape (tuple): Volume dimensions
        num_labels (int): Number of distinct labels to create

    Returns:
        mask (np.ndarray): 3D segmentation mask
    """
    mask = np.zeros(shape, dtype=np.int32)

    # Create ellipsoids at different positions
    centers = [
        (shape[0] // 2, shape[1] // 2, shape[2] // 2),  # Center
        (shape[0] // 3, shape[1] // 3, shape[2] // 2),  # Upper left
    ]

    radii = [
        (20, 25, 30),  # Label 1: elongated ellipsoid
        (15, 15, 20),  # Label 2: rounder ellipsoid
    ]

    for label_idx in range(min(num_labels, len(centers))):
        center = centers[label_idx]
        radius = radii[label_idx]

        # Create coordinate grids
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]

        # Ellipsoid equation: (x-cx)²/a² + (y-cy)²/b² + (z-cz)²/c² <= 1
        ellipsoid = (
            ((x - center[2]) / radius[2]) ** 2 +
            ((y - center[1]) / radius[1]) ** 2 +
            ((z - center[0]) / radius[0]) ** 2
        ) <= 1

        mask[ellipsoid] = label_idx + 1

    return mask


def example_single_label():
    """Example: Convert single-label segmentation to .off"""
    print("=" * 60)
    print("Example 1: Single Label Segmentation")
    print("=" * 60)

    # Create synthetic data
    print("\nCreating synthetic segmentation (single label)...")
    mask = create_synthetic_segmentation(num_labels=1)

    # Save as numpy array
    output_dir = Path("example_output")
    output_dir.mkdir(exist_ok=True)

    input_file = output_dir / "single_label_seg.npy"
    np.save(input_file, mask)
    print(f"Saved segmentation to {input_file}")
    print(f"  Shape: {mask.shape}")
    print(f"  Labels: {np.unique(mask)}")

    # Convert to .off
    print("\nConverting to .off format...")
    off_dir = output_dir / "off"
    output_files = convert_segmentation_to_off(
        input_path=input_file,
        output_dir=off_dir,
        smooth=True,
        prefix="synthetic"
    )

    print(f"\n✓ Generated {len(output_files)} .off file(s):")
    for f in output_files:
        print(f"  - {f}")


def example_multi_label():
    """Example: Convert multi-label segmentation to separate .off files"""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Label Segmentation")
    print("=" * 60)

    # Create synthetic data
    print("\nCreating synthetic segmentation (2 labels)...")
    mask = create_synthetic_segmentation(num_labels=2)

    # Save as numpy array
    output_dir = Path("example_output")
    output_dir.mkdir(exist_ok=True)

    input_file = output_dir / "multi_label_seg.npy"
    np.save(input_file, mask)
    print(f"Saved segmentation to {input_file}")
    print(f"  Shape: {mask.shape}")
    print(f"  Labels: {np.unique(mask)}")

    # Convert to .off - each label becomes a separate file
    print("\nConverting to .off format (separate file per label)...")
    off_dir = output_dir / "off"
    output_files = convert_segmentation_to_off(
        input_path=input_file,
        output_dir=off_dir,
        labels=None,  # Extract all labels
        smooth=True,
        prefix="multi_organ"
    )

    print(f"\n✓ Generated {len(output_files)} .off file(s):")
    for f in output_files:
        print(f"  - {f}")


def example_with_simplification():
    """Example: Convert with mesh simplification"""
    print("\n" + "=" * 60)
    print("Example 3: Conversion with Mesh Simplification")
    print("=" * 60)

    # Create synthetic data
    print("\nCreating synthetic segmentation...")
    mask = create_synthetic_segmentation(num_labels=1)

    # Save as numpy array
    output_dir = Path("example_output")
    input_file = output_dir / "seg_for_simplification.npy"
    np.save(input_file, mask)

    # Convert with simplification
    print("\nConverting to .off with simplification...")
    off_dir = output_dir / "off"
    output_files = convert_segmentation_to_off(
        input_path=input_file,
        output_dir=off_dir,
        smooth=True,
        simplify=True,
        target_faces=2000,  # Reduce to ~2000 faces
        prefix="simplified"
    )

    print(f"\n✓ Generated {len(output_files)} .off file(s):")
    for f in output_files:
        print(f"  - {f}")


def example_batch_processing():
    """Example: Batch convert multiple segmentation files"""
    print("\n" + "=" * 60)
    print("Example 4: Batch Processing")
    print("=" * 60)

    # Create multiple synthetic segmentations
    print("\nCreating 3 synthetic segmentations...")
    output_dir = Path("example_output") / "batch_input"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, 4):
        # Vary the size slightly for each
        shape = (100 + i * 10, 100 + i * 10, 100 + i * 10)
        mask = create_synthetic_segmentation(shape=shape, num_labels=1)

        input_file = output_dir / f"subject_{i:03d}.npy"
        np.save(input_file, mask)
        print(f"  Created {input_file.name}: shape {mask.shape}")

    # Batch convert
    print("\nBatch converting all .npy files...")
    from segmentation_to_off import batch_convert

    off_dir = Path("example_output") / "off_batch"
    all_files = batch_convert(
        input_dir=output_dir,
        output_dir=off_dir,
        pattern="*.npy",
        smooth=True,
        simplify=True,
        target_faces=3000
    )

    print(f"\n✓ Batch conversion complete!")
    print(f"  Generated {len(all_files)} .off file(s) in {off_dir}")


def visualize_example():
    """Example: Load and display mesh statistics"""
    print("\n" + "=" * 60)
    print("Example 5: Mesh Inspection")
    print("=" * 60)

    off_dir = Path("example_output") / "off"
    off_files = list(off_dir.glob("*.off"))

    if not off_files:
        print("No .off files found. Run previous examples first.")
        return

    print(f"\nFound {len(off_files)} .off file(s):")

    try:
        from utils.shape_util import read_shape

        for off_file in off_files[:5]:  # Show first 5
            verts, faces = read_shape(str(off_file))
            print(f"\n  {off_file.name}:")
            print(f"    Vertices: {len(verts)}")
            print(f"    Faces: {len(faces)}")
            print(f"    Bounds: [{verts.min(axis=0)} - {verts.max(axis=0)}]")

    except Exception as e:
        print(f"Could not load meshes: {e}")
        print("Make sure you're running from the repository root.")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("SEGMENTATION TO .OFF CONVERSION EXAMPLES")
    print("=" * 60)
    print("\nThis script demonstrates various use cases for converting")
    print("3D segmentation masks to .off format meshes.\n")

    # Run examples
    example_single_label()
    example_multi_label()
    example_with_simplification()
    example_batch_processing()
    visualize_example()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print(f"\nOutput files saved to: example_output/")
    print("\nNext steps:")
    print("  1. Inspect the generated .off files in MeshLab or similar")
    print("  2. Run preprocessing: python preprocess.py --data_root example_output/ --n_eig 200")
    print("  3. Configure options/train/*.yaml with your data paths")
    print("  4. Train FUSS model: python train.py --opt options/train/your_config.yaml")


if __name__ == "__main__":
    main()