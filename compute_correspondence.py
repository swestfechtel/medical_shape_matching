"""
Compute point-wise correspondences between shapes using a trained FUSS model.

This script loads a trained FUSS model and computes dense point-to-point
correspondences between a reference shape and one or more target shapes.

Usage:
    # Single correspondence
    python compute_correspondence.py \
        --model experiments/fuss_pancreas/models/final.pth \
        --config options/test/pancreas.yaml \
        --reference data/pancreas/off/shape_001.off \
        --target data/pancreas/off/shape_050.off \
        --output results/correspondences/

    # Batch correspondences (reference to all targets)
    python compute_correspondence.py \
        --model experiments/fuss_pancreas/models/final.pth \
        --config options/test/pancreas.yaml \
        --reference data/pancreas/off/shape_001.off \
        --target_dir data/pancreas/off/ \
        --output results/correspondences/

    # With visualization
    python compute_correspondence.py \
        --model experiments/fuss_pancreas/models/final.pth \
        --config options/test/pancreas.yaml \
        --reference data/pancreas/off/shape_001.off \
        --target data/pancreas/off/shape_050.off \
        --output results/correspondences/ \
        --visualize
"""

import os
import argparse
import numpy as np
import scipy.io as sio
from pathlib import Path
from glob import glob
from tqdm import tqdm

import torch

from models import build_model
from utils.options import parse_options
from utils.shape_util import read_shape, write_off
from utils.tensor_util import to_device, to_numpy
from utils.fmap_util import fmap2pointmap
from utils.texture_util import write_obj_pair
from datasets.shape_dataset import get_spectral_ops


def load_shape(shape_path, num_evecs=40, cache_dir=None):
    """
    Load a shape from .off file and compute spectral operators.

    Args:
        shape_path (str): Path to .off file
        num_evecs (int): Number of eigenvectors to compute
        cache_dir (str): Directory to cache spectral operators

    Returns:
        item (dict): Dictionary containing shape data
    """
    shape_path = Path(shape_path)
    verts, faces = read_shape(str(shape_path))

    item = {
        'name': shape_path.stem,
        'verts': torch.from_numpy(verts).float(),
        'faces': torch.from_numpy(faces).long()
    }

    # Compute spectral operators
    item = get_spectral_ops(item, num_evecs=num_evecs, cache_dir=cache_dir)

    return item


def compute_correspondence(model, reference, target, device='cuda'):
    """
    Compute point-wise correspondence between reference and target shapes.

    Args:
        model: Trained FUSS model
        reference (dict): Reference shape data
        target (dict): Target shape data
        device (str): Device to run computation on

    Returns:
        p2p_ref_to_tgt (np.ndarray): Point-to-point map from reference to target [N_ref]
        p2p_tgt_to_ref (np.ndarray): Point-to-point map from target to reference [N_tgt]
        P_ref_to_tgt (np.ndarray): Soft permutation matrix [N_ref, N_tgt]
        P_tgt_to_ref (np.ndarray): Soft permutation matrix [N_tgt, N_ref]
        C_ref_to_tgt (np.ndarray): Functional map [K, K]
        C_tgt_to_ref (np.ndarray): Functional map [K, K]
    """
    model.eval()

    # Move data to device and add batch dimension
    ref_data = {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) else v
                for k, v in reference.items()}
    tgt_data = {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) else v
                for k, v in target.items()}

    with torch.no_grad():
        # Extract features
        feat_ref = model.networks['feature_extractor'](ref_data['verts'], ref_data['faces'])
        feat_tgt = model.networks['feature_extractor'](tgt_data['verts'], tgt_data['faces'])

        # Compute soft permutation matrices
        P_rt, P_tr = model.compute_permutation_matrix(feat_ref, feat_tgt, bidirectional=True)
        P_rt = P_rt.squeeze(0)  # [N_ref, N_tgt]
        P_tr = P_tr.squeeze(0)  # [N_tgt, N_ref]

        # Get spectral operators
        evecs_ref = ref_data['evecs'].squeeze(0)
        evecs_tgt = tgt_data['evecs'].squeeze(0)
        evecs_trans_ref = ref_data['evecs_trans'].squeeze(0)
        evecs_trans_tgt = tgt_data['evecs_trans'].squeeze(0)

        # Compute functional maps from soft permutation
        C_tr = evecs_trans_ref @ (P_rt @ evecs_tgt)  # ref to tgt in spectral domain
        C_rt = evecs_trans_tgt @ (P_tr @ evecs_ref)  # tgt to ref in spectral domain

        # Convert functional maps to point-to-point maps
        p2p_ref_to_tgt = fmap2pointmap(C_tr, evecs_tgt, evecs_ref)
        p2p_tgt_to_ref = fmap2pointmap(C_rt, evecs_ref, evecs_tgt)

        # Convert to numpy
        p2p_ref_to_tgt = to_numpy(p2p_ref_to_tgt)
        p2p_tgt_to_ref = to_numpy(p2p_tgt_to_ref)
        P_rt = to_numpy(P_rt)
        P_tr = to_numpy(P_tr)
        C_tr = to_numpy(C_tr)
        C_rt = to_numpy(C_rt)

    return p2p_ref_to_tgt, p2p_tgt_to_ref, P_rt, P_tr, C_tr, C_rt


def save_correspondence(output_path, reference, target,
                       p2p_ref_to_tgt, p2p_tgt_to_ref,
                       P_rt, P_tr, C_rt, C_tr):
    """
    Save correspondence results to .mat file.

    Args:
        output_path (str): Output file path
        reference (dict): Reference shape data
        target (dict): Target shape data
        p2p_ref_to_tgt (np.ndarray): Point map reference to target
        p2p_tgt_to_ref (np.ndarray): Point map target to reference
        P_rt (np.ndarray): Soft permutation matrix ref to target
        P_tr (np.ndarray): Soft permutation matrix target to ref
        C_rt (np.ndarray): Functional map target to ref
        C_tr (np.ndarray): Functional map ref to target
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'reference_name': reference['name'],
        'target_name': target['name'],
        # Point-to-point maps (1-indexed for MATLAB compatibility)
        'p2p_ref_to_tgt': p2p_ref_to_tgt + 1,
        'p2p_tgt_to_ref': p2p_tgt_to_ref + 1,
        # Soft permutation matrices
        'P_ref_to_tgt': P_rt,
        'P_tgt_to_ref': P_tr,
        # Functional maps
        'C_ref_to_tgt': C_tr,
        'C_tgt_to_ref': C_rt,
    }

    sio.savemat(str(output_path), save_dict)


def visualize_correspondence(output_dir, reference, target, p2p_ref_to_tgt, p2p_tgt_to_ref):
    """
    Create visualization of correspondences as .obj files with texture mapping.

    Args:
        output_dir (str): Output directory
        reference (dict): Reference shape data
        target (dict): Target shape data
        p2p_ref_to_tgt (np.ndarray): Point map reference to target
        p2p_tgt_to_ref (np.ndarray): Point map target to reference
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    verts_ref = to_numpy(reference['verts'])
    faces_ref = to_numpy(reference['faces'])
    verts_tgt = to_numpy(target['verts'])
    faces_tgt = to_numpy(target['faces'])

    ref_name = reference['name']
    tgt_name = target['name']

    # Create .obj files with texture mapping
    file_ref = output_dir / f'{ref_name}.obj'
    file_mapped = output_dir / f'{ref_name}_to_{tgt_name}.obj'

    # Copy texture file if it exists
    texture_file = Path(__file__).parent / 'figures' / 'texture.png'
    if texture_file.exists():
        import shutil
        shutil.copy(texture_file, output_dir / 'texture.png')
        texture_name = 'texture.png'
    else:
        texture_name = None

    # Write reference and mapped target
    write_obj_pair(str(file_ref), str(file_mapped),
                   verts_ref, faces_ref, verts_tgt, faces_tgt,
                   p2p_tgt_to_ref, texture_name)

    # Also create the reverse mapping
    file_tgt = output_dir / f'{tgt_name}.obj'
    file_mapped_rev = output_dir / f'{tgt_name}_to_{ref_name}.obj'

    write_obj_pair(str(file_tgt), str(file_mapped_rev),
                   verts_tgt, faces_tgt, verts_ref, faces_ref,
                   p2p_ref_to_tgt, texture_name)


def deform_shape(reference, target, p2p_ref_to_tgt):
    """
    Deform reference shape to target using point-to-point correspondence.

    Args:
        reference (dict): Reference shape data
        target (dict): Target shape data
        p2p_ref_to_tgt (np.ndarray): Point map reference to target

    Returns:
        deformed_verts (np.ndarray): Deformed vertices [N_ref, 3]
    """
    verts_ref = to_numpy(reference['verts'])
    verts_tgt = to_numpy(target['verts'])

    # Transfer target positions to reference mesh
    deformed_verts = verts_tgt[p2p_ref_to_tgt]

    return deformed_verts


def load_trained_model(model_path, config_path):
    """
    Load a trained FUSS model from checkpoint.

    Args:
        model_path (str): Path to model checkpoint (.pth file)
        config_path (str): Path to config file (.yaml file)

    Returns:
        model: Loaded model
        device: Device the model is on
    """
    # Parse config
    opt = parse_options(root_path=os.getcwd(), is_train=False)

    # Update config with checkpoint path
    opt['path']['resume_state'] = model_path
    opt['path']['resume'] = False  # Only load model weights, not training state

    # Build model (automatically loads checkpoint)
    model = build_model(opt)

    device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')

    print(f"Loaded model from {model_path}")
    print(f"Running on device: {device}")

    return model, device, opt


def main():
    parser = argparse.ArgumentParser(
        description='Compute point-wise correspondences using trained FUSS model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model and config
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file (.yaml file)')

    # Input shapes
    parser.add_argument('--reference', type=str, required=True,
                       help='Path to reference shape (.off file)')
    parser.add_argument('--target', type=str,
                       help='Path to target shape (.off file)')
    parser.add_argument('--target_dir', type=str,
                       help='Directory containing target shapes (batch mode)')

    # Output
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for correspondences')

    # Options
    parser.add_argument('--num_evecs', type=int, default=40,
                       help='Number of eigenvectors to use (default: 40)')
    parser.add_argument('--cache_dir', type=str,
                       help='Directory to cache spectral operators')
    parser.add_argument('--visualize', action='store_true',
                       help='Create .obj visualizations of correspondences')
    parser.add_argument('--save_deformed', action='store_true',
                       help='Save deformed reference shape')

    args = parser.parse_args()

    # Validate inputs
    if args.target is None and args.target_dir is None:
        parser.error("Either --target or --target_dir must be specified")

    if args.target is not None and args.target_dir is not None:
        parser.error("Cannot specify both --target and --target_dir")

    # Load model
    print("Loading trained model...")
    os.environ['opt'] = args.config  # Set config path for parse_options
    model, device, opt = load_trained_model(args.model, args.config)

    # Set cache directory
    cache_dir = args.cache_dir
    if cache_dir is None:
        # Try to use the cache from config
        if 'networks' in opt and 'feature_extractor' in opt['networks']:
            cache_dir = opt['networks']['feature_extractor'].get('cache_dir')

    # Load reference shape
    print(f"\nLoading reference shape: {args.reference}")
    reference = load_shape(args.reference, num_evecs=args.num_evecs, cache_dir=cache_dir)
    print(f"  Vertices: {reference['verts'].shape[0]}")
    print(f"  Faces: {reference['faces'].shape[0]}")

    # Get target shapes
    if args.target:
        target_files = [args.target]
    else:
        target_files = sorted(glob(os.path.join(args.target_dir, '*.off')))
        # Exclude reference if it's in the directory
        ref_path = str(Path(args.reference).resolve())
        target_files = [f for f in target_files if str(Path(f).resolve()) != ref_path]

    print(f"\nProcessing {len(target_files)} target shape(s)...")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each target
    for target_file in tqdm(target_files, desc="Computing correspondences"):
        # Load target shape
        target = load_shape(target_file, num_evecs=args.num_evecs, cache_dir=cache_dir)

        # Compute correspondence
        p2p_rt, p2p_tr, P_rt, P_tr, C_tr, C_rt = compute_correspondence(
            model, reference, target, device=device
        )

        # Save results
        output_name = f"{reference['name']}_to_{target['name']}.mat"
        output_path = output_dir / output_name
        save_correspondence(
            output_path, reference, target,
            p2p_rt, p2p_tr, P_rt, P_tr, C_rt, C_tr
        )

        # Visualize if requested
        if args.visualize:
            vis_dir = output_dir / 'visualizations'
            visualize_correspondence(vis_dir, reference, target, p2p_rt, p2p_tr)

        # Save deformed shape if requested
        if args.save_deformed:
            deformed_verts = deform_shape(reference, target, p2p_rt)
            deformed_file = output_dir / 'deformed' / f"{reference['name']}_deformed_to_{target['name']}.off"
            deformed_file.parent.mkdir(exist_ok=True)
            write_off(str(deformed_file), deformed_verts, to_numpy(reference['faces']))

    print(f"\n✓ Correspondences saved to {output_dir}")
    if args.visualize:
        print(f"✓ Visualizations saved to {output_dir / 'visualizations'}")
    if args.save_deformed:
        print(f"✓ Deformed shapes saved to {output_dir / 'deformed'}")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Reference shape: {reference['name']}")
    print(f"Number of targets: {len(target_files)}")
    print(f"Outputs:")
    print(f"  - Correspondence files (.mat): {output_dir}")
    if args.visualize:
        print(f"  - Visualizations (.obj): {output_dir / 'visualizations'}")
    if args.save_deformed:
        print(f"  - Deformed shapes (.off): {output_dir / 'deformed'}")


if __name__ == '__main__':
    main()