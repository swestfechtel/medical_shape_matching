"""
Python API for computing shape correspondences with FUSS.

This module provides a simple Python interface for computing correspondences
without using the command-line interface.
"""

import os
import torch
import numpy as np
from pathlib import Path

from models import build_model
from utils.options import parse_options
from utils.shape_util import read_shape
from utils.tensor_util import to_numpy
from utils.fmap_util import fmap2pointmap
from datasets.shape_dataset import get_spectral_ops


class CorrespondenceComputer:
    """
    High-level API for computing shape correspondences using trained FUSS model.

    Example:
        >>> computer = CorrespondenceComputer(
        ...     model_path='experiments/fuss_pancreas/models/final.pth',
        ...     config_path='options/test/pancreas.yaml'
        ... )
        >>>
        >>> # Compute correspondence between two shapes
        >>> results = computer.compute(
        ...     reference_path='data/pancreas/off/shape_001.off',
        ...     target_path='data/pancreas/off/shape_050.off'
        ... )
        >>>
        >>> # Access results
        >>> p2p = results['p2p_ref_to_target']  # Point-to-point map
        >>> P = results['P_ref_to_target']      # Soft permutation matrix
        >>> C = results['C_ref_to_target']      # Functional map
    """

    def __init__(self, model_path, config_path, num_evecs=40, cache_dir=None, device=None):
        """
        Initialize correspondence computer.

        Args:
            model_path (str): Path to trained model checkpoint
            config_path (str): Path to configuration file
            num_evecs (int): Number of eigenvectors to use
            cache_dir (str): Directory to cache spectral operators
            device (str): Device to run on ('cuda' or 'cpu', None=auto)
        """
        self.num_evecs = num_evecs
        self.cache_dir = cache_dir

        # Load model
        os.environ['opt'] = config_path
        opt = parse_options(root_path=os.getcwd(), is_train=False)
        opt['path']['resume_state'] = model_path
        opt['path']['resume'] = False

        self.model = build_model(opt)
        self.model.eval()

        # Set device
        if device is None:
            self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        else:
            self.device = torch.device(device)

        # Try to get cache_dir from config if not provided
        if self.cache_dir is None:
            if 'networks' in opt and 'feature_extractor' in opt['networks']:
                self.cache_dir = opt['networks']['feature_extractor'].get('cache_dir')

        print(f"CorrespondenceComputer initialized")
        print(f"  Model: {model_path}")
        print(f"  Device: {self.device}")
        print(f"  Num eigenvectors: {num_evecs}")

    def load_shape(self, shape_path):
        """
        Load shape from file and compute spectral operators.

        Args:
            shape_path (str): Path to .off file

        Returns:
            dict: Shape data dictionary
        """
        shape_path = Path(shape_path)
        verts, faces = read_shape(str(shape_path))

        item = {
            'name': shape_path.stem,
            'verts': torch.from_numpy(verts).float(),
            'faces': torch.from_numpy(faces).long()
        }

        item = get_spectral_ops(item, num_evecs=self.num_evecs, cache_dir=self.cache_dir)

        return item

    def compute(self, reference_path=None, target_path=None,
                reference_data=None, target_data=None):
        """
        Compute correspondence between reference and target shapes.

        Args:
            reference_path (str): Path to reference .off file
            target_path (str): Path to target .off file
            reference_data (dict): Pre-loaded reference shape data
            target_data (dict): Pre-loaded target shape data

        Returns:
            dict: Dictionary containing:
                - p2p_ref_to_target: Point-to-point map [N_ref]
                - p2p_target_to_ref: Point-to-point map [N_tgt]
                - P_ref_to_target: Soft permutation matrix [N_ref, N_tgt]
                - P_target_to_ref: Soft permutation matrix [N_tgt, N_ref]
                - C_ref_to_target: Functional map [K, K]
                - C_target_to_ref: Functional map [K, K]
                - reference_verts: Reference vertices [N_ref, 3]
                - target_verts: Target vertices [N_tgt, 3]
                - reference_faces: Reference faces [F_ref, 3]
                - target_faces: Target faces [F_tgt, 3]
        """
        # Load shapes if paths provided
        if reference_data is None:
            if reference_path is None:
                raise ValueError("Must provide either reference_path or reference_data")
            reference_data = self.load_shape(reference_path)

        if target_data is None:
            if target_path is None:
                raise ValueError("Must provide either target_path or target_data")
            target_data = self.load_shape(target_path)

        # Move to device and add batch dimension
        ref_data = {k: v.unsqueeze(0).to(self.device) if torch.is_tensor(v) else v
                    for k, v in reference_data.items()}
        tgt_data = {k: v.unsqueeze(0).to(self.device) if torch.is_tensor(v) else v
                    for k, v in target_data.items()}

        with torch.no_grad():
            # Extract features
            feat_ref = self.model.networks['feature_extractor'](
                ref_data['verts'], ref_data['faces']
            )
            feat_tgt = self.model.networks['feature_extractor'](
                tgt_data['verts'], tgt_data['faces']
            )

            # Compute soft permutation matrices
            P_rt, P_tr = self.model.compute_permutation_matrix(
                feat_ref, feat_tgt, bidirectional=True
            )
            P_rt = P_rt.squeeze(0)
            P_tr = P_tr.squeeze(0)

            # Get spectral operators
            evecs_ref = ref_data['evecs'].squeeze(0)
            evecs_tgt = tgt_data['evecs'].squeeze(0)
            evecs_trans_ref = ref_data['evecs_trans'].squeeze(0)
            evecs_trans_tgt = tgt_data['evecs_trans'].squeeze(0)

            # Compute functional maps
            C_tr = evecs_trans_ref @ (P_rt @ evecs_tgt)
            C_rt = evecs_trans_tgt @ (P_tr @ evecs_ref)

            # Convert to point-to-point maps
            p2p_rt = fmap2pointmap(C_tr, evecs_tgt, evecs_ref)
            p2p_tr = fmap2pointmap(C_rt, evecs_ref, evecs_tgt)

        # Return results
        return {
            'p2p_ref_to_target': to_numpy(p2p_rt),
            'p2p_target_to_ref': to_numpy(p2p_tr),
            'P_ref_to_target': to_numpy(P_rt),
            'P_target_to_ref': to_numpy(P_tr),
            'C_ref_to_target': to_numpy(C_tr),
            'C_target_to_ref': to_numpy(C_rt),
            'reference_verts': to_numpy(reference_data['verts']),
            'target_verts': to_numpy(target_data['verts']),
            'reference_faces': to_numpy(reference_data['faces']),
            'target_faces': to_numpy(target_data['faces']),
            'reference_name': reference_data['name'],
            'target_name': target_data['name'],
        }

    def transfer_labels(self, reference_labels, correspondence_result):
        """
        Transfer per-vertex labels from reference to target using correspondence.

        Args:
            reference_labels (np.ndarray): Labels on reference vertices [N_ref]
            correspondence_result (dict): Result from compute()

        Returns:
            np.ndarray: Labels transferred to target vertices [N_tgt]
        """
        p2p_tr = correspondence_result['p2p_target_to_ref']
        target_labels = reference_labels[p2p_tr]
        return target_labels

    def transfer_features(self, reference_features, correspondence_result):
        """
        Transfer per-vertex features from reference to target using correspondence.

        Args:
            reference_features (np.ndarray): Features on reference vertices [N_ref, D]
            correspondence_result (dict): Result from compute()

        Returns:
            np.ndarray: Features transferred to target vertices [N_tgt, D]
        """
        p2p_tr = correspondence_result['p2p_target_to_ref']
        target_features = reference_features[p2p_tr]
        return target_features

    def deform_reference_to_target(self, correspondence_result):
        """
        Deform reference mesh to match target shape using correspondence.

        Args:
            correspondence_result (dict): Result from compute()

        Returns:
            np.ndarray: Deformed reference vertices [N_ref, 3]
        """
        p2p_rt = correspondence_result['p2p_ref_to_target']
        target_verts = correspondence_result['target_verts']
        deformed_verts = target_verts[p2p_rt]
        return deformed_verts

    def compute_correspondence_error(self, correspondence_result, ground_truth_p2p=None):
        """
        Compute correspondence error metrics.

        Args:
            correspondence_result (dict): Result from compute()
            ground_truth_p2p (np.ndarray): Ground truth correspondence [N_ref]

        Returns:
            dict: Error metrics
        """
        ref_verts = correspondence_result['reference_verts']
        tgt_verts = correspondence_result['target_verts']
        p2p_rt = correspondence_result['p2p_ref_to_target']

        # Deformed reference
        deformed = tgt_verts[p2p_rt]

        # Geodesic error (if ground truth provided)
        metrics = {}

        if ground_truth_p2p is not None:
            # Percentage of correct correspondences
            correct = (p2p_rt == ground_truth_p2p).sum()
            metrics['accuracy'] = correct / len(p2p_rt)

        # Average Euclidean distance (as proxy for correspondence quality)
        # Note: This assumes similar poses, not valid for large deformations
        if ref_verts.shape == tgt_verts.shape:
            euclidean_dist = np.linalg.norm(ref_verts - deformed, axis=1)
            metrics['mean_euclidean_error'] = euclidean_dist.mean()
            metrics['median_euclidean_error'] = np.median(euclidean_dist)

        return metrics


def example_usage():
    """Example usage of the CorrespondenceComputer API."""

    # Initialize computer
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
    print("\nCorrespondence Results:")
    print(f"  Reference: {results['reference_name']}")
    print(f"  Target: {results['target_name']}")
    print(f"  Reference vertices: {len(results['reference_verts'])}")
    print(f"  Target vertices: {len(results['target_verts'])}")
    print(f"  Point-to-point map shape: {results['p2p_ref_to_target'].shape}")
    print(f"  Soft permutation matrix shape: {results['P_ref_to_target'].shape}")
    print(f"  Functional map shape: {results['C_ref_to_target'].shape}")

    # Deform reference to target
    deformed_verts = computer.deform_reference_to_target(results)
    print(f"\nDeformed reference vertices shape: {deformed_verts.shape}")

    # Transfer labels example
    # Simulate some labels on reference vertices
    reference_labels = np.random.randint(0, 5, size=len(results['reference_verts']))
    target_labels = computer.transfer_labels(reference_labels, results)
    print(f"\nTransferred labels shape: {target_labels.shape}")

    return results


if __name__ == '__main__':
    example_usage()