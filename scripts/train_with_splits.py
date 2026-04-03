#!/usr/bin/env python
"""Train diffusion policy with compositional train/val/test splits.

Key differences from train_diffusion_policy.py:
  1. Loads split indices from splits_compositional.json
  2. Trains on specified train split only
  3. Validates on specified val split
  4. Supports evaluation on multiple test splits (trajectory/scene/full)

Usage:
    python scripts/train_with_splits.py --config configs/train.yaml --split_file data/demos/splits_compositional.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

# Import training functions from existing script
sys.path.append(str(Path(__file__).parent))
from train_diffusion_policy import (
    load_demos,
    create_dataloaders,
    train_epoch,
    validate,
    NoiseNet,
    DiffusionPolicyConfig,
)


def load_split_indices(split_file: Path) -> Dict:
    """Load train/val/test split indices from JSON."""
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    return split_data


def filter_demos_by_indices(demos: Dict, indices: List[int]) -> Dict:
    """Extract subset of demos by episode indices."""
    filtered = {}
    
    for key in demos.keys():
        data = demos[key]
        if key in ["obs", "actions"]:
            # (N, T, D) arrays
            filtered[key] = data[indices]
        elif key in ["episode_lengths", "labels", "config_ids"]:
            # (N,) arrays
            filtered[key] = data[indices]
        else:
            # Keep other metadata as-is
            filtered[key] = data
    
    return filtered


def create_validation_split(train_indices: List[int], val_config_id: int, 
                            all_config_ids: np.ndarray) -> tuple[List[int], List[int]]:
    """Create validation split by holding out one config from train set.
    
    Returns:
        train_final: indices for training (excluding val config)
        val: indices for validation (val config only)
    """
    train_indices = np.array(train_indices)
    config_ids = all_config_ids[train_indices]
    
    # Split by config
    val_mask = config_ids == val_config_id
    
    val_indices = train_indices[val_mask].tolist()
    train_final_indices = train_indices[~val_mask].tolist()
    
    return train_final_indices, val_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train.yaml")
    ap.add_argument("--split_file", type=str, default="data/demos/splits_compositional.json")
    ap.add_argument("--val_config", type=int, default=7, 
                   help="Config ID to use for validation (from train set)")
    ap.add_argument("--demo_path", type=str, default="data/demos/demos.npz")
    ap.add_argument("--output_dir", type=str, default="runs/compositional_split")
    args = ap.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load splits
    print("\n" + "="*60)
    print("COMPOSITIONAL SPLIT TRAINING")
    print("="*60)
    
    split_data = load_split_indices(Path(args.split_file))
    splits = split_data["splits"]
    
    print(f"\nSplit strategy: {split_data['strategy']}")
    print(f"Description: {split_data['description']}")
    
    # Load full demo dataset
    print(f"\nLoading demos from: {args.demo_path}")
    all_demos = dict(np.load(args.demo_path, allow_pickle=True))
    
    # Get train indices and create validation split
    train_indices_full = splits["train"]
    config_ids = all_demos.get("config_ids", np.zeros(len(all_demos["labels"])))
    
    train_indices, val_indices = create_validation_split(
        train_indices_full, args.val_config, config_ids
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train (final):      {len(train_indices)} episodes")
    print(f"  Validation:         {len(val_indices)} episodes (config {args.val_config})")
    print(f"  Test-trajectory:    {len(splits['test_trajectory'])} episodes")
    print(f"  Test-scene:         {len(splits['test_scene'])} episodes")
    print(f"  Test-full:          {len(splits['test_full'])} episodes")
    
    # Filter demos
    train_demos = filter_demos_by_indices(all_demos, train_indices)
    val_demos = filter_demos_by_indices(all_demos, val_indices)
    
    # Verify balance
    train_left = np.sum(train_demos["labels"] == "left")
    train_right = np.sum(train_demos["labels"] == "right")
    print(f"\nTrain balance: {train_left}L + {train_right}R = {train_left+train_right} "
          f"({train_left/(train_left+train_right):.1%} left)")
    
    val_left = np.sum(val_demos["labels"] == "left")
    val_right = np.sum(val_demos["labels"] == "right")
    print(f"Val balance:   {val_left}L + {val_right}R = {val_left+val_right} "
          f"({val_left/(val_left+val_right):.1%} left)")
    
    # Save split info
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    split_info = {
        "strategy": split_data["strategy"],
        "description": split_data["description"],
        "train_indices": train_indices,
        "val_indices": val_indices,
        "val_config": args.val_config,
        "test_trajectory_indices": splits["test_trajectory"],
        "test_scene_indices": splits["test_scene"],
        "test_full_indices": splits["test_full"],
    }
    
    with open(output_dir / "split_info.json", 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✓ Split info saved to: {output_dir / 'split_info.json'}")
    print("\nNow call train_diffusion_policy.py with custom train/val demo files")
    print("(Full integration requires modifying train_diffusion_policy.py to accept indices)")


if __name__ == "__main__":
    main()
