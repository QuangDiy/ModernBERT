#!/usr/bin/env python3
"""
Merge multiple MDS subset folders into unified train/val/train_small splits.

This script takes multiple MDS datasets with subset folders and merges them into
a single dataset with proper train/val/train_small splits.

Usage:
    # From local directories
    python tools/merge_mds_subsets.py \
        --source_dir ./source_data \
        --output_dir ./data \
        --train_ratio 0.9 \
        --val_ratio 0.1 \
        --train_small_ratio 0.05 \
        --datasets FineWeb2-vie-mds FineWiki-mds
    
    # Download from Hugging Face and merge
    python tools/merge_mds_subsets.py \
        --hf_repos QuangDuy/FineWiki-mds QuangDuy/FineWeb2-vie-mds \
        --output_dir ./data \
        --train_ratio 0.9 \
        --val_ratio 0.1 \
        --train_small_ratio 0.05
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
from tqdm import tqdm


def download_from_huggingface(repo_id: str, cache_dir: Optional[Path] = None, 
                               token: Optional[str] = None) -> Path:
    """
    Download MDS dataset from Hugging Face repository.
    
    Args:
        repo_id: Hugging Face repository ID (e.g., 'QuangDuy/FineWiki-mds')
        cache_dir: Optional cache directory for downloads (default: ~/.cache/huggingface)
        token: Optional Hugging Face API token for private repos
        
    Returns:
        Path to the downloaded dataset directory
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download from Hugging Face. "
            "Install it with: pip install huggingface_hub"
        )
    
    print(f"Downloading {repo_id} from Hugging Face...")
    
    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir else None,
        token=token,
        resume_download=True,
    )
    
    print(f"  Downloaded to: {local_path}")
    return Path(local_path)


def discover_subset_folders(dataset_path: Path) -> List[Path]:
    """
    Discover all subset folders matching the pattern XXX_XXXXX.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        List of paths to subset folders, sorted by name
    """
    pattern = re.compile(r'^\d{3}_\d{5}$')
    subset_folders = []
    
    if not dataset_path.exists():
        print(f"Warning: Dataset path {dataset_path} does not exist")
        return []
    
    for item in dataset_path.iterdir():
        if item.is_dir() and pattern.match(item.name):
            index_file = item / "index.json"
            if index_file.exists():
                subset_folders.append(item)
            else:
                print(f"Warning: Found subset folder {item} but no index.json, skipping")
    
    return sorted(subset_folders)


def load_index_file(index_path: Path) -> Dict:
    """Load and parse an MDS index.json file."""
    with open(index_path, 'r') as f:
        return json.load(f)


def get_shard_info(subset_folder: Path) -> List[Dict]:
    """
    Get information about all shards in a subset folder.
    
    Args:
        subset_folder: Path to subset folder containing index.json
        
    Returns:
        List of shard info dictionaries
    """
    index_file = subset_folder / "index.json"
    index_data = load_index_file(index_file)
    
    shards = []
    for shard in index_data.get('shards', []):
        shard_info = {
            'subset_folder': subset_folder,
            'shard_data': shard,
            'raw_data': shard.get('raw_data', {}),
            'zip_data': shard.get('zip_data', {}),
            'samples': shard.get('samples', 0),
        }
        shards.append(shard_info)
    
    return shards


def split_shards(all_shards: List[Dict], train_ratio: float, val_ratio: float, 
                 train_small_ratio: float, seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split shards into train, val, and train_small sets.
    
    Args:
        all_shards: List of all shard info dictionaries
        train_ratio: Ratio of shards for training (e.g., 0.9)
        val_ratio: Ratio of shards for validation (e.g., 0.1)
        train_small_ratio: Ratio of train shards for train_small (e.g., 0.05)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_shards, val_shards, train_small_shards)
    """
    random.seed(seed)
    shuffled_shards = all_shards.copy()
    random.shuffle(shuffled_shards)
    
    total_shards = len(shuffled_shards)
    train_count = int(total_shards * train_ratio)
    
    train_shards = shuffled_shards[:train_count]
    val_shards = shuffled_shards[train_count:]
    
    train_small_count = max(1, int(len(train_shards) * train_small_ratio)) if train_shards else 0
    train_small_shards = train_shards[:train_small_count]
    
    return train_shards, val_shards, train_small_shards


def create_merged_split(output_split_dir: Path, shards: List[Dict], use_symlinks: bool = True):
    """
    Create a merged split directory with index.json and shard references.
    
    Args:
        output_split_dir: Path to output split directory (e.g., ./data/train)
        shards: List of shard info dictionaries to include
        use_symlinks: Whether to use symlinks (True) or copy files (False)
    """
    if not shards:
        print(f"  Warning: No shards to create split at {output_split_dir}")
        return
    
    output_split_dir.mkdir(parents=True, exist_ok=True)
    
    merged_shards = []
    
    symlink_failed = False
    
    print(f"  Processing {len(shards)} shards...")
    for idx, shard_info in enumerate(tqdm(shards, desc="  Creating shard references")):
        subset_folder = shard_info['subset_folder']
        shard_data = shard_info['shard_data']
        
        new_shard = shard_data.copy()
        
        if 'raw_data' in shard_data and shard_data['raw_data']:
            raw_basename = shard_data['raw_data'].get('basename', '')
            if raw_basename:
                source_file = subset_folder / raw_basename
                if source_file.exists():
                    dest_file = output_split_dir / f"shard.{idx:05d}.mds"
                    if use_symlinks and not symlink_failed:
                        try:
                            if dest_file.exists() or dest_file.is_symlink():
                                dest_file.unlink()
                            dest_file.symlink_to(source_file.absolute())
                        except (OSError, NotImplementedError) as e:
                            if idx == 0: 
                                print(f"\n  Warning: Symlinks not available ({e})")
                                print(f"  Falling back to copying files...")
                            symlink_failed = True
                            shutil.copy2(source_file, dest_file)
                    else:
                        shutil.copy2(source_file, dest_file)
                    new_shard['raw_data']['basename'] = dest_file.name
        
        if 'zip_data' in shard_data and shard_data['zip_data']:
            zip_basename = shard_data['zip_data'].get('basename', '')
            if zip_basename:
                source_file = subset_folder / zip_basename
                if source_file.exists():
                    ext = ''.join(source_file.suffixes) 
                    dest_file = output_split_dir / f"shard.{idx:05d}{ext}"
                    if use_symlinks and not symlink_failed:
                        try:
                            if dest_file.exists() or dest_file.is_symlink():
                                dest_file.unlink()
                            dest_file.symlink_to(source_file.absolute())
                        except (OSError, NotImplementedError) as e:
                            if idx == 0:  
                                print(f"\n  Warning: Symlinks not available ({e})")
                                print(f"  Falling back to copying files...")
                            symlink_failed = True
                            shutil.copy2(source_file, dest_file)
                    else:
                        shutil.copy2(source_file, dest_file)
                    new_shard['zip_data']['basename'] = dest_file.name
        
        merged_shards.append(new_shard)
    
    first_subset = shards[0]['subset_folder']
    template_index = load_index_file(first_subset / "index.json")
    
    merged_index = {
        'version': template_index.get('version', 2),
        'shards': merged_shards,
    }
    
    for key in template_index:
        if key not in merged_index:
            merged_index[key] = template_index[key]
    
    index_path = output_split_dir / "index.json"
    with open(index_path, 'w') as f:
        json.dump(merged_index, f, indent=2)
    
    print(f"  Created index.json with {len(merged_shards)} shards")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple MDS subset folders into unified train/val/train_small splits"
    )
    
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        '--source_dir',
        type=str,
        help='Source directory containing dataset folders (e.g., ./source_data)'
    )
    source_group.add_argument(
        '--hf_repos',
        type=str,
        nargs='+',
        help='Hugging Face repository IDs to download (e.g., QuangDuy/FineWiki-mds QuangDuy/FineWeb2-vie-mds)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for merged splits (e.g., ./data)'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        help='List of dataset folder names to merge (required when using --source_dir)'
    )
    parser.add_argument(
        '--hf_cache_dir',
        type=str,
        default=None,
        help='Cache directory for Hugging Face downloads (default: ~/.cache/huggingface)'
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='Hugging Face API token for private repositories'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.9,
        help='Ratio of data for training (default: 0.9)'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.1,
        help='Ratio of data for validation (default: 0.1)'
    )
    parser.add_argument(
        '--train_small_ratio',
        type=float,
        default=0.05,
        help='Ratio of train data for train_small split (default: 0.05)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible splits (default: 42)'
    )
    parser.add_argument(
        '--no-symlinks',
        action='store_true',
        help='Copy files instead of creating symlinks'
    )
    parser.add_argument(
        '--only-train-small',
        action='store_true',
        help='Only create train_small split without train/val (uses train_small_ratio from all data)'
    )
    
    args = parser.parse_args()
    
    if args.source_dir and not args.datasets:
        parser.error("--datasets is required when using --source_dir")
    
    if not args.only_train_small:
        if abs(args.train_ratio + args.val_ratio - 1.0) > 0.001:
            parser.error(f"train_ratio ({args.train_ratio}) + val_ratio ({args.val_ratio}) must equal 1.0")
    
    if args.train_small_ratio > 1.0 or args.train_small_ratio <= 0:
        parser.error(f"train_small_ratio must be between 0 and 1.0")
    
    output_dir = Path(args.output_dir)
    
    dataset_paths = []
    
    if args.hf_repos:
        print("="*60)
        print("Downloading datasets from Hugging Face")
        print("="*60)
        
        cache_dir = Path(args.hf_cache_dir) if args.hf_cache_dir else None
        
        for repo_id in args.hf_repos:
            try:
                dataset_path = download_from_huggingface(
                    repo_id=repo_id,
                    cache_dir=cache_dir,
                    token=args.hf_token
                )
                dataset_paths.append(dataset_path)
                print(f"✓ Successfully downloaded {repo_id}")
            except Exception as e:
                print(f"✗ Failed to download {repo_id}: {e}")
                return 1
        
        print()
    else:
        source_dir = Path(args.source_dir)
        if not source_dir.exists():
            parser.error(f"Source directory {source_dir} does not exist")
        
        for dataset_name in args.datasets:
            dataset_path = source_dir / dataset_name
            if not dataset_path.exists():
                print(f"Warning: Dataset path {dataset_path} does not exist, skipping")
                continue
            dataset_paths.append(dataset_path)
    
    if not dataset_paths:
        print("Error: No valid dataset paths found")
        return 1
    
    print("="*60)
    print(f"Merging MDS datasets to {output_dir}")
    print("="*60)
    print("="*60)
    print(f"Merging MDS datasets to {output_dir}")
    print("="*60)
    
    if args.only_train_small:
        print(f"Mode: Only creating train_small split ({args.train_small_ratio * 100}% of all data)")
    else:
        print(f"Split ratios: train={args.train_ratio}, val={args.val_ratio}, train_small={args.train_small_ratio}")
    
    print(f"Random seed: {args.seed}")
    print()
    
    all_shards = []
    for dataset_path in dataset_paths:
        dataset_name = dataset_path.name
        print(f"Processing dataset: {dataset_name}")
        
        subset_folders = discover_subset_folders(dataset_path)
        print(f"  Found {len(subset_folders)} subset folders")
        
        for subset_folder in subset_folders:
            print(f"    - {subset_folder.name}")
            shards = get_shard_info(subset_folder)
            all_shards.extend(shards)
            print(f"      ({len(shards)} shards)")
        print()
    
    print(f"Total shards collected: {len(all_shards)}")
    total_samples = sum(s['samples'] for s in all_shards)
    print(f"Total samples: {total_samples:,}")
    print()
    
    use_symlinks = not args.no_symlinks
    
    if args.only_train_small:
        print(f"Creating train_small split from all data...")
        random.seed(args.seed)
        shuffled_shards = all_shards.copy()
        random.shuffle(shuffled_shards)
        
        train_small_count = max(1, int(len(shuffled_shards) * args.train_small_ratio)) if shuffled_shards else 0
        train_small_shards = shuffled_shards[:train_small_count]
        
        print(f"  Train_small: {len(train_small_shards)} shards ({sum(s['samples'] for s in train_small_shards):,} samples)")
        print()
        
        print(f"Creating train_small split ({'symlinks' if use_symlinks else 'copying files'})...")
        create_merged_split(output_dir / "train_small", train_small_shards, use_symlinks)
        
        print("\n" + "="*60)
        print("✓ Merge complete!")
        print(f"✓ Output directory: {output_dir.absolute()}")
        print("="*60)
        print("\nYou can now use this data with your config:")
        print(f"  data_local: {output_dir}")
        print("  split: train_small")
    else:
        print("Splitting shards into train/val/train_small...")
        train_shards, val_shards, train_small_shards = split_shards(
            all_shards,
            args.train_ratio,
            args.val_ratio,
            args.train_small_ratio,
            args.seed
        )
        
        print(f"  Train: {len(train_shards)} shards ({sum(s['samples'] for s in train_shards):,} samples)")
        print(f"  Val: {len(val_shards)} shards ({sum(s['samples'] for s in val_shards):,} samples)")
        print(f"  Train_small: {len(train_small_shards)} shards ({sum(s['samples'] for s in train_small_shards):,} samples)")
        print()
        
        print(f"Creating merged splits ({'symlinks' if use_symlinks else 'copying files'})...")
        
        print("\nCreating train split...")
        create_merged_split(output_dir / "train", train_shards, use_symlinks)
        
        print("\nCreating val split...")
        create_merged_split(output_dir / "val", val_shards, use_symlinks)
        
        print("\nCreating train_small split...")
        create_merged_split(output_dir / "train_small", train_small_shards, use_symlinks)
        
        print("\n" + "="*60)
        print("✓ Merge complete!")
        print(f"✓ Output directory: {output_dir.absolute()}")
        print("="*60)
        print("\nYou can now use this data with your config:")
        print(f"  data_local: {output_dir}")
        print("  train split: train")
        print("  eval split: val")
        print("  quick test split: train_small")
    
    return 0


if __name__ == "__main__":
    exit(main())
