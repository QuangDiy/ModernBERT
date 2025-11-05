"""Quick diagnostic script for MDS data issues.

Run this to quickly check if your data is properly formatted.

Usage:
    python tools/quick_data_check.py
"""

import json
import os
from pathlib import Path


def check_data_directory(data_dir: str):
    """Quick check of data directory structure."""
    data_path = Path(data_dir)
    
    print(f"\n{'='*60}")
    print(f"CHECKING: {data_dir}")
    print(f"{'='*60}\n")
    
    if not data_path.exists():
        print(f"❌ ERROR: Directory does not exist: {data_dir}")
        return False
    
    print(f"✓ Directory exists: {data_dir}")
    
    index_path = data_path / "index.json"
    if not index_path.exists():
        print(f"❌ ERROR: index.json not found in {data_dir}")
        return False
    
    print(f"✓ Found index.json")
    
    try:
        with open(index_path, 'r') as f:
            index = json.load(f)
        print(f"✓ index.json is valid JSON")
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: index.json is corrupted: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: Cannot read index.json: {e}")
        return False
    
    shards = index.get('shards', [])
    print(f"✓ Index lists {len(shards)} shards")
    
    if not shards:
        print(f"❌ ERROR: No shards defined in index.json")
        return False
    
    missing_shards = []
    corrupted_shards = []
    total_samples = 0
    
    for i, shard_info in enumerate(shards):
        shard_name = shard_info.get('raw_data', {}).get('basename', '')
        if not shard_name:
            print(f"⚠️  Warning: Shard {i} has no basename")
            continue
        
        shard_path = data_path / shard_name
        expected_samples = shard_info.get('samples', 0)
        total_samples += expected_samples
        
        if not shard_path.exists():
            missing_shards.append(shard_name)
        elif shard_path.stat().st_size == 0:
            corrupted_shards.append(f"{shard_name} (0 bytes)")
        else:
            size_mb = shard_path.stat().st_size / 1024 / 1024
            if i < 3:  # Show details for first 3 shards
                print(f"  ✓ {shard_name}: {size_mb:.2f} MB, {expected_samples:,} samples")
    
    if len(shards) > 3:
        print(f"  ... and {len(shards) - 3} more shards")
    
    print(f"\n📊 Summary:")
    print(f"  Total shards: {len(shards)}")
    print(f"  Total expected samples: {total_samples:,}")
    print(f"  Missing shards: {len(missing_shards)}")
    print(f"  Corrupted shards: {len(corrupted_shards)}")
    
    if missing_shards:
        print(f"\n❌ MISSING SHARD FILES:")
        for shard in missing_shards[:10]:
            print(f"  - {shard}")
        if len(missing_shards) > 10:
            print(f"  ... and {len(missing_shards) - 10} more")
        return False
    
    if corrupted_shards:
        print(f"\n❌ CORRUPTED SHARD FILES (0 bytes):")
        for shard in corrupted_shards[:10]:
            print(f"  - {shard}")
        if len(corrupted_shards) > 10:
            print(f"  ... and {len(corrupted_shards) - 10} more")
        return False
    
    print(f"\n✅ All shard files present and non-empty")
    return True


def main():
    print("\n" + "="*60)
    print("MDS DATA QUICK CHECK")
    print("="*60)
    
    # Check both train and val directories
    train_ok = check_data_directory("./data/train")
    val_ok = check_data_directory("./data/val")
    
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    
    if train_ok and val_ok:
        print("\n✅ DATA CHECK PASSED")
        print("\nYour data appears to be properly formatted.")
        print("If you're still getting errors, the shard contents may be corrupted.")
        print("\nRun the full validation:")
        print("  python tools/validate_mds_data.py --data-path ./data/train --max-shards 5")
    else:
        print("\n❌ DATA CHECK FAILED")
        print("\n🔧 NEXT STEPS:")
        print("1. If files are missing: re-copy or re-download your data")
        print("2. If index.json is corrupted: regenerate it from source data")
        print("3. If shards are corrupted: delete ./data and regenerate from source")
        print("\n📖 See README_VIETNAMESE_TRAINING.md for data preparation instructions")


if __name__ == '__main__':
    main()
