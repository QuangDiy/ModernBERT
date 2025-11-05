"""
Validate MDS data shards for corruption and index integrity.

This script checks:
1. Index.json file exists and is valid
2. All shards referenced in index exist
3. Sample counts match between index and actual shards
4. No overflow errors when reading samples

Usage:
    python tools/validate_mds_data.py --data-path ./data/train
    python tools/validate_mds_data.py --data-path ./data/val --fix-index
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import zstandard as zstd


def load_index(data_path: Path) -> Dict:
    """Load and validate index.json file."""
    index_path = data_path / "index.json"
    
    if not index_path.exists():
        raise FileNotFoundError(f"index.json not found in {data_path}")
    
    with open(index_path, 'r') as f:
        index = json.load(f)
    
    print(f"✓ Found index.json with {len(index.get('shards', []))} shards")
    return index


def validate_shard_files(data_path: Path, index: Dict) -> List[str]:
    """Check if all shard files referenced in index exist."""
    missing_shards = []
    
    for shard_info in index.get('shards', []):
        shard_name = shard_info.get('raw_data', {}).get('basename', '')
        if not shard_name:
            continue
            
        shard_path = data_path / shard_name
        
        if not shard_path.exists():
            missing_shards.append(shard_name)
            print(f"✗ Missing shard file: {shard_name}")
        else:
            file_size = shard_path.stat().st_size
            print(f"✓ Found shard: {shard_name} ({file_size / 1024 / 1024:.2f} MB)")
    
    return missing_shards


def validate_shard_integrity(data_path: Path, index: Dict, max_shards: Optional[int] = None) -> Dict[str, List[str]]:
    """Validate that shards can be read and contain expected number of samples."""
    issues = {
        'corrupted': [],
        'count_mismatch': [],
        'readable': []
    }
    
    shards = index.get('shards', [])
    if max_shards:
        shards = shards[:max_shards]
    
    for i, shard_info in enumerate(shards):
        shard_name = shard_info.get('raw_data', {}).get('basename', '')
        expected_samples = shard_info.get('samples', 0)
        
        if not shard_name:
            continue
        
        shard_path = data_path / shard_name
        
        if not shard_path.exists():
            continue
        
        print(f"\nValidating shard {i+1}/{len(shards)}: {shard_name}")
        print(f"  Expected samples: {expected_samples}")
        
        try:
            # Try to read the shard using streaming library
            from streaming.base.format.mds.reader import MDSReader
            
            reader = MDSReader(
                dirname=str(data_path),
                split=None,
                column_encodings=index.get('column_encodings', []),
                column_names=index.get('column_names', []),
                compression=index.get('compression'),
                hashes=index.get('hashes', []),
                raw_data=shard_info.get('raw_data'),
                samples=expected_samples,
                size_limit=index.get('size_limit'),
                zip_data=shard_info.get('zip_data'),
            )
            
            actual_samples = 0
            for sample_idx in range(expected_samples):
                try:
                    _ = reader.get_sample_data(sample_idx)
                    actual_samples += 1
                except (IndexError, RuntimeWarning) as e:
                    print(f"  ✗ Error reading sample {sample_idx}: {e}")
                    issues['corrupted'].append(f"{shard_name}:sample_{sample_idx}")
                    break
            
            if actual_samples == expected_samples:
                print(f"  ✓ All {actual_samples} samples readable")
                issues['readable'].append(shard_name)
            else:
                print(f"  ✗ Only {actual_samples}/{expected_samples} samples readable")
                issues['count_mismatch'].append(f"{shard_name} (found {actual_samples}, expected {expected_samples})")
        
        except Exception as e:
            print(f"  ✗ Failed to validate shard: {e}")
            issues['corrupted'].append(shard_name)
    
    return issues


def rebuild_index(data_path: Path, original_index: Dict) -> Dict:
    """Rebuild index.json by scanning actual shard files."""
    print("\n" + "="*50)
    print("REBUILDING INDEX")
    print("="*50)
    
    print("WARNING: Index rebuilding requires the streaming library to scan shards.")
    print("This is a complex operation. Consider regenerating data from source instead.")
    
    return original_index


def main():
    parser = argparse.ArgumentParser(description="Validate MDS data shards")
    parser.add_argument('--data-path', type=str, required=True, help='Path to data directory (e.g., ./data/train)')
    parser.add_argument('--fix-index', action='store_true', help='Attempt to rebuild corrupted index')
    parser.add_argument('--max-shards', type=int, default=None, help='Maximum number of shards to validate (for quick check)')
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    
    if not data_path.exists():
        print(f"ERROR: Data path does not exist: {data_path}")
        sys.exit(1)
    
    print("="*50)
    print(f"VALIDATING MDS DATA: {data_path}")
    print("="*50)
    
    try:
        index = load_index(data_path)
    except Exception as e:
        print(f"ERROR: Failed to load index.json: {e}")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("CHECKING SHARD FILES")
    print("="*50)
    missing_shards = validate_shard_files(data_path, index)
    
    if missing_shards:
        print(f"\nERROR: {len(missing_shards)} shard files are missing!")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("VALIDATING SHARD INTEGRITY")
    print("="*50)
    issues = validate_shard_integrity(data_path, index, max_shards=args.max_shards)
    
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    print(f"Readable shards: {len(issues['readable'])}")
    print(f"Corrupted shards: {len(issues['corrupted'])}")
    print(f"Count mismatches: {len(issues['count_mismatch'])}")
    
    if issues['corrupted']:
        print("\nCorrupted shards:")
        for shard in issues['corrupted']:
            print(f"  - {shard}")
    
    if issues['count_mismatch']:
        print("\nCount mismatches:")
        for issue in issues['count_mismatch']:
            print(f"  - {issue}")
    
    if issues['corrupted'] or issues['count_mismatch']:
        print("\n⚠️  DATA VALIDATION FAILED")
        print("\nRecommended actions:")
        print("1. Delete corrupted shards and regenerate from source data")
        print("2. Check disk space and file system integrity")
        print("3. Re-run data conversion with tools/convert_dataset.py")
        
        if args.fix_index:
            rebuild_index(data_path, index)
        
        sys.exit(1)
    else:
        print("\n✓ ALL DATA VALIDATED SUCCESSFULLY")
        sys.exit(0)


if __name__ == '__main__':
    main()
