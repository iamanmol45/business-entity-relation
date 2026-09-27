"""
Export SQLite postings to memory-mapped sorted binary arrays (.npy).
"""

import sys
import gc
import time
import zlib
import array
import sqlite3
import numpy as np
from pathlib import Path

def hash64(s_bytes: bytes) -> int:
    return (zlib.crc32(s_bytes) << 32) | zlib.adler32(s_bytes)

def export_all_strategies(db_path: Path, out_dir: Path):
    print("=" * 80, flush=True)
    print("EXPORTING ALL 8 BLOCKING STRATEGIES TO MMAP-READY NUMPY ARRAYS", flush=True)
    print("=" * 80, flush=True)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA cache_size = 50000")
    
    # Check what strategies already exist
    for strat_id in range(8):
        k_file = out_dir / f"strat_{strat_id}_keys.npy"
        o_file = out_dir / f"strat_{strat_id}_offsets.npy"
        c_file = out_dir / f"strat_{strat_id}_counts.npy"
        p_file = out_dir / f"strat_{strat_id}_postings.npy"
        
        if k_file.exists() and o_file.exists() and c_file.exists() and p_file.exists():
            print(f"Strategy {strat_id} already exported. Skipping.", flush=True)
            continue
            
        print(f"\nProcessing Strategy {strat_id}...", flush=True)
        t0 = time.time()
        cur = conn.cursor()
        cur.execute("SELECT key, eid FROM postings WHERE strat_id = ? ORDER BY key", (strat_id,))
        
        # Use compact C-level arrays to prevent millions of PyObject allocations
        keys_arr = array.array('Q')
        offsets_arr = array.array('I')
        counts_arr = array.array('H')
        postings_arr = array.array('I')
        
        cur_key = None
        cur_count = 0
        
        for key, eid in cur:
            is_s3 = eid.startswith("S3-")
            eid_int = int(eid[3:])
            val = (eid_int | 0x80000000) if is_s3 else eid_int
            
            if key != cur_key:
                if cur_key is not None:
                    counts_arr.append(min(cur_count, 500))
                cur_key = key
                cur_count = 1
                keys_arr.append(hash64(key.encode("utf-8")))
                offsets_arr.append(len(postings_arr))
                postings_arr.append(val)
            else:
                cur_count += 1
                if cur_count <= 500:
                    postings_arr.append(val)
                    
        if cur_key is not None:
            counts_arr.append(min(cur_count, 500))
            
        t1 = time.time()
        print(f"  Fetched & hashed {len(keys_arr):,} unique keys, {len(postings_arr):,} postings in {t1-t0:.1f}s", flush=True)
        
        # Zero-copy buffer conversion to numpy arrays
        arr_keys = np.frombuffer(keys_arr, dtype=np.uint64).copy()
        arr_offsets = np.frombuffer(offsets_arr, dtype=np.uint32).copy()
        arr_counts = np.frombuffer(counts_arr, dtype=np.uint16).copy()
        arr_postings = np.frombuffer(postings_arr, dtype=np.uint32).copy()
        
        # Release array buffers
        del keys_arr, offsets_arr, counts_arr, postings_arr
        gc.collect()
        
        # Sort by 64-bit key hash for C binary search (np.searchsorted)
        sort_order = np.argsort(arr_keys)
        arr_keys = arr_keys[sort_order]
        arr_offsets = arr_offsets[sort_order]
        arr_counts = arr_counts[sort_order]
        del sort_order
        gc.collect()
        
        # Save separate .npy files for direct mmap
        np.save(k_file, arr_keys)
        np.save(o_file, arr_offsets)
        np.save(c_file, arr_counts)
        np.save(p_file, arr_postings)
        
        t2 = time.time()
        total_mb = (k_file.stat().st_size + o_file.stat().st_size + c_file.stat().st_size + p_file.stat().st_size) / (1024**2)
        print(f"  Strategy {strat_id} saved to disk in {t2-t1:.1f}s | Total Size: {total_mb:.1f} MB (Total Time: {t2-t0:.1f}s)", flush=True)
        
        # Free memory immediately
        del arr_keys, arr_offsets, arr_counts, arr_postings
        gc.collect()
        
    conn.close()
    print("\n" + "=" * 80, flush=True)
    print("ALL 8 STRATEGIES SUCCESSFULLY EXPORTED TO MMAP BINARY ARRAYS!", flush=True)
    print("=" * 80 + "\n", flush=True)

if __name__ == "__main__":
    db = Path("outputs/test_cache/test_entity_index.db")
    out = Path("outputs/test_cache")
    export_all_strategies(db, out)

