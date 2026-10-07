import os
import sys
import subprocess
import glob
import time
import pandas as pd

TOTAL_SHARDS = 13
METADATA_DIR = "data/metadata_shards"
FINAL_METADATA = "data/raw_metadata.csv"

def main():
    print("=" * 65)
    print("ORCHESTRATING 13-SHARD DATASET EXTRACTION VIA ISOLATED WORKERS")
    print("=" * 65)
    start_time = time.time()
    
    os.makedirs(METADATA_DIR, exist_ok=True)
    
    for i in range(TOTAL_SHARDS):
        print(f"\n[Shard {i+1:02d}/{TOTAL_SHARDS:02d}] Launching isolated extraction process...")
        t0 = time.time()
        ret = subprocess.run([sys.executable, "scripts/extract_shard_worker.py", str(i)])
        if ret.returncode != 0:
            print(f"FATAL: Shard {i} worker failed with exit code {ret.returncode}")
            sys.exit(1)
        print(f"[Shard {i+1:02d}/{TOTAL_SHARDS:02d}] Completed in {time.time() - t0:.2f}s")
        
    print("\n" + "=" * 65)
    print("MERGING METADATA SHARDS INTO data/raw_metadata.csv")
    print("=" * 65)
    
    shard_files = sorted(glob.glob(os.path.join(METADATA_DIR, "shard_*.csv")))
    all_dfs = [pd.read_csv(f) for f in shard_files]
    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df.to_csv(FINAL_METADATA, index=False)
    
    elapsed = time.time() - start_time
    print(f"Extraction & Merge Complete in {elapsed:.2f}s!")
    print(f"Total Images: {len(merged_df)}")
    print(f"Unique Subjects: {merged_df['subject_id'].nunique()}")
    print("Class Counts:")
    print(merged_df['class_name'].value_counts().to_string())
    print(f"Master metadata saved to: {FINAL_METADATA}")
    print("=" * 65)

if __name__ == "__main__":
    main()
