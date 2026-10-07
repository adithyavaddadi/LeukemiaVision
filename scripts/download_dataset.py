import os
import sys
import gc
import json
import time
import csv
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

# Use Windows system memory pool to prevent virtual memory fragmentation
pa.set_memory_pool(pa.system_memory_pool())

REPO_ID = "dwb2023/cnmc-leukemia-2019"
TOTAL_FILES = 13
DATA_DIR = "data"
RAW_IMG_DIR = os.path.join(DATA_DIR, "raw", "images")
METADATA_OUT = os.path.join(DATA_DIR, "raw_metadata.csv")

COLS_TO_LOAD = ['image', 'subject_id', 'class_label', 'original_image_name', 'fold']

def main():
    os.makedirs(RAW_IMG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(METADATA_OUT), exist_ok=True)
    
    print("=" * 65)
    print("DOWNLOADING & EXTRACTING C-NMC 2019 (PURE PYARROW ZERO-COPY)")
    print(f"Repository: {REPO_ID}")
    print("=" * 65)
    
    start_time = time.time()
    total_extracted = 0
    
    # Open CSV writer incrementally so we never accumulate large metadata in RAM
    with open(METADATA_OUT, "w", newline="", encoding="utf-8") as f_meta:
        writer = csv.DictWriter(f_meta, fieldnames=[
            "image_path", "filename", "subject_id", "class_name", "numeric_label", "original_fold"
        ])
        writer.writeheader()
        
        for i in range(TOTAL_FILES):
            shard_filename = f"data/train-{i:05d}-of-00013.parquet"
            print(f"\n[Shard {i+1:02d}/{TOTAL_FILES:02d}] Fetching {shard_filename}...")
            
            shard_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=shard_filename,
                repo_type="dataset"
            )
            print(f"  Downloaded/Cached: {os.path.basename(shard_path)} ({os.path.getsize(shard_path) / (1024*1024):.2f} MB)")
            
            pf = pq.ParquetFile(shard_path)
            shard_count = 0
            
            for rg in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg, columns=COLS_TO_LOAD)
                
                c_img = tbl.column('image')
                c_subj = tbl.column('subject_id')
                c_cls = tbl.column('class_label')
                c_orig = tbl.column('original_image_name')
                c_fold = tbl.column('fold')
                
                rows_in_rg = len(tbl)
                for row_idx in range(rows_in_rg):
                    subj_id = str(c_subj[row_idx].as_py()).strip()
                    class_raw = str(c_cls[row_idx].as_py()).strip().lower()
                    orig_name = str(c_orig[row_idx].as_py()).strip()
                    fold_val = int(c_fold[row_idx].as_py())
                    
                    numeric_label = 1 if class_raw in ['all', 'cancer'] else 0
                    label_name = "ALL" if numeric_label == 1 else "HEM"
                    
                    sub_folder = os.path.join(RAW_IMG_DIR, label_name, f"subject_{subj_id}")
                    os.makedirs(sub_folder, exist_ok=True)
                    img_dest = os.path.join(sub_folder, orig_name)
                    
                    if not os.path.exists(img_dest):
                        img_obj = c_img[row_idx].as_py()
                        img_bytes = img_obj['bytes'] if isinstance(img_obj, dict) else img_obj
                        with open(img_dest, "wb") as f_img:
                            f_img.write(img_bytes)
                            
                    writer.writerow({
                        "image_path": os.path.relpath(img_dest, start="."),
                        "filename": orig_name,
                        "subject_id": subj_id,
                        "class_name": label_name,
                        "numeric_label": numeric_label,
                        "original_fold": fold_val
                    })
                    
                    shard_count += 1
                    total_extracted += 1
                    
                del tbl, c_img, c_subj, c_cls, c_orig, c_fold
                gc.collect()
                
            print(f"  -> Extracted {shard_count} images from shard {i+1} (Cumulative: {total_extracted})")
            
    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print("EXTRACTION COMPLETE")
    print(f"Total images: {total_extracted}")
    print(f"Metadata file: {METADATA_OUT}")
    print(f"Total time elapsed: {elapsed:.2f} seconds")
    print("=" * 65)

if __name__ == "__main__":
    main()
