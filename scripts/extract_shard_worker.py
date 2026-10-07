import os
import sys
import gc
import csv
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

pa.set_memory_pool(pa.system_memory_pool())

REPO_ID = "dwb2023/cnmc-leukemia-2019"
DATA_DIR = "data"
RAW_IMG_DIR = os.path.join(DATA_DIR, "raw", "images")
METADATA_DIR = os.path.join(DATA_DIR, "metadata_shards")
COLS = ['image', 'subject_id', 'class_label', 'original_image_name', 'fold']

def process_shard(shard_idx):
    os.makedirs(RAW_IMG_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)
    
    shard_filename = f"data/train-{shard_idx:05d}-of-00013.parquet"
    out_csv = os.path.join(METADATA_DIR, f"shard_{shard_idx:02d}.csv")
    
    shard_path = hf_hub_download(repo_id=REPO_ID, filename=shard_filename, repo_type="dataset")
    pf = pq.ParquetFile(shard_path)
    
    extracted = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=[
            "image_path", "filename", "subject_id", "class_name", "numeric_label", "original_fold"
        ])
        writer.writeheader()
        
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=COLS)
            c_img = tbl.column('image')
            c_subj = tbl.column('subject_id')
            c_cls = tbl.column('class_label')
            c_orig = tbl.column('original_image_name')
            c_fold = tbl.column('fold')
            
            for i in range(len(tbl)):
                subj_id = str(c_subj[i].as_py()).strip()
                class_raw = str(c_cls[i].as_py()).strip().lower()
                orig_name = str(c_orig[i].as_py()).strip()
                fold_val = int(c_fold[i].as_py())
                
                numeric_label = 1 if class_raw in ['all', 'cancer'] else 0
                label_name = "ALL" if numeric_label == 1 else "HEM"
                
                sub_folder = os.path.join(RAW_IMG_DIR, label_name, f"subject_{subj_id}")
                os.makedirs(sub_folder, exist_ok=True)
                img_dest = os.path.join(sub_folder, orig_name)
                
                if not os.path.exists(img_dest):
                    img_obj = c_img[i].as_py()
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
                extracted += 1
                
            del tbl, c_img, c_subj, c_cls, c_orig, c_fold
            gc.collect()
            
    print(f"Shard {shard_idx} finished successfully. Extracted {extracted} images.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_shard_worker.py <shard_idx>")
        sys.exit(1)
    process_shard(int(sys.argv[1]))
