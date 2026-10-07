import os
import sys
import glob
import json
import hashlib
import time
from collections import Counter
from PIL import Image
import pandas as pd

DATA_DIR = "data"
RAW_IMG_DIR = os.path.join(DATA_DIR, "raw", "images")
METADATA_FILE = os.path.join(DATA_DIR, "raw_metadata.csv")
REPORT_JSON = os.path.join("artifacts", "dataset_report.json")
REPORT_MD = os.path.join("artifacts", "dataset_report.md")

def file_hash(filepath, chunk_size=65536):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def inspect_dataset():
    os.makedirs("artifacts", exist_ok=True)
    start_time = time.time()
    
    if not os.path.exists(METADATA_FILE):
        print(f"Error: Metadata file {METADATA_FILE} not found. Run scripts/download_dataset.py first.")
        sys.exit(1)
        
    df = pd.read_csv(METADATA_FILE)
    total_records = len(df)
    print(f"Loaded {total_records} records from {METADATA_FILE}...")
    
    hashes = {}
    duplicates = []
    corrupted = []
    dimensions = []
    channels_list = []
    modes_list = []
    extensions = Counter()
    
    print("Auditing image integrity, dimensions, channels, and hashes...")
    for idx, row in df.iterrows():
        img_path = row['image_path']
        ext = os.path.splitext(img_path)[1].lower()
        extensions[ext] += 1
        
        if not os.path.exists(img_path):
            corrupted.append({"path": img_path, "error": "File does not exist"})
            continue
            
        try:
            # Check hash
            h = file_hash(img_path)
            if h in hashes:
                duplicates.append({"original": hashes[h], "duplicate": img_path, "hash": h})
            else:
                hashes[h] = img_path
                
            # Verify image readability and dimensions
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                w, h_dim = im.size
                dimensions.append((w, h_dim))
                modes_list.append(im.mode)
                channels_list.append(len(im.getbands()))
        except Exception as e:
            corrupted.append({"path": img_path, "error": str(e)})
            
    # Statistics
    class_counts = df['class_name'].value_counts().to_dict()
    subject_counts = df.groupby('class_name')['subject_id'].nunique().to_dict()
    total_subjects = df['subject_id'].nunique()
    per_subject_cells = df.groupby('subject_id').size().to_dict()
    dim_counter = Counter(dimensions)
    mode_counter = Counter(modes_list)
    channel_counter = Counter(channels_list)
    
    report_dict = {
        "dataset_name": "C-NMC 2019 (Hugging Face dwb2023/cnmc-leukemia-2019)",
        "total_images": total_records,
        "valid_images": total_records - len(corrupted),
        "corrupted_images_count": len(corrupted),
        "corrupted_images": corrupted,
        "exact_hash_duplicates_count": len(duplicates),
        "total_unique_subjects": total_subjects,
        "class_distribution": class_counts,
        "subject_distribution_by_class": subject_counts,
        "extensions": dict(extensions),
        "color_modes": dict(mode_counter),
        "channel_counts": dict(channel_counter),
        "top_resolutions": [
            {"width": w, "height": h, "count": count}
            for (w, h), count in dim_counter.most_common(5)
        ],
        "per_subject_cell_summary": {
            "min_cells": int(min(per_subject_cells.values())) if per_subject_cells else 0,
            "max_cells": int(max(per_subject_cells.values())) if per_subject_cells else 0,
            "mean_cells": float(round(sum(per_subject_cells.values()) / len(per_subject_cells), 2)) if per_subject_cells else 0
        },
        "inspection_duration_seconds": round(time.time() - start_time, 2)
    }
    
    with open(REPORT_JSON, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2)
        
    md_content = f"""# C-NMC 2019 Dataset Inspection & Audit Report
**Standard**: Master Execution Prompt v2 - Milestone 2  
**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}  

---

## 1. Executive Summary & Verification
- **Total Images Inspected**: {report_dict['total_images']}
- **Valid & Readable Images**: {report_dict['valid_images']}
- **Corrupted / Unreadable Images**: {report_dict['corrupted_images_count']}
- **Exact Hash Duplicates**: {report_dict['exact_hash_duplicates_count']}
- **Total Unique Subjects**: {report_dict['total_unique_subjects']}
- **Inspection Duration**: {report_dict['inspection_duration_seconds']} seconds

---

## 2. Class & Subject Distribution
| Class | Label Meaning | Total Cells | Percent of Cells | Unique Subjects |
| :--- | :--- | :---: | :---: | :---: |
| **ALL** | Malignant B-lymphoblast | {class_counts.get('ALL', 0)} | {class_counts.get('ALL', 0)/total_records*100:.1f}% | {subject_counts.get('ALL', 0)} |
| **HEM** | Benign / Normal Cell | {class_counts.get('HEM', 0)} | {class_counts.get('HEM', 0)/total_records*100:.1f}% | {subject_counts.get('HEM', 0)} |
| **Total** | | **{total_records}** | **100.0%** | **{total_subjects}** |

> **Class Imbalance Ratio**: ALL to HEM is **{class_counts.get('ALL', 0)/max(1, class_counts.get('HEM', 1)):.2f} : 1**.

---

## 3. Image Characteristics
- **File Extensions**: {dict(extensions)}
- **Color Modes**: {dict(mode_counter)}
- **Channels**: {dict(channel_counter)}
- **Dominant Resolutions**:
"""
    for item in report_dict['top_resolutions']:
        md_content += f"  - {item['width']} × {item['height']} px: {item['count']} images ({item['count']/total_records*100:.1f}%)\n"

    md_content += f"""
---

## 4. Subject-Level Statistics
- **Mean Cells per Subject**: {report_dict['per_subject_cell_summary']['mean_cells']}
- **Min Cells for a Subject**: {report_dict['per_subject_cell_summary']['min_cells']}
- **Max Cells for a Subject**: {report_dict['per_subject_cell_summary']['max_cells']}

---

## 5. Audit Conclusions for Milestone 3 (Splits)
1. **Subject-Stratified Splitting is Essential**: Because subjects contain multiple correlated cells (up to {report_dict['per_subject_cell_summary']['max_cells']} cells per subject), random cell splitting would cause catastrophic data leakage.
2. **Locked Test Set Allocation**: We will hold out 15 subjects (~10 ALL, ~5 HEM) as the locked test set.
3. **Cross-Validation**: 5-Fold StratifiedGroupKFold on remaining 58 subjects.
"""

    with open(REPORT_MD, 'w', encoding='utf-8') as f:
        f.write(md_content)
        
    print(f"Inspection complete. Artifacts written to:\n - {REPORT_JSON}\n - {REPORT_MD}")

if __name__ == "__main__":
    inspect_dataset()
