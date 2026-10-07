import os
import sys
import hashlib
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

METADATA_FILE = "data/raw_metadata.csv"
SPLITS_FILE = "data/splits.csv"
SPLIT_HASH_FILE = "data/split_hash.json"

def create_splits(seed=42, n_test_subjects=15, n_folds=5):
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(f"Metadata file {METADATA_FILE} not found. Run dataset extraction first.")
        
    df = pd.read_csv(METADATA_FILE)
    print(f"Loaded {len(df)} samples from {METADATA_FILE}.")

    # Filter out corrupted files if reported
    report_file = os.path.join("artifacts", "dataset_report.json")
    if os.path.exists(report_file):
        with open(report_file, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        corrupted_paths = {c['path'] for c in report_data.get('corrupted_images', [])}
        if corrupted_paths:
            prev_len = len(df)
            df = df[~df['image_path'].isin(corrupted_paths)].copy()
            print(f"Filtered out {prev_len - len(df)} corrupted/unreadable images. Remaining valid: {len(df)}")

    # Identify unique subjects and their predominant class
    subject_info = df.groupby('subject_id').agg(
        class_name=('class_name', lambda s: s.mode()[0]),
        numeric_label=('numeric_label', lambda s: s.mode()[0]),
        cell_count=('image_path', 'count')
    ).reset_index()
    
    total_subjects = len(subject_info)
    print(f"Total unique subjects: {total_subjects}")
    print("Subject breakdown by class:")
    print(subject_info['class_name'].value_counts())
    
    # 1. Carve out Held-out Locked Test Set (15 subjects) stratified by class
    test_ratio = n_test_subjects / total_subjects
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    
    dev_subj_idx, test_subj_idx = next(sss.split(subject_info, subject_info['numeric_label']))
    
    dev_subjects = set(subject_info.iloc[dev_subj_idx]['subject_id'])
    test_subjects = set(subject_info.iloc[test_subj_idx]['subject_id'])
    
    print(f"\nHeld-out Test Subjects: {len(test_subjects)} subjects")
    print(f"Development Subjects: {len(dev_subjects)} subjects")
    
    # 2. 5-Fold StratifiedGroupKFold on remaining development subjects
    dev_df = df[df['subject_id'].isin(dev_subjects)].copy()
    test_df = df[df['subject_id'].isin(test_subjects)].copy()
    
    test_df['split'] = 'test'
    test_df['fold'] = -1  # Test set is not in any CV fold
    
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Prepare dev groups
    dev_groups = dev_df['subject_id'].values
    dev_y = dev_df['numeric_label'].values
    dev_df['fold'] = -1
    dev_df['split'] = 'dev'
    
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(dev_df, dev_y, dev_groups)):
        # For evaluation tracking, assign fold number
        dev_df.iloc[val_idx, dev_df.columns.get_loc('fold')] = fold_idx
        
    # Combine full dataset
    splits_df = pd.concat([dev_df, test_df], ignore_index=True)
    
    # Audit zero leakage
    print("\n" + "=" * 50)
    print("LEAKAGE AUDIT VERIFICATION")
    print("=" * 50)
    
    # Check 1: Dev vs Test
    test_subjs = set(splits_df[splits_df['split'] == 'test']['subject_id'])
    dev_subjs = set(splits_df[splits_df['split'] == 'dev']['subject_id'])
    assert len(test_subjs.intersection(dev_subjs)) == 0, "FATAL: Subject leakage between test and dev!"
    print("PASS: Zero subject overlap between Held-out Test and Development set.")
    
    # Check 2: Cross-fold leakage
    for f in range(n_folds):
        val_f_subjs = set(splits_df[(splits_df['split'] == 'dev') & (splits_df['fold'] == f)]['subject_id'])
        train_f_subjs = set(splits_df[(splits_df['split'] == 'dev') & (splits_df['fold'] != f)]['subject_id'])
        assert len(val_f_subjs.intersection(train_f_subjs)) == 0, f"FATAL: Subject leakage in fold {f}!"
        print(f"PASS: Fold {f} zero subject overlap (Val subjects: {len(val_f_subjs)}, Train subjects: {len(train_f_subjs)}).")
        
    # Save splits.csv
    splits_df = splits_df[['image_path', 'filename', 'subject_id', 'class_name', 'numeric_label', 'split', 'fold']]
    splits_df.to_csv(SPLITS_FILE, index=False)
    
    # Calculate SHA256 of splits.csv
    with open(SPLITS_FILE, 'rb') as f:
        split_hash = hashlib.sha256(f.read()).hexdigest()
        
    hash_record = {
        "split_file": SPLITS_FILE,
        "split_hash_sha256": split_hash,
        "seed": seed,
        "total_images": len(splits_df),
        "test_images": len(test_df),
        "dev_images": len(dev_df),
        "n_test_subjects": len(test_subjects),
        "n_dev_subjects": len(dev_subjects),
        "n_folds": n_folds
    }
    
    with open(SPLIT_HASH_FILE, 'w') as f:
        json.dump(hash_record, f, indent=2)
        
    print(f"\nSplits saved to {SPLITS_FILE}")
    print(f"Split Hash: {split_hash}")
    return hash_record

if __name__ == "__main__":
    create_splits()
