import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve, auc, matthews_corrcoef,
    brier_score_loss, confusion_matrix, roc_curve
)
from sklearn.calibration import calibration_curve

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

def cluster_bootstrap_ci(df_results, n_bootstraps=1000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    unique_subjects = df_results['subject_id'].unique()
    n_subjects = len(unique_subjects)
    
    boot_metrics = {
        'accuracy': [], 'all_sensitivity': [], 'hem_specificity': [],
        'balanced_acc': [], 'roc_auc': [], 'f1': []
    }
    
    for _ in range(n_bootstraps):
        sampled_subjs = rng.choice(unique_subjects, size=n_subjects, replace=True)
        # Sample rows corresponding to chosen clusters
        sample_df = pd.concat([df_results[df_results['subject_id'] == s] for s in sampled_subjs], ignore_index=True)
        
        y_t = sample_df['target'].values
        y_p = sample_df['prob'].values
        y_b = (y_p >= 0.5).astype(int)
        
        if len(np.unique(y_t)) < 2:
            continue
            
        boot_metrics['accuracy'].append(accuracy_score(y_t, y_b))
        boot_metrics['all_sensitivity'].append(recall_score(y_t, y_b, pos_label=1, zero_division=0))
        boot_metrics['hem_specificity'].append(recall_score(y_t, y_b, pos_label=0, zero_division=0))
        boot_metrics['balanced_acc'].append(balanced_accuracy_score(y_t, y_b))
        boot_metrics['f1'].append(f1_score(y_t, y_b, pos_label=1, zero_division=0))
        try:
            boot_metrics['roc_auc'].append(roc_auc_score(y_t, y_p))
        except:
            pass
            
    ci_results = {}
    for k, v in boot_metrics.items():
        if v:
            low = float(np.percentile(v, 100 * (alpha / 2)))
            high = float(np.percentile(v, 100 * (1 - alpha / 2)))
            mean_val = float(np.mean(v))
            ci_results[k] = {"mean": mean_val, "ci_95_low": low, "ci_95_high": high}
    return ci_results

def evaluate_predictions(y_true, y_prob, subject_ids=None, threshold=0.5, save_dir="artifacts"):
    os.makedirs(save_dir, exist_ok=True)
    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    
    # 1. Base Metrics
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    sens = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    spec = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_prob))
    ece = expected_calibration_error(y_true, y_prob)
    
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except:
        roc_auc = 0.5
        
    p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(r_curve, p_curve))
    
    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['HEM (Normal)', 'ALL (Malignant)'],
                yticklabels=['HEM (Normal)', 'ALL (Malignant)'])
    plt.title(f'Confusion Matrix (Threshold = {threshold:.2f})')
    plt.ylabel('Ground Truth')
    plt.xlabel('AI Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # 3. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='#2563eb', lw=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300)
    plt.close()
    
    # 4. Precision-Recall Curve Plot
    plt.figure(figsize=(6, 5))
    plt.plot(r_curve, p_curve, label=f'PR Curve (AUC = {pr_auc:.4f})', color='#7c3aed', lw=2)
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (PPV)')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'), dpi=300)
    plt.close()
    
    # 5. Sensitivity vs Specificity vs Threshold Plot
    thresh_grid = np.linspace(0.01, 0.99, 100)
    sens_grid, spec_grid = [], []
    for t in thresh_grid:
        p_t = (y_prob >= t).astype(int)
        sens_grid.append(recall_score(y_true, p_t, pos_label=1, zero_division=0))
        spec_grid.append(recall_score(y_true, p_t, pos_label=0, zero_division=0))
        
    plt.figure(figsize=(7, 5))
    plt.plot(thresh_grid, sens_grid, label='ALL Sensitivity', color='#dc2626', lw=2)
    plt.plot(thresh_grid, spec_grid, label='HEM Specificity', color='#059669', lw=2)
    plt.axvline(x=threshold, color='black', linestyle=':', label=f'Frozen Threshold ({threshold:.2f})')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Metric Score')
    plt.title('Sensitivity vs Specificity Tradeoff')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'threshold_curve.png'), dpi=300)
    plt.close()
    
    # 6. Cluster Bootstrap CIs
    ci_dict = {}
    if subject_ids is not None:
        df_res = pd.DataFrame({'target': y_true, 'prob': y_prob, 'subject_id': subject_ids})
        ci_dict = cluster_bootstrap_ci(df_res)
        
    results = {
        "threshold": threshold,
        "metrics": {
            "accuracy": acc, "balanced_accuracy": bal_acc, "all_sensitivity": sens,
            "hem_specificity": spec, "precision": prec, "f1_score": f1,
            "roc_auc": roc_auc, "pr_auc": pr_auc, "mcc": mcc,
            "brier_score": brier, "ece": ece
        },
        "confusion_matrix": cm.tolist(),
        "cluster_bootstrap_95_ci": ci_dict
    }
    
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
        
    return results
