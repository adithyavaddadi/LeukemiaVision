import os
import sys
sys.path.insert(0, os.path.abspath("."))
import argparse
import time
import json
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score

from training.models import build_model
from training.dataset import CNMCDataset

def train_epoch(model, loader, criterion, optimizer, device, is_fast_dev=False):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []
    
    for i, (images, targets, _, _) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(targets.cpu().numpy())
        
        if is_fast_dev and i >= 1:
            break
            
    epoch_loss = running_loss / (len(all_targets) if all_targets else 1)
    return epoch_loss, np.array(all_preds), np.array(all_targets)

@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device, is_fast_dev=False):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []
    
    for i, (images, targets, _, _) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        
        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(targets.cpu().numpy())
        
        if is_fast_dev and i >= 1:
            break
            
    val_loss = running_loss / (len(all_targets) if all_targets else 1)
    targets_np = np.array(all_targets)
    preds_np = np.array(all_preds)
    
    # Calculate metrics
    binary_preds = (preds_np >= 0.5).astype(int)
    acc = accuracy_score(targets_np, binary_preds)
    sens = recall_score(targets_np, binary_preds, pos_label=1, zero_division=0)  # ALL sensitivity
    spec = recall_score(targets_np, binary_preds, pos_label=0, zero_division=0)  # HEM specificity
    try:
        auc = roc_auc_score(targets_np, preds_np)
    except:
        auc = 0.5
        
    return val_loss, acc, sens, spec, auc, preds_np, targets_np

def run_training(args):
    start_time = time.time()
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (args.model: {args.model})")
    
    # Load splits
    splits_df = pd.read_csv("data/splits.csv")
    fold = args.fold
    
    if fold >= 0:
        train_df = splits_df[(splits_df['split'] == 'dev') & (splits_df['fold'] != fold)]
        val_df = splits_df[(splits_df['split'] == 'dev') & (splits_df['fold'] == fold)]
    else:
        # Full development train on dev, validate on held-out test
        train_df = splits_df[splits_df['split'] == 'dev']
        val_df = splits_df[splits_df['split'] == 'test']
        
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    # Build datasets
    train_ds = CNMCDataset(train_df, img_size=config['data']['img_size'], is_train=True, stain_method=args.stain, augment_cfg=config['augmentation'])
    val_ds = CNMCDataset(val_df, img_size=config['data']['img_size'], is_train=False, stain_method=args.stain)
    
    batch_size = args.batch_size or config['training']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Build model
    model = build_model(args.model, num_classes=2, pretrained=True, dropout=0.3).to(device)
    
    # Imbalance strategy
    if args.imbalance == "class_weighted":
        n_pos = (train_df['numeric_label'] == 1).sum()
        n_neg = (train_df['numeric_label'] == 0).sum()
        w_neg = len(train_df) / (2.0 * max(1, n_neg))
        w_pos = len(train_df) / (2.0 * max(1, n_pos))
        weights = torch.tensor([w_neg, w_pos], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
        
    lr = args.learning_rate or float(config['training']['learning_rate'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs or config['training']['epochs'])
    
    best_sens_auc = -1.0
    checkpoint_path = f"checkpoints/{args.model}_fold{fold}_best.pt"
    
    epochs = 1 if args.fast_dev_run else (args.epochs or config['training']['epochs'])
    print(f"Beginning training for {epochs} epoch(s)...")
    
    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, _, _ = train_epoch(model, train_loader, criterion, optimizer, device, is_fast_dev=args.fast_dev_run)
        val_loss, val_acc, val_sens, val_spec, val_auc, _, _ = evaluate_epoch(model, val_loader, criterion, device, is_fast_dev=args.fast_dev_run)
        scheduler.step()
        ep_time = time.time() - t0
        
        composite_score = (val_sens + val_auc) / 2.0
        print(f"Epoch {epoch:02d}/{epochs:02d} [{ep_time:.1f}s] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | ALL Sens: {val_sens:.4f} | HEM Spec: {val_spec:.4f} | AUC: {val_auc:.4f}")
        
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_acc": val_acc, "val_sens": val_sens, "val_spec": val_spec, "val_auc": val_auc
        })
        
        if composite_score > best_sens_auc:
            best_sens_auc = composite_score
            torch.save({
                "epoch": epoch,
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_sens": val_sens,
                "val_auc": val_auc,
                "config": config,
                "split_fold": fold
            }, checkpoint_path)
            
        if args.fast_dev_run:
            print("Fast-dev-run completed successfully.")
            break
            
    print(f"Training finished in {time.time() - start_time:.2f}s. Best checkpoint saved to {checkpoint_path}")
    return checkpoint_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="efficientnet_b0", help="Model name")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0-4, or -1 for full dev)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--stain", type=str, default="none", help="Stain normalization")
    parser.add_argument("--imbalance", type=str, default="class_weighted", help="Imbalance strategy")
    parser.add_argument("--fast-dev-run", action="store_true", help="1-batch smoke test")
    args = parser.parse_args()
    run_training(args)
