import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from training.models.efficientnet import EfficientNetB0Classifier
from training.dataset import CNMCDataset
from training.export_onnx import export_to_onnx

def train_production_model(max_batches=150, epochs=2):
    print("=== Training EfficientNet-B0 Production Weights ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Dataset & Loader
    splits_df = pd.read_csv("data/splits.csv")
    train_df = splits_df[splits_df['split'] == 'dev']
    
    # Stratified balance sample for fast high-accuracy calibration
    all_df = train_df[train_df['numeric_label'] == 1].sample(n=min(len(train_df[train_df['numeric_label'] == 1]), 800), random_state=42)
    hem_df = train_df[train_df['numeric_label'] == 0].sample(n=min(len(train_df[train_df['numeric_label'] == 0]), 800), random_state=42)
    balanced_df = pd.concat([all_df, hem_df]).sample(frac=1.0, random_state=42)

    dataset = CNMCDataset(balanced_df, is_train=True)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    # 2. Build model with ImageNet pre-training
    model = EfficientNetB0Classifier(num_classes=2, pretrained=True)
    model.freeze_backbone()  # Fast convergence on classifier head
    model.to(device)

    # 3. Train classifier head
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        t0 = time.time()
        for i, (images, targets, _, _) in enumerate(loader):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += images.size(0)

            if i >= max_batches:
                break
        
        acc = correct / total if total > 0 else 0
        avg_loss = total_loss / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} [{time.time()-t0:.1f}s] - Loss: {avg_loss:.4f} | Accuracy: {acc*100:.2f}%")

    # 4. Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/efficientnet_b0_production.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved trained weights to {ckpt_path}")

    # 5. Export to production ONNX
    export_to_onnx(model_name="efficientnet_b0", checkpoint_path=ckpt_path, output_onnx_path="models/production.onnx")
    print("=== Production Model Successfully Trained and Exported ===")

if __name__ == "__main__":
    train_production_model()
