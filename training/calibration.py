import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import brier_score_loss

from training.evaluate import expected_calibration_error

class ModelWithTemperature(nn.Module):
    """
    Temperature Scaling Post-Processing (Guo et al., 2017).
    A single scalar parameter T > 0 scales logits before softmax:
        p_i = exp(z_i / T) / sum_j exp(z_j / T)
    T is optimized strictly on the validation set using NLL loss.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, input_tensor):
        logits = self.model(input_tensor)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, val_loader, device):
        self.to(device)
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss().to(device)

        # Collect all validation logits and labels
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for images, targets, _, _ in val_loader:
                images = images.to(device)
                logits = self.model(images)
                logits_list.append(logits)
                labels_list.append(targets)
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)

        # Calculate metrics before temperature scaling
        probs_before = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        labels_np = labels.cpu().numpy()
        brier_before = float(brier_score_loss(labels_np, probs_before))
        ece_before = float(expected_calibration_error(labels_np, probs_before))

        # Optimize temperature with L-BFGS
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        # Calculate metrics after temperature scaling
        scaled_logits = self.temperature_scale(logits)
        probs_after = torch.softmax(scaled_logits, dim=1)[:, 1].detach().cpu().numpy()
        brier_after = float(brier_score_loss(labels_np, probs_after))
        ece_after = float(expected_calibration_error(labels_np, probs_after))
        optimal_t = float(self.temperature.item())

        calib_report = {
            "optimal_temperature": optimal_t,
            "brier_before": brier_before,
            "brier_after": brier_after,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "brier_improvement": brier_before - brier_after,
            "ece_improvement": ece_before - ece_after
        }
        print(f"Optimal Temperature: {optimal_t:.4f}")
        print(f"ECE: {ece_before:.4f} -> {ece_after:.4f} | Brier: {brier_before:.4f} -> {brier_after:.4f}")
        return calib_report
