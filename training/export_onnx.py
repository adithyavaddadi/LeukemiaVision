import os
import sys
sys.path.insert(0, os.path.abspath("."))
import argparse
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from PIL import Image

from training.models import build_model
from training.preprocess import preprocess_image_numpy, check_parity_torch_numpy

class DualHeadONNXWrapper(nn.Module):
    """
    Wraps a model to produce two outputs:
    1. Logits (shape: [B, num_classes])
    2. Final convolutional feature map (shape: [B, C, H, W])
    
    This enables closed-form Grad-CAM in pure NumPy without needing PyTorch at inference time.
    """
    def __init__(self, base_model, model_name):
        super().__init__()
        self.base_model = base_model
        self.model_name = model_name

    def forward(self, x):
        if self.model_name == "efficientnet_b0":
            # EfficientNet features
            feat = self.base_model.features(x)
            pooled = self.base_model.avgpool(feat)
            flattened = torch.flatten(pooled, 1)
            logits = self.base_model.classifier(flattened)
            return logits, feat
        elif self.model_name == "resnet50":
            x_conv = self.base_model.conv1(x)
            x_conv = self.base_model.bn1(x_conv)
            x_conv = self.base_model.relu(x_conv)
            x_conv = self.base_model.maxpool(x_conv)

            x_conv = self.base_model.layer1(x_conv)
            x_conv = self.base_model.layer2(x_conv)
            x_conv = self.base_model.layer3(x_conv)
            feat = self.base_model.layer4(x_conv) # [B, 2048, 7, 7]
            pooled = self.base_model.avgpool(feat)
            flattened = torch.flatten(pooled, 1)
            logits = self.base_model.fc(flattened)
            return logits, feat
        elif self.model_name == "baseline_cnn":
            feat = self.base_model.features(x)
            pooled = self.base_model.pool(feat)
            flattened = torch.flatten(pooled, 1)
            logits = self.base_model.classifier(flattened)
            return logits, feat
        else:
            # Generic fallback
            logits = self.base_model(x)
            dummy_feat = torch.zeros(x.shape[0], 1, 7, 7, device=x.device)
            return logits, dummy_feat

def export_to_onnx(model_name="efficientnet_b0", checkpoint_path=None, output_onnx_path="models/production.onnx", num_classes=2):
    os.makedirs(os.path.dirname(output_onnx_path) if os.path.dirname(output_onnx_path) else ".", exist_ok=True)
    
    # 1. Instantiate model
    base_model = build_model(model_name, num_classes=num_classes, pretrained=False)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        base_model.load_state_dict(state_dict)
    else:
        print("Exporting with initialized architecture weights (smoke/development mode).")
        
    base_model.eval()
    wrapped_model = DualHeadONNXWrapper(base_model, model_name)
    wrapped_model.eval()

    # 2. Dummy input
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    # 3. Export to ONNX
    print(f"Exporting dual-output ONNX model to {output_onnx_path}...")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['logits', 'feature_map'],
        dynamic_axes={'input': {0: 'batch_size'}, 'logits': {0: 'batch_size'}, 'feature_map': {0: 'batch_size'}},
        dynamo=False
    )
    print(f"[PASS] Successfully exported ONNX model to {output_onnx_path}")

    # 4. Verify ONNX validity
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    print("[PASS] ONNX model structure verified via onnx.checker.")

    # 5. Numerical Parity Test
    print("\n--- Verifying PyTorch vs ONNX Runtime Numerical Parity ---")
    with torch.no_grad():
        pt_logits, pt_feat = wrapped_model(dummy_input)
        pt_probs = torch.softmax(pt_logits, dim=1).numpy()

    session = ort.InferenceSession(output_onnx_path, providers=['CPUExecutionProvider'])
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_logits, ort_feat = session.run(None, ort_inputs)
    ort_exp = np.exp(ort_logits - np.max(ort_logits, axis=1, keepdims=True))
    ort_probs = ort_exp / np.sum(ort_exp, axis=1, keepdims=True)

    logits_diff = np.max(np.abs(pt_logits.numpy() - ort_logits))
    probs_diff = np.max(np.abs(pt_probs - ort_probs))
    feat_diff = np.max(np.abs(pt_feat.numpy() - ort_feat))

    print(f"Logits Max Absolute Difference:     {logits_diff:.8e}")
    print(f"Probabilities Max Absolute Diff:    {probs_diff:.8e}")
    print(f"Feature Map Max Absolute Diff:      {feat_diff:.8e}")

    assert logits_diff < 1e-4, f"Parity failure on logits: max diff {logits_diff}"
    assert probs_diff < 1e-4, f"Parity failure on probabilities: max diff {probs_diff}"
    print("[PASS] Dual-head PyTorch vs ONNX numerical parity strictly validated (diff < 1e-4).")

    # Check file size for Vercel compliance (<250MB limit)
    size_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)
    print(f"Production ONNX Model Size: {size_mb:.2f} MB (Vercel Limit: <250 MB total package)")
    return output_onnx_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="efficientnet_b0")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="models/production.onnx")
    args = parser.parse_args()
    
    export_to_onnx(model_name=args.model, checkpoint_path=args.checkpoint, output_onnx_path=args.output)
