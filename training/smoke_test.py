import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np
from PIL import Image
from training.models import build_model
from training.preprocess import preprocess_image_numpy, check_parity_torch_numpy

def run_smoke_tests():
    print("=== Running Model & Preprocessing Smoke Tests ===")
    
    # 1. Preprocessing Test & Parity Test
    check_parity_torch_numpy()
    dummy_img = Image.fromarray(np.uint8(np.random.rand(450, 450, 3) * 255))
    tensor_np, _ = preprocess_image_numpy(dummy_img)
    assert tensor_np.shape == (1, 3, 224, 224), f"Unexpected preprocessed shape: {tensor_np.shape}"
    print("[PASS] Preprocessing: aspect-safe padding + resize to (1, 3, 224, 224)")

    # 2. Test All Architectures
    models = ["baseline_cnn", "resnet50", "efficientnet_b0", "cnn_transformer"]
    dummy_batch = torch.randn(2, 3, 224, 224)
    dummy_target = torch.tensor([0, 1], dtype=torch.long)
    criterion = torch.nn.CrossEntropyLoss()

    for m_name in models:
        print(f"Testing model: {m_name}...")
        # build without pretrained weights for fast smoke testing
        model = build_model(m_name, num_classes=2, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
            assert out.shape == (2, 2), f"{m_name} output shape mismatch: {out.shape}"
        
        # Test backward pass
        model.train()
        model.zero_grad()
        out = model(dummy_batch)
        loss = criterion(out, dummy_target)
        loss.backward()
        print(f"  [PASS] {m_name} forward + backward pass successful. Loss: {loss.item():.4f}")

    print("=== ALL SMOKE TESTS COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    run_smoke_tests()
