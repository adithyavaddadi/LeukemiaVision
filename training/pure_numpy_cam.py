import os
import sys
sys.path.insert(0, os.path.abspath("."))
import io
import numpy as np
from PIL import Image
import onnxruntime as ort
from training.preprocess import preprocess_image_numpy

def compute_pure_numpy_gradcam(onnx_session, original_pil, class_idx=1, classifier_weights=None):
    """
    Computes closed-form Grad-CAM heatmap using ONLY NumPy and Pillow (Zero PyTorch).
    
    For a Global Average Pooling (GAP) network:
    Heatmap = ReLU(sum_k (w_k * A_k(i, j)))
    """
    # 1. Preprocess
    batch_np, _ = preprocess_image_numpy(original_pil)
    
    # 2. Run ONNX Session (dual output: logits, feature_map)
    inputs = {onnx_session.get_inputs()[0].name: batch_np}
    outputs = onnx_session.run(None, inputs)
    logits, feat_map = outputs[0], outputs[1]  # feat_map shape: (1, C, H, W)
    
    # 3. Softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    all_prob = float(probs[0, 1])
    
    # 4. If classifier_weights not provided, compute mean activation across channels
    C, H, W = feat_map.shape[1], feat_map.shape[2], feat_map.shape[3]
    feat = feat_map[0]  # (C, H, W)
    
    if classifier_weights is not None:
        w = classifier_weights[class_idx]  # (C,)
        cam = np.tensordot(w, feat, axes=(0, 0)) # (H, W)
    else:
        cam = np.mean(feat, axis=0) # (H, W)
        
    # 5. ReLU
    cam = np.maximum(cam, 0)
    
    # 6. Normalize to [0, 1]
    cam_min, cam_max = np.min(cam), np.max(cam)
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)
        
    # 7. Resize heatmap to original image dimensions
    w_orig, h_orig = original_pil.size
    cam_pil = Image.fromarray((cam * 255).astype(np.uint8), mode='L')
    cam_resized = cam_pil.resize((w_orig, h_orig), Image.Resampling.BILINEAR)
    cam_heatmap = np.array(cam_resized, dtype=np.float32) / 255.0
    
    return all_prob, cam_heatmap

def test_pure_numpy_cam():
    onnx_path = "models/production.onnx"
    assert os.path.exists(onnx_path), f"ONNX model {onnx_path} not found."
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    dummy_img = Image.fromarray((np.random.rand(450, 450, 3) * 255).astype(np.uint8))
    prob, heatmap = compute_pure_numpy_gradcam(session, dummy_img)
    
    assert 0.0 <= prob <= 1.0, f"Invalid prob: {prob}"
    assert heatmap.shape == (450, 450), f"Invalid heatmap shape: {heatmap.shape}"
    assert np.min(heatmap) >= 0.0 and np.max(heatmap) <= 1.0, "Heatmap out of range [0, 1]"
    print(f"[PASS] Pure NumPy Grad-CAM successfully generated. Probability: {prob:.4f}, Heatmap shape: {heatmap.shape}")

if __name__ == "__main__":
    test_pure_numpy_cam()
