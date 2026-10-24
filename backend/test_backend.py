import os
import sys
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

# Add workspace and backend to path
sys.path.insert(0, os.path.abspath("backend"))
from app.main import app

def test_api():
    client = TestClient(app)
    
    # 1. Health check
    print("1. Testing GET /health...")
    resp = client.get("/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    print("   [PASS]", resp.json())
    
    # 2. Model Info
    print("2. Testing GET /model-info...")
    resp = client.get("/model-info")
    assert resp.status_code == 200, f"Model info failed: {resp.text}"
    print("   [PASS] Model Name:", resp.json().get("model_name"))
    
    # 3. Predict Endpoint with valid image
    print("3. Testing POST /predict with sample image...")
    dummy_img = Image.fromarray((np.random.rand(450, 450, 3) * 255).astype(np.uint8))
    img_bytes = io.BytesIO()
    dummy_img.save(img_bytes, format="BMP")
    img_bytes.seek(0)
    
    files = {"file": ("test_cell.bmp", img_bytes.getvalue(), "image/bmp")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 200, f"Predict failed: {resp.text}"
    data = resp.json()
    print("   [PASS] Prediction Result:")
    print("          Review State:", data.get("review_state"))
    print("          Predicted Class:", data.get("predicted_class"))
    print("          Probabilities:", data.get("probabilities"))
    print("          Inference Latency:", data.get("inference_latency_ms"), "ms")
    print("          Has Grad-CAM overlay:", "overlay_b64" in data.get("explainability", {}))
    print("          Has Disclaimer:", "clinical_disclaimer" in data)

    # 4. Predict Endpoint with corrupted/invalid payload
    print("4. Testing POST /predict with invalid file...")
    files_bad = {"file": ("bad.txt", b"not an image", "text/plain")}
    resp_bad = client.post("/predict", files=files_bad)
    assert resp_bad.status_code == 400, f"Expected 400 for bad image, got: {resp_bad.status_code}"
    print("   [PASS] Successfully rejected corrupted/non-image upload with 400.")

    # 5. Predict Endpoint with uncropped whole-slide smear image
    smear_path = os.path.join("data", "sample_slides", "allbloodsmear_650.jpg")
    if os.path.exists(smear_path):
        print("5. Testing POST /predict with uncropped whole-slide smear...")
        with open(smear_path, "rb") as f_smear:
            resp_smear = client.post("/predict", files={"file": ("smear.jpg", f_smear.read(), "image/jpeg")})
        assert resp_smear.status_code == 200, f"Smear test failed: {resp_smear.text}"
        data_smear = resp_smear.json()
        assert data_smear.get("is_widefield_smear") is True, "Failed to identify wide-field smear"
        assert len(data_smear.get("cells", [])) >= 2, "Failed to auto-segment cells from smear"
        print(f"   [PASS] Whole smear auto-segmented: {data_smear.get('total_cells_detected')} cells detected ({data_smear.get('blasts_detected')} blasts, {data_smear.get('blast_index_pct')}% blast index).")
        print(f"   [PASS] Generated annotated smear overlay: {bool(data_smear.get('annotated_smear_b64'))}")

    print("\n=== ALL FASTAPI BACKEND INTEGRATION TESTS PASSED ===")

if __name__ == "__main__":
    test_api()
