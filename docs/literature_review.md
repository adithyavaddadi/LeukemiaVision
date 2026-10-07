# Comprehensive Literature Review & Benchmark Matrix
**Course / Major Project**: AI-Assisted Leukemia Screening Prototype  
**Supervisor**: Mr. Balika Mahesh (Assistant Professor, Dept of AI & ML)  
**Dataset**: C-NMC 2019 (ISBI 2019 ALL Challenge)  

---

## 1. Comparative Literature Matrix

| Dimension | Base Paper (ALL-Net / Thiriveedhi et al.) | Research Paper 1 (Almahdawi et al., 2025) | Research Paper 2 (Yan et al., Cancer Sci 2025) | **Our Project (LeukemiaVision 2026)** |
| :--- | :--- | :--- | :--- | :--- |
| **Domain & Modality** | Microscopic single-cell blood smears | Blood smear microscopic images | Bone marrow & peripheral smear cells | Peripheral blood smear single-cell crops (C-NMC 2019) |
| **Target Classes** | ALL (malignant) vs HEM (benign) | ALL vs AML vs Normal | Multi-lineage hematological malignancies | Binary: ALL (B-lymphoblast) vs HEM (benign) |
| **Dataset Size & Scope** | ~3,200 images (subset) | ~10,000 images | Multi-center cohorts | **10,661 images across 73 subjects** |
| **Splitting Protocol** | Naive random cell split (**High leakage risk**) | Random train/test split | Multi-center validation | **Strict 5-Fold StratifiedGroupKFold (Zero subject leakage)** |
| **Core Architecture** | Custom Shallow CNN | ResNet-50 / DenseNet | Hybrid Vision Transformer (ViT) | **4-Model Suite: Baseline CNN, ResNet-50, EfficientNet-B0, Hybrid CNN-Transformer** |
| **Model Explainability** | None (Black Box) | Post-hoc Grad-CAM | Attention Rollout | **Dual-Head Closed-Form NumPy Grad-CAM + Randomization Sanity Check** |
| **Calibration & Review States** | Raw Argmax softmax | Raw Softmax threshold | Decision tree thresholding | **Temperature Scaling, ECE, 4-Tier Calibrated Review States** |
| **Statistical Rigor** | Single-point Accuracy | Accuracy & Macro F1 | Area Under ROC Curve | **Cluster-Bootstrap 95% Confidence Intervals (B=1000)** |
| **Deployment Engine** | Offline Python script | Heavy PyTorch container | Cloud GPU server | **Vercel Serverless ONNX CPU Engine (<40ms, <120MB bundle)** |

---

## 2. Critical Research Gaps Identified in Prior Work
1. **The Data Leakage Vulnerability**: Prior literature frequently partitioned single-cell blood images randomly at the cell level. Because multiple cells originate from the same patient smear, models learned slide staining and illumination artifacts rather than true cytological morphology.
2. **The Clinical Black-Box Problem**: Many previous papers reported high accuracy (98%+) without explainability maps or calibration, rendering models unsafe for clinical decision support.
3. **The Heavy Inference Bottleneck**: Prior implementations relied on 800MB+ PyTorch dependencies requiring GPU hardware, making real-time serverless deployment infeasible.

---

## 3. Our Architectural Breakthroughs
- **Subject-Stratified Isolation**: Cryptographically locked held-out test partition and 5-fold cross-validation with zero subject overlap.
- **Closed-Form GAP Grad-CAM**: Derived mathematical equivalence for Global Average Pooling networks enabling instant Grad-CAM in pure NumPy on CPU.
- **Calibrated Review States**: Shift from binary forced choice to 4 medical review states (`ALL pattern detected`, `HEM/normal pattern detected`, `Low confidence: manual review required`, `Invalid image`).
