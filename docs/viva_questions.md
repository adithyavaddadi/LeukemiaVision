# 20-Question Technical Viva Defense Guide
**Major Project**: AI-Assisted Leukemia Screening Prototype  
**Department**: Dept of Artificial Intelligence & Machine Learning  
**Supervisor**: Mr. Balika Mahesh  

---

### Q1: What is the primary clinical objective of this project?
**Answer**: To provide an AI-assisted decision-support prototype that screens peripheral blood smear single-cell crops for Acute Lymphoblastic Leukemia (ALL vs normal HEM). It does not provide autonomous diagnosis; all predictions are mapped to calibrated review states accompanied by visual Grad-CAM explanations and a clinical disclaimer.

### Q2: What dataset did you use, and what is its class distribution?
**Answer**: The C-NMC 2019 dataset (ISBI 2019 ALL Challenge) containing 10,661 single-cell images across 73 unique subjects. The class distribution is 7,272 ALL malignant cells (68.2%) and 3,389 HEM normal cells (31.8%), representing a 2.15:1 class imbalance ratio.

### Q3: What is "Subject Data Leakage", and how did you prevent it?
**Answer**: In single-cell hematology datasets, multiple cell images are cropped from smears belonging to the same patient. Randomly splitting cells across train and test sets results in the model memorizing patient-specific staining, slide illumination, and blood background rather than biological cell morphology. We prevented this by grouping images by `subject_id` and using **5-Fold StratifiedGroupKFold**, ensuring zero subject overlap between training, validation, and test partitions.

### Q4: Why is raw accuracy a misleading metric for medical datasets?
**Answer**: In imbalanced medical datasets, a naive model predicting the majority class can achieve high accuracy while failing to detect malignant cases (false negatives). In clinical oncology, missing a malignant blast (low sensitivity) carries severe clinical risk. We prioritize **ALL Sensitivity ($\ge 95\%$)**, **HEM Specificity**, **Balanced Accuracy**, and **ROC-AUC**.

### Q5: What 4 model architectures did you evaluate?
**Answer**:
1. Custom 4-Block Baseline CNN (~1.2M params)
2. ResNet-50 with Deep Residual Blocks (~23.5M params)
3. EfficientNet-B0 with Compound Scaling (~4.0M params)
4. Hybrid CNN-Transformer combining convolutional feature extraction with multi-head self-attention (~18.2M params).

### Q6: Why was EfficientNet-B0 chosen as the production model?
**Answer**: EfficientNet-B0 achieves high sensitivity (>95%) and ROC-AUC (0.962) while having only 4.0 million parameters. Its ONNX export is only 15.2 MB and executes in ~28 ms on CPU, easily satisfying Vercel's serverless size (<250MB) and latency limits.

### Q7: How does the Hybrid CNN-Transformer work?
**Answer**: A CNN stem extracts local low-level texture features into a $7 \times 7$ spatial map. A $1 \times 1$ convolution projects these into 49 patch tokens, which are concatenated with a learnable class token, augmented with positional embeddings, and passed through a 4-layer Transformer Encoder. This allows the model to capture both local chromatin texture and global morphological context.

### Q8: What is Grad-CAM, and how does it work?
**Answer**: Gradient-weighted Class Activation Mapping (Grad-CAM) uses the gradients of the target class logit flowing into the final convolutional layer to calculate spatial importance weights ($\alpha_k$), generating a visual heatmap highlighting the discriminative image regions.

### Q9: How did you implement Grad-CAM on Vercel without installing PyTorch?
**Answer**: PyTorch wheels exceed 800MB (exceeding Vercel's 250MB limit). For networks using Global Average Pooling (GAP), the gradient $\alpha_k$ is mathematically constant ($\frac{w_k}{HW}$). We exported a dual-head ONNX model returning both logits and the final convolutional feature tensor, computing the weighted activation sum in pure NumPy with zero PyTorch dependency in ~30ms.

### Q10: What sanity checks did you perform on Grad-CAM?
**Answer**: We conducted weight randomization sanity checks (Adebayo et al.). When model weights are replaced with random initialization, the Grad-CAM heatmap degrades into diffuse noise, proving that our saliency maps genuinely reflect learned features.

### Q11: What is Model Calibration and why is it necessary?
**Answer**: Modern deep networks often produce overconfident probabilities (e.g., 99% confident even when incorrect). We applied **Temperature Scaling** ($z / T$) to align predicted softmax probabilities with empirical accuracy, minimizing Expected Calibration Error (ECE) and Brier score.

### Q12: What are the 4 Calibrated Review States?
**Answer**:
1. `ALL pattern detected` ($P(\text{ALL}) \ge 0.65$)
2. `HEM/normal pattern detected` ($P(\text{ALL}) \le 0.35$)
3. `Low confidence: manual review required` ($0.35 < P(\text{ALL}) < 0.65$)
4. `Invalid image` (corrupt, out-of-distribution, or malformed dimensions).

### Q13: What is the purpose of the 95% Sensitivity Target?
**Answer**: In leukemia screening, minimizing False Negatives (missed cancer) is paramount. We set the high-confidence operating threshold to ensure that at least 95% of true ALL cases are flagged for clinical review.

### Q14: What statistical method did you use to compute confidence intervals?
**Answer**: We used **1,000-iteration Cluster Bootstrap** at the subject level. Entire patient clusters are resampled with replacement, accounting for correlation among cells from the same patient.

### Q15: How does your preprocessing pipeline preserve cell morphology?
**Answer**: Standard bilinear stretching distorts the nuclear-to-cytoplasmic (N:C) ratio. We use **aspect-safe square padding** (mirroring/padding border) followed by bicubic resizing to $224 \times 224$ and ImageNet standardization.

### Q16: How is data privacy maintained in the deployment?
**Answer**: The backend operates in memory without logging or saving patient cell images to disk. Buffers are discarded immediately after inference, aligning with HIPAA principles.

### Q17: How did you handle the Vercel 4.5MB payload limit?
**Answer**: The React frontend uses an HTML5 Canvas downscaler to downscale high-resolution microscope images (longest side $\le 1024$px) before transmission, ensuring uploads stay well below 800 KB.

### Q18: What is the purpose of Docker containerization?
**Answer**: We created multi-stage Dockerfiles (`Dockerfile.backend`, `Dockerfile.frontend`) and `docker-compose.yml` to ensure reproducible, isolated execution across any operating system or cloud provider.

### Q19: What are the primary limitations of the current prototype?
**Answer**: The system processes pre-cropped single-cell images from the C-NMC 2019 dataset. Future expansion requires an automated cell detection/segmentation module for whole slide imaging (WSI) and external validation on multi-center clinical cohorts.

### Q20: How does this project adhere to ethical medical AI guidelines?
**Answer**: The interface and API strictly frame all outputs as clinical decision support. Every response contains clear review states, explainability maps, operating confidence thresholds, and prominent medical disclaimers.
