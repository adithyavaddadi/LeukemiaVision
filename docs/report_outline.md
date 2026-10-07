# Major Project Report Outline: AI-Assisted Leukemia Screening Prototype
**Department**: Artificial Intelligence & Machine Learning  
**Supervisor**: Mr. Balika Mahesh (Assistant Professor)  
**Authors**: A. Rachana (23N81A6690), G. Himaresh (23N81A66A1), V. Adithya (23N81A66C7), B. Bharath Kumar (23N81A66B5)  

---

## Chapter 1: Introduction & Clinical Background
1.1 Background of Acute Lymphoblastic Leukemia (ALL)  
1.2 Clinical Diagnosis via Peripheral Blood Smear (PBS) Cytology  
1.3 Challenges in Manual Microscopic Examination (Fatigue, Inter-Observer Variability)  
1.4 Problem Statement & Clinical Decision-Support Framing  
1.5 Major Objectives & Scope  

## Chapter 2: Literature Survey & Research Gaps
2.1 Traditional Image Processing Approaches in Hematology  
2.2 Deep Learning Architectures (CNNs, ResNets, DenseNets, ViTs)  
2.3 Critical Research Gap: Data Leakage in Single-Cell Datasets  
2.4 The Black-Box Dilemma in Medical AI  
2.5 Comparative Matrix of Existing Methodologies  

## Chapter 3: Dataset Curation & Leakage-Free Splitting
3.1 C-NMC 2019 Dataset Description (ISBI 2019 ALL Challenge)  
3.2 Metadata Structure & Subject Isolation (73 Unique Subjects, 10,661 Images)  
3.3 Mathematical Formulation of Subject Leakage  
3.4 5-Fold StratifiedGroupKFold Partitioning Protocol  
3.5 Image Preprocessing: Aspect-Safe Padding & Color Normalization  

## Chapter 4: Model Exploration & Mathematical Formulations
4.1 Architectural Suite Overview:
   - 4.1.1 Model 1: Custom 4-Block Baseline CNN
   - 4.1.2 Model 2: Deep Residual Network (ResNet-50)
   - 4.1.3 Model 3: Compound-Scaled EfficientNet-B0
   - 4.1.4 Model 4: Hybrid CNN-Transformer Architecture  
4.2 Loss Formulations: Class-Weighted Cross-Entropy & Focal Loss  
4.3 Optimization Strategy, Learning Rate Scheduling & Early Stopping  

## Chapter 5: Explainability, Calibration & Decision Thresholds
5.1 PyTorch Grad-CAM Saliency & Randomization Sanity Checks  
5.2 Closed-Form Mathematical Derivation for Zero-PyTorch GAP Grad-CAM  
5.3 Model Calibration via Temperature Scaling  
5.4 Expected Calibration Error (ECE) and Brier Score Optimization  
5.5 4-Tier Calibrated Clinical Review States Formulation  

## Chapter 6: Production Engineering & Cloud Deployment
6.1 Dual-Head ONNX Export & Numerical Parity Auditing ($< 10^{-4}$)  
6.2 FastAPI Asynchronous REST Engine Architecture  
6.3 Modern Dark Glassmorphic React Frontend with Client-Side Downscaling  
6.4 Multi-Stage Docker Containerization  
6.5 Vercel Serverless Optimization (<250MB Package Compliance)  

## Chapter 7: Results, Statistical Analysis & Viva Defense
7.1 5-Fold Cross-Validation Performance Comparison  
7.2 1,000-Iteration Cluster-Bootstrap 95% Confidence Intervals  
7.3 Inference Latency & Memory Footprint Benchmark  
7.4 Error Analysis & Cytological Saliency Map Inspection  
7.5 Conclusion & Future Scope (Whole Slide Imaging & Multi-Center Validation)  
