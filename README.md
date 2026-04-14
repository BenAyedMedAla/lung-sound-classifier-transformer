# Geometry-Aware Optimization for Respiratory Sound Classification (AST + SAM)

[![arXiv](https://img.shields.io/badge/arXiv-2502.22564-b31b1b.svg)](https://arxiv.org/abs/2512.22564)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-blue)](https://huggingface.co/papers/2512.22564)

This repository contains the official PyTorch implementation of the paper: **"Geometry-Aware Optimization for Respiratory Sound Classification: Enhancing Sensitivity with SAM-Optimized Audio Spectrogram Transformers"**.

We introduce a robust framework integrating the **Audio Spectrogram Transformer (AST)** with **Sharpness-Aware Minimization (SAM)** and a signal-preserving cyclic padding strategy to achieve State-of-the-Art (SOTA) results on the ICBHI 2017 dataset.

## 🏆 Key Results

Our proposed method achieves superior performance on the ICBHI 2017 Official Split, specifically targeting high sensitivity for reliable clinical screening.

| Metric | Score |
| :--- | :--- |
| **Sensitivity (Se)** | **68.31%** |
| **Specificity (Sp)** | **67.89%** |
| **ICBHI Score** | **68.10%** |

## 📂 Project Structure

```text
ICBHI-AST-SAM/
├── data/                   # Place dataset files here (see instructions below)
├── checkpoints/            # Saved models
├── src/                    # Source code (Model, Dataset, SAM)
├── preprocess.py           # Data preparation script
├── train.py                # Training loop
├── evaluate.py             # Evaluation and Visualization
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

```bash
git clone https://github.com/Atakanisik/ICBHI-AST-SAM.git
cd ICBHI-AST-SAM
```

```bash
pip install -r requirements.txt
```

##  Dataset Preparation

Due to licensing restrictions, we cannot distribute the ICBHI 2017 dataset directly in this repository. Please follow these steps to set up the data:

1.  **Download the Database:**
    * Download the full dataset zip file from the [Official ICBHI Challenge Website](https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip).
    * Extract the contents. You should have a folder named `ICBHI_final_database` containing `.wav` and `.txt` files.
    * Move this folder inside the `data/` directory.

2.  **Download the Split File:**
    * Download the official train/test split file: [ICBHI_challenge_train_test.txt](https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_challenge_train_test.txt).
    * Place this text file directly inside the `data/` directory.

**Your `data/` directory should look like this:**
```text
data/
├── ICBHI_final_database/
│   ├── 101_1b1_Al_sc_Meditron.wav
│   ├── 101_1b1_Al_sc_Meditron.txt
│   └── ... (920 files)
└── ICBHI_challenge_train_test.txt
```
1. Preprocessing

Run the preprocessing script to generate the spectrogram-ready .npz file. This applies Cyclic Padding to ensure all samples are fixed to 8 seconds without information loss.

```bash
python preprocess.py
```
2. Training

Train the AST model using the SAM optimizer. The script automatically saves the best model based on the ICBHI Score (Average of Sensitivity and Specificity).

```bash
python train.py --epochs 20 --batch_size 8 --lr 1e-5
```
3. Evaluation

Evaluate the trained model on the official test set and generate the Confusion Matrix (Figure 2 in the paper).

```bash
python evaluate.py --model_path ./checkpoints/best_model.pth
```
⚠️ Note on Reproducibility


The results presented in the paper were obtained using mixed-precision (FP16) inference on an NVIDIA Tesla L4 GPU. Due to hardware differences and the non-deterministic nature of some CUDA operations, slight variations (±0.5%) in Sensitivity/Specificity metrics may be observed when retraining from scratch or running on different hardware.




