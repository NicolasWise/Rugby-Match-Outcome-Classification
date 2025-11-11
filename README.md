# 🏉 Rugby Match Outcome Classification

This project predicts international rugby test match outcomes using machine learning.  
Two models are implemented:
1. **Naïve Bayes Baseline** (`NiaveBayesBaseline.py`)
2. **Feedforward Neural Network (FFN)** (`RugbyCLF_MLP.ipynb`)

---

## Project Structure

```
├── RugbyCLF_MLP.ipynb        # Neural network model (TensorFlow)
├── NiaveBayesBaseline.py     # Baseline Naïve Bayes classifier
├── results.csv               # Dataset
└── README.md                 # This file
```

---

## Setup Instructions

### Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate     # On Mac/Linux
venv\Scripts\activate        # On Windows
```

### Install Dependencies
All experiments can be reproduced using Python 3.10+ and the following packages:

> pip install -r requirements.txt


---

## Running the Experiments

### 1️⃣ Naïve Bayes Baseline
Run the baseline model directly:
```bash
python NiaveBayesBaseline.py
```

**Output:**  
- Validation and test accuracies  
- Classification report and confusion matrix  

---

### 2️⃣ Feedforward Neural Network (FFN)
Launch Jupyter and open the notebook:
```bash
jupyter notebook RugbyCLF_MLP.ipynb
```

Run all cells in sequence to:
- Load and preprocess the data  
- Train the FFN using TensorFlow/Keras  
- Evaluate model performance (accuracy, loss curves, confusion matrix)  

**Final Architecture:**
- Layer 1: 128 neurons (ReLU)
- Layer 2: 64 neurons (ReLU)
- Output Layer: 3 neurons (Softmax)
- Optimizer: Adam  
- Loss: Cross-Entropy  
- Epochs: 100  
- Dropout: 0.3 
- Learning rate: 0.0005
- Weight Decay: 0.001

---

## Evaluation Metrics

The following metrics are logged:
- **Validation Accuracy** — performance on the held-out validation set  
- **Test Accuracy** — final generalization performance  
- **Confusion Matrix** — to inspect per-class performance (Home Win / Loss / Draw)

Results can be found at the bottom of the notebook.

---

## Author
**Nicolas Wise (WSXNIC001)**
**Alexander White (WHTALE015)** 
University of Cape Town — 2025  
Course: CSC4025Z — Artificial Intelligence

---

## Citation
If you reuse this code or analysis, please cite appropriately:
> Wise, N. White, A. (2025). *Predicting Rugby Match Outcomes Using Feedforward Neural Networks and Naïve Bayes Baselines*.  
> University of Cape Town, Department of Computer Science.
