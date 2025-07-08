#  Total Perspective Vortex — Brain-Computer Interface with EEG & ML


## 🧠 Introduction

This project is about designing a **brain-computer interface (BCI)** using **EEG data** and **machine learning**. From raw brain signals, we aim to infer the user's motor intention: whether they imagine moving a **hand** or a **foot**, in near **real-time**.

We process open EEG datasets and implement dimensionality reduction techniques (e.g., CSP, PCA) to transform signals into meaningful features before classification.

## 🎯 Goals

-   🧹 Parse and filter raw EEG data.
    
-   📉 Implement a dimensionality reduction algorithm (CSP).
    
-   🔁 Use scikit-learn’s pipeline for full treatment flow.
    
-   ⏱️ Simulate real-time classification of EEG signal chunks (< 2s).

## 🚀 Getting Started
Clone the repository
```
git clone https://github.com/m3L4n/total-perspective-vortex.git
cd total-perspective-vortex
```

Install dependencies
```
python -m venv [path of your wanted venv folder]
pip install -r requirements.txt
```

## 🧪 Usage

### Train a model
```
python mybci.py 4 14 train
```
### Predict from a stream (simulated)
```
python mybci.py 4 14 predict
```
### Global evaluation across multiple subjects
```
python mybci.py
```

## 📈 Evaluation

-   Accuracy is measured across subjects and experiments
    
-   Cross-validation is used to evaluate the entire ML pipeline
    
-   Real-time prediction must happen within 2 seconds of signal input
    
-   Example output:
```
epoch 01: [2] [1] False
epoch 02: [1] [1] True
...
Accuracy: 0.7666
```
Average classification scores are computed per experiment and overall:
```
Experiment 0: accuracy = 0.6991
Experiment 1: accuracy = 0.8718
...
Overall accuracy: 0.8261
```

