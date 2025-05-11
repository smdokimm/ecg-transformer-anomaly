# ECG Transformer Anomaly Detection

A modular Transformer-based Autoencoder for anomaly detection on the ECG5000 dataset.  
Built with PyTorch and optimized with Weights & Biases (W&B) for training visualization and experiment tracking.

## 📁 Project Structure
- `.venv/` — virtual environment (not tracked in git)
- `requirements.txt` — list of required packages

## 🧪 Features
- Custom `MultiHeadAttention`, `PositionalEncoding`
- Early stopping and best model saving
- Confusion matrix, ROC, and reconstruction plots via W&B

## 📊 Dataset
- [ECG5000](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000)  
  UCR time-series classification archive

## 🚀 Training
```bash
python main.py
