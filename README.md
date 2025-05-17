# ECG Transformer Anomaly Detection

A modular Transformer-based Autoencoder for anomaly detection on the ECG5000 dataset.
Built with PyTorch and optimized with Weights & Biases (W\&B) for training visualization and experiment tracking.

## 📁 Project Structure

* `.venv/` — virtual environment (not tracked in git)
* `requirements.txt` — list of required packages
* `data/` — ECG5000 dataset files (`ECG5000_T.txt`, `ECG5000_V.txt`, `ECG5000_TE.txt`)
* `models/`, `train/`, `utils/` — source code modules
* `main.py`, `config.py` — entrypoint and configuration

## 🧪 Features

* Custom `MultiHeadAttention`, `PositionalEncoding` implementations
* Early stopping with best model checkpointing
* Automatic thresholding and anomaly evaluation (accuracy, precision, recall, F1, AUC)
* W\&B logging for metrics and visualizations (confusion matrix, ROC curve, histograms, KDE, signal reconstructions, learning/training curves)

## 📊 Dataset

* [ECG5000](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000)
  UCR time-series classification archive

## 📋 Prerequisites

* Linux-based GPU cluster
* Python 3.10
* `virtualenv` installed in your home (e.g. `pip install --user virtualenv`)

## 🔧 Setup

```bash
# 1) Clone & enter
git clone https://github.com/smdokimm/ecg-transformer-anomaly.git
cd ecg-transformer-anomaly

# 2) Create virtualenv
~/.local/bin/virtualenv venv

# 3) Activate it
source venv/bin/activate

# 4) Verify activation
echo "$VIRTUAL_ENV"      # → /homes/<you>/ecg-transformer-anomaly/venv
which python             # → …/venv/bin/python

# 5) Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 🚀 Run (single GPU)

```bash
source venv/bin/activate
python main.py
```

## ⚙️ Run (3-way parallel on GPUs 0, 1, 2)

```bash
# Option A: 3 terminal panes
cd ecg-transformer-anomaly; source venv/bin/activate; export CUDA_VISIBLE_DEVICES=0; python main.py
cd ecg-transformer-anomaly; source venv/bin/activate; export CUDA_VISIBLE_DEVICES=1; python main.py
cd ecg-transformer-anomaly; source venv/bin/activate; export CUDA_VISIBLE_DEVICES=2; python main.py

# Option B: single-shell loop
cd ecg-transformer-anomaly
source venv/bin/activate
for GPU in 0 1 2; do
  (export CUDA_VISIBLE_DEVICES=$GPU; python main.py) &
done
wait
```

## 🛑 Monitoring & Cleanup

```bash
# Monitor CPU/Memory
htop

# Monitor GPU usage (refresh every 2s)
nvidia-smi -l 2

# Cancel *only your* Slurm jobs
scancel -u $USER    # cancels jobs submitted to Slurm under your username

# Alternatively, kill all your processes (incl. non-Slurm)
pkill -u $USER

# Finally, logout
exit
```
