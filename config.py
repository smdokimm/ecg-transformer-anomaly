# === config.py ===
import os

config = {
    "project_name": "ecg-transformer-anomaly2",
    "epochs": 100,
    "patience": 5,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "seq_len": 140,
    "d_model": 64,
    "nhead": 16,
    "ff_dim": 16,
    "num_layers": 2,
    "dropout": 0.2,
    "data_dir": "data",
    "train_file": "ECG5000_T.txt",
    "valid_file": "ECG5000_V.txt",
    "test_file":  "ECG5000_TE.txt"
}

def get_data_path(filename):
    return os.path.join(config["data_dir"], filename)
