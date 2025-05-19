import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_ecg_data(file_path):
    arr = np.loadtxt(file_path)
    labels = arr[:, 0]
    signals = arr[:, 1:]
    return signals, labels

def load_dataset():
    data_path = "data"
    train_file = os.path.join(data_path, "ECG5000_T.txt")
    valid_file = os.path.join(data_path, "ECG5000_V.txt")
    test_file  = os.path.join(data_path, "ECG5000_TE.txt")

    train_signals, train_labels = load_ecg_data(train_file)
    valid_signals, valid_labels = load_ecg_data(valid_file)
    test_signals,  test_labels  = load_ecg_data(test_file)

    train_idx = np.where(train_labels == 1)[0]
    valid_idx = np.where(valid_labels == 1)[0]

    return (
        (train_signals[train_idx], train_labels[train_idx]),
        (valid_signals[valid_idx], valid_labels[valid_idx]),
        (test_signals, test_labels)
    )

def make_loader(
    signals: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False
) -> DataLoader:
    if signals.ndim == 2:
        signals = signals[..., None]

    tensor_x = torch.from_numpy(signals).float()
    tensor_y = torch.from_numpy(labels).long()

    return DataLoader(
        TensorDataset(tensor_x, tensor_y),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
