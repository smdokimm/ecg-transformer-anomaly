import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate(model, loader, threshold, device):
    model.eval()
    test_losses = []
    test_labels_list = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            recon = model(x_batch)
            batch_mae = torch.mean(torch.abs(recon - x_batch), dim=1).cpu().numpy()
            test_losses.extend(batch_mae.tolist())
            test_labels_list.extend(y_batch.numpy().tolist())

    test_losses = np.array(test_losses).ravel()
    test_labels_list = np.array(test_labels_list).ravel()
    bin_test_labels = (test_labels_list != 1).astype(int)
    pred_anomaly = (test_losses > threshold).astype(int)

    acc  = accuracy_score(bin_test_labels, pred_anomaly)
    prec = precision_score(bin_test_labels, pred_anomaly, zero_division=0)
    rec  = recall_score(bin_test_labels, pred_anomaly, zero_division=0)
    f1   = f1_score(bin_test_labels, pred_anomaly, zero_division=0)
    auc  = roc_auc_score(bin_test_labels, test_losses)

    return {
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "test_losses": test_losses,
        "test_labels": bin_test_labels,
        "preds": pred_anomaly
    }
