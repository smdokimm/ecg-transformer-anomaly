import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.metrics import roc_curve
import torch

# --- Confusion Matrix ---
def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    wandb.log({"Confusion Matrix": wandb.Image(fig)})
    plt.close(fig)

# --- ROC Curve ---
def plot_roc_curve(y_true, scores, auc_score):
    fpr, tpr, _ = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    ax.plot([0,1], [0,1], '--')
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid()
    wandb.log({"ROC Curve": wandb.Image(fig)})
    plt.close(fig)

# --- Reconstruction Error Histograms ---
def plot_histograms(test_losses, threshold, bin_test_labels):
    normal_losses = test_losses[bin_test_labels == 0]
    anomaly_losses = test_losses[bin_test_labels == 1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))

    ax1.hist(test_losses, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(x=threshold, color='red', linestyle='--', label=f"Threshold={threshold:.4f}")
    ax1.set_title("Reconstruction Error (All Test Samples)")
    ax1.legend()
    ax1.grid(axis='y')

    ax2.hist(normal_losses, bins=50, alpha=0.5, edgecolor='black', label="Normal")
    ax2.hist(anomaly_losses, bins=50, alpha=0.5, edgecolor='black', label="Anomaly")
    ax2.axvline(x=threshold, color='blue', linestyle='--', label=f"Threshold={threshold:.4f}")
    ax2.set_title("Reconstruction Error (Normal vs. Anomaly)")
    ax2.legend()
    ax2.grid(axis='y')

    plt.tight_layout()
    wandb.log({"Reconstruction Error Histogram": wandb.Image(fig)})
    plt.close(fig)

# --- KDE Plot ---
def plot_kde(test_losses, threshold, bin_test_labels):
    normal = test_losses[bin_test_labels == 0]
    anomaly = test_losses[bin_test_labels == 1]
    fig, ax = plt.subplots(figsize=(8,5))
    sns.kdeplot(x=normal, fill=True, alpha=0.5, label="Normal", ax=ax)
    sns.kdeplot(x=anomaly, fill=True, alpha=0.5, label="Anomaly", ax=ax)
    ax.axvline(x=threshold, linestyle='--', label=f"Threshold={threshold:.4f}")
    ax.set_title("Reconstruction Error KDE")
    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)
    wandb.log({"Reconstruction Error KDE": wandb.Image(fig)})
    plt.close(fig)

# --- Signal Reconstruction Visualization ---
def plot_signal_reconstruction(model, train_loader, test_loader, bin_test_labels, device):
    model.eval()
    # Normal sample
    normal_batch, _ = next(iter(train_loader))
    normal = normal_batch[0:1].to(device)
    recon_n = model(normal).cpu().detach().numpy().squeeze()
    orig_n = normal.cpu().numpy().squeeze()

    # Anomalous sample
    all_test = []
    all_labels = []
    for x, y in test_loader:
        all_test.append(x)
        all_labels.append(y)
    X = torch.cat(all_test, dim=0)
    Y = torch.cat(all_labels, dim=0)
    idx_anom = (Y != 1).nonzero().flatten()
    if len(idx_anom) > 0:
        x_a = X[idx_anom[0]:idx_anom[0]+1].to(device)
    else:
        x_a = X[0:1].to(device)
    recon_a = model(x_a).cpu().detach().numpy().squeeze()
    orig_a = x_a.cpu().numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,5))
    ax1.plot(orig_a, label='Actual Anomaly')
    ax1.plot(recon_a, label='Reconstructed Anomaly')
    ax1.set_title('Anomalous Signal Reconstruction')
    ax1.legend()

    ax2.plot(orig_n, label='Actual Normal')
    ax2.plot(recon_n, label='Reconstructed Normal')
    ax2.set_title('Normal Signal Reconstruction')
    ax2.legend()

    plt.tight_layout()
    wandb.log({"Signal Reconstruction": wandb.Image(fig)})
    plt.close(fig)

# --- Learning Curve ---
def plot_learning_curve(train_sizes, train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(train_sizes, train_losses, marker='o', label='Train Loss')
    ax.plot(train_sizes, val_losses, marker='o', label='Val Loss')
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('MAE (L1 Loss)')
    ax.set_title('Learning Curve')
    ax.grid(True)
    ax.legend()
    wandb.log({"Learning Curve": wandb.Image(fig)})
    plt.close(fig)

# --- Training Curve ---
def plot_training_curve(train_epoch_loss, val_epoch_loss):
    fig, ax = plt.subplots(figsize=(8,5))
    epochs = list(range(1, len(train_epoch_loss)+1))
    ax.plot(epochs, train_epoch_loss, label='Train Loss')
    ax.plot(epochs, val_epoch_loss, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Curve')
    ax.grid(True)
    ax.legend()
    wandb.log({"Training Curve": wandb.Image(fig)})
    plt.close(fig)