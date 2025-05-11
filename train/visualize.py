import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.metrics import roc_curve

def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    wandb.log({"Confusion Matrix": wandb.Image(fig)})
    plt.close(fig)

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