import os
import wandb
import numpy as np
import torch
from config import config
from utils.data_loader import load_dataset, make_loader
from train.training import train
from train.evaluate import evaluate
from train.visualize import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_histograms,
    plot_kde,
    plot_signal_reconstruction,
    plot_learning_curve,
    plot_training_curve,
)
from utils.introspection import register_hooks, layer_shapes

# --- Initialize W&B ---
os.environ["WANDB_API_KEY"] = "d346a4dc3445453305891dfd3d50b166c3ca8e45"
wandb.login()
run = wandb.init(project=config["project_name"], config=config)

# --- Load Data ---
(train_signals, train_labels), (valid_signals, valid_labels), (test_signals, test_labels) = load_dataset()
config["in_channels"] = 1 if train_signals.ndim == 2 else train_signals.shape[-1]
train_loader = make_loader(train_signals, train_labels, batch_size=config["batch_size"])
valid_loader = make_loader(valid_signals, valid_labels, batch_size=config["batch_size"])
# Prepare test loader
from torch.utils.data import DataLoader, TensorDataset
tensor_x = torch.from_numpy(test_signals[..., None]).float()
tensor_y = torch.from_numpy(test_labels).long()
test_loader = DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=config["batch_size"])

# --- Model and Hooks ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.transformer import ECGTransformerAutoencoder
model = ECGTransformerAutoencoder(config).to(device)
register_hooks(model)
# Dummy forward to capture layer output shapes
_ = model(torch.zeros(1, config["seq_len"], config["in_channels"], device=device))
print("=== Layer Output Shapes ===")
for name, shape in layer_shapes.items():
    print(f"{name:50s} -> {shape}")

# --- Training ---
model, train_epoch_loss_list, val_epoch_loss_list = train(
    config,
    (train_signals, train_labels),
    (valid_signals, valid_labels)
)

# --- Threshold Computation ---
model.eval()
train_mae = []
with torch.no_grad():
    for x_batch, _ in train_loader:
        x_batch = x_batch.to(device)
        recon = model(x_batch)
        batch_mae = torch.mean(torch.abs(recon - x_batch), dim=1).cpu().numpy()
        train_mae.extend(batch_mae)
train_mae = np.array(train_mae)
threshold = train_mae.mean() + 2.0 * train_mae.std()
wandb.log({"threshold": threshold})

# --- Evaluation ---
results = evaluate(model, test_loader, threshold, device)
wandb.log({
    "test_acc": results["acc"],
    "test_precision": results["precision"],
    "test_recall": results["recall"],
    "test_f1": results["f1"],
    "test_auc": results["auc"]
})

# --- Visualization ---
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(results["test_labels"], results["preds"])
plot_confusion_matrix(cm, ["Normal", "Anomaly"])
plot_roc_curve(results["test_labels"], results["test_losses"], results["auc"])
plot_histograms(results["test_losses"], threshold, results["test_labels"])
plot_kde(results["test_losses"], threshold, results["test_labels"])
plot_signal_reconstruction(model, train_loader, test_loader, results["test_labels"], device)

# --- Learning Curve ---
train_sizes = list(np.linspace(100, len(train_signals), num=5, dtype=int))
plot_learning_curve(train_sizes, train_epoch_loss_list[:5], val_epoch_loss_list[:5])
# --- Training Curve ---
plot_training_curve(train_epoch_loss_list, val_epoch_loss_list)

wandb.finish()