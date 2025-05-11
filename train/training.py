import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from utils.data_loader import make_loader
from models.transformer import ECGTransformerAutoencoder


def train(config, train_data, valid_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ECGTransformerAutoencoder(config).to(device)
    criterion = nn.L1Loss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    train_loader = make_loader(*train_data, batch_size=config["batch_size"])
    valid_loader = make_loader(*valid_data, batch_size=config["batch_size"])

    best_val_loss = float('inf')
    counter = 0
    best_model_state = None

    train_epoch_loss_list = []
    val_epoch_loss_list = []

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_losses = []
        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            recon = model(x_batch)
            loss = criterion(recon, x_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_val, _ in valid_loader:
                x_val = x_val.to(device)
                recon_val = model(x_val)
                val_loss = criterion(recon_val, x_val)
                val_losses.append(val_loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        train_epoch_loss_list.append(avg_train_loss)
        val_epoch_loss_list.append(avg_val_loss)

        print(f"Epoch [{epoch}/{config['epochs']}]  Train Loss: {avg_train_loss:.6f}  Val Loss: {avg_val_loss:.6f}")
        wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= config["patience"]:
                print(f"Early stopping triggered at epoch={epoch}. Best Val Loss={best_val_loss:.6f}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Best model weights loaded (Val Loss =", best_val_loss, ")")

    return model, train_epoch_loss_list, val_epoch_loss_list