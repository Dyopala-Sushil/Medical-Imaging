import torch
import matplotlib.pyplot as plt


def plot_lr_curves(path2checkpoint):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(path2checkpoint, map_location=device)
    val_interval = 1
    plt.figure(figsize=(15, 5))

    # Subplot for training loss
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Dice Loss")
    num_epoch = [i + 1 for i in range(len(checkpoint["train_loss"]))]
    train_loss = checkpoint["train_loss"]
    plt.xlabel("#Epochs")
    plt.ylabel("Train Dice Loss")
    plt.plot(num_epoch, train_loss)

    # Subplot for validation score
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice Score")
    num_epoch = [val_interval * (i + 1) for i in range(len(checkpoint["val_dice"]))]
    val_score = checkpoint["val_dice"]
    plt.xlabel("#Epochs")
    plt.ylabel("Val Dice Score")
    plt.plot(num_epoch, val_score)
    plt.show()