import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root, "outputs")
    fig_dir = os.path.join(root, "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.Compose([transforms.ToTensor()])

    train_full = datasets.MNIST(root=os.path.join(root, "model_store"), train=True, download=True, transform=tfm)
    train_ds, val_ds = random_split(train_full, [54000, 6000], generator=torch.Generator().manual_seed(42))
    test_ds = datasets.MNIST(root=os.path.join(root, "model_store"), train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    model = SimpleCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, 6):
        model.train()
        train_loss = 0.0
        train_total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = x.size(0)
            train_loss += float(loss.item()) * bs
            train_total += bs

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                val_correct += int((pred == y).sum().item())
                val_total += y.numel()

        history.append([epoch, train_loss / train_total, val_correct / val_total])
        print(f"epoch={epoch} train_loss={train_loss/train_total:.5f} val_acc={val_correct/val_total:.4f}")

    y_true, y_pred = [], []
    feature_map_sample = None
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(y.numpy().tolist())

            if feature_map_sample is None:
                fmap = F.relu(model.conv1(x[:1])).cpu().numpy()[0]
                feature_map_sample = fmap

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float((y_true == y_pred).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["accuracy", acc])
        writer.writerow(["precision_macro", float(precision)])
        writer.writerow(["recall_macro", float(recall)])
        writer.writerow(["f1_macro", float(f1)])

    with open(os.path.join(out_dir, "train_history.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_acc"])
        writer.writerows(history)

    plt.figure(figsize=(6.8, 4.4))
    epochs = [h[0] for h in history]
    losses = [h[1] for h in history]
    accs = [h[2] for h in history]
    plt.plot(epochs, losses, label="train loss", linewidth=2)
    plt.plot(epochs, accs, label="val acc", linewidth=2)
    plt.xlabel("Epoch")
    plt.legend()
    plt.title("CNN Training Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "training_curve.png"), dpi=180)
    plt.close()

    if feature_map_sample is not None:
        plt.figure(figsize=(8, 8))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(feature_map_sample[i], cmap="viridis")
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "feature_maps_conv1.png"), dpi=180)
        plt.close()

    torch.save(model.state_dict(), os.path.join(root, "model_store", "cnn_mnist.pt"))
    print(f"test accuracy={acc:.4f}, precision={precision:.4f}, recall={recall:.4f}")


if __name__ == "__main__":
    main()
