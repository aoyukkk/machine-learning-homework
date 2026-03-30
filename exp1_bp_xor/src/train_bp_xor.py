import csv
import os

import matplotlib.pyplot as plt
import numpy as np


class BPXOR:
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1, lr=0.5, seed=42):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 0.8, size=(input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.w2 = rng.normal(0, 0.8, size=(hidden_dim, output_dim))
        self.b2 = np.zeros((1, output_dim))
        self.lr = lr

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def mse(y_hat, y):
        return np.mean((y_hat - y) ** 2)

    def forward(self, x):
        z1 = x @ self.w1 + self.b1
        a1 = self.sigmoid(z1)
        z2 = a1 @ self.w2 + self.b2
        y_hat = self.sigmoid(z2)
        return z1, a1, z2, y_hat

    def backward(self, x, y, cache):
        _, a1, _, y_hat = cache
        n = x.shape[0]

        d_y_hat = 2.0 * (y_hat - y) / n
        d_z2 = d_y_hat * y_hat * (1.0 - y_hat)
        d_w2 = a1.T @ d_z2
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)

        d_a1 = d_z2 @ self.w2.T
        d_z1 = d_a1 * a1 * (1.0 - a1)
        d_w1 = x.T @ d_z1
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        self.w2 -= self.lr * d_w2
        self.b2 -= self.lr * d_b2
        self.w1 -= self.lr * d_w1
        self.b1 -= self.lr * d_b1

    def train(self, x, y, epochs=12000):
        losses = []
        for ep in range(epochs):
            cache = self.forward(x)
            y_hat = cache[-1]
            loss = self.mse(y_hat, y)
            losses.append(loss)
            self.backward(x, y, cache)
            if (ep + 1) % 2000 == 0:
                print(f"epoch={ep+1:5d} loss={loss:.6f}")
        return losses


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root, "outputs")
    fig_dir = os.path.join(root, "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([[0], [1], [1], [0]], dtype=np.float64)

    model = BPXOR(lr=0.6, hidden_dim=4)
    losses = model.train(x, y, epochs=12000)
    y_pred = model.forward(x)[-1]

    with open(os.path.join(out_dir, "history.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "mse_loss"])
        for i, val in enumerate(losses, start=1):
            writer.writerow([i, float(val)])

    with open(os.path.join(out_dir, "predictions.txt"), "w", encoding="utf-8") as f:
        for xi, yi, pi in zip(x, y, y_pred):
            f.write(f"x={xi.tolist()} y={float(yi[0]):.1f} pred={float(pi[0]):.4f}\n")

    plt.figure(figsize=(7, 4))
    plt.plot(losses, color="#1368ce", linewidth=2)
    plt.title("BP XOR Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "loss_curve.png"), dpi=180)
    plt.close()

    print("Final predictions:")
    for xi, pi in zip(x, y_pred):
        print(f"x={xi.tolist()} -> {float(pi[0]):.4f}")


if __name__ == "__main__":
    main()
