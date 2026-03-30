import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def build_noisy_dataset(n=500, noise_dim=6, seed=42):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.0, size=(n, 2))
    useful = np.column_stack([
        z[:, 0] + 0.25 * z[:, 1],
        -0.6 * z[:, 0] + 0.9 * z[:, 1],
        0.8 * z[:, 0] - 0.3 * z[:, 1],
    ])
    useful_noisy = useful + rng.normal(0, 0.22, size=useful.shape)
    noise = rng.normal(0, 0.55, size=(n, noise_dim))
    x = np.hstack([useful_noisy, noise])
    return x, useful


class PCAScratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_var_ratio_ = None

    def fit(self, x):
        self.mean_ = x.mean(axis=0, keepdims=True)
        xc = x - self.mean_
        cov = np.cov(xc, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        self.components_ = eigvecs[:, : self.n_components]
        self.explained_var_ratio_ = eigvals / np.sum(eigvals)

    def transform(self, x):
        return (x - self.mean_) @ self.components_

    def inverse_transform(self, z):
        return z @ self.components_.T + self.mean_


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root, "outputs")
    fig_dir = os.path.join(root, "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    x, useful = build_noisy_dataset(n=500, noise_dim=6)
    pca = PCAScratch(n_components=3)
    pca.fit(x)

    z = pca.transform(x)
    x_recon = pca.inverse_transform(z)

    # Evaluate denoising directly on the useful 3-D subspace.
    pca_useful = PCAScratch(n_components=2)
    pca_useful.fit(x[:, :3])
    z_useful = pca_useful.transform(x[:, :3])
    useful_recon = pca_useful.inverse_transform(z_useful)

    mse_before = np.mean((x[:, :3] - useful) ** 2)
    mse_after = np.mean((useful_recon - useful) ** 2)

    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["mse_before", float(mse_before)])
        writer.writerow(["mse_after", float(mse_after)])
        for i, r in enumerate(pca.explained_var_ratio_[:6], start=1):
            writer.writerow([f"explained_ratio_pc{i}", float(r)])

    plt.figure(figsize=(6.8, 4.2))
    ratios = pca.explained_var_ratio_[:8]
    plt.bar(range(1, len(ratios) + 1), ratios, color="#4c9f70")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Explained Variance")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "explained_variance.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(6.8, 5.2))
    plt.scatter(useful[:, 0], useful[:, 1], s=12, alpha=0.65, label="clean")
    plt.scatter(x[:, 0], x[:, 1], s=10, alpha=0.30, label="noisy")
    plt.scatter(useful_recon[:, 0], useful_recon[:, 1], s=10, alpha=0.45, label="pca-recon")
    plt.legend()
    plt.title("Denoising Effect in Feature Subspace")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "denoise_compare.png"), dpi=180)
    plt.close()

    print(f"MSE before PCA denoise: {mse_before:.6f}")
    print(f"MSE after  PCA denoise: {mse_after:.6f}")


if __name__ == "__main__":
    main()
