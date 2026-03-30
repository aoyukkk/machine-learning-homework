import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def generate_data(seed=42):
    rng = np.random.default_rng(seed)
    centers = np.array([[-3, -2], [-3, 3], [3, -1], [3, 3]], dtype=np.float64)
    points = []
    labels = []
    for i, c in enumerate(centers):
        pts = rng.normal(loc=c, scale=0.8, size=(50, 2))
        points.append(pts)
        labels.extend([i] * 50)
    x = np.vstack(points)
    y = np.array(labels)
    return x, y


class KMeansScratch:
    def __init__(self, k=4, max_iter=100, tol=1e-5, seed=42):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.centers = None
        self.sse_history = []

    def fit(self, x):
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(x.shape[0], size=self.k, replace=False)
        centers = x[idx].copy()

        for _ in range(self.max_iter):
            dists = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
            cluster_id = np.argmin(dists, axis=1)

            new_centers = np.zeros_like(centers)
            for i in range(self.k):
                cluster_points = x[cluster_id == i]
                if len(cluster_points) == 0:
                    new_centers[i] = centers[i]
                else:
                    new_centers[i] = cluster_points.mean(axis=0)

            sse = np.sum((x - new_centers[cluster_id]) ** 2)
            self.sse_history.append(float(sse))

            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift < self.tol:
                break

        self.centers = centers
        return cluster_id


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root, "outputs")
    fig_dir = os.path.join(root, "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    x, _ = generate_data()
    model = KMeansScratch(k=4, max_iter=100, tol=1e-5)
    pred = model.fit(x)

    with open(os.path.join(out_dir, "sse_history.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "sse"])
        for i, v in enumerate(model.sse_history, start=1):
            writer.writerow([i, v])

    np.savetxt(os.path.join(out_dir, "cluster_centers.txt"), model.centers, fmt="%.6f")

    plt.figure(figsize=(6.8, 5.5))
    for i in range(4):
        pts = x[pred == i]
        plt.scatter(pts[:, 0], pts[:, 1], s=18, alpha=0.75, label=f"cluster {i}")
    plt.scatter(model.centers[:, 0], model.centers[:, 1], c="black", s=220, marker="X", label="centers")
    plt.legend()
    plt.title("K-means Clustering Result")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "kmeans_clusters.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(6.8, 4.2))
    plt.plot(model.sse_history, color="#cc5500", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("SSE")
    plt.title("K-means SSE Convergence")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "sse_curve.png"), dpi=180)
    plt.close()

    print(f"Finished in {len(model.sse_history)} iterations")


if __name__ == "__main__":
    main()
