import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment


def generate_gmm_data(seed=7):
    rng = np.random.default_rng(seed)
    means = np.array([[-3, -2], [0, 3.5], [3, -1]], dtype=np.float64)
    covs = np.array([
        [[1.1, 0.2], [0.2, 0.7]],
        [[0.8, -0.1], [-0.1, 1.0]],
        [[0.7, 0.0], [0.0, 0.9]],
    ])
    n_each = [160, 170, 170]
    xs, ys = [], []
    for i in range(3):
        pts = rng.multivariate_normal(means[i], covs[i], size=n_each[i])
        xs.append(pts)
        ys.extend([i] * n_each[i])
    return np.vstack(xs), np.array(ys)


class EMGMM:
    def __init__(self, k=3, max_iter=300, tol=1e-6, seed=42):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.pi = None
        self.mu = None
        self.sigma = None
        self.log_likelihood = []

    def _gaussian_pdf(self, x, mu, sigma):
        d = x.shape[1]
        eps = 1e-6
        sigma = sigma + eps * np.eye(d)
        det = np.linalg.det(sigma)
        inv = np.linalg.inv(sigma)
        diff = x - mu
        exponent = -0.5 * np.sum((diff @ inv) * diff, axis=1)
        coeff = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)
        return coeff * np.exp(exponent)

    def fit(self, x):
        n, d = x.shape
        rng = np.random.default_rng(self.seed)
        self.pi = np.ones(self.k) / self.k
        self.mu = x[rng.choice(n, self.k, replace=False)]
        self.sigma = np.array([np.cov(x, rowvar=False) for _ in range(self.k)])

        prev_ll = -np.inf
        for _ in range(self.max_iter):
            # E-step
            resp = np.zeros((n, self.k))
            for j in range(self.k):
                resp[:, j] = self.pi[j] * self._gaussian_pdf(x, self.mu[j], self.sigma[j])
            resp_sum = np.sum(resp, axis=1, keepdims=True) + 1e-12
            resp /= resp_sum

            # M-step
            nk = np.sum(resp, axis=0)
            self.pi = nk / n
            for j in range(self.k):
                self.mu[j] = np.sum(resp[:, j:j+1] * x, axis=0) / nk[j]
                diff = x - self.mu[j]
                self.sigma[j] = (resp[:, j:j+1] * diff).T @ diff / nk[j]

            mix_pdf = np.zeros(n)
            for j in range(self.k):
                mix_pdf += self.pi[j] * self._gaussian_pdf(x, self.mu[j], self.sigma[j])
            ll = np.sum(np.log(mix_pdf + 1e-12))
            self.log_likelihood.append(float(ll))

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

    def predict(self, x):
        probs = np.zeros((x.shape[0], self.k))
        for j in range(self.k):
            probs[:, j] = self.pi[j] * self._gaussian_pdf(x, self.mu[j], self.sigma[j])
        return np.argmax(probs, axis=1)


def clustering_accuracy(y_true, y_pred, k):
    w = np.zeros((k, k), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        w[yt, yp] += 1
    row, col = linear_sum_assignment(-w)
    return w[row, col].sum() / len(y_true)


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root, "outputs")
    fig_dir = os.path.join(root, "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    x, y_true = generate_gmm_data()
    model = EMGMM(k=3, max_iter=300, tol=1e-6)
    model.fit(x)
    y_pred = model.predict(x)
    acc = clustering_accuracy(y_true, y_pred, 3)

    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["cluster_accuracy", float(acc)])
        writer.writerow(["final_log_likelihood", float(model.log_likelihood[-1])])

    plt.figure(figsize=(6.8, 4.2))
    plt.plot(model.log_likelihood, color="#5b4b8a", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.title("EM-GMM Convergence")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "log_likelihood_curve.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(6.8, 5.4))
    for i in range(3):
        pts = x[y_pred == i]
        plt.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.7, label=f"cluster {i}")
    plt.scatter(model.mu[:, 0], model.mu[:, 1], c="black", marker="X", s=230, label="means")
    plt.title("EM-GMM Clustering Result")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "gmm_clusters.png"), dpi=180)
    plt.close()

    print(f"Clustering accuracy (best mapping): {acc:.4f}")


if __name__ == "__main__":
    main()
