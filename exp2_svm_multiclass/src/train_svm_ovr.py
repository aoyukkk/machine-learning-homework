import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class BinaryLinearSVM:
    def __init__(self, lr=0.01, c=1.0, epochs=3000, seed=42):
        self.lr = lr
        self.c = c
        self.epochs = epochs
        self.rng = np.random.default_rng(seed)
        self.w = None
        self.b = 0.0

    def fit(self, x, y):
        n, d = x.shape
        self.w = self.rng.normal(0, 0.01, size=d)
        self.b = 0.0
        for _ in range(self.epochs):
            margins = y * (x @ self.w + self.b)
            mask = margins < 1.0
            grad_w = self.w - self.c * np.mean((y[mask, None] * x[mask]), axis=0) if np.any(mask) else self.w
            grad_b = -self.c * np.mean(y[mask]) if np.any(mask) else 0.0
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def decision_function(self, x):
        return x @ self.w + self.b


class OneVsRestSVM:
    def __init__(self, lr=0.01, c=1.0, epochs=3000):
        self.lr = lr
        self.c = c
        self.epochs = epochs
        self.models = {}
        self.classes_ = None

    def fit(self, x, y):
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            y_binary = np.where(y == cls, 1.0, -1.0)
            model = BinaryLinearSVM(lr=self.lr, c=self.c, epochs=self.epochs, seed=int(cls) + 7)
            model.fit(x, y_binary)
            self.models[int(cls)] = model

    def predict(self, x):
        scores = np.column_stack([self.models[int(cls)].decision_function(x) for cls in self.classes_])
        return self.classes_[np.argmax(scores, axis=1)]


def plot_confusion(cm, labels, path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (Iris, OVR-SVM)")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root, "outputs")
    fig_dir = os.path.join(root, "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    data = load_iris()
    x, y = data.data, data.target
    class_names = data.target_names

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    svm = OneVsRestSVM(lr=0.005, c=4.0, epochs=12000)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)

    acc = (y_pred == y_test).mean()
    cm = confusion_matrix(y_test, y_pred)

    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["accuracy", float(acc)])

    np.savetxt(os.path.join(out_dir, "confusion_matrix.txt"), cm, fmt="%d")
    plot_confusion(cm, class_names, os.path.join(fig_dir, "confusion_matrix.png"))

    print(f"Test accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()
