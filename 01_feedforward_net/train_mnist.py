"""
Demo 2: MNIST Handwritten Digit Classification
================================================
Train a FeedforwardNet on 70,000 handwritten digits (28×28 pixels).
Architecture: 784 → 256 → 128 → 10

Expected accuracy: ~96-97% with only NumPy (no frameworks!)

Reference: Zemke, AML Lecture 4, Slide 35 (Keras/TensorFlow MNIST example)
Here we do it from scratch instead.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import gzip
import struct
from pathlib import Path

from feedforward_net import FeedforwardNet


# ── Load MNIST ────────────────────────────────────────────────────────────
def load_mnist(path: str = "data"):
    """Download and load MNIST dataset."""
    import urllib.request

    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images":  "t10k-images-idx3-ubyte.gz",
        "test_labels":  "t10k-labels-idx1-ubyte.gz",
    }

    os.makedirs(path, exist_ok=True)

    for name, filename in files.items():
        filepath = os.path.join(path, filename)
        if not os.path.exists(filepath):
            print(f"  Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)

    def read_images(filepath):
        with gzip.open(filepath, "rb") as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(num, rows * cols).T.astype(np.float64) / 255.0

    def read_labels(filepath):
        with gzip.open(filepath, "rb") as f:
            magic, num = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    X_train = read_images(os.path.join(path, files["train_images"]))
    y_train_raw = read_labels(os.path.join(path, files["train_labels"]))
    X_test = read_images(os.path.join(path, files["test_images"]))
    y_test_raw = read_labels(os.path.join(path, files["test_labels"]))

    # One-hot encoding
    def one_hot(labels, num_classes=10):
        y = np.zeros((num_classes, len(labels)))
        for i, label in enumerate(labels):
            y[label, i] = 1
        return y

    return X_train, one_hot(y_train_raw), X_test, one_hot(y_test_raw), y_test_raw


def load_mnist_sklearn():
    """Fallback: load MNIST via scikit-learn."""
    from sklearn.datasets import fetch_openml

    print("  Loading MNIST via scikit-learn (may take a moment)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.T / 255.0
    y_raw = mnist.target.astype(int)

    X_train, X_test = X[:, :60000], X[:, 60000:]
    y_raw_train, y_raw_test = y_raw[:60000], y_raw[60000:]

    def one_hot(labels, num_classes=10):
        y = np.zeros((num_classes, len(labels)))
        for i, label in enumerate(labels):
            y[label, i] = 1
        return y

    return X_train, one_hot(y_raw_train), X_test, one_hot(y_raw_test), y_raw_test


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  MNIST — Feedforward Net from Scratch")
    print("=" * 60)

    # Load data
    print("\n📦 Loading MNIST dataset...")
    try:
        X_train, y_train, X_test, y_test, y_test_raw = load_mnist()
    except Exception:
        X_train, y_train, X_test, y_test, y_test_raw = load_mnist_sklearn()

    print(f"  Training samples: {X_train.shape[1]:,}")
    print(f"  Test samples:     {X_test.shape[1]:,}")
    print(f"  Input dimension:  {X_train.shape[0]} (28×28 pixels)")

    # ── Build Network ─────────────────────────────────────────────────
    # Architecture: 784 → 256 → 128 → 10
    net = FeedforwardNet(
        layer_sizes=[784, 256, 128, 10],
        activation="relu",
        output="softmax",
        init_method="he",
        seed=42,
    )
    net.summary()

    # ── Train ─────────────────────────────────────────────────────────
    print("\n🚀 Training...")
    history = net.train(
        X_train, y_train,
        epochs=20,
        batch_size=64,
        optimizer="adam",
        lr=0.001,
        l2_lambda=1e-4,
        dropout_rate=0.2,
        loss_fn="cross_entropy",
        verbose=True,
        X_val=X_test,
        y_val=y_test,
    )

    # ── Final Evaluation ──────────────────────────────────────────────
    test_loss, test_acc = net.evaluate(X_test, y_test)
    print("\n" + "=" * 60)
    print(f"  ✅ Final Test Accuracy: {test_acc * 100:.2f}%")
    print(f"  📉 Final Test Loss:     {test_loss:.4f}")
    print("=" * 60)

    # ── Plots ─────────────────────────────────────────────────────────
    os.makedirs("figures", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Loss curves
    axes[0].plot(history["train_loss"], label="Train", color="#2563eb", linewidth=2)
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Validation", color="#dc2626", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2) Accuracy curves
    axes[1].plot(
        [a * 100 for a in history["train_acc"]],
        label="Train", color="#2563eb", linewidth=2,
    )
    if "val_acc" in history:
        axes[1].plot(
            [a * 100 for a in history["val_acc"]],
            label="Validation", color="#dc2626", linewidth=2,
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training & Validation Accuracy", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3) Sample predictions
    predictions = net.predict(X_test[:, :25])
    for idx in range(25):
        ax = fig.add_subplot(1, 3, 3) if idx == 0 else axes[2]

    axes[2].set_visible(False)
    fig_samples, axes_s = plt.subplots(5, 5, figsize=(8, 8))
    for idx in range(25):
        ax = axes_s[idx // 5, idx % 5]
        img = X_test[:, idx].reshape(28, 28)
        pred = predictions[idx]
        true = y_test_raw[idx]
        ax.imshow(img, cmap="gray")
        color = "#16a34a" if pred == true else "#dc2626"
        ax.set_title(f"{pred}", color=color, fontsize=14, fontweight="bold")
        ax.axis("off")

    fig_samples.suptitle(
        "Sample Predictions (green=correct, red=wrong)",
        fontsize=14, fontweight="bold",
    )
    fig_samples.tight_layout()
    fig_samples.savefig("figures/mnist_samples.png", dpi=150, bbox_inches="tight")

    fig.tight_layout()
    fig.savefig("figures/mnist_training.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    print("\n📊 Plots saved to figures/")
