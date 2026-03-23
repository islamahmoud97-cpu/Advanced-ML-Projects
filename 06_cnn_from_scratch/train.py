"""
Demo: CNN from Scratch on Handwritten Digits
===============================================
Based on: Zemke, AML Lectures 4-6

Architecture (inspired by LeNet-5, Lecture 5):
  Conv(1→8, 3×3) → ReLU → MaxPool(2×2) →
  Conv(8→16, 3×3) → ReLU → MaxPool(2×2) →
  Flatten → Dense(→64) → ReLU → Dense(→10) → Softmax

Dataset: sklearn digits (8×8) for quick demo, MNIST (28×28) for full training.

Run: python train.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

from layers import Conv2D, MaxPool2D, ReLU, Flatten, Dense, Softmax
from model import CNN


def load_digits_sklearn():
    """Load sklearn digits (8×8 images, 10 classes) — always available."""
    from sklearn.datasets import load_digits
    
    digits = load_digits()
    X = digits.data.reshape(-1, 1, 8, 8).astype(np.float64) / 16.0
    y_raw = digits.target
    
    # One-hot encode
    y = np.zeros((len(y_raw), 10))
    y[np.arange(len(y_raw)), y_raw] = 1
    
    # Split 80/20
    n_train = int(0.8 * len(X))
    perm = np.random.permutation(len(X))
    X, y, y_raw = X[perm], y[perm], y_raw[perm]
    
    return (X[:n_train], y[:n_train], X[n_train:], y[n_train:], y_raw[n_train:])


def load_mnist():
    """Try to load MNIST (28×28). Falls back to sklearn digits if unavailable."""
    try:
        import gzip, struct, urllib.request
        
        base_url = "http://yann.lecun.com/exdb/mnist/"
        files = {
            "train_img": "train-images-idx3-ubyte.gz",
            "train_lbl": "train-labels-idx1-ubyte.gz",
            "test_img": "t10k-images-idx3-ubyte.gz",
            "test_lbl": "t10k-labels-idx1-ubyte.gz",
        }
        
        os.makedirs("data", exist_ok=True)
        for name, fn in files.items():
            path = os.path.join("data", fn)
            if not os.path.exists(path):
                print(f"    Downloading {fn}...")
                urllib.request.urlretrieve(base_url + fn, path)
        
        def read_imgs(p):
            with gzip.open(p) as f:
                _, n, r, c = struct.unpack(">IIII", f.read(16))
                return np.frombuffer(f.read(), np.uint8).reshape(n, 1, r, c).astype(np.float64) / 255.0
        
        def read_lbls(p):
            with gzip.open(p) as f:
                struct.unpack(">II", f.read(8))
                return np.frombuffer(f.read(), np.uint8)
        
        X_train = read_imgs(os.path.join("data", files["train_img"]))
        y_raw_train = read_lbls(os.path.join("data", files["train_lbl"]))
        X_test = read_imgs(os.path.join("data", files["test_img"]))
        y_raw_test = read_lbls(os.path.join("data", files["test_lbl"]))
        
        # Use subset for speed (full MNIST is slow with NumPy-only CNN)
        n_train = min(5000, len(X_train))
        n_test = min(1000, len(X_test))
        
        def one_hot(labels):
            y = np.zeros((len(labels), 10))
            y[np.arange(len(labels)), labels] = 1
            return y
        
        return (X_train[:n_train], one_hot(y_raw_train[:n_train]),
                X_test[:n_test], one_hot(y_raw_test[:n_test]), y_raw_test[:n_test])
    
    except Exception as e:
        print(f"    MNIST not available ({e}), using sklearn digits instead.")
        return load_digits_sklearn()


def train_and_evaluate():
    """Main training pipeline."""
    np.random.seed(42)
    
    print("=" * 60)
    print("  CNN from Scratch — Handwritten Digit Classification")
    print("=" * 60)
    
    # ── Load Data ─────────────────────────────────────────────────
    print("\n  Loading data...")
    X_train, y_train, X_test, y_test, y_test_raw = load_digits_sklearn()
    
    N, C, H, W = X_train.shape
    print(f"  Train: {N} samples | Test: {X_test.shape[0]} samples")
    print(f"  Image size: {C}×{H}×{W}")
    
    # ── Build CNN ─────────────────────────────────────────────────
    # For 8×8 input:
    # Conv(1→8, 3×3, pad=1) → 8×8×8 → Pool → 8×4×4
    # Conv(8→16, 3×3, pad=1) → 16×4×4 → Pool → 16×2×2
    # Flatten → 64 → Dense → 10
    
    model = CNN([
        Conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
        ReLU(),
        MaxPool2D(pool_size=2, stride=2),
        Conv2D(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
        ReLU(),
        MaxPool2D(pool_size=2, stride=2),
        Flatten(),
        Dense(in_features=16 * 2 * 2, out_features=32),
        ReLU(),
        Dense(in_features=32, out_features=10),
        Softmax(),
    ])
    
    model.summary()
    
    # ── Train ─────────────────────────────────────────────────────
    print("\n  Training...")
    start = time.time()
    
    history = model.train(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        lr=0.002,
        verbose=True,
    )
    
    elapsed = time.time() - start
    print(f"\n  Training time: {elapsed:.1f}s")
    
    # ── Evaluate ──────────────────────────────────────────────────
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\n  Test Accuracy: {test_acc*100:.1f}%")
    print(f"  Test Loss:     {test_loss:.4f}")
    
    return model, history, X_test, y_test, y_test_raw


def plot_results(model, history, X_test, y_test, y_test_raw):
    """Generate all plots."""
    os.makedirs("figures", exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss curve
    axes[0].plot(history["train_loss"], color="#2563eb", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss", fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot([a*100 for a in history["train_acc"]], color="#16a34a", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training Accuracy", fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    
    # Sample predictions
    predictions = model.predict(X_test[:25])
    axes[2].axis("off")
    axes[2].set_title("Sample Predictions", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig("figures/training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Sample grid
    fig, axes_s = plt.subplots(5, 5, figsize=(8, 8))
    for idx in range(25):
        ax = axes_s[idx // 5, idx % 5]
        img = X_test[idx, 0]
        pred = predictions[idx]
        true = y_test_raw[idx]
        ax.imshow(img, cmap="gray")
        color = "#16a34a" if pred == true else "#dc2626"
        ax.set_title(f"P:{pred} T:{true}", color=color, fontsize=10, fontweight="bold")
        ax.axis("off")
    
    fig.suptitle("CNN Predictions (green=correct, red=wrong)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print("  ✅ figures/training_curves.png")
    print("  ✅ figures/predictions.png")


def plot_filters(model, save_path=None):
    """Visualize the learned convolutional filters."""
    # First conv layer filters
    conv1 = model.layers[0]
    filters = conv1.W  # (out_channels, in_channels, kH, kW)
    n_filters = filters.shape[0]
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.ravel()):
        if i < n_filters:
            f = filters[i, 0]  # First input channel
            ax.imshow(f, cmap="RdBu_r", vmin=-f.max(), vmax=f.max())
            ax.set_title(f"Filter {i+1}", fontsize=11, fontweight="bold")
        ax.axis("off")
    
    fig.suptitle("Learned Conv1 Filters (3×3)\nCNN discovers edge detectors automatically!",
                fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {save_path}")


def plot_im2col_explanation(save_path=None):
    """Visual explanation of how im2col works."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1) Original image with sliding window
    ax = axes[0]
    img = np.random.rand(5, 5)
    ax.imshow(img, cmap="Blues", vmin=0, vmax=1)
    
    # Highlight 3×3 patches
    import matplotlib.patches as patches
    colors_patch = ["#dc2626", "#16a34a", "#2563eb"]
    for idx, (r, c) in enumerate([(0, 0), (0, 1), (1, 0)]):
        rect = patches.Rectangle((c-0.5, r-0.5), 3, 3, linewidth=3,
                                  edgecolor=colors_patch[idx % 3], facecolor='none',
                                  linestyle=['solid', 'dashed', 'dotted'][idx % 3])
        ax.add_patch(rect)
    
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{img[i,j]:.1f}", ha="center", va="center", fontsize=9)
    
    ax.set_title("1. Input Image (5×5)\nSliding 3×3 window", fontsize=13, fontweight="bold")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    
    # 2) im2col result
    ax = axes[1]
    # Each patch flattened into a column
    col_data = np.zeros((9, 9))
    for col_idx, (r, c) in enumerate([(r, c) for r in range(3) for c in range(3)]):
        patch = img[r:r+3, c:c+3].ravel()
        col_data[:, col_idx] = patch
    
    ax.imshow(col_data, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_title("2. im2col Result\nEach column = one patch", fontsize=13, fontweight="bold")
    ax.set_xlabel("Patch index")
    ax.set_ylabel("Pixel index in patch")
    
    # 3) Matrix multiplication
    ax = axes[2]
    ax.text(0.5, 0.7, "Convolution =\nW @ im2col(X) + b",
            fontsize=20, ha="center", va="center", fontweight="bold",
            transform=ax.transAxes, color="#2563eb")
    ax.text(0.5, 0.35, "(M, C·k·k) @ (C·k·k, N·H'·W')\n\n= GEMM (BLAS Level 3)\n= FAST!",
            fontsize=14, ha="center", va="center", transform=ax.transAxes,
            fontfamily="monospace")
    ax.text(0.5, 0.08, "Lecture 6: 'Can we use GEMM for convolution? YES!'",
            fontsize=10, ha="center", va="center", transform=ax.transAxes,
            style="italic", color="gray")
    ax.axis("off")
    ax.set_title("3. GEMM Convolution\nNaive: 6 loops → im2col: 1 matmul", fontsize=13, fontweight="bold")
    
    fig.suptitle("How im2col Works — Converting Convolution to Matrix Multiplication\n[Lecture 6, Slides 24-30]",
                fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {save_path}")


if __name__ == "__main__":
    model, history, X_test, y_test, y_test_raw = train_and_evaluate()
    
    print("\n  Generating plots...")
    plot_results(model, history, X_test, y_test, y_test_raw)
    plot_filters(model, save_path="figures/filters.png")
    plot_im2col_explanation(save_path="figures/im2col_explained.png")
    
    print("\n✨ Done!")
