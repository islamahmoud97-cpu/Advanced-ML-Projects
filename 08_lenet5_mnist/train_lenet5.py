"""
LeNet-5 — The CNN That Started It All
========================================
Based on: Zemke, AML Lecture 5, Slides 3-8

"LeNet-5 by LeCun et al. from 1998 was used on industrial scale
 for deciphering handwritten digits on checks and can be considered
 the first successful CNN." — Lecture 5

Original Architecture (LeCun et al., 1998):
  Input (1, 32, 32)
  → Conv(6, 5×5) → TanH → AvgPool(2×2)        → (6, 14, 14)
  → Conv(16, 5×5) → TanH → AvgPool(2×2)       → (16, 5, 5)
  → Flatten → Dense(120) → TanH
  → Dense(84) → TanH
  → Dense(10) → Softmax

Modern Version (used here):
  - ReLU instead of TanH (Lecture 1: better gradient flow)
  - MaxPool instead of AvgPool (Lecture 4: stronger features)
  - Adam instead of SGD (Lecture 3: faster convergence)
  - Input 28×28 (MNIST) instead of 32×32

Run: python train_lenet5.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time


def build_lenet5_original():
    """
    Original LeNet-5 (1998) — as close to the paper as possible.
    Uses TanH and AvgPool to match the original.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),

        # Pad to 32×32 (original LeNet-5 used 32×32 input)
        layers.ZeroPadding2D(padding=2),

        # C1: Conv(6, 5×5) → TanH
        layers.Conv2D(6, kernel_size=5, activation='tanh', name='C1_Conv'),
        # S2: AvgPool(2×2)
        layers.AveragePooling2D(pool_size=2, name='S2_AvgPool'),

        # C3: Conv(16, 5×5) → TanH
        layers.Conv2D(16, kernel_size=5, activation='tanh', name='C3_Conv'),
        # S4: AvgPool(2×2)
        layers.AveragePooling2D(pool_size=2, name='S4_AvgPool'),

        # Flatten
        layers.Flatten(name='Flatten'),

        # C5: Dense(120) → TanH
        layers.Dense(120, activation='tanh', name='C5_Dense'),
        # F6: Dense(84) → TanH
        layers.Dense(84, activation='tanh', name='F6_Dense'),
        # Output: Dense(10) → Softmax
        layers.Dense(10, activation='softmax', name='Output'),
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def build_lenet5_modern():
    """
    Modern LeNet-5 — with improvements from Lectures 1-3:
    - ReLU (better gradient flow — Lecture 1)
    - MaxPool (stronger features — Lecture 4)
    - Batch Normalization (Lecture 3)
    - Dropout (Lecture 3)
    - He initialization (Lecture 3)
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),

        # Conv Block 1
        layers.Conv2D(6, kernel_size=5, padding='same',
                      kernel_initializer='he_normal', name='Conv1'),
        layers.BatchNormalization(name='BN1'),
        layers.Activation('relu', name='ReLU1'),
        layers.MaxPooling2D(pool_size=2, name='MaxPool1'),

        # Conv Block 2
        layers.Conv2D(16, kernel_size=5, padding='valid',
                      kernel_initializer='he_normal', name='Conv2'),
        layers.BatchNormalization(name='BN2'),
        layers.Activation('relu', name='ReLU2'),
        layers.MaxPooling2D(pool_size=2, name='MaxPool2'),

        # Dense layers
        layers.Flatten(name='Flatten'),
        layers.Dense(120, kernel_initializer='he_normal', name='Dense1'),
        layers.Activation('relu', name='ReLU3'),
        layers.Dropout(0.5, name='Dropout1'),

        layers.Dense(84, kernel_initializer='he_normal', name='Dense2'),
        layers.Activation('relu', name='ReLU4'),
        layers.Dropout(0.3, name='Dropout2'),

        layers.Dense(10, activation='softmax', name='Output'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def load_mnist():
    """Load and preprocess MNIST."""
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshape to (N, 28, 28, 1) and normalize to [0, 1]
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return X_train, y_train, X_test, y_test


def train_and_compare():
    """Train both original and modern LeNet-5, compare results."""

    print("=" * 60)
    print("  LeNet-5 on MNIST — Original vs Modern")
    print("=" * 60)

    # Load data
    print("\n  Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    results = {}

    # ── Original LeNet-5 ──────────────────────────────────────────
    print("\n" + "=" * 40)
    print("  1/2: Original LeNet-5 (1998)")
    print("=" * 40)
    model_orig = build_lenet5_original()
    model_orig.summary()

    start = time.time()
    hist_orig = model_orig.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=15,
        batch_size=128,
        verbose=1,
    )
    time_orig = time.time() - start

    loss_orig, acc_orig = model_orig.evaluate(X_test, y_test, verbose=0)
    results["original"] = {
        "history": hist_orig.history,
        "test_acc": acc_orig,
        "test_loss": loss_orig,
        "time": time_orig,
        "model": model_orig,
    }
    print(f"\n  Original → Test Accuracy: {acc_orig*100:.2f}%  ({time_orig:.1f}s)")

    # ── Modern LeNet-5 ────────────────────────────────────────────
    print("\n" + "=" * 40)
    print("  2/2: Modern LeNet-5 (BN + Dropout + ReLU)")
    print("=" * 40)
    model_mod = build_lenet5_modern()
    model_mod.summary()

    start = time.time()
    hist_mod = model_mod.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=15,
        batch_size=128,
        verbose=1,
    )
    time_mod = time.time() - start

    loss_mod, acc_mod = model_mod.evaluate(X_test, y_test, verbose=0)
    results["modern"] = {
        "history": hist_mod.history,
        "test_acc": acc_mod,
        "test_loss": loss_mod,
        "time": time_mod,
        "model": model_mod,
    }
    print(f"\n  Modern → Test Accuracy: {acc_mod*100:.2f}%  ({time_mod:.1f}s)")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COMPARISON")
    print("=" * 60)
    print(f"  {'':20s} {'Original':>12s} {'Modern':>12s}")
    print(f"  {'Test Accuracy':20s} {acc_orig*100:>11.2f}% {acc_mod*100:>11.2f}%")
    print(f"  {'Test Loss':20s} {loss_orig:>12.4f} {loss_mod:>12.4f}")
    print(f"  {'Training Time':20s} {time_orig:>11.1f}s {time_mod:>11.1f}s")
    print("=" * 60)

    return results


def plot_results(results, save_dir="figures"):
    """Generate comparison plots."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Training accuracy
    ax = axes[0, 0]
    ax.plot(results["original"]["history"]["accuracy"],
            color="#ea580c", linewidth=2, label="Original (TanH + AvgPool)")
    ax.plot(results["modern"]["history"]["accuracy"],
            color="#2563eb", linewidth=2, label="Modern (ReLU + BN + Dropout)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training Accuracy", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation accuracy
    ax = axes[0, 1]
    ax.plot(results["original"]["history"]["val_accuracy"],
            color="#ea580c", linewidth=2, label="Original")
    ax.plot(results["modern"]["history"]["val_accuracy"],
            color="#2563eb", linewidth=2, label="Modern")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Training loss
    ax = axes[1, 0]
    ax.plot(results["original"]["history"]["loss"],
            color="#ea580c", linewidth=2, label="Original")
    ax.plot(results["modern"]["history"]["loss"],
            color="#2563eb", linewidth=2, label="Modern")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Comparison bar chart
    ax = axes[1, 1]
    names = ["Original\n(1998)", "Modern\n(+BN+DO+ReLU)"]
    accs = [results["original"]["test_acc"] * 100, results["modern"]["test_acc"] * 100]
    colors = ["#ea580c", "#2563eb"]
    bars = ax.bar(names, accs, color=colors, alpha=0.8, edgecolor="white", width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 2,
                f"{acc:.2f}%", ha="center", va="top", fontsize=14,
                fontweight="bold", color="white")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Final Comparison", fontweight="bold")
    ax.set_ylim(95, 100)
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(
        "LeNet-5: Original (1998) vs Modern — MNIST\n"
        "[Lecture 5: First successful CNN on industrial scale]",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {save_dir}/comparison.png")

    # ── Sample predictions ────────────────────────────────────────
    from tensorflow.keras.datasets import mnist as mnist_data
    (_, _), (X_test_raw, y_test_raw) = mnist_data.load_data()
    X_test_input = X_test_raw.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    model = results["modern"]["model"]
    preds = np.argmax(model.predict(X_test_input[:25], verbose=0), axis=1)

    fig, axes_s = plt.subplots(5, 5, figsize=(10, 10))
    for idx in range(25):
        ax = axes_s[idx // 5, idx % 5]
        ax.imshow(X_test_raw[idx], cmap="gray")
        color = "#16a34a" if preds[idx] == y_test_raw[idx] else "#dc2626"
        ax.set_title(f"P:{preds[idx]} T:{y_test_raw[idx]}",
                     color=color, fontsize=11, fontweight="bold")
        ax.axis("off")

    fig.suptitle("LeNet-5 Modern — Sample Predictions",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "predictions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {save_dir}/predictions.png")

    # ── Learned filters ───────────────────────────────────────────
    conv1_weights = model.layers[0].get_weights()[0]  # (5, 5, 1, 6)

    fig, axes_f = plt.subplots(1, 6, figsize=(15, 3))
    for i in range(6):
        axes_f[i].imshow(conv1_weights[:, :, 0, i], cmap="RdBu_r")
        axes_f[i].set_title(f"Filter {i+1}", fontweight="bold")
        axes_f[i].axis("off")

    fig.suptitle("Learned First-Layer Filters — CNN discovers edge detectors!",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "filters.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {save_dir}/filters.png")


if __name__ == "__main__":
    results = train_and_compare()
    print("\n  Generating plots...")
    plot_results(results)
    print("\n✨ Done!")
