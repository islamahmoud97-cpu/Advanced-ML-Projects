"""
Activation Function Benchmark
===============================
Train a simple FNN with different activation functions and compare:
  - Convergence speed
  - Final accuracy
  - Gradient health during training

Uses the Moons dataset (non-linear, 2D) for fast training & clear visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from activations import TRAINABLE_ACTIVATIONS


# ═══════════════════════════════════════════════════════════════════════════
# Minimal FNN (self-contained for this benchmark)
# ═══════════════════════════════════════════════════════════════════════════
class MiniFNN:
    def __init__(self, sizes, act_name="relu", seed=42):
        np.random.seed(seed)
        self.sizes = sizes
        self.act, self.act_d = TRAINABLE_ACTIVATIONS[act_name]
        self.W = []
        self.b = []
        for i in range(len(sizes) - 1):
            self.W.append(np.random.randn(sizes[i+1], sizes[i]) * np.sqrt(2.0 / sizes[i]))
            self.b.append(np.zeros((sizes[i+1], 1)))

    def forward(self, X):
        self.a = [X]
        self.z = []
        for i in range(len(self.W)):
            zi = self.W[i] @ self.a[i] + self.b[i]
            self.z.append(zi)
            if i == len(self.W) - 1:
                # sigmoid output for binary classification
                ai = 1.0 / (1.0 + np.exp(-np.clip(zi, -500, 500)))
            else:
                ai = self.act(zi)
            self.a.append(ai)
        return self.a[-1]

    def train_step(self, X, y, lr=0.01):
        N = X.shape[1]
        out = self.forward(X)

        # BCE loss
        eps = 1e-12
        loss = -(1/N) * np.sum(y * np.log(out + eps) + (1 - y) * np.log(1 - out + eps))

        # Backprop
        delta = (out - y) / N
        grad_norms = []

        for i in range(len(self.W) - 1, -1, -1):
            dW = delta @ self.a[i].T
            db = np.sum(delta, axis=1, keepdims=True)
            grad_norms.append(np.linalg.norm(dW))

            if i > 0:
                delta = self.W[i].T @ delta * self.act_d(self.z[i-1])

            self.W[i] -= lr * dW
            self.b[i] -= lr * db

        return loss, list(reversed(grad_norms))


# ═══════════════════════════════════════════════════════════════════════════
# Dataset: Two Moons
# ═══════════════════════════════════════════════════════════════════════════
def make_moons(n_samples=1000, noise=0.15, seed=42):
    np.random.seed(seed)
    n = n_samples // 2

    # Upper moon
    theta1 = np.linspace(0, np.pi, n)
    x1 = np.cos(theta1) + np.random.randn(n) * noise
    y1 = np.sin(theta1) + np.random.randn(n) * noise

    # Lower moon
    theta2 = np.linspace(0, np.pi, n)
    x2 = 1 - np.cos(theta2) + np.random.randn(n) * noise
    y2 = -np.sin(theta2) + 0.5 + np.random.randn(n) * noise

    X = np.vstack([np.hstack([x1, x2]), np.hstack([y1, y2])])
    y = np.hstack([np.zeros(n), np.ones(n)]).reshape(1, -1)
    return X, y


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════════════
def run_benchmark():
    X, y = make_moons(n_samples=1000, noise=0.15)

    activations_to_test = ["sigmoid", "tanh", "relu", "leaky_relu", "elu", "swish"]
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]

    epochs = 300
    arch = [2, 32, 16, 1]  # 3 layers deep to see gradient effects

    results = {}

    for act_name, color in zip(activations_to_test, colors):
        net = MiniFNN(arch, act_name=act_name, seed=42)
        losses = []
        all_grad_norms = []

        for epoch in range(epochs):
            loss, grads = net.train_step(X, y, lr=0.05)
            losses.append(loss)
            all_grad_norms.append(grads)

        # Final accuracy
        pred = (net.forward(X) > 0.5).astype(float)
        acc = np.mean(pred == y)

        results[act_name] = {
            "losses": losses,
            "grad_norms": all_grad_norms,
            "accuracy": acc,
            "color": color,
            "net": net,
        }
        print(f"  {act_name:12s}  →  Acc: {acc*100:.1f}%  |  Final Loss: {losses[-1]:.4f}")

    return X, y, results


def plot_benchmark(X, y, results):
    os.makedirs("figures", exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # ── Row 1: Loss, Gradient Health, Accuracy Bar ────────────────────
    # Loss curves
    ax = axes[0, 0]
    for name, r in results.items():
        ax.plot(r["losses"], color=r["color"], linewidth=2, label=name, alpha=0.85)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Convergence Speed", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    # Gradient norms (first layer — most affected by vanishing gradient)
    ax = axes[0, 1]
    for name, r in results.items():
        layer0_grads = [g[0] for g in r["grad_norms"]]
        ax.plot(layer0_grads, color=r["color"], linewidth=1.5, label=name, alpha=0.7)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Gradient Norm (Layer 1)", fontsize=12)
    ax.set_title("Gradient Health — First Layer", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_yscale("log")

    # Accuracy bar chart
    ax = axes[0, 2]
    names = list(results.keys())
    accs = [results[n]["accuracy"] * 100 for n in names]
    bar_colors = [results[n]["color"] for n in names]
    bars = ax.barh(names, accs, color=bar_colors, alpha=0.8, edgecolor="white", height=0.6)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() - 5, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%", va="center", ha="right", fontsize=11,
                fontweight="bold", color="white")
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title("Final Accuracy", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.grid(True, alpha=0.2, axis="x")

    # ── Row 2: Decision boundaries ───────────────────────────────────
    xx, yy = np.meshgrid(np.linspace(X[0].min()-0.5, X[0].max()+0.5, 200),
                         np.linspace(X[1].min()-0.5, X[1].max()+0.5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()].T

    for idx, (name, r) in enumerate(list(results.items())[:3]):
        ax = axes[1, idx]
        Z = r["net"].forward(grid)
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 20),
                    cmap="RdBu_r", alpha=0.6)
        ax.contour(xx, yy, Z, levels=[0.5], colors=["black"], linewidths=2)

        mask0 = y.ravel() == 0
        mask1 = y.ravel() == 1
        ax.scatter(X[0, mask0], X[1, mask0], c="#2563eb", s=8, alpha=0.6)
        ax.scatter(X[0, mask1], X[1, mask1], c="#dc2626", s=8, alpha=0.6)
        ax.set_title(f"Decision Boundary: {name}", fontsize=13, fontweight="bold")
        ax.set_aspect("equal")

    fig.suptitle(
        "Activation Function Benchmark — Two Moons Dataset\n"
        "[Lectures 1-3: Activation Functions, Backpropagation, Optimization]",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig("figures/benchmark.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n✅ figures/benchmark.png saved")


if __name__ == "__main__":
    print("=" * 60)
    print("  Activation Function Benchmark")
    print("=" * 60)
    print(f"\n  Architecture: [2, 32, 16, 1]")
    print(f"  Dataset: Two Moons (1000 samples)")
    print(f"  Epochs: 300\n")

    X, y, results = run_benchmark()
    plot_benchmark(X, y, results)
    print("\nDone!")
