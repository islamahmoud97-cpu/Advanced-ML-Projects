"""
Training Benchmark — Initialization Impact on Real Training
==============================================================
Train the SAME network architecture with different initializations
and compare convergence speed and final accuracy.

Dataset: Two Moons (non-linear binary classification)
Architecture: [2, 64, 64, 64, 64, 1] — 4 hidden layers (deep enough to see effects)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from initializations import (
    zeros, random_normal, xavier_normal, he_normal,
    lecun_normal, orthogonal,
)


def make_moons(n=1000, noise=0.15, seed=42):
    np.random.seed(seed)
    n_half = n // 2
    t1 = np.linspace(0, np.pi, n_half)
    X1 = np.column_stack([np.cos(t1), np.sin(t1)]) + np.random.randn(n_half, 2) * noise
    t2 = np.linspace(0, np.pi, n_half)
    X2 = np.column_stack([1-np.cos(t2), -np.sin(t2)+0.5]) + np.random.randn(n_half, 2) * noise
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n_half), np.zeros(n_half)])
    return X, y


class BenchmarkNet:
    """Simple FNN for benchmarking initialization methods."""
    
    def __init__(self, sizes, init_fn, activation="relu"):
        self.sizes = sizes
        self.W = []
        self.b = []
        
        if activation == "relu":
            self.act = lambda z: np.maximum(0, z)
            self.act_d = lambda z: (z > 0).astype(float)
        else:
            self.act = lambda z: np.tanh(z)
            self.act_d = lambda z: 1 - np.tanh(z)**2
        
        for i in range(len(sizes) - 1):
            shape = (sizes[i+1], sizes[i])
            W = init_fn(shape)
            self.W.append(W)
            self.b.append(np.zeros((sizes[i+1], 1)))
    
    def forward(self, X):
        self.a_list = [X.T]
        self.z_list = []
        a = X.T
        for i in range(len(self.W)):
            z = self.W[i] @ a + self.b[i]
            self.z_list.append(z)
            if i == len(self.W) - 1:
                a = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
            else:
                a = self.act(z)
            self.a_list.append(a)
        return a
    
    def train_epoch(self, X, y, lr=0.01):
        N = len(y)
        out = self.forward(X)
        y_row = y.reshape(1, -1)
        
        eps = 1e-12
        loss = -(1/N) * np.sum(y_row * np.log(out + eps) + (1-y_row) * np.log(1-out + eps))
        
        delta = (out - y_row) / N
        grad_norms = []
        
        for i in range(len(self.W) - 1, -1, -1):
            dW = delta @ self.a_list[i].T
            db = np.sum(delta, axis=1, keepdims=True)
            grad_norms.append(np.linalg.norm(dW))
            
            if i > 0:
                delta = self.W[i].T @ delta * self.act_d(self.z_list[i-1])
            
            self.W[i] -= lr * dW
            self.b[i] -= lr * db
        
        pred = (out > 0.5).astype(float)
        acc = np.mean(pred == y_row)
        
        return loss, acc, list(reversed(grad_norms))


def run_benchmark(save_path=None):
    """Train with each init and compare."""
    X, y = make_moons(1000, noise=0.15)
    
    arch = [2, 64, 64, 64, 64, 1]  # 4 hidden layers
    epochs = 300
    
    inits = [
        ("Zeros", zeros, "#6b7280"),
        ("Random (σ=0.01)", lambda s: random_normal(s, std=0.01), "#a3a3a3"),
        ("LeCun", lecun_normal, "#ea580c"),
        ("Xavier", xavier_normal, "#2563eb"),
        ("He", he_normal, "#dc2626"),
        ("Orthogonal", orthogonal, "#16a34a"),
    ]
    
    results = {}
    
    for name, init_fn, color in inits:
        np.random.seed(42)
        net = BenchmarkNet(arch, init_fn, activation="relu")
        losses, accs, all_grads = [], [], []
        
        for epoch in range(epochs):
            loss, acc, grads = net.train_epoch(X, y, lr=0.1)
            losses.append(loss)
            accs.append(acc)
            all_grads.append(grads)
        
        results[name] = {
            "losses": losses, "accs": accs, "grads": all_grads,
            "color": color, "final_acc": accs[-1], "final_loss": losses[-1],
        }
        print(f"  {name:20s} → Acc: {accs[-1]*100:.1f}%  |  Loss: {losses[-1]:.4f}")
    
    # ── Plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Loss curves
    ax = axes[0, 0]
    for name, r in results.items():
        ax.plot(r["losses"], color=r["color"], linewidth=2, alpha=0.8, label=name)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 1.5)
    
    # Accuracy curves
    ax = axes[0, 1]
    for name, r in results.items():
        ax.plot([a*100 for a in r["accs"]], color=r["color"], linewidth=2, alpha=0.8, label=name)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Training Accuracy", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    
    # Gradient norms (first hidden layer — most affected)
    ax = axes[1, 0]
    for name, r in results.items():
        layer0_grads = [g[0] for g in r["grads"]]
        ax.semilogy(layer0_grads, color=r["color"], linewidth=1.5, alpha=0.7, label=name)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Gradient Norm — Layer 1 (log)", fontsize=12)
    ax.set_title("Gradient Health — First Layer", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, which="both")
    
    # Final accuracy bar chart
    ax = axes[1, 1]
    names = list(results.keys())
    accs_final = [results[n]["final_acc"] * 100 for n in names]
    bar_colors = [results[n]["color"] for n in names]
    bars = ax.barh(names, accs_final, color=bar_colors, alpha=0.8, edgecolor="white", height=0.6)
    for bar, acc in zip(bars, accs_final):
        ax.text(max(bar.get_width() - 8, 2), bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%", va="center", ha="right" if acc > 15 else "left",
                fontsize=11, fontweight="bold", color="white" if acc > 15 else "black")
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title("Final Accuracy After 300 Epochs", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.grid(True, alpha=0.2, axis="x")
    
    fig.suptitle(
        "Training Benchmark: [2, 64, 64, 64, 64, 1] with ReLU — Two Moons Dataset\n"
        "[Lecture 3: Wrong initialization = no learning]",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  ✅ {save_path}")
    plt.close()
    
    return results


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    
    print("=" * 60)
    print("  Training Benchmark — Weight Initialization")
    print("=" * 60)
    print(f"\n  Architecture: [2, 64, 64, 64, 64, 1]")
    print(f"  Activation: ReLU")
    print(f"  Dataset: Two Moons (1000 samples)")
    print(f"  Epochs: 300\n")
    
    run_benchmark(save_path="figures/training_benchmark.png")
    
    print("\n✨ Done!")
