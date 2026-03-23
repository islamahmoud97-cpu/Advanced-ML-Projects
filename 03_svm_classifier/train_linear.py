"""
Demo 1: Linear SVM — Margin & Support Vectors
=================================================
Based on: Zemke, AML Lecture 1, Slides 8-14

Key concepts demonstrated:
  1. Separating hyperplane: wᵀx + b = 0
  2. Margin = 2/||w||₂ (maximize this!)
  3. Support vectors: points closest to the boundary
  4. Optimization: min ½||w||² s.t. yᵢ(wᵀxᵢ + b) ≥ 1

Run: python train_linear.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from svm import SVM
from datasets import make_linear


def plot_linear_svm(X, y, svm, title="Linear SVM", save_path=None):
    """Plot decision boundary, margin, and support vectors."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Decision boundary mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    Z = svm.decision_function(grid).reshape(xx.shape)
    
    # Decision regions
    ax.contourf(xx, yy, Z, levels=[-1e10, 0, 1e10], colors=["#dbeafe", "#fecaca"], alpha=0.5)
    
    # Margin boundaries: f(x) = -1 and f(x) = +1
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=["#2563eb", "#1e293b", "#dc2626"],
               linewidths=[1.5, 2.5, 1.5], linestyles=["--", "-", "--"])
    
    # Margin band
    ax.contourf(xx, yy, Z, levels=[-1, 1], colors=["#fef3c7"], alpha=0.3)
    
    # Data points
    mask_pos = y == 1
    mask_neg = y == -1
    ax.scatter(X[mask_pos, 0], X[mask_pos, 1], c="#2563eb", s=60, edgecolors="white",
               linewidths=1, label="Class +1", zorder=3)
    ax.scatter(X[mask_neg, 0], X[mask_neg, 1], c="#dc2626", s=60, edgecolors="white",
               linewidths=1, label="Class −1", zorder=3)
    
    # Highlight support vectors
    sv = svm.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=200, facecolors="none", edgecolors="#f59e0b",
               linewidths=2.5, label=f"Support Vectors ({svm.n_support_})", zorder=4)
    
    # Margin annotation
    margin = svm.get_margin()
    if margin is not None:
        ax.set_title(f"{title}\nMargin = 2/||w|| = {margin:.3f}  |  "
                     f"Support Vectors: {svm.n_support_}",
                     fontsize=14, fontweight="bold")
    else:
        ax.set_title(f"{title}  |  Support Vectors: {svm.n_support_}",
                     fontsize=14, fontweight="bold")
    
    ax.set_xlabel("x₁", fontsize=12)
    ax.set_ylabel("x₂", fontsize=12)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path} saved")
    plt.close()


def plot_c_comparison(X, y, save_path=None):
    """Show effect of C parameter on margin and support vectors."""
    C_values = [0.01, 0.1, 1.0, 100.0]
    
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    
    for ax, C in zip(axes, C_values):
        svm = SVM(kernel="linear", C=C, max_iter=200)
        svm.fit(X, y, verbose=False)
        
        h = 0.05
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, levels=[-1e10, 0, 1e10], colors=["#dbeafe", "#fecaca"], alpha=0.5)
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=["#2563eb", "#1e293b", "#dc2626"],
                   linewidths=[1, 2, 1], linestyles=["--", "-", "--"])
        ax.contourf(xx, yy, Z, levels=[-1, 1], colors=["#fef3c7"], alpha=0.3)
        
        mask_pos = y == 1
        mask_neg = y == -1
        ax.scatter(X[mask_pos, 0], X[mask_pos, 1], c="#2563eb", s=30, edgecolors="white", linewidths=0.5)
        ax.scatter(X[mask_neg, 0], X[mask_neg, 1], c="#dc2626", s=30, edgecolors="white", linewidths=0.5)
        
        sv = svm.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=120, facecolors="none", edgecolors="#f59e0b", linewidths=2)
        
        margin = svm.get_margin()
        margin_str = f"Margin: {margin:.2f}" if margin else "—"
        ax.set_title(f"C = {C}\n{margin_str}  |  SVs: {svm.n_support_}",
                     fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2)
    
    fig.suptitle("Effect of C Parameter on SVM Decision Boundary\n"
                 "Small C → wide margin (soft) | Large C → narrow margin (hard)",
                 fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path} saved")
    plt.close()


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    
    print("=" * 60)
    print("  Linear SVM — Margin & Support Vectors")
    print("=" * 60)
    
    # Generate data
    X, y = make_linear(n_samples=200, noise=0.4)
    print(f"\n  Dataset: {len(y)} samples, {(y==1).sum()} positive, {(y==-1).sum()} negative")
    
    # Train
    print("\n  Training Linear SVM (C=1.0)...")
    svm = SVM(kernel="linear", C=1.0, max_iter=200)
    svm.fit(X, y)
    
    acc = svm.score(X, y)
    print(f"  Accuracy: {acc*100:.1f}%")
    
    # Plot
    plot_linear_svm(X, y, svm, title="Linear SVM — Maximum Margin Classifier",
                    save_path="figures/linear_svm.png")
    
    # C parameter comparison
    print("\n  Comparing C parameter values...")
    plot_c_comparison(X, y, save_path="figures/c_parameter.png")
    
    print("\n✨ Done!")
