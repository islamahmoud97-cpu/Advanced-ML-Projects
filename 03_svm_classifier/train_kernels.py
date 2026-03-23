"""
Demo 2: The Kernel Trick — Non-Linear Classification
=======================================================
Based on: Zemke, AML Lecture 1

Key insight from Lecture 1:
  XOR is NOT linearly separable → no hyperplane wᵀx + b = 0 exists.

Solution: The Kernel Trick
  Instead of finding a linear boundary in the original space,
  map data to a HIGHER-dimensional space where it IS separable.

  K(x, z) = φ(x)ᵀ · φ(z)

  The kernel computes the dot product in the high-dimensional
  space WITHOUT explicitly computing φ(x).

This script demonstrates:
  1. Linear SVM fails on XOR/circles/moons
  2. RBF kernel solves all of them
  3. Effect of γ (gamma) on decision boundary
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from svm import SVM
from datasets import make_xor, make_circles, make_moons, make_spiral, make_linear


def plot_kernel_comparison(save_path=None):
    """Compare Linear vs RBF vs Polynomial on 4 datasets."""
    datasets = [
        ("Linear Data", make_linear(200, noise=0.4)),
        ("XOR Pattern", make_xor(200, noise=0.3)),
        ("Circles", make_circles(200, noise=0.12)),
        ("Moons", make_moons(200, noise=0.15)),
    ]

    kernels = [
        ("Linear", {"kernel": "linear", "C": 1.0}),
        ("Polynomial (d=3)", {"kernel": "polynomial", "C": 1.0, "degree": 3}),
        ("RBF (γ=1)", {"kernel": "rbf", "C": 1.0, "gamma": 1.0}),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(18, 22))

    for row, (data_name, (X, y)) in enumerate(datasets):
        for col, (kern_name, params) in enumerate(kernels):
            ax = axes[row, col]

            svm = SVM(**params, max_iter=300)
            svm.fit(X, y, verbose=False)
            acc = svm.score(X, y)

            # Decision boundary
            h = 0.05
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            ax.contourf(xx, yy, Z, levels=np.linspace(-3, 3, 30), cmap="RdBu_r", alpha=0.5)
            ax.contour(xx, yy, Z, levels=[0], colors=["black"], linewidths=2)

            mask_pos = y == 1
            mask_neg = y == -1
            ax.scatter(X[mask_pos, 0], X[mask_pos, 1], c="#2563eb", s=20, edgecolors="white", linewidths=0.5)
            ax.scatter(X[mask_neg, 0], X[mask_neg, 1], c="#dc2626", s=20, edgecolors="white", linewidths=0.5)

            sv = svm.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], s=80, facecolors="none", edgecolors="#f59e0b", linewidths=1.5)

            # Colorize title based on accuracy
            color = "#16a34a" if acc > 0.90 else "#ea580c" if acc > 0.75 else "#dc2626"
            ax.set_title(f"{kern_name}\nAcc: {acc*100:.0f}%  |  SVs: {svm.n_support_}",
                         fontsize=11, fontweight="bold", color=color)
            ax.grid(True, alpha=0.15)

            if col == 0:
                ax.set_ylabel(data_name, fontsize=13, fontweight="bold", rotation=90, labelpad=15)

    fig.suptitle(
        "The Kernel Trick: Linear vs Polynomial vs RBF\n"
        "[Lecture 1: Linear SVM fails on non-linear data → Kernel maps to higher dimension]",
        fontsize=16, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path} saved")
    plt.close()


def plot_gamma_effect(save_path=None):
    """Show effect of γ on RBF kernel decision boundary."""
    X, y = make_moons(300, noise=0.15)
    gammas = [0.1, 0.5, 1.0, 5.0, 20.0, 100.0]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for ax, gamma in zip(axes.ravel(), gammas):
        svm = SVM(kernel="rbf", C=10.0, gamma=gamma, max_iter=300)
        svm.fit(X, y, verbose=False)
        acc = svm.score(X, y)

        h = 0.03
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        ax.contourf(xx, yy, Z, levels=np.linspace(-3, 3, 30), cmap="RdBu_r", alpha=0.5)
        ax.contour(xx, yy, Z, levels=[0], colors=["black"], linewidths=2)

        mask_pos = y == 1
        mask_neg = y == -1
        ax.scatter(X[mask_pos, 0], X[mask_pos, 1], c="#2563eb", s=15, edgecolors="white", linewidths=0.3)
        ax.scatter(X[mask_neg, 0], X[mask_neg, 1], c="#dc2626", s=15, edgecolors="white", linewidths=0.3)

        label = ""
        if gamma <= 0.1:
            label = "(underfitting)"
        elif gamma >= 20:
            label = "(overfitting)"
        else:
            label = "(good fit)"

        ax.set_title(f"γ = {gamma}  {label}\nAcc: {acc*100:.0f}%  |  SVs: {svm.n_support_}",
                     fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.15)

    fig.suptitle(
        "Effect of γ (Gamma) on RBF Kernel — Two Moons Dataset\n"
        "Small γ → smooth boundary (underfit)  |  Large γ → complex boundary (overfit)",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path} saved")
    plt.close()


def plot_kernel_trick_intuition(save_path=None):
    """
    Visual intuition: How the kernel trick transforms 2D data to 3D.
    
    For circles data with polynomial kernel K(x,z) = (xᵀz)²:
    φ(x) = [x₁², x₂², √2·x₁·x₂]ᵀ
    
    In this higher-dimensional space, the data IS linearly separable!
    """
    X, y = make_circles(300, noise=0.08, seed=42)

    fig = plt.figure(figsize=(18, 6))

    # Original 2D space
    ax1 = fig.add_subplot(131)
    mask_pos = y == 1
    mask_neg = y == -1
    ax1.scatter(X[mask_pos, 0], X[mask_pos, 1], c="#2563eb", s=30, edgecolors="white", linewidths=0.5, label="+1")
    ax1.scatter(X[mask_neg, 0], X[mask_neg, 1], c="#dc2626", s=30, edgecolors="white", linewidths=0.5, label="−1")
    ax1.set_title("Original 2D Space\n(NOT linearly separable)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("x₁")
    ax1.set_ylabel("x₂")
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    ax1.set_aspect("equal")

    # Transformed 3D space: φ(x) = [x₁², x₂², √2·x₁·x₂]
    ax2 = fig.add_subplot(132, projection="3d")
    phi_X = np.column_stack([X[:, 0]**2, X[:, 1]**2, np.sqrt(2) * X[:, 0] * X[:, 1]])
    ax2.scatter(phi_X[mask_pos, 0], phi_X[mask_pos, 1], phi_X[mask_pos, 2],
                c="#2563eb", s=20, alpha=0.7, label="+1")
    ax2.scatter(phi_X[mask_neg, 0], phi_X[mask_neg, 1], phi_X[mask_neg, 2],
                c="#dc2626", s=20, alpha=0.7, label="−1")

    # Separating hyperplane in 3D
    xx3d = np.linspace(0, phi_X[:, 0].max(), 20)
    yy3d = np.linspace(0, phi_X[:, 1].max(), 20)
    XX3d, YY3d = np.meshgrid(xx3d, yy3d)
    # Plane: x₁² + x₂² = threshold
    threshold = 0.5
    ZZ3d = np.full_like(XX3d, 0.0)
    ax2.plot_surface(XX3d, YY3d, ZZ3d + threshold, alpha=0.15, color="#f59e0b")

    ax2.set_title("After φ: 3D Feature Space\n(NOW linearly separable!)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("x₁²")
    ax2.set_ylabel("x₂²")
    ax2.set_zlabel("√2·x₁x₂")
    ax2.legend()

    # RBF SVM back in 2D
    ax3 = fig.add_subplot(133)
    svm = SVM(kernel="rbf", C=10.0, gamma=1.0, max_iter=300)
    svm.fit(X, y, verbose=False)

    h = 0.03
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax3.contourf(xx, yy, Z, levels=np.linspace(-3, 3, 30), cmap="RdBu_r", alpha=0.5)
    ax3.contour(xx, yy, Z, levels=[0], colors=["black"], linewidths=2)
    ax3.scatter(X[mask_pos, 0], X[mask_pos, 1], c="#2563eb", s=30, edgecolors="white", linewidths=0.5)
    ax3.scatter(X[mask_neg, 0], X[mask_neg, 1], c="#dc2626", s=30, edgecolors="white", linewidths=0.5)
    sv = svm.support_vectors_
    ax3.scatter(sv[:, 0], sv[:, 1], s=100, facecolors="none", edgecolors="#f59e0b", linewidths=2)
    acc = svm.score(X, y)
    ax3.set_title(f"RBF SVM Result in 2D\nAcc: {acc*100:.0f}%  |  SVs: {svm.n_support_}",
                  fontsize=13, fontweight="bold")
    ax3.set_xlabel("x₁")
    ax3.set_ylabel("x₂")
    ax3.grid(True, alpha=0.2)
    ax3.set_aspect("equal")

    fig.suptitle(
        "The Kernel Trick Intuition: φ maps data to a space where it becomes linearly separable",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path} saved")
    plt.close()


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    
    print("=" * 60)
    print("  The Kernel Trick — Non-Linear SVM")
    print("=" * 60)
    
    print("\n📊 1/3: Kernel comparison (Linear vs Poly vs RBF)...")
    plot_kernel_comparison(save_path="figures/kernel_comparison.png")
    
    print("\n📊 2/3: Gamma effect on RBF kernel...")
    plot_gamma_effect(save_path="figures/gamma_effect.png")
    
    print("\n📊 3/3: Kernel trick 3D intuition...")
    plot_kernel_trick_intuition(save_path="figures/kernel_intuition.png")
    
    print("\n✨ Done!")
