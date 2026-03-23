"""
Visualization Utilities
========================
Generate publication-quality plots for the README.

1. Activation functions comparison
2. Weight initialization distributions
3. Optimizer convergence comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from activations import (
    sigmoid, sigmoid_derivative,
    tanh, tanh_derivative,
    relu, relu_derivative,
    leaky_relu, leaky_relu_derivative,
    elu, elu_derivative,
    swish, swish_derivative,
    softplus, softplus_derivative,
)


def plot_activations():
    """Plot all activation functions and their derivatives side by side."""
    x = np.linspace(-4, 4, 500)

    funcs = [
        ("Sigmoid", sigmoid, sigmoid_derivative, "#2563eb"),
        ("TanH", tanh, tanh_derivative, "#dc2626"),
        ("ReLU", relu, relu_derivative, "#16a34a"),
        ("Leaky ReLU", leaky_relu, leaky_relu_derivative, "#9333ea"),
        ("ELU", elu, elu_derivative, "#ea580c"),
        ("Swish", swish, swish_derivative, "#0891b2"),
        ("SoftPlus", softplus, softplus_derivative, "#be185d"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for name, fn, fn_deriv, color in funcs:
        axes[0].plot(x, fn(x), label=name, color=color, linewidth=2)
        axes[1].plot(x, fn_deriv(x), label=f"{name}'", color=color, linewidth=2)

    axes[0].set_title("Activation Functions σ(x)", fontsize=16, fontweight="bold")
    axes[0].set_ylabel("σ(x)", fontsize=13)
    axes[0].legend(fontsize=11, ncol=4, loc="upper left")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color="black", linewidth=0.5)
    axes[0].axvline(x=0, color="black", linewidth=0.5)

    axes[1].set_title("Derivatives σ'(x) — needed for Backpropagation", fontsize=16, fontweight="bold")
    axes[1].set_xlabel("x", fontsize=13)
    axes[1].set_ylabel("σ'(x)", fontsize=13)
    axes[1].legend(fontsize=11, ncol=4, loc="upper left")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color="black", linewidth=0.5)
    axes[1].axvline(x=0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("figures/activation_functions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ figures/activation_functions.png saved")


def plot_initializations():
    """Compare He vs Xavier vs Orthogonal weight distributions."""
    np.random.seed(42)
    n_in, n_out = 512, 256

    # He initialization
    W_he = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)

    # Xavier initialization
    W_xavier = np.random.randn(n_out, n_in) * np.sqrt(2.0 / (n_in + n_out))

    # Orthogonal initialization
    A = np.random.randn(n_out, n_in)
    U, _, Vt = np.linalg.svd(A, full_matrices=False)
    W_ortho = U

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, W, name, color in [
        (axes[0], W_he, f"He/Kaiming\nσ = √(2/{n_in}) = {np.sqrt(2/n_in):.4f}", "#2563eb"),
        (axes[1], W_xavier, f"Xavier/Glorot\nσ = √(2/{n_in+n_out}) = {np.sqrt(2/(n_in+n_out)):.4f}", "#dc2626"),
        (axes[2], W_ortho, "Orthogonal (SVD)\nSingular values = 1", "#16a34a"),
    ]:
        ax.hist(W.ravel(), bins=80, color=color, alpha=0.7, density=True, edgecolor="white")
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Density")
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.text(
            0.95, 0.95,
            f"mean={W.mean():.4f}\nstd={W.std():.4f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    fig.suptitle(
        "Weight Initialization Comparison [Lecture 3, Slides 2-5]",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig("figures/weight_initialization.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ figures/weight_initialization.png saved")


def plot_optimizer_comparison():
    """
    Compare optimizers on the Rosenbrock function:
    f(x,y) = (a-x)² + b(y-x²)²    with a=1, b=100
    Minimum at (1, 1).
    """
    # Rosenbrock function
    def rosenbrock(x, y, a=1, b=100):
        return (a - x) ** 2 + b * (y - x**2) ** 2

    def rosenbrock_grad(x, y, a=1, b=100):
        dx = -2 * (a - x) + 2 * b * (y - x**2) * (-2 * x)
        dy = 2 * b * (y - x**2)
        return np.array([dx, dy])

    optimizers = {
        "SGD (η=0.001)": {"lr": 0.001, "type": "sgd"},
        "Momentum (γ=0.9)": {"lr": 0.001, "type": "momentum"},
        "RMSprop": {"lr": 0.01, "type": "rmsprop"},
        "Adam": {"lr": 0.01, "type": "adam"},
    }

    colors = ["#6b7280", "#2563eb", "#ea580c", "#16a34a"]
    n_steps = 500

    fig, ax = plt.subplots(figsize=(12, 9))

    # Contour of Rosenbrock
    xx = np.linspace(-2, 2, 300)
    yy = np.linspace(-1, 3, 300)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = rosenbrock(XX, YY)
    ax.contour(XX, YY, ZZ, levels=np.logspace(-1, 3.5, 30), cmap="Greys", alpha=0.4)
    ax.contourf(XX, YY, ZZ, levels=np.logspace(-1, 3.5, 30), cmap="Greys", alpha=0.1)

    for (name, config), color in zip(optimizers.items(), colors):
        x, y = -1.5, 2.5  # starting point
        path_x, path_y = [x], [y]

        # Optimizer state
        if config["type"] == "sgd":
            for _ in range(n_steps):
                grad = rosenbrock_grad(x, y)
                x -= config["lr"] * grad[0]
                y -= config["lr"] * grad[1]
                path_x.append(x)
                path_y.append(y)

        elif config["type"] == "momentum":
            vx, vy = 0, 0
            gamma = 0.9
            for _ in range(n_steps):
                grad = rosenbrock_grad(x, y)
                vx = gamma * vx + config["lr"] * grad[0]
                vy = gamma * vy + config["lr"] * grad[1]
                x -= vx
                y -= vy
                path_x.append(x)
                path_y.append(y)

        elif config["type"] == "rmsprop":
            sx, sy = 0, 0
            beta, eps = 0.9, 1e-8
            for _ in range(n_steps):
                grad = rosenbrock_grad(x, y)
                sx = beta * sx + (1 - beta) * grad[0] ** 2
                sy = beta * sy + (1 - beta) * grad[1] ** 2
                x -= config["lr"] * grad[0] / (np.sqrt(sx) + eps)
                y -= config["lr"] * grad[1] / (np.sqrt(sy) + eps)
                path_x.append(x)
                path_y.append(y)

        elif config["type"] == "adam":
            mx, my, vx, vy = 0, 0, 0, 0
            b1, b2, eps = 0.9, 0.999, 1e-8
            for t in range(1, n_steps + 1):
                grad = rosenbrock_grad(x, y)
                mx = b1 * mx + (1 - b1) * grad[0]
                my = b1 * my + (1 - b1) * grad[1]
                vx = b2 * vx + (1 - b2) * grad[0] ** 2
                vy = b2 * vy + (1 - b2) * grad[1] ** 2
                mxh = mx / (1 - b1**t)
                myh = my / (1 - b1**t)
                vxh = vx / (1 - b2**t)
                vyh = vy / (1 - b2**t)
                x -= config["lr"] * mxh / (np.sqrt(vxh) + eps)
                y -= config["lr"] * myh / (np.sqrt(vyh) + eps)
                path_x.append(x)
                path_y.append(y)

        ax.plot(path_x, path_y, color=color, linewidth=1.5, alpha=0.8, label=name)
        ax.scatter(path_x[0], path_y[0], color=color, s=80, zorder=5, marker="o", edgecolors="white")
        ax.scatter(path_x[-1], path_y[-1], color=color, s=80, zorder=5, marker="*", edgecolors="white")

    # Mark the minimum
    ax.scatter(1, 1, color="#dc2626", s=200, zorder=10, marker="*", edgecolors="white", linewidths=1.5)
    ax.annotate("Minimum (1,1)", xy=(1, 1), xytext=(1.3, 0.5), fontsize=11,
                arrowprops=dict(arrowstyle="->", color="#dc2626"), color="#dc2626")

    ax.set_xlabel("x", fontsize=13)
    ax.set_ylabel("y", fontsize=13)
    ax.set_title("Optimizer Comparison on Rosenbrock Function\n[Lecture 3, Slides 6-22]",
                 fontsize=15, fontweight="bold")
    ax.legend(fontsize=12, loc="upper left")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 3)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("figures/optimizer_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ figures/optimizer_comparison.png saved")


# ── Run all ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    print("Generating visualizations...\n")
    plot_activations()
    plot_initializations()
    plot_optimizer_comparison()
    print("\n✨ All figures generated!")
