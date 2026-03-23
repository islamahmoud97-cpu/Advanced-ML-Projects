"""
Vanishing Gradient Problem — Demonstration
=============================================
Based on: Zemke, AML Lecture 1 & 3

The vanishing gradient problem occurs when:
  - Sigmoid/TanH derivatives are << 1 for large |x|
  - During backpropagation, gradients are MULTIPLIED layer by layer
  - After many layers: gradient → 0 → network stops learning

This script demonstrates:
  1. WHY sigmoid causes vanishing gradients (max derivative = 0.25)
  2. HOW ReLU fixes it (derivative = 1 for x > 0)
  3. WHAT happens to gradients through 10, 20, 50 layers
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from activations import sigmoid, sigmoid_deriv, tanh_fn, tanh_deriv
from activations import relu, relu_deriv, swish, swish_deriv, elu, elu_deriv


def simulate_gradient_flow(activation_deriv, n_layers, n_samples=10000):
    """
    Simulate gradient flow through n_layers.

    In backpropagation, the gradient at layer i is:
        δᵢ = σ'(zᵢ) ⊙ (Wᵢ₊₁ᵀ · δᵢ₊₁)

    For simplicity, assume W ≈ I (identity) and z ~ N(0, 1).
    Then the gradient magnitude after L layers ≈ ∏ σ'(zₗ)

    Returns: array of gradient magnitudes at each layer.
    """
    # Random pre-activations z ~ N(0, 1)
    z_samples = np.random.randn(n_samples)

    gradient_magnitudes = []
    cumulative_gradient = np.ones(n_samples)

    for layer in range(n_layers):
        z = np.random.randn(n_samples)  # fresh z per layer
        deriv = activation_deriv(z)
        cumulative_gradient *= deriv
        gradient_magnitudes.append(np.mean(np.abs(cumulative_gradient)))

    return np.array(gradient_magnitudes)


def plot_vanishing_gradient():
    """Generate the vanishing gradient comparison plot."""
    np.random.seed(42)
    n_layers = 50

    activations = [
        ("Sigmoid", sigmoid_deriv, "#2563eb"),
        ("TanH", tanh_deriv, "#dc2626"),
        ("ReLU", relu_deriv, "#16a34a"),
        ("ELU", elu_deriv, "#ea580c"),
        ("Swish", swish_deriv, "#0891b2"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Plot 1: Gradient magnitude vs. layer depth ────────────────────
    ax = axes[0]
    for name, deriv_fn, color in activations:
        grads = simulate_gradient_flow(deriv_fn, n_layers)
        ax.semilogy(range(1, n_layers + 1), grads, color=color,
                     linewidth=2.5, label=name, alpha=0.85)

    ax.set_xlabel("Layer Depth", fontsize=13)
    ax.set_ylabel("Gradient Magnitude (log scale)", fontsize=13)
    ax.set_title("Vanishing Gradient Through 50 Layers", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    ax.axhline(y=1e-10, color="red", linestyle="--", alpha=0.4, linewidth=1)
    ax.annotate("Gradient effectively zero", xy=(30, 1e-10),
                fontsize=10, color="red", alpha=0.6)
    ax.set_xlim(1, n_layers)

    # ── Plot 2: Derivative histograms ─────────────────────────────────
    ax2 = axes[1]
    z = np.random.randn(50000)

    for name, deriv_fn, color in activations:
        derivs = deriv_fn(z)
        ax2.hist(derivs, bins=100, alpha=0.4, color=color, label=name,
                 density=True, range=(-0.5, 1.5))

    ax2.axvline(x=0.25, color="#2563eb", linestyle="--", alpha=0.5, linewidth=1.5)
    ax2.annotate("max σ'(sigmoid) = 0.25", xy=(0.26, 3), fontsize=10, color="#2563eb")

    ax2.axvline(x=1.0, color="#16a34a", linestyle="--", alpha=0.5, linewidth=1.5)
    ax2.annotate("ReLU' = 1 for z > 0", xy=(1.02, 2), fontsize=10, color="#16a34a")

    ax2.set_xlabel("Derivative value σ'(z)", fontsize=13)
    ax2.set_ylabel("Density", fontsize=13)
    ax2.set_title("Distribution of Derivatives for z ~ N(0,1)", fontsize=15, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.5, 1.5)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/vanishing_gradient.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ figures/vanishing_gradient.png saved")


def plot_why_relu_works():
    """
    Visual explanation: Why ReLU solved deep learning.

    Key insight from Lecture 1:
    - Sigmoid: max derivative = 0.25 → after 10 layers: 0.25^10 ≈ 9.5e-7
    - ReLU: derivative = 1 (for x > 0) → gradient flows unchanged
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    x = np.linspace(-4, 4, 500)

    # ── Sigmoid problem ───────────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(x, sigmoid(x), alpha=0.15, color="#2563eb")
    ax.plot(x, sigmoid(x), color="#2563eb", linewidth=2.5, label="σ(x)")
    ax.plot(x, sigmoid_deriv(x), color="#dc2626", linewidth=2.5, label="σ'(x)")
    ax.axhline(y=0.25, color="#dc2626", linestyle="--", alpha=0.5)
    ax.annotate("max = 0.25\n→ gradient shrinks\n   every layer!",
                xy=(0, 0.25), xytext=(1.5, 0.6),
                fontsize=11, color="#dc2626", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#dc2626"))
    ax.set_title("Sigmoid: Vanishing Gradient", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.1, 1.1)

    # ── ReLU solution ─────────────────────────────────────────────────
    ax = axes[1]
    ax.fill_between(x[x >= 0], relu(x[x >= 0]), alpha=0.15, color="#16a34a")
    ax.plot(x, relu(x), color="#16a34a", linewidth=2.5, label="ReLU(x)")
    ax.plot(x, relu_deriv(x), color="#dc2626", linewidth=2.5, label="ReLU'(x)")
    ax.axhline(y=1, color="#dc2626", linestyle="--", alpha=0.5)
    ax.annotate("derivative = 1\n→ gradient flows\n   unchanged!",
                xy=(2, 1), xytext=(2.5, 2.5),
                fontsize=11, color="#16a34a", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#16a34a"))
    ax.annotate("BUT: dead zone\n(gradient = 0)",
                xy=(-2, 0), xytext=(-3.5, 1.5),
                fontsize=10, color="#6b7280",
                arrowprops=dict(arrowstyle="->", color="#6b7280"))
    ax.set_title("ReLU: The Fix", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.5, 4)

    # ── Gradient after N layers ───────────────────────────────────────
    ax = axes[2]
    layers = np.arange(1, 21)

    # Sigmoid: best case gradient = 0.25^L
    sigmoid_grad = 0.25 ** layers

    # ReLU: ~50% neurons active → 0.5^L (but gradient = 1 for active)
    relu_grad = 0.5 ** layers  # probability all active

    # Swish: average derivative ≈ 0.5-0.8
    swish_grad = 0.6 ** layers

    ax.semilogy(layers, sigmoid_grad, "o-", color="#2563eb", linewidth=2, markersize=6, label="Sigmoid")
    ax.semilogy(layers, relu_grad, "s-", color="#16a34a", linewidth=2, markersize=6, label="ReLU")
    ax.semilogy(layers, swish_grad, "^-", color="#0891b2", linewidth=2, markersize=6, label="Swish")

    ax.axhline(y=1e-7, color="red", linestyle="--", alpha=0.4)
    ax.annotate("Gradient effectively dead", xy=(10, 1e-7), fontsize=10, color="red", alpha=0.6)

    ax.set_xlabel("Number of Layers", fontsize=13)
    ax.set_ylabel("Gradient Magnitude", fontsize=13)
    ax.set_title("Gradient Decay: Sigmoid vs ReLU vs Swish", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("figures/why_relu_works.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ figures/why_relu_works.png saved")


if __name__ == "__main__":
    print("Generating vanishing gradient analysis...\n")
    plot_vanishing_gradient()
    plot_why_relu_works()
    print("\nDone!")
