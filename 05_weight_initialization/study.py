"""
Signal Propagation Study
=========================
Based on: Zemke, AML Lecture 3, Slides 9-14

Key Question: How do activations and gradients flow through a DEEP network?

The answer depends on weight initialization:
  - Too small σ → activations shrink to 0 (vanishing signals)
  - Too large σ → activations explode to ±∞ (exploding signals)
  - Xavier/He  → activations stay stable (Var ≈ constant)

This script propagates a random input through a 20-layer network
and measures the activation statistics at each layer.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from initializations import (
    zeros, random_normal, xavier_normal, he_normal,
    lecun_normal, orthogonal, ALL_INITS,
)


def propagate_signal(init_fn, n_layers=20, n_in=256, n_hidden=256,
                     activation="relu", n_samples=500):
    """
    Forward-pass a random input through n_layers and record statistics.
    
    Returns dict with:
      - means: mean activation at each layer
      - stds: std of activations at each layer
      - histograms: activation distributions
    """
    # Activation functions
    if activation == "relu":
        act = lambda z: np.maximum(0, z)
    elif activation == "tanh":
        act = lambda z: np.tanh(z)
    elif activation == "sigmoid":
        act = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    else:
        act = lambda z: z  # linear
    
    # Random input
    x = np.random.randn(n_in, n_samples)
    
    means = [np.mean(np.abs(x))]
    stds = [np.std(x)]
    histograms = [x.ravel().copy()]
    
    a = x
    for layer in range(n_layers):
        shape = (n_hidden, a.shape[0])
        W = init_fn(shape)
        z = W @ a
        a = act(z)
        
        means.append(np.mean(np.abs(a)))
        stds.append(np.std(a))
        histograms.append(a.ravel().copy())
    
    return {"means": means, "stds": stds, "histograms": histograms}


def plot_signal_propagation(save_path=None):
    """
    Main visualization: how activations flow through 20 layers.
    Compare Zeros, Random(0.01), LeCun, Xavier, He, Orthogonal.
    """
    np.random.seed(42)
    n_layers = 20
    
    inits_to_test = [
        ("Random (σ=0.01)", lambda s: random_normal(s, std=0.01), "#a3a3a3"),
        ("Random (σ=1.0)", lambda s: random_normal(s, std=1.0), "#525252"),
        ("LeCun", lecun_normal, "#ea580c"),
        ("Xavier", xavier_normal, "#2563eb"),
        ("He", he_normal, "#dc2626"),
        ("Orthogonal", orthogonal, "#16a34a"),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # ── Plot 1: Activation std per layer (ReLU) ──────────────────────
    ax = axes[0, 0]
    for name, init_fn, color in inits_to_test:
        result = propagate_signal(init_fn, n_layers, activation="relu")
        ax.semilogy(result["stds"], "o-", color=color, linewidth=2, markersize=5,
                    alpha=0.8, label=name)
    
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.annotate("Ideal: Var ≈ 1", xy=(15, 1.0), fontsize=10, color="gray")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Std of Activations (log)", fontsize=12)
    ax.set_title("Signal Propagation — ReLU\nHe initialization keeps variance stable",
                fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, which="both")
    ax.set_xlim(0, n_layers)
    
    # ── Plot 2: Activation std per layer (TanH) ──────────────────────
    ax = axes[0, 1]
    for name, init_fn, color in inits_to_test:
        result = propagate_signal(init_fn, n_layers, activation="tanh")
        ax.semilogy(result["stds"], "o-", color=color, linewidth=2, markersize=5,
                    alpha=0.8, label=name)
    
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Std of Activations (log)", fontsize=12)
    ax.set_title("Signal Propagation — TanH\nXavier initialization keeps variance stable",
                fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, which="both")
    ax.set_xlim(0, n_layers)
    
    # ── Plot 3: Activation histograms (ReLU + He) ────────────────────
    ax = axes[1, 0]
    result_he = propagate_signal(he_normal, n_layers, activation="relu")
    layers_to_show = [0, 1, 5, 10, 15, 19]
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(layers_to_show)))
    
    for idx, layer in enumerate(layers_to_show):
        data = result_he["histograms"][layer]
        data_clipped = data[np.abs(data) < np.percentile(np.abs(data), 99)]
        ax.hist(data_clipped, bins=80, alpha=0.4, color=cmap[idx],
                density=True, label=f"Layer {layer}")
    
    ax.set_xlabel("Activation Value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Activation Distributions — He Init + ReLU\nDistributions stay healthy across layers",
                fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    
    # ── Plot 4: Activation histograms (Random σ=0.01 + ReLU) ────────
    ax = axes[1, 1]
    result_bad = propagate_signal(lambda s: random_normal(s, std=0.01), n_layers, activation="relu")
    
    for idx, layer in enumerate(layers_to_show):
        data = result_bad["histograms"][layer]
        data_clipped = data[np.abs(data) < max(np.percentile(np.abs(data), 99), 1e-10)]
        if len(data_clipped) > 0:
            ax.hist(data_clipped, bins=80, alpha=0.4, color=cmap[idx],
                    density=True, label=f"Layer {layer}")
    
    ax.set_xlabel("Activation Value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Activation Distributions — Random(0.01) + ReLU\nSignal collapses to zero after a few layers!",
                fontsize=13, fontweight="bold", color="#dc2626")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    
    fig.suptitle(
        "Weight Initialization — Signal Propagation Through 20 Layers\n"
        "[Lecture 3: Xavier/Glorot (2010), He/Kaiming (2015)]",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


def plot_gradient_flow(save_path=None):
    """
    Backward pass: how gradients flow through 20 layers.
    
    Gradient at layer i: δᵢ = σ'(zᵢ) ⊙ (Wᵢ₊₁ᵀ δᵢ₊₁)
    
    If W too small: gradient → 0 (vanishing)
    If W too large: gradient → ∞ (exploding)
    """
    np.random.seed(42)
    n_layers = 20
    n = 256
    n_samples = 500
    
    inits = [
        ("Random (σ=0.01)", lambda s: random_normal(s, std=0.01), "#a3a3a3"),
        ("Xavier", xavier_normal, "#2563eb"),
        ("He", he_normal, "#dc2626"),
        ("Orthogonal", orthogonal, "#16a34a"),
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    for ax, act_name, title in [
        (axes[0], "relu", "Gradient Flow — ReLU"),
        (axes[1], "tanh", "Gradient Flow — TanH"),
    ]:
        if act_name == "relu":
            act_deriv = lambda z: (z > 0).astype(float)
        else:
            act_deriv = lambda z: 1 - np.tanh(z)**2
        
        for name, init_fn, color in inits:
            # Forward pass (store z values)
            x = np.random.randn(n, n_samples) * 0.5
            z_list = []
            W_list = []
            a = x
            for _ in range(n_layers):
                W = init_fn((n, n))
                W_list.append(W)
                z = W @ a
                z_list.append(z)
                if act_name == "relu":
                    a = np.maximum(0, z)
                else:
                    a = np.tanh(z)
            
            # Backward pass
            delta = np.random.randn(n, n_samples)  # random output gradient
            grad_norms = [np.linalg.norm(delta) / n_samples]
            
            for i in range(n_layers - 1, -1, -1):
                delta = act_deriv(z_list[i]) * (W_list[i].T @ delta)
                grad_norms.insert(0, np.linalg.norm(delta) / n_samples)
            
            ax.semilogy(grad_norms, "o-", color=color, linewidth=2,
                        markersize=4, alpha=0.8, label=name)
        
        ax.set_xlabel("Layer (backward)", fontsize=12)
        ax.set_ylabel("Gradient Norm (log)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2, which="both")
    
    fig.suptitle(
        "Gradient Flow Through 20 Layers — Forward vs Backward\n"
        "Proper initialization keeps gradients from vanishing or exploding",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


def plot_weight_distributions(save_path=None):
    """Visualize the weight distributions for each initialization method."""
    np.random.seed(42)
    shape = (256, 256)
    
    methods = [
        ("Random (σ=0.01)", random_normal(shape, std=0.01), "#a3a3a3"),
        ("LeCun", lecun_normal(shape), "#ea580c"),
        ("Xavier", xavier_normal(shape), "#2563eb"),
        ("He", he_normal(shape), "#dc2626"),
        ("Orthogonal", orthogonal(shape), "#16a34a"),
    ]
    
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    
    for ax, (name, W, color) in zip(axes, methods):
        ax.hist(W.ravel(), bins=80, color=color, alpha=0.7, density=True, edgecolor="white")
        ax.set_title(f"{name}\nmean={W.mean():.4f}, std={W.std():.4f}",
                    fontsize=11, fontweight="bold", color=color)
        ax.axvline(x=0, color="black", linewidth=0.5, alpha=0.3)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("Weight Value")
    
    axes[0].set_ylabel("Density")
    
    fig.suptitle(
        "Weight Distributions — Shape (256, 256)\n"
        "[Lecture 3, Slide 13: Xavier σ=√(2/(nᵢₙ+nₒᵤₜ)), He σ=√(2/nᵢₙ)]",
        fontsize=15, fontweight="bold", y=1.04,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


def plot_singular_values(save_path=None):
    """
    Compare singular value spectra of initialization methods.
    
    For orthogonal init: all singular values = 1 (perfect norm preservation).
    For random init: circular law distribution.
    """
    np.random.seed(42)
    n = 256
    
    methods = [
        ("Random (σ=1/√n)", lambda: np.random.randn(n, n) / np.sqrt(n), "#a3a3a3"),
        ("Xavier", lambda: xavier_normal((n, n)), "#2563eb"),
        ("He", lambda: he_normal((n, n)), "#dc2626"),
        ("Orthogonal", lambda: orthogonal((n, n)), "#16a34a"),
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    
    for ax, (name, init_fn, color) in zip(axes, methods):
        W = init_fn()
        svd_vals = np.linalg.svd(W, compute_uv=False)
        
        ax.bar(range(len(svd_vals)), svd_vals, color=color, alpha=0.7, width=1.0)
        ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.4, linewidth=1)
        ax.set_title(f"{name}\nσ_max={svd_vals[0]:.2f}, σ_min={svd_vals[-1]:.4f}",
                    fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("Index")
        ax.set_xlim(0, n)
        ax.grid(True, alpha=0.2)
    
    axes[0].set_ylabel("Singular Value")
    
    fig.suptitle(
        "Singular Value Spectra — Orthogonal Has ALL σᵢ = 1\n"
        "[Lecture 3, Slide 5: Orthogonal via SVD preserves norms perfectly]",
        fontsize=15, fontweight="bold", y=1.04,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    
    print("=" * 60)
    print("  Weight Initialization Study")
    print("=" * 60)
    
    print("\n📊 1/4: Signal propagation through 20 layers...")
    plot_signal_propagation(save_path="figures/signal_propagation.png")
    
    print("\n📊 2/4: Gradient flow analysis...")
    plot_gradient_flow(save_path="figures/gradient_flow.png")
    
    print("\n📊 3/4: Weight distributions...")
    plot_weight_distributions(save_path="figures/weight_distributions.png")
    
    print("\n📊 4/4: Singular value spectra...")
    plot_singular_values(save_path="figures/singular_values.png")
    
    print("\n✨ All figures generated!")
