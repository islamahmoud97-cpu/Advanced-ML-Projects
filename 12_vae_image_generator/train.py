"""
VAE Image Generator — Train, Generate, and Explore the Latent Space
=====================================================================
Based on: Zemke, AML Lecture 11

This script:
  1. Trains a VAE on handwritten digits
  2. Generates new digits by sampling from N(0, I)
  3. Visualizes the 2D latent space (each digit class = cluster)
  4. Interpolates between digits in latent space
  5. Shows the latent space grid (manifold)

Run: python train.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from vae import VAE


def load_data():
    """Load sklearn digits (8×8)."""
    from sklearn.datasets import load_digits
    digits = load_digits()
    X = digits.data / 16.0  # normalize to [0, 1]
    y = digits.target
    n = int(0.8 * len(X))
    return X[:n], y[:n], X[n:], y[n:]


def plot_training_curves(history, save_path=None):
    """Plot loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(history["loss"], color="#2563eb", linewidth=2)
    axes[0].set_title("Total Loss (Recon + KL)", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history["recon_loss"], color="#dc2626", linewidth=2)
    axes[1].set_title("Reconstruction Loss (MSE)", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(history["kl_loss"], color="#16a34a", linewidth=2)
    axes[2].set_title("KL Divergence", fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle("VAE Training — Loss = Reconstruction + KL Divergence\n"
                 "[Lecture 11: KL pushes latent distribution towards N(0,I)]",
                 fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


def plot_reconstructions(vae, X_test, y_test, n=10, save_path=None):
    """Show original vs reconstructed images."""
    x_recon = vae.reconstruct(X_test[:n])
    
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for i in range(n):
        axes[0, i].imshow(X_test[i].reshape(8, 8), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=12, fontweight="bold")
        axes[0, i].set_title(f"{y_test[i]}", fontsize=11, color="#2563eb")
        
        axes[1, i].imshow(x_recon[i].reshape(8, 8), cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Reconstructed", fontsize=12, fontweight="bold")
    
    fig.suptitle("VAE Reconstruction — Encode → Latent Space → Decode\n"
                 "[Lecture 11: f_autoencoder = f_decoder ∘ f_encoder]",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


def plot_generated_samples(vae, save_path=None):
    """Generate new images by sampling z ~ N(0, I)."""
    samples = vae.generate(n_samples=50)
    
    fig, axes = plt.subplots(5, 10, figsize=(16, 8))
    for i in range(50):
        ax = axes[i // 10, i % 10]
        ax.imshow(samples[i].reshape(8, 8), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    
    fig.suptitle("Generated Digits — Sampled from z ~ N(0, I)\n"
                 "The VAE learns to generate NEW digits that don't exist in the training set!",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


def plot_latent_space(vae, X, y, save_path=None):
    """
    Visualize the 2D latent space — each digit class forms a cluster.
    
    This is the KEY insight of VAEs: the latent space is SMOOTH and CONTINUOUS.
    Nearby points in latent space produce similar images.
    """
    z = vae.encode_to_latent(X)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10",
                        s=15, alpha=0.7, edgecolors="none")
    
    # Add colorbar with digit labels
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.set_label("Digit Class", fontsize=12)
    
    # Add class centers
    for digit in range(10):
        mask = y == digit
        cx, cy = z[mask, 0].mean(), z[mask, 1].mean()
        ax.annotate(str(digit), (cx, cy), fontsize=16, fontweight="bold",
                   ha="center", va="center",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                            edgecolor="gray", alpha=0.8))
    
    ax.set_xlabel("z₁ (Latent Dimension 1)", fontsize=13)
    ax.set_ylabel("z₂ (Latent Dimension 2)", fontsize=13)
    ax.set_title("2D Latent Space — Each Digit Forms a Cluster\n"
                 "Nearby points → similar images (smooth, continuous space!)",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


def plot_latent_manifold(vae, save_path=None):
    """
    Decode a grid of points in latent space to see the manifold.
    
    This shows what the decoder "sees" at each point in 2D latent space.
    Walking through the grid smoothly transitions between digit styles.
    """
    n = 15  # grid size
    figure = np.zeros((8 * n, 8 * n))
    
    # Grid over [-3, 3] × [-3, 3]
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)[::-1]
    
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z = np.array([[xi], [yi]])
            x_decoded, _ = vae.decode(z)
            digit = x_decoded.ravel().reshape(8, 8)
            figure[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = digit
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(figure, cmap="gray", vmin=0, vmax=1)
    
    # Add axis labels
    tick_pos = np.arange(0, 8 * n, 8) + 4
    ax.set_xticks(tick_pos[::3])
    ax.set_xticklabels([f"{x:.1f}" for x in grid_x[::3]])
    ax.set_yticks(tick_pos[::3])
    ax.set_yticklabels([f"{y:.1f}" for y in grid_y[::3]])
    ax.set_xlabel("z₁", fontsize=13)
    ax.set_ylabel("z₂", fontsize=13)
    
    ax.set_title("Latent Space Manifold — Decoded Grid [-3, 3]²\n"
                 "Walking through latent space smoothly transitions between digits",
                 fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


def plot_interpolation(vae, X, y, save_path=None):
    """Interpolate between two digits in latent space."""
    # Find one of each digit
    digit_examples = {}
    for i in range(len(X)):
        if y[i] not in digit_examples:
            digit_examples[y[i]] = X[i]
        if len(digit_examples) == 10:
            break
    
    # Pairs to interpolate
    pairs = [(0, 1), (3, 8), (4, 9), (2, 7)]
    n_steps = 10
    
    fig, axes = plt.subplots(len(pairs), n_steps, figsize=(n_steps * 1.5, len(pairs) * 1.8))
    
    for row, (d1, d2) in enumerate(pairs):
        x1 = digit_examples[d1].reshape(1, -1)
        x2 = digit_examples[d2].reshape(1, -1)
        
        z1 = vae.encode_to_latent(x1).ravel()
        z2 = vae.encode_to_latent(x2).ravel()
        
        for col, alpha in enumerate(np.linspace(0, 1, n_steps)):
            z_interp = (1 - alpha) * z1 + alpha * z2
            z_col = z_interp.reshape(-1, 1)
            x_decoded, _ = vae.decode(z_col)
            
            ax = axes[row, col]
            ax.imshow(x_decoded.ravel().reshape(8, 8), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            
            if col == 0:
                ax.set_title(f"{d1}", fontsize=12, fontweight="bold", color="#2563eb")
            elif col == n_steps - 1:
                ax.set_title(f"{d2}", fontsize=12, fontweight="bold", color="#dc2626")
        
        axes[row, 0].set_ylabel(f"{d1}→{d2}", fontsize=11, fontweight="bold")
    
    fig.suptitle("Latent Space Interpolation — Smooth Transitions Between Digits\n"
                 "[Lecture 11: interpolation in latent space]",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


if __name__ == "__main__":
    np.random.seed(42)
    os.makedirs("figures", exist_ok=True)
    
    print("=" * 60)
    print("  VAE Image Generator — Variational Autoencoder")
    print("=" * 60)
    
    # Load data
    print("\n  Loading data...")
    X_train, y_train, X_test, y_test = load_data()
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Input dim: {X_train.shape[1]} (8×8 images)")
    
    # Build VAE with 2D latent space
    vae = VAE(
        input_dim=64,
        hidden_dims=[128, 64],
        latent_dim=2,
        learning_rate=0.001,
    )
    
    n_params = sum(w.size for w in vae.enc_W) + sum(b.size for b in vae.enc_b) + \
               vae.W_mu.size + vae.b_mu.size + vae.W_logvar.size + vae.b_logvar.size + \
               sum(w.size for w in vae.dec_W) + sum(b.size for b in vae.dec_b)
    print(f"  Parameters: {n_params:,}")
    print(f"  Latent dim: {vae.latent_dim}")
    
    # Train
    print("\n  Training...\n")
    start = time.time()
    history = vae.train(X_train, epochs=100, batch_size=64, verbose=True)
    elapsed = time.time() - start
    print(f"\n  Training time: {elapsed:.1f}s")
    
    # Generate plots
    print("\n  Generating plots...")
    
    plot_training_curves(history, save_path="figures/training.png")
    plot_reconstructions(vae, X_test, y_test, save_path="figures/reconstructions.png")
    plot_generated_samples(vae, save_path="figures/generated.png")
    plot_latent_space(vae, X_train, y_train, save_path="figures/latent_space.png")
    plot_latent_manifold(vae, save_path="figures/manifold.png")
    plot_interpolation(vae, X_train, y_train, save_path="figures/interpolation.png")
    
    print("\n✨ Done!")
