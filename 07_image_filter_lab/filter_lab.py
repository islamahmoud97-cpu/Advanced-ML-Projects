"""
Image Filter Lab — Apply & Visualize All Classic Filters
==========================================================
Based on: Zemke, AML Lecture 4, Slides 8-20

From Lecture 4:
  - Derivative filter: d = [-1, 1] / h  → edge detection
  - Laplacian: d = [-1, 2, -1] / h²     → 2nd derivative, all edges
  - Averaging: d = [1, 1, 1] / 3        → low-pass / blur

This script:
  1. Generates test images (shapes, gradients, textures)
  2. Applies all 14 filters
  3. Shows Sobel edge magnitude + direction
  4. Demonstrates filter composition (Gaussian → Laplacian = LoG)
  5. Shows how CNN first layers learn similar filters

Run: python filter_lab.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from kernels import (
    ALL_KERNELS, EDGE_KERNELS, BLUR_KERNELS,
    SOBEL_X, SOBEL_Y, LAPLACIAN, LAPLACIAN_8,
    BOX_BLUR_3, GAUSSIAN_3, GAUSSIAN_5,
    SHARPEN, EMBOSS, IDENTITY,
    convolve2d, apply_filter, make_gaussian_kernel,
)


# ═══════════════════════════════════════════════════════════════════════════
# Test Image Generators
# ═══════════════════════════════════════════════════════════════════════════
def make_shapes_image(size=128):
    """Generate test image with geometric shapes."""
    img = np.zeros((size, size))
    
    # Rectangle
    img[20:50, 20:60] = 1.0
    
    # Circle
    y, x = np.ogrid[:size, :size]
    circle = ((x - 90)**2 + (y - 35)**2) < 20**2
    img[circle] = 0.8
    
    # Triangle
    for i in range(30):
        img[70+i, 50-i:50+i+1] = 0.6
    
    # Gradient bar
    img[110:120, 10:118] = np.linspace(0, 1, 108)
    
    # Diagonal line
    for i in range(60):
        if 65+i < size and 65+i < size:
            img[65+i, 65+i] = 1.0
            if 66+i < size:
                img[65+i, 66+i] = 0.7
    
    return img


def make_checkerboard(size=128, block=16):
    """Checkerboard pattern — tests frequency response."""
    img = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if ((i // block) + (j // block)) % 2 == 0:
                img[i, j] = 1.0
    return img


def make_noisy_image(size=128, noise_level=0.3):
    """Shapes with added Gaussian noise — tests denoising."""
    img = make_shapes_image(size)
    noise = np.random.randn(size, size) * noise_level
    return np.clip(img + noise, 0, 1)


def make_gradient_image(size=128):
    """Smooth gradient — tests derivative filters."""
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    return np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) * 0.5 + 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Visualization Functions
# ═══════════════════════════════════════════════════════════════════════════
def plot_all_filters_grid(save_path=None):
    """Apply all 14 filters to the shapes image — big grid."""
    img = make_shapes_image(128)
    
    n_filters = len(ALL_KERNELS)
    n_cols = 5
    n_rows = (n_filters + 1 + n_cols - 1) // n_cols  # +1 for original
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 4))
    
    # Original
    axes.ravel()[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes.ravel()[0].set_title("Original", fontsize=12, fontweight="bold")
    axes.ravel()[0].axis("off")
    
    for idx, kernel_info in enumerate(ALL_KERNELS):
        ax = axes.ravel()[idx + 1]
        result = convolve2d(img, kernel_info.kernel)
        
        if kernel_info.category == "edge":
            ax.imshow(np.abs(result), cmap="hot")
        elif kernel_info.category == "emboss":
            ax.imshow(result, cmap="gray")
        else:
            ax.imshow(np.clip(result, 0, 1), cmap="gray")
        
        ax.set_title(f"{kernel_info.name}\n({kernel_info.category})",
                     fontsize=10, fontweight="bold", color=kernel_info.color)
        ax.axis("off")
    
    # Hide empty axes
    for idx in range(len(ALL_KERNELS) + 1, len(axes.ravel())):
        axes.ravel()[idx].axis("off")
    
    fig.suptitle(
        "All 14 Image Filters Applied to Test Image\n"
        "[Lecture 4: Derivative, Laplacian, and Averaging Filters]",
        fontsize=16, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


def plot_sobel_edge_detection(save_path=None):
    """
    Sobel edge detection: combine X and Y gradients.
    
    Magnitude: |G| = √(Gx² + Gy²)
    Direction: θ = atan2(Gy, Gx)
    """
    img = make_shapes_image(128)
    
    gx = convolve2d(img, SOBEL_X.kernel)
    gy = convolve2d(img, SOBEL_Y.kernel)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("Original Image", fontsize=13, fontweight="bold")
    
    axes[0, 1].imshow(np.abs(gx), cmap="hot")
    axes[0, 1].set_title("Sobel X (∂/∂x)\nVertical edges", fontsize=13, fontweight="bold", color="#2563eb")
    
    axes[0, 2].imshow(np.abs(gy), cmap="hot")
    axes[0, 2].set_title("Sobel Y (∂/∂y)\nHorizontal edges", fontsize=13, fontweight="bold", color="#7c3aed")
    
    axes[1, 0].imshow(magnitude, cmap="hot")
    axes[1, 0].set_title("Edge Magnitude\n|G| = √(Gx² + Gy²)", fontsize=13, fontweight="bold", color="#dc2626")
    
    axes[1, 1].imshow(direction, cmap="hsv")
    axes[1, 1].set_title("Edge Direction\nθ = atan2(Gy, Gx)", fontsize=13, fontweight="bold", color="#ea580c")
    
    # Thresholded edges
    threshold = 0.3 * magnitude.max()
    edges = (magnitude > threshold).astype(float)
    axes[1, 2].imshow(edges, cmap="gray")
    axes[1, 2].set_title(f"Thresholded Edges\n(threshold = {threshold:.2f})", fontsize=13, fontweight="bold")
    
    for ax in axes.ravel():
        ax.axis("off")
    
    fig.suptitle(
        "Sobel Edge Detection — Gradient Magnitude & Direction\n"
        "[Lecture 4: Derivative filter d = [-1, 0, 1] extended to 2D]",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


def plot_denoising(save_path=None):
    """Show how blur filters remove noise."""
    np.random.seed(42)
    img_clean = make_shapes_image(128)
    img_noisy = make_noisy_image(128, noise_level=0.3)
    
    filters = [
        ("Noisy Input", img_noisy),
        ("Box 3x3", convolve2d(img_noisy, BOX_BLUR_3.kernel)),
        ("Gaussian σ=1", convolve2d(img_noisy, GAUSSIAN_3.kernel)),
        ("Gaussian σ=1.5 (5x5)", convolve2d(img_noisy, GAUSSIAN_5.kernel)),
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Top: filtered results
    axes[0, 0].imshow(img_clean, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Clean Original", fontsize=12, fontweight="bold")
    
    for idx, (name, result) in enumerate(filters):
        ax = axes[0, idx] if idx == 0 else axes[0, idx]
        if idx == 0:
            ax = axes[0, 1]
            ax.imshow(np.clip(result, 0, 1), cmap="gray")
            ax.set_title(name, fontsize=12, fontweight="bold", color="#dc2626")
        else:
            ax = axes[0, idx + 1] if idx < 3 else axes[0, 3]
            # Fix for proper indexing
    
    # Redo properly
    for ax in axes.ravel():
        ax.axis("off")
    
    images = [
        ("Clean Original", img_clean),
        ("Noisy (σ=0.3)", img_noisy),
        ("Box Blur 3×3", np.clip(convolve2d(img_noisy, BOX_BLUR_3.kernel), 0, 1)),
        ("Gaussian σ=1.5", np.clip(convolve2d(img_noisy, GAUSSIAN_5.kernel), 0, 1)),
    ]
    
    colors = ["#16a34a", "#dc2626", "#2563eb", "#ea580c"]
    
    for idx, ((name, im), color) in enumerate(zip(images, colors)):
        # Image
        axes[0, idx].imshow(im, cmap="gray", vmin=0, vmax=1)
        axes[0, idx].set_title(name, fontsize=12, fontweight="bold", color=color)
        axes[0, idx].axis("off")
        
        # Difference from clean
        diff = np.abs(im - img_clean)
        mse = np.mean(diff**2)
        axes[1, idx].imshow(diff, cmap="hot", vmin=0, vmax=0.5)
        axes[1, idx].set_title(f"Error (MSE={mse:.4f})", fontsize=11, fontweight="bold")
        axes[1, idx].axis("off")
    
    fig.suptitle(
        "Denoising with Blur Filters — [Lecture 4: Averaging = Low-Pass Filter]",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


def plot_kernel_visualizations(save_path=None):
    """Visualize all kernel matrices as heatmaps."""
    kernels_to_show = [
        IDENTITY, SOBEL_X, SOBEL_Y, LAPLACIAN,
        BOX_BLUR_3, GAUSSIAN_3, SHARPEN, EMBOSS,
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for ax, k in zip(axes.ravel(), kernels_to_show):
        im = ax.imshow(k.kernel, cmap="RdBu_r", vmin=-k.kernel.max(), vmax=k.kernel.max())
        ax.set_title(f"{k.name}", fontsize=11, fontweight="bold", color=k.color)
        
        # Show values in cells
        kH, kW = k.kernel.shape
        for i in range(kH):
            for j in range(kW):
                val = k.kernel[i, j]
                color = "white" if abs(val) > k.kernel.max() * 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8 if kH <= 3 else 6, color=color)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.suptitle(
        "Kernel Matrices — What Each Filter Looks Like\n"
        "Red = positive weight, Blue = negative weight",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


def plot_filter_composition(save_path=None):
    """
    Show filter composition: Gaussian + Laplacian = Laplacian of Gaussian (LoG).
    This is how real edge detectors work (reduces noise sensitivity).
    """
    img = make_shapes_image(128)
    
    # Direct Laplacian (noisy on real images)
    laplacian_direct = convolve2d(img, LAPLACIAN.kernel)
    
    # Gaussian first, then Laplacian (LoG — much cleaner)
    blurred = convolve2d(img, GAUSSIAN_3.kernel)
    log_result = convolve2d(blurred, LAPLACIAN.kernel)
    
    # Sharpen = Identity + Laplacian
    sharpened = convolve2d(img, SHARPEN.kernel)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("Original", fontsize=13, fontweight="bold")
    
    axes[0, 1].imshow(np.abs(laplacian_direct), cmap="hot")
    axes[0, 1].set_title("Laplacian (direct)\nSensitive to noise", fontsize=13, fontweight="bold", color="#dc2626")
    
    axes[0, 2].imshow(np.abs(log_result), cmap="hot")
    axes[0, 2].set_title("Gaussian → Laplacian (LoG)\nMuch cleaner!", fontsize=13, fontweight="bold", color="#16a34a")
    
    axes[1, 0].imshow(np.clip(convolve2d(img, GAUSSIAN_3.kernel), 0, 1), cmap="gray")
    axes[1, 0].set_title("Step 1: Gaussian Blur\nSmooth away noise", fontsize=13, fontweight="bold", color="#2563eb")
    
    axes[1, 1].imshow(np.clip(sharpened, 0, 1), cmap="gray")
    axes[1, 1].set_title("Sharpen = Identity + Laplacian\nEnhance edges", fontsize=13, fontweight="bold", color="#ea580c")
    
    # Show the math
    axes[1, 2].axis("off")
    math_text = (
        "Filter Composition:\n\n"
        "LoG = Gaussian * Laplacian\n"
        "(smooth first, then detect edges)\n\n"
        "Sharpen = Identity + alpha * Laplacian\n"
        "(enhance edges while keeping image)\n\n"
        "Key insight from Lecture 4:\n"
        "Convolution is ASSOCIATIVE\n"
        "(A * B) * C = A * (B * C)"
    )
    axes[1, 2].text(0.5, 0.5, math_text, fontsize=13, ha="center", va="center",
                    transform=axes[1, 2].transAxes, fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="#f8fafc", edgecolor="#e2e8f0"))
    
    for ax in axes.ravel():
        if ax.images:
            ax.axis("off")
    
    fig.suptitle(
        "Filter Composition — Combining Filters for Better Results",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    
    print("=" * 60)
    print("  Image Filter Lab")
    print("=" * 60)
    
    print("\n  1/5: All 14 filters grid...")
    plot_all_filters_grid(save_path="figures/all_filters.png")
    
    print("\n  2/5: Sobel edge detection...")
    plot_sobel_edge_detection(save_path="figures/sobel_edges.png")
    
    print("\n  3/5: Denoising comparison...")
    plot_denoising(save_path="figures/denoising.png")
    
    print("\n  4/5: Kernel matrix visualizations...")
    plot_kernel_visualizations(save_path="figures/kernel_matrices.png")
    
    print("\n  5/5: Filter composition (LoG, Sharpen)...")
    plot_filter_composition(save_path="figures/filter_composition.png")
    
    print("\n✨ All figures generated!")
