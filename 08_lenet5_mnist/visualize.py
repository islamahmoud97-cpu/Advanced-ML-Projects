"""
LeNet-5 Architecture Diagram
==============================
Generates a visual architecture diagram using only matplotlib.
No TensorFlow required for this file.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def plot_architecture(save_path=None):
    """Draw LeNet-5 architecture as a professional diagram."""
    fig, ax = plt.subplots(figsize=(22, 8))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Layer definitions: (x, name, shape, color, width, height)
    layers = [
        (1.0, "Input\n28×28×1", "#64748b", 1.0, 3.5),
        (3.5, "Conv1\n6×5×5\n→ 28×28×6", "#2563eb", 1.2, 4.0),
        (5.5, "BN+ReLU", "#60a5fa", 0.6, 3.5),
        (6.8, "MaxPool\n2×2\n→ 14×14×6", "#16a34a", 0.8, 3.0),
        (8.5, "Conv2\n16×5×5\n→ 10×10×16", "#2563eb", 1.2, 3.5),
        (10.5, "BN+ReLU", "#60a5fa", 0.6, 3.0),
        (11.8, "MaxPool\n2×2\n→ 5×5×16", "#16a34a", 0.8, 2.5),
        (13.5, "Flatten\n→ 400", "#f59e0b", 0.6, 2.0),
        (15.0, "Dense\n120\nReLU+DO", "#dc2626", 1.0, 2.5),
        (17.0, "Dense\n84\nReLU+DO", "#dc2626", 1.0, 2.2),
        (19.0, "Dense\n10\nSoftmax", "#9333ea", 1.0, 1.8),
    ]

    y_center = 4.0

    for x, label, color, width, height in layers:
        rect = patches.FancyBboxPatch(
            (x, y_center - height/2), width, height,
            boxstyle="round,pad=0.1", facecolor=color, alpha=0.8,
            edgecolor="white", linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(x + width/2, y_center, label,
                ha="center", va="center", fontsize=8, color="white",
                fontweight="bold", linespacing=1.3)

    # Arrows
    arrow_y = y_center
    arrow_xs = [2.0, 4.7, 6.1, 7.6, 9.7, 11.1, 12.6, 14.1, 16.0, 18.0]
    for ax_pos in arrow_xs:
        ax.annotate("", xy=(ax_pos + 0.3, arrow_y), xytext=(ax_pos, arrow_y),
                     arrowprops=dict(arrowstyle="->", color="#334155", lw=2))

    # Title
    ax.text(11, 7.3,
            "LeNet-5 Architecture — Original (1998) vs Modern",
            ha="center", va="center", fontsize=18, fontweight="bold")

    # Legend
    legend_items = [
        ("#2563eb", "Convolution"), ("#16a34a", "Pooling"),
        ("#dc2626", "Dense"), ("#60a5fa", "BN+ReLU"),
        ("#f59e0b", "Flatten"), ("#9333ea", "Output"),
    ]
    for i, (color, label) in enumerate(legend_items):
        x_leg = 1.0 + i * 3.5
        rect = patches.Rectangle((x_leg, 0.3), 0.4, 0.4, facecolor=color, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x_leg + 0.6, 0.5, label, fontsize=10, va="center")

    # Parameter count
    ax.text(11, 1.2,
            "Original: ~61K params (TanH + AvgPool)  |  "
            "Modern: ~62K params (ReLU + BN + MaxPool + Dropout)",
            ha="center", fontsize=11, color="#64748b", style="italic")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


def plot_lenet5_timeline(save_path=None):
    """Timeline of CNN history from Lecture 5."""
    fig, ax = plt.subplots(figsize=(20, 6))

    events = [
        (1980, "Neocognitron\n(Fukushima)", "#6b7280", "First CNN concept"),
        (1989, "Backprop in CNN\n(LeCun et al.)", "#2563eb", "First trained CNN"),
        (1998, "LeNet-5\n(LeCun et al.)", "#dc2626", "Industrial scale\ncheck digit reading"),
        (2012, "AlexNet\n(Krizhevsky)", "#16a34a", "ImageNet winner\nGPU + ReLU + Dropout"),
        (2014, "VGGNet\n(Simonyan)", "#9333ea", "Very deep (19 layers)\nSmall 3×3 filters"),
        (2014, "GoogLeNet\n(Szegedy)", "#ea580c", "Inception modules\n22 layers"),
        (2015, "ResNet\n(He et al.)", "#0891b2", "152 layers!\nSkip connections"),
    ]

    ax.set_xlim(1975, 2020)
    ax.set_ylim(-1, 5)

    # Timeline line
    ax.plot([1978, 2018], [0, 0], color="#1e293b", linewidth=3, zorder=1)

    for i, (year, name, color, desc) in enumerate(events):
        y_pos = 1.5 if i % 2 == 0 else -0.8
        y_line = 0.3 if i % 2 == 0 else -0.3

        ax.scatter(year, 0, s=150, color=color, zorder=5, edgecolors="white", linewidths=2)
        ax.plot([year, year], [y_line, y_pos - 0.2 if y_pos > 0 else y_pos + 0.5],
                color=color, linewidth=1.5, linestyle="--", alpha=0.5)

        bbox = dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.15, edgecolor=color)
        ax.text(year, y_pos, f"{name}\n({year})\n{desc}",
                ha="center", va="center" if y_pos > 0 else "center",
                fontsize=9, fontweight="bold", bbox=bbox)

    # Highlight LeNet-5
    rect = patches.FancyBboxPatch((1994, -0.15), 8, 0.3,
                                    boxstyle="round", facecolor="#dc2626", alpha=0.1,
                                    edgecolor="#dc2626", linewidth=2, linestyle="--")
    ax.add_patch(rect)

    ax.set_title("CNN History — From Neocognitron to ResNet\n[Lecture 5: History of CNN]",
                fontsize=16, fontweight="bold")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_yticks([])
    ax.grid(True, alpha=0.1, axis="x")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    print("Generating architecture diagrams...\n")
    plot_architecture(save_path="figures/architecture.png")
    plot_lenet5_timeline(save_path="figures/cnn_timeline.png")
    print("\nDone!")
