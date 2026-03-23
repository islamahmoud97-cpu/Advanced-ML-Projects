"""
Generate All Figures
=====================
Creates publication-quality plots for the README:
  1. Complete activation function comparison (all 9)
  2. Properties comparison table (as image)
  3. Zoomed views near x=0 (critical region)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

from activations import ALL_ACTIVATIONS


def plot_all_activations():
    """Full comparison: all activation functions + derivatives."""
    x = np.linspace(-5, 5, 1000)

    colors = [
        "#6b7280", "#2563eb", "#dc2626", "#16a34a", "#9333ea",
        "#ea580c", "#be185d", "#0891b2", "#a16207",
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    for idx, (act, color) in enumerate(zip(ALL_ACTIVATIONS, colors)):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        y_func = act.forward(x)
        ax.plot(x, y_func, color=color, linewidth=3, label="σ(x)")

        if act.derivative is not None:
            y_deriv = act.derivative(x)
            ax.plot(x, y_deriv, color=color, linewidth=2, linestyle="--",
                    alpha=0.6, label="σ'(x)")

        ax.set_title(act.name, fontsize=14, fontweight="bold", color=color)
        ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color="black", linewidth=0.5, alpha=0.3)
        ax.grid(True, alpha=0.15)
        ax.legend(fontsize=10, loc="upper left")
        ax.set_xlim(-5, 5)

        # Add range annotation
        if act.output_range[1] == float('inf'):
            range_str = f"Range: [{act.output_range[0]}, ∞)"
        elif act.output_range[0] == float('-inf'):
            range_str = f"Range: (-∞, ∞)"
        else:
            range_str = f"Range: [{act.output_range[0]}, {act.output_range[1]}]"

        props_str = range_str
        if act.zero_centered:
            props_str += "  |  Zero-centered ✓"
        if act.smooth:
            props_str += "  |  Smooth ✓"

        ax.text(0.02, 0.02, props_str, transform=ax.transAxes,
                fontsize=8, color="gray", style="italic")

    fig.suptitle(
        "Complete Activation Function Reference — AML Lecture 1, TUHH",
        fontsize=18, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig("figures/all_activations.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ figures/all_activations.png saved")


def plot_zoomed_comparison():
    """
    Zoomed-in view near x=0 — where activation behavior differs most.
    This is the critical region for gradient flow.
    """
    x = np.linspace(-2, 2, 500)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    key_acts = [
        ("Sigmoid", ALL_ACTIVATIONS[1], "#2563eb"),
        ("TanH", ALL_ACTIVATIONS[2], "#dc2626"),
        ("ReLU", ALL_ACTIVATIONS[3], "#16a34a"),
        ("Leaky ReLU", ALL_ACTIVATIONS[4], "#9333ea"),
        ("ELU", ALL_ACTIVATIONS[5], "#ea580c"),
        ("Swish", ALL_ACTIVATIONS[7], "#0891b2"),
    ]

    # Functions
    ax = axes[0]
    for name, act, color in key_acts:
        y = act.forward(x)
        ax.plot(x, y, color=color, linewidth=2.5, label=name)

    ax.plot(x, x, color="gray", linewidth=1, linestyle=":", alpha=0.5, label="y = x (identity)")
    ax.set_title("Activation Functions — Zoomed Near Origin", fontsize=14, fontweight="bold")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("σ(x)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 2)
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color="black", linewidth=0.5, alpha=0.3)

    # Derivatives
    ax = axes[1]
    for name, act, color in key_acts:
        if act.derivative is not None:
            y = act.derivative(x)
            ax.plot(x, y, color=color, linewidth=2.5, label=f"{name}'")

    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(y=0.25, color="#2563eb", linestyle=":", alpha=0.4, linewidth=1)
    ax.annotate("Sigmoid max = 0.25", xy=(0.5, 0.25), fontsize=9, color="#2563eb", alpha=0.7)
    ax.annotate("Ideal gradient = 1", xy=(0.5, 1.02), fontsize=9, color="gray", alpha=0.7)

    ax.set_title("Derivatives — Where Gradients Flow", fontsize=14, fontweight="bold")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("σ'(x)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.2, 1.3)

    plt.tight_layout()
    plt.savefig("figures/zoomed_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ figures/zoomed_comparison.png saved")


def plot_properties_table():
    """Generate a visual comparison table of activation properties."""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis("off")

    headers = ["Function", "Range", "Zero-\ncentered", "Smooth", "Monotonic", "Max σ'", "Best for"]
    rows = [
        ["Heaviside", "[0, 1]", "✗", "✗", "✓", "0", "—"],
        ["Sigmoid", "(0, 1)", "✗", "✓", "✓", "0.25", "Output (binary)"],
        ["TanH", "(-1, 1)", "✓", "✓", "✓", "1.0", "RNNs, hidden layers"],
        ["ReLU", "[0, ∞)", "✗", "✗", "✓", "1.0", "Default for CNNs/FNNs"],
        ["Leaky ReLU", "(-∞, ∞)", "~", "✗", "✓", "1.0", "Avoiding dead neurons"],
        ["ELU", "(-α, ∞)", "✓", "✓", "✓", "1.0", "Noise-robust training"],
        ["SoftPlus", "(0, ∞)", "✗", "✓", "✓", "1.0", "Smooth ReLU approx."],
        ["Swish", "(-0.28, ∞)", "~", "✓", "✗", "1.1", "SOTA deep networks"],
    ]

    colors_row = ["#f3f4f6", "white"] * 4
    header_color = "#1e293b"

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style headers
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold", fontsize=11)
        cell.set_edgecolor("white")

    # Style rows
    row_colors_map = [
        "#6b7280", "#2563eb", "#dc2626", "#16a34a",
        "#9333ea", "#ea580c", "#be185d", "#0891b2",
    ]
    for i in range(len(rows)):
        for j in range(len(headers)):
            cell = table[i + 1, j]
            cell.set_facecolor(colors_row[i % 2])
            cell.set_edgecolor("#e5e7eb")
            if j == 0:
                cell.set_text_props(fontweight="bold", color=row_colors_map[i])

    fig.suptitle(
        "Activation Function Properties — Quick Reference",
        fontsize=16, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("figures/properties_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ figures/properties_table.png saved")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    print("Generating figures...\n")
    plot_all_activations()
    plot_zoomed_comparison()
    plot_properties_table()
    print("\n✨ All figures generated!")
