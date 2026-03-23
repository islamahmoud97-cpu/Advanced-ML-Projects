"""
Interactive Activation Function Explorer
==========================================
Based on: Zemke, AML Lecture 1, Slides 18-28

Run this script to interactively explore activation functions:
  - Toggle functions on/off
  - See the function, its derivative, and properties side by side
  - Zoom in/out to observe behavior near zero and at extremes

Usage:
    python explorer.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, RadioButtons
from activations import ALL_ACTIVATIONS


def run_explorer():
    """Launch the interactive activation function explorer."""

    x = np.linspace(-6, 6, 1000)

    # Color palette
    colors = [
        "#6b7280",  # Heaviside — gray
        "#2563eb",  # Sigmoid — blue
        "#dc2626",  # TanH — red
        "#16a34a",  # ReLU — green
        "#9333ea",  # Leaky ReLU — purple
        "#ea580c",  # ELU — orange
        "#be185d",  # SoftPlus — pink
        "#0891b2",  # Swish — teal
        "#a16207",  # Abs — brown
    ]

    # ── Setup Figure ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor("#fafafa")
    fig.suptitle(
        "Activation Function Explorer  —  AML Lecture 1",
        fontsize=18, fontweight="bold", y=0.98,
    )

    # Layout: [checkboxes | function plot | derivative plot]
    ax_check = fig.add_axes([0.01, 0.15, 0.14, 0.75])
    ax_func  = fig.add_axes([0.20, 0.12, 0.37, 0.78])
    ax_deriv = fig.add_axes([0.62, 0.12, 0.37, 0.78])

    # ── Checkboxes ────────────────────────────────────────────────────
    labels = [a.name for a in ALL_ACTIVATIONS]
    # Start with Sigmoid, TanH, ReLU, Swish visible
    initial_visible = [False, True, True, True, False, False, False, True, False]

    check = CheckButtons(ax_check, labels, initial_visible)
    ax_check.set_title("Toggle Functions", fontsize=11, fontweight="bold")

    # Color the checkbox labels
    for i, (label, color) in enumerate(zip(check.labels, colors)):
        label.set_color(color)
        label.set_fontsize(10)
        label.set_fontweight("bold")

    # ── Plot lines ────────────────────────────────────────────────────
    func_lines = []
    deriv_lines = []

    for i, act in enumerate(ALL_ACTIVATIONS):
        visible = initial_visible[i]
        y_func = act.forward(x)
        lf, = ax_func.plot(x, y_func, color=colors[i], linewidth=2.5,
                           label=act.name, visible=visible, alpha=0.85)
        func_lines.append(lf)

        if act.derivative is not None:
            y_deriv = act.derivative(x)
        else:
            y_deriv = np.zeros_like(x)
        ld, = ax_deriv.plot(x, y_deriv, color=colors[i], linewidth=2.5,
                            label=f"{act.name}'", visible=visible, alpha=0.85)
        deriv_lines.append(ld)

    # ── Style axes ────────────────────────────────────────────────────
    for ax, title in [(ax_func, "σ(x) — Activation Functions"),
                      (ax_deriv, "σ'(x) — Derivatives (for Backprop)")]:
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("x", fontsize=12)
        ax.axhline(y=0, color="black", linewidth=0.6, alpha=0.5)
        ax.axvline(x=0, color="black", linewidth=0.6, alpha=0.5)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(-6, 6)

    ax_func.set_ylim(-2, 4)
    ax_func.set_ylabel("σ(x)", fontsize=12)
    ax_deriv.set_ylim(-0.5, 1.5)
    ax_deriv.set_ylabel("σ'(x)", fontsize=12)

    # Reference lines
    for val in [0, 1, -1]:
        ax_func.axhline(y=val, color="gray", linewidth=0.4, linestyle="--", alpha=0.4)
    ax_deriv.axhline(y=1, color="gray", linewidth=0.4, linestyle="--", alpha=0.4)
    ax_deriv.axhline(y=0.25, color="gray", linewidth=0.4, linestyle="--", alpha=0.3)
    ax_deriv.annotate("max σ'(sigmoid) = 0.25", xy=(3, 0.25), fontsize=8, color="gray", alpha=0.6)

    # ── Interaction ───────────────────────────────────────────────────
    def toggle(label):
        idx = labels.index(label)
        visible = not func_lines[idx].get_visible()
        func_lines[idx].set_visible(visible)
        deriv_lines[idx].set_visible(visible)
        fig.canvas.draw_idle()

    check.on_clicked(toggle)

    # ── Info text ─────────────────────────────────────────────────────
    fig.text(0.01, 0.02,
             "Click checkboxes to toggle functions  |  "
             "Based on Zemke, AML Lecture 1, TUHH WS 2025/26",
             fontsize=9, color="gray", style="italic")

    plt.show()


if __name__ == "__main__":
    run_explorer()
