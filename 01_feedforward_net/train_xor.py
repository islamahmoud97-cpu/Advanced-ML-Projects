"""
Demo 1: XOR Problem
=====================
XOR is NOT linearly separable — you need at least one hidden layer.
This is the classic proof that a single-layer perceptron is insufficient.

Reference: Zemke, AML Lecture 1, Slide 9 (Boolean functions)

Expected output:
    Input [0,0] → ~0   (target: 0)
    Input [0,1] → ~1   (target: 1)
    Input [1,0] → ~1   (target: 1)
    Input [1,1] → ~0   (target: 0)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from feedforward_net import FeedforwardNet

# ── Data ──────────────────────────────────────────────────────────────────
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]], dtype=np.float64)

y = np.array([[1, 0, 0, 1],   # class 0 (XOR = 0)
              [0, 1, 1, 0]],   # class 1 (XOR = 1)
             dtype=np.float64)

# ── Network: 2 → 8 → 2 ──────────────────────────────────────────────────
net = FeedforwardNet(
    layer_sizes=[2, 8, 2],
    activation="tanh",
    output="softmax",
    init_method="xavier",
    seed=42,
)
net.summary()

# ── Train ─────────────────────────────────────────────────────────────────
print("\nTraining on XOR problem...")
history = net.train(
    X, y,
    epochs=500,
    batch_size=4,
    optimizer="adam",
    lr=0.01,
    loss_fn="cross_entropy",
    verbose=True,
)

# ── Test ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 40)
print("Results:")
print("=" * 40)
for i in range(4):
    pred = net.forward(X[:, i : i + 1])
    label = np.argmax(pred, axis=0)[0]
    confidence = pred[label, 0]
    target = np.argmax(y[:, i])
    status = "✓" if label == target else "✗"
    print(
        f"  Input: [{X[0,i]:.0f}, {X[1,i]:.0f}]  →  "
        f"XOR = {label}  (confidence: {confidence:.4f})  "
        f"target: {target}  {status}"
    )

# ── Plot: Training Loss ──────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
axes[0].plot(history["train_loss"], color="#2563eb", linewidth=2)
axes[0].set_xlabel("Epoch", fontsize=12)
axes[0].set_ylabel("Cross-Entropy Loss", fontsize=12)
axes[0].set_title("Training Loss — XOR Problem", fontsize=14, fontweight="bold")
axes[0].grid(True, alpha=0.3)

# Decision boundary
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()].T
Z = net.predict(grid).reshape(xx.shape)

axes[1].contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=["#dbeafe", "#fecaca"], alpha=0.7)
axes[1].contour(xx, yy, Z, levels=[0.5], colors=["#1e40af"], linewidths=2)

# Plot data points
colors = ["#2563eb", "#dc2626"]
labels_text = ["XOR = 0", "XOR = 1"]
for cls in [0, 1]:
    mask = np.argmax(y, axis=0) == cls
    axes[1].scatter(
        X[0, mask], X[1, mask],
        c=colors[cls], s=200, edgecolors="white", linewidths=2,
        label=labels_text[cls], zorder=5,
    )

axes[1].set_xlabel("x₁", fontsize=12)
axes[1].set_ylabel("x₂", fontsize=12)
axes[1].set_title("Decision Boundary — XOR", fontsize=14, fontweight="bold")
axes[1].legend(fontsize=11)
axes[1].set_xlim(-0.5, 1.5)
axes[1].set_ylim(-0.5, 1.5)
axes[1].set_aspect("equal")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/xor_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n📊 Plot saved to figures/xor_results.png")
