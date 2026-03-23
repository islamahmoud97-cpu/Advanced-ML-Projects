"""
Loss Functions & Their Derivatives
====================================
Based on: Zemke, Advanced Machine Learning, TUHH WS 2025/26
Lecture 2, Slide 14

Loss functions measure the discrepancy between the network output f(x)
and the target y. The derivative is needed for backpropagation.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Mean Squared Error (MSE)
# C(p) = 1/(2N) · Σ_j ||f(x_j) - y_j||²       [L2, Slide 14]
# ---------------------------------------------------------------------------
def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    N = y_true.shape[1]
    return (1.0 / (2.0 * N)) * np.sum((y_pred - y_true) ** 2)


def mse_derivative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """∂C/∂a_m = (a_m - y) / N"""
    N = y_true.shape[1]
    return (y_pred - y_true) / N


# ---------------------------------------------------------------------------
# Cross-Entropy Loss (for classification with softmax output)
# C = -1/N · Σ_j Σ_k y_{jk} · ln(ŷ_{jk})
# ---------------------------------------------------------------------------
def cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    N = y_true.shape[1]
    eps = 1e-12  # avoid log(0)
    return -(1.0 / N) * np.sum(y_true * np.log(y_pred + eps))


def cross_entropy_derivative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Combined with softmax output layer:
    ∂C/∂z_m = a_m - y   (softmax + cross-entropy simplification)
    """
    N = y_true.shape[1]
    return (y_pred - y_true) / N


# ---------------------------------------------------------------------------
# Binary Cross-Entropy (for single-output binary classification)
# C = -1/N · Σ [y·ln(ŷ) + (1-y)·ln(1-ŷ)]
# ---------------------------------------------------------------------------
def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    N = y_true.shape[1]
    eps = 1e-12
    return -(1.0 / N) * np.sum(
        y_true * np.log(y_pred + eps) + (1.0 - y_true) * np.log(1.0 - y_pred + eps)
    )


def binary_cross_entropy_derivative(
    y_pred: np.ndarray, y_true: np.ndarray
) -> np.ndarray:
    eps = 1e-12
    N = y_true.shape[1]
    return (1.0 / N) * (-(y_true / (y_pred + eps)) + (1.0 - y_true) / (1.0 - y_pred + eps))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
LOSSES = {
    "mse":            (mse, mse_derivative),
    "cross_entropy":  (cross_entropy, cross_entropy_derivative),
    "bce":            (binary_cross_entropy, binary_cross_entropy_derivative),
}


def get_loss(name: str):
    """Return (loss_fn, derivative_fn) by name."""
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {list(LOSSES.keys())}")
    return LOSSES[name]
