"""
Activation Functions & Their Derivatives
=========================================
Based on: Zemke, Advanced Machine Learning, TUHH WS 2025/26
Lecture 1, Slides 18-28

Each function operates element-wise on NumPy arrays.
Derivatives are needed for backpropagation (Lecture 2).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Sigmoid / Logistic  —  σ(x) = 1 / (1 + e^{-x})
# Derivative: σ'(x) = σ(x) · (1 - σ(x))        [L1, Slide 19]
# ---------------------------------------------------------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    z_safe = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_safe))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1.0 - s)


# ---------------------------------------------------------------------------
# Hyperbolic Tangent  —  tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})
# Derivative: tanh'(x) = 1 - tanh²(x)            [L1, Slide 21]
# ---------------------------------------------------------------------------
def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def tanh_derivative(z: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(z) ** 2


# ---------------------------------------------------------------------------
# ReLU  —  max(0, x)
# Derivative: Heaviside (1 if x > 0, else 0)      [L1, Slide 23]
# ---------------------------------------------------------------------------
def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.float64)


# ---------------------------------------------------------------------------
# Leaky ReLU  —  max(αx, x)  with α = 0.01
# Derivative: 1 if x > 0, else α                  [L1, Slide 24]
# ---------------------------------------------------------------------------
def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(z > 0, z, alpha * z)


def leaky_relu_derivative(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(z > 0, 1.0, alpha)


# ---------------------------------------------------------------------------
# ELU  —  x if x > 0, else α(e^x - 1)            [L1, Slide 25]
# ---------------------------------------------------------------------------
def elu(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(z > 0, z, alpha * (np.exp(z) - 1.0))


def elu_derivative(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(z > 0, 1.0, alpha * np.exp(z))


# ---------------------------------------------------------------------------
# Swish / SiLU  —  x · σ(x)                       [L1, Slide 26]
# ---------------------------------------------------------------------------
def swish(z: np.ndarray) -> np.ndarray:
    return z * sigmoid(z)


def swish_derivative(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s + z * s * (1.0 - s)


# ---------------------------------------------------------------------------
# SoftPlus  —  ln(1 + e^x)                        [L1, Slide 25]
# ---------------------------------------------------------------------------
def softplus(z: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)


def softplus_derivative(z: np.ndarray) -> np.ndarray:
    return sigmoid(z)


# ---------------------------------------------------------------------------
# Softmax  —  e^{z_i} / Σ e^{z_j}  (for output layer classification)
# Applied column-wise (each column = one sample)
# ---------------------------------------------------------------------------
def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - np.max(z, axis=0, keepdims=True))  # numerical stability
    return e / np.sum(e, axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# Registry: map names to (function, derivative) pairs
# ---------------------------------------------------------------------------
ACTIVATIONS = {
    "sigmoid":    (sigmoid, sigmoid_derivative),
    "tanh":       (tanh, tanh_derivative),
    "relu":       (relu, relu_derivative),
    "leaky_relu": (leaky_relu, leaky_relu_derivative),
    "elu":        (elu, elu_derivative),
    "swish":      (swish, swish_derivative),
    "softplus":   (softplus, softplus_derivative),
}


def get_activation(name: str):
    """Return (activation_fn, derivative_fn) by name."""
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Choose from: {list(ACTIVATIONS.keys())}"
        )
    return ACTIVATIONS[name]
