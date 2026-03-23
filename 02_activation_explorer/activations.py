"""
Activation Functions — Complete Collection
============================================
Based on: Zemke, Advanced Machine Learning, TUHH WS 2025/26
Lecture 1, Slides 18-28

Every activation function includes:
  - Forward pass:   σ(z)
  - Derivative:     σ'(z)   (needed for backpropagation)
  - LaTeX formula for documentation
  - Properties (range, monotonicity, zero-centered, etc.)

All functions operate element-wise on NumPy arrays.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Optional


@dataclass
class ActivationInfo:
    """Metadata for an activation function."""
    name: str
    forward: Callable
    derivative: Callable
    formula: str              # LaTeX-style formula
    derivative_formula: str   # LaTeX-style derivative
    output_range: Tuple[float, float]
    zero_centered: bool
    monotonic: bool
    smooth: bool              # differentiable everywhere?
    year: Optional[int]       # year of introduction
    authors: str
    pros: list
    cons: list


# ═══════════════════════════════════════════════════════════════════════════
# 1. HEAVISIDE / STEP FUNCTION  [L1, Slide 18]
# ═══════════════════════════════════════════════════════════════════════════
def heaviside(z):
    return (z >= 0).astype(np.float64)

def heaviside_deriv(z):
    return np.zeros_like(z, dtype=np.float64)  # 0 everywhere (not useful for backprop)

HEAVISIDE = ActivationInfo(
    name="Heaviside (Step)",
    forward=heaviside, derivative=heaviside_deriv,
    formula=r"$H(x) = \begin{cases} 1 & x \geq 0 \\ 0 & x < 0 \end{cases}$",
    derivative_formula=r"$H'(x) = 0$ (not differentiable at 0)",
    output_range=(0, 1), zero_centered=False, monotonic=True, smooth=False,
    year=1943, authors="McCulloch & Pitts",
    pros=["Simple binary output", "Historical importance"],
    cons=["Zero gradient everywhere → cannot train with backprop", "Not differentiable at 0"],
)


# ═══════════════════════════════════════════════════════════════════════════
# 2. SIGMOID / LOGISTIC  [L1, Slide 19]
# ═══════════════════════════════════════════════════════════════════════════
def sigmoid(z):
    z_safe = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_safe))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1.0 - s)

SIGMOID = ActivationInfo(
    name="Sigmoid (Logistic)",
    forward=sigmoid, derivative=sigmoid_deriv,
    formula=r"$\sigma(x) = \frac{1}{1 + e^{-x}}$",
    derivative_formula=r"$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$",
    output_range=(0, 1), zero_centered=False, monotonic=True, smooth=True,
    year=1943, authors="McCulloch & Pitts / widespread use",
    pros=["Smooth & differentiable", "Output ∈ (0,1) → probability interpretation"],
    cons=["Vanishing gradient for |x| >> 0 (max σ' = 0.25)", "Not zero-centered", "Saturates both ends"],
)


# ═══════════════════════════════════════════════════════════════════════════
# 3. HYPERBOLIC TANGENT (TanH)  [L1, Slide 21]
# ═══════════════════════════════════════════════════════════════════════════
def tanh_fn(z):
    return np.tanh(z)

def tanh_deriv(z):
    return 1.0 - np.tanh(z) ** 2

TANH = ActivationInfo(
    name="TanH",
    forward=tanh_fn, derivative=tanh_deriv,
    formula=r"$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$",
    derivative_formula=r"$\tanh'(x) = 1 - \tanh^2(x)$",
    output_range=(-1, 1), zero_centered=True, monotonic=True, smooth=True,
    year=None, authors="Classical mathematics",
    pros=["Zero-centered → faster convergence", "Stronger gradients than sigmoid (max = 1.0)"],
    cons=["Still suffers from vanishing gradient for |x| >> 0", "Saturates both ends"],
)


# ═══════════════════════════════════════════════════════════════════════════
# 4. ReLU (Rectified Linear Unit)  [L1, Slide 23]
# ═══════════════════════════════════════════════════════════════════════════
def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(np.float64)

RELU = ActivationInfo(
    name="ReLU",
    forward=relu, derivative=relu_deriv,
    formula=r"$\text{ReLU}(x) = \max(0, x)$",
    derivative_formula=r"$\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$",
    output_range=(0, float('inf')), zero_centered=False, monotonic=True, smooth=False,
    year=2000, authors="Hahnloser et al. (2000), popularized by Nair & Hinton (2010)",
    pros=["No vanishing gradient for x > 0", "Computationally efficient", "Sparse activation"],
    cons=["Dying ReLU: neurons with x < 0 have zero gradient forever", "Not zero-centered", "Not differentiable at 0"],
)


# ═══════════════════════════════════════════════════════════════════════════
# 5. LEAKY ReLU  [L1, Slide 24]
# ═══════════════════════════════════════════════════════════════════════════
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_deriv(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)

LEAKY_RELU = ActivationInfo(
    name="Leaky ReLU",
    forward=leaky_relu, derivative=leaky_relu_deriv,
    formula=r"$\text{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}$  ($\alpha = 0.01$)",
    derivative_formula=r"$\text{LeakyReLU}'(x) = \begin{cases} 1 & x > 0 \\ \alpha & x \leq 0 \end{cases}$",
    output_range=(float('-inf'), float('inf')), zero_centered=True, monotonic=True, smooth=False,
    year=2013, authors="Maas et al.",
    pros=["Fixes dying ReLU problem (small gradient for x < 0)", "Nearly as fast as ReLU"],
    cons=["α is a hyperparameter", "Not smooth at x = 0"],
)


# ═══════════════════════════════════════════════════════════════════════════
# 6. ELU (Exponential Linear Unit)  [L1, Slide 25]
# ═══════════════════════════════════════════════════════════════════════════
def elu(z, alpha=1.0):
    return np.where(z > 0, z, alpha * (np.exp(z) - 1.0))

def elu_deriv(z, alpha=1.0):
    return np.where(z > 0, 1.0, alpha * np.exp(z))

ELU = ActivationInfo(
    name="ELU",
    forward=elu, derivative=elu_deriv,
    formula=r"$\text{ELU}(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}$",
    derivative_formula=r"$\text{ELU}'(x) = \begin{cases} 1 & x > 0 \\ \alpha e^x & x \leq 0 \end{cases}$",
    output_range=(-1, float('inf')), zero_centered=True, monotonic=True, smooth=True,
    year=2015, authors="Clevert, Unterthiner & Hochreiter",
    pros=["Smooth everywhere", "Zero-centered mean activations", "Robust to noise"],
    cons=["Slower than ReLU (exp computation)", "Saturates for large negative values"],
)


# ═══════════════════════════════════════════════════════════════════════════
# 7. SOFTPLUS  [L1, Slide 25]
# ═══════════════════════════════════════════════════════════════════════════
def softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)  # numerically stable

def softplus_deriv(z):
    return sigmoid(z)  # SoftPlus' = Sigmoid!

SOFTPLUS = ActivationInfo(
    name="SoftPlus",
    forward=softplus, derivative=softplus_deriv,
    formula=r"$\text{SoftPlus}(x) = \ln(1 + e^x)$",
    derivative_formula=r"$\text{SoftPlus}'(x) = \sigma(x)$ (= Sigmoid!)",
    output_range=(0, float('inf')), zero_centered=False, monotonic=True, smooth=True,
    year=2001, authors="Dugas et al.",
    pros=["Smooth approximation of ReLU", "Always differentiable", "Derivative = sigmoid"],
    cons=["Slower than ReLU", "Not zero-centered"],
)


# ═══════════════════════════════════════════════════════════════════════════
# 8. SWISH / SiLU  [L1, Slide 26]
# ═══════════════════════════════════════════════════════════════════════════
def swish(z):
    return z * sigmoid(z)

def swish_deriv(z):
    s = sigmoid(z)
    return s + z * s * (1.0 - s)

SWISH = ActivationInfo(
    name="Swish (SiLU)",
    forward=swish, derivative=swish_deriv,
    formula=r"$\text{Swish}(x) = x \cdot \sigma(x)$",
    derivative_formula=r"$\text{Swish}'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x))$",
    output_range=(-0.278, float('inf')), zero_centered=True, monotonic=False, smooth=True,
    year=2017, authors="Ramachandran, Zoph & Le (Google Brain)",
    pros=["Self-gated (no extra params)", "Smooth & non-monotonic", "Found via NAS (neural architecture search)"],
    cons=["Slightly more expensive than ReLU", "Non-monotonic (unusual)"],
)


# ═══════════════════════════════════════════════════════════════════════════
# 9. SOFTMAX  (Output layer for classification)
# ═══════════════════════════════════════════════════════════════════════════
def softmax(z):
    e = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e / np.sum(e, axis=0, keepdims=True)

SOFTMAX_INFO = ActivationInfo(
    name="Softmax",
    forward=softmax, derivative=None,
    formula=r"$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$",
    derivative_formula=r"Jacobian matrix (not element-wise)",
    output_range=(0, 1), zero_centered=False, monotonic=True, smooth=True,
    year=None, authors="Boltzmann distribution / widespread",
    pros=["Outputs form a probability distribution (sum = 1)", "Standard for multi-class classification"],
    cons=["Only for output layer", "Sensitive to large inputs (overflow)"],
)


# ═══════════════════════════════════════════════════════════════════════════
# 10. ABS FAMILY  [L1, Slide 27]
# ═══════════════════════════════════════════════════════════════════════════
def abs_fn(z):
    return np.abs(z)

def abs_deriv(z):
    return np.sign(z).astype(np.float64)

ABS = ActivationInfo(
    name="|x| (Absolute Value)",
    forward=abs_fn, derivative=abs_deriv,
    formula=r"$|x|$",
    derivative_formula=r"$\text{sign}(x)$",
    output_range=(0, float('inf')), zero_centered=False, monotonic=False, smooth=False,
    year=None, authors="Classical",
    pros=["Symmetric", "Simple"],
    cons=["Not monotonic", "Not differentiable at 0", "Rarely used in practice"],
)


# ═══════════════════════════════════════════════════════════════════════════
# REGISTRY — All activation functions in one place
# ═══════════════════════════════════════════════════════════════════════════
ALL_ACTIVATIONS = [
    HEAVISIDE, SIGMOID, TANH, RELU, LEAKY_RELU,
    ELU, SOFTPLUS, SWISH, ABS,
]

# Only the commonly used ones (for training demos)
TRAINABLE_ACTIVATIONS = {
    "sigmoid":    (sigmoid, sigmoid_deriv),
    "tanh":       (tanh_fn, tanh_deriv),
    "relu":       (relu, relu_deriv),
    "leaky_relu": (leaky_relu, leaky_relu_deriv),
    "elu":        (elu, elu_deriv),
    "swish":      (swish, swish_deriv),
    "softplus":   (softplus, softplus_deriv),
}
