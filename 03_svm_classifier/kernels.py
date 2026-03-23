"""
Kernel Functions for SVM
=========================
Based on: Zemke, AML Lecture 1, Slides 8-14

The kernel trick allows SVM to classify non-linearly separable data
by mapping inputs into a higher-dimensional space WITHOUT explicitly
computing the transformation.

    K(x, z) = φ(x)ᵀ · φ(z)

Instead of computing φ(x) and then the dot product, we compute K directly.
"""

import numpy as np


def linear_kernel(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Linear Kernel: K(x, z) = xᵀz
    
    No transformation — equivalent to standard dot product.
    Use for linearly separable data.
    """
    return x @ z.T


def polynomial_kernel(x: np.ndarray, z: np.ndarray, degree: int = 3, c: float = 1.0) -> np.ndarray:
    """
    Polynomial Kernel: K(x, z) = (xᵀz + c)^d
    
    Maps to feature space of all monomials up to degree d.
    - degree=2: quadratic decision boundaries (circles, ellipses)
    - degree=3: cubic decision boundaries
    - c: free parameter (c=0 → homogeneous polynomial)
    """
    return (x @ z.T + c) ** degree


def rbf_kernel(x: np.ndarray, z: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    RBF (Radial Basis Function) / Gaussian Kernel:
    K(x, z) = exp(-γ · ||x - z||²)
    
    Maps to INFINITE-dimensional feature space.
    - Small γ: smooth, wide decision boundaries (underfitting)
    - Large γ: complex, tight boundaries (overfitting)
    
    This is the most popular kernel for non-linear SVM.
    """
    # Efficient computation of pairwise squared distances
    # ||x - z||² = ||x||² + ||z||² - 2xᵀz
    x_sq = np.sum(x ** 2, axis=1, keepdims=True)
    z_sq = np.sum(z ** 2, axis=1, keepdims=True)
    dist_sq = x_sq + z_sq.T - 2 * (x @ z.T)
    return np.exp(-gamma * dist_sq)


def sigmoid_kernel(x: np.ndarray, z: np.ndarray, alpha: float = 0.01, c: float = 0.0) -> np.ndarray:
    """
    Sigmoid Kernel: K(x, z) = tanh(α · xᵀz + c)
    
    Related to neural networks (two-layer perceptron).
    Note: NOT a valid Mercer kernel for all α, c values.
    """
    return np.tanh(alpha * (x @ z.T) + c)


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════
KERNELS = {
    "linear":     linear_kernel,
    "polynomial": polynomial_kernel,
    "rbf":        rbf_kernel,
    "sigmoid":    sigmoid_kernel,
}


def get_kernel(name: str):
    """Return kernel function by name."""
    if name not in KERNELS:
        raise ValueError(f"Unknown kernel '{name}'. Choose from: {list(KERNELS.keys())}")
    return KERNELS[name]
