"""
Dataset Generators
===================
Generate 2D datasets for SVM classification demos.
"""

import numpy as np


def make_linear(n_samples=200, noise=0.3, seed=42):
    """Linearly separable data — perfect for linear SVM."""
    np.random.seed(seed)
    n = n_samples // 2
    X1 = np.random.randn(n, 2) * noise + np.array([1.5, 1.5])
    X2 = np.random.randn(n, 2) * noise + np.array([-1.5, -1.5])
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n), -np.ones(n)])
    return X, y


def make_xor(n_samples=200, noise=0.3, seed=42):
    """
    XOR pattern — NOT linearly separable.
    Requires a non-linear kernel (RBF or polynomial).
    [Lecture 1, Slide 9: XOR cannot be solved by single hyperplane]
    """
    np.random.seed(seed)
    n = n_samples // 4
    X1 = np.random.randn(n, 2) * noise + np.array([1, 1])
    X2 = np.random.randn(n, 2) * noise + np.array([-1, -1])
    X3 = np.random.randn(n, 2) * noise + np.array([1, -1])
    X4 = np.random.randn(n, 2) * noise + np.array([-1, 1])
    X = np.vstack([X1, X2, X3, X4])
    y = np.hstack([np.ones(n), np.ones(n), -np.ones(n), -np.ones(n)])
    return X, y


def make_circles(n_samples=200, noise=0.1, seed=42):
    """Concentric circles — requires RBF kernel."""
    np.random.seed(seed)
    n = n_samples // 2
    theta_outer = np.random.uniform(0, 2 * np.pi, n)
    r_outer = 2.0 + np.random.randn(n) * noise
    X_outer = np.column_stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)])

    theta_inner = np.random.uniform(0, 2 * np.pi, n)
    r_inner = 0.5 + np.random.randn(n) * noise
    X_inner = np.column_stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)])

    X = np.vstack([X_outer, X_inner])
    y = np.hstack([-np.ones(n), np.ones(n)])
    return X, y


def make_moons(n_samples=200, noise=0.15, seed=42):
    """Two interleaving half-circles."""
    np.random.seed(seed)
    n = n_samples // 2
    theta1 = np.linspace(0, np.pi, n)
    X1 = np.column_stack([np.cos(theta1), np.sin(theta1)]) + np.random.randn(n, 2) * noise
    theta2 = np.linspace(0, np.pi, n)
    X2 = np.column_stack([1 - np.cos(theta2), -np.sin(theta2) + 0.5]) + np.random.randn(n, 2) * noise
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n), -np.ones(n)])
    return X, y


def make_spiral(n_samples=200, noise=0.25, seed=42):
    """Spiral pattern — very challenging, requires RBF with small gamma."""
    np.random.seed(seed)
    n = n_samples // 2
    theta = np.linspace(0, 3 * np.pi, n)
    r = np.linspace(0.5, 3, n)
    X1 = np.column_stack([r * np.cos(theta), r * np.sin(theta)]) + np.random.randn(n, 2) * noise
    X2 = np.column_stack([-r * np.cos(theta), -r * np.sin(theta)]) + np.random.randn(n, 2) * noise
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n), -np.ones(n)])
    return X, y


DATASETS = {
    "linear":  make_linear,
    "xor":     make_xor,
    "circles": make_circles,
    "moons":   make_moons,
    "spiral":  make_spiral,
}
