"""
Optimizers — Variants of Stochastic Gradient Descent
======================================================
Based on: Zemke, Advanced Machine Learning, TUHH WS 2025/26
Lecture 3, Slides 6-22

Each optimizer stores its internal state and applies a parameter update rule.
"""

import numpy as np
from typing import List


class SGD:
    """
    Vanilla Stochastic Gradient Descent
    p ← p - η · ∇C(p)                            [L2, Slide 20]
    """

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def initialize(self, weights: List[np.ndarray], biases: List[np.ndarray]):
        pass  # No state needed

    def update(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        dW: List[np.ndarray],
        db: List[np.ndarray],
    ):
        for i in range(len(weights)):
            weights[i] -= self.lr * dW[i]
            biases[i] -= self.lr * db[i]


class Momentum:
    """
    SGD with Momentum                              [L3, Slide 8]
    v ← γ·v + η·∇C(p)
    p ← p - v
    Typical: γ = 0.9
    """

    def __init__(self, lr: float = 0.01, gamma: float = 0.9):
        self.lr = lr
        self.gamma = gamma
        self.vW = None
        self.vb = None

    def initialize(self, weights: List[np.ndarray], biases: List[np.ndarray]):
        self.vW = [np.zeros_like(w) for w in weights]
        self.vb = [np.zeros_like(b) for b in biases]

    def update(self, weights, biases, dW, db):
        for i in range(len(weights)):
            self.vW[i] = self.gamma * self.vW[i] + self.lr * dW[i]
            self.vb[i] = self.gamma * self.vb[i] + self.lr * db[i]
            weights[i] -= self.vW[i]
            biases[i] -= self.vb[i]


class NAG:
    """
    Nesterov Accelerated Gradient                  [L3, Slide 10]
    Compute gradient at the "look-ahead" position.
    Better convergence for RNNs.
    """

    def __init__(self, lr: float = 0.01, gamma: float = 0.9):
        self.lr = lr
        self.gamma = gamma
        self.vW = None
        self.vb = None

    def initialize(self, weights, biases):
        self.vW = [np.zeros_like(w) for w in weights]
        self.vb = [np.zeros_like(b) for b in biases]

    def update(self, weights, biases, dW, db):
        for i in range(len(weights)):
            vW_prev = self.vW[i].copy()
            vb_prev = self.vb[i].copy()
            self.vW[i] = self.gamma * self.vW[i] + self.lr * dW[i]
            self.vb[i] = self.gamma * self.vb[i] + self.lr * db[i]
            weights[i] -= -self.gamma * vW_prev + (1 + self.gamma) * self.vW[i]
            biases[i] -= -self.gamma * vb_prev + (1 + self.gamma) * self.vb[i]


class RMSprop:
    """
    RMSprop — Hinton (unpublished)                 [L3, Slide 16]
    Exponentially decaying average of squared gradients.
    s ← β·s + (1-β)·(∇C)²
    p ← p - η · ∇C / (√s + ε)
    """

    def __init__(self, lr: float = 0.001, beta: float = 0.9, eps: float = 1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.sW = None
        self.sb = None

    def initialize(self, weights, biases):
        self.sW = [np.zeros_like(w) for w in weights]
        self.sb = [np.zeros_like(b) for b in biases]

    def update(self, weights, biases, dW, db):
        for i in range(len(weights)):
            self.sW[i] = self.beta * self.sW[i] + (1 - self.beta) * dW[i] ** 2
            self.sb[i] = self.beta * self.sb[i] + (1 - self.beta) * db[i] ** 2
            weights[i] -= self.lr * dW[i] / (np.sqrt(self.sW[i]) + self.eps)
            biases[i] -= self.lr * db[i] / (np.sqrt(self.sb[i]) + self.eps)


class Adam:
    """
    Adam — Adaptive Moment Estimation               [L3, Slide 18]
    Combines Momentum + RMSprop + Bias correction.

    m ← β₁·m + (1-β₁)·∇C          (1st moment — mean)
    v ← β₂·v + (1-β₂)·(∇C)²       (2nd moment — variance)
    m̂ = m / (1-β₁ᵗ)                (bias correction)
    v̂ = v / (1-β₂ᵗ)
    p ← p - η · m̂ / (√v̂ + ε)

    Defaults: η=0.001, β₁=0.9, β₂=0.999, ε=1e-8
    "Standard in TensorFlow" — L3
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.mW = None
        self.mb = None
        self.vW = None
        self.vb = None

    def initialize(self, weights, biases):
        self.t = 0
        self.mW = [np.zeros_like(w) for w in weights]
        self.mb = [np.zeros_like(b) for b in biases]
        self.vW = [np.zeros_like(w) for w in weights]
        self.vb = [np.zeros_like(b) for b in biases]

    def update(self, weights, biases, dW, db):
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)

        for i in range(len(weights)):
            # First moment
            self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * dW[i]
            self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * db[i]
            # Second moment
            self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * dW[i] ** 2
            self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * db[i] ** 2
            # Update (bias correction folded into lr_t)
            weights[i] -= lr_t * self.mW[i] / (np.sqrt(self.vW[i]) + self.eps)
            biases[i] -= lr_t * self.mb[i] / (np.sqrt(self.vb[i]) + self.eps)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
OPTIMIZERS = {
    "sgd":      SGD,
    "momentum": Momentum,
    "nag":      NAG,
    "rmsprop":  RMSprop,
    "adam":      Adam,
}


def get_optimizer(name: str, **kwargs):
    """Create optimizer by name with keyword arguments."""
    if name not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{name}'. Choose from: {list(OPTIMIZERS.keys())}"
        )
    return OPTIMIZERS[name](**kwargs)
