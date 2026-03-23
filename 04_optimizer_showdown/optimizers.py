"""
Optimizers — From Scratch
===========================
Based on: Zemke, Advanced Machine Learning, TUHH WS 2025/26
Lecture 3, Slides 6-22

Every optimizer implemented with:
  - Pure NumPy (no frameworks)
  - Mathematical formula from the lecture
  - Step-by-step update tracking for visualization

All optimizers follow the same interface:
  1. __init__(lr, ...)    → set hyperparameters
  2. step(grad)           → update parameters, return new position
  3. get_path()           → return full optimization path
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class OptimizerResult:
    """Stores the optimization trajectory for visualization."""
    name: str
    color: str
    path: List[np.ndarray] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)


class SGD:
    """
    Vanilla Stochastic Gradient Descent           [L2, Slide 20]
    
    Update rule:
        p ← p - η · ∇C(p)
    
    The simplest optimizer. Learning rate η must be tuned carefully:
    - Too large → diverges
    - Too small → painfully slow
    """
    
    def __init__(self, lr: float = 0.01):
        self.lr = lr
        self.name = f"SGD (η={lr})"
        self.color = "#6b7280"
    
    def optimize(self, f, grad_f, x0, n_steps=500):
        x = x0.copy()
        result = OptimizerResult(name=self.name, color=self.color)
        result.path.append(x.copy())
        
        for _ in range(n_steps):
            g = grad_f(x)
            x = x - self.lr * g
            result.path.append(x.copy())
            result.losses.append(f(x))
            result.grad_norms.append(np.linalg.norm(g))
        
        return result


class MomentumSGD:
    """
    SGD with Momentum                              [L3, Slide 8]
    
    Fixes: oscillating slow convergence in ravines.
    
    Update rule:
        v ← γ·v + η·∇C(p)
        p ← p - v
    
    The velocity v accumulates past gradients → smoother trajectory.
    Typical: γ = 0.9 (momentum coefficient)
    
    Analogy: a ball rolling downhill gains speed (momentum).
    """
    
    def __init__(self, lr: float = 0.01, gamma: float = 0.9):
        self.lr = lr
        self.gamma = gamma
        self.name = f"Momentum (γ={gamma})"
        self.color = "#2563eb"
    
    def optimize(self, f, grad_f, x0, n_steps=500):
        x = x0.copy()
        v = np.zeros_like(x)
        result = OptimizerResult(name=self.name, color=self.color)
        result.path.append(x.copy())
        
        for _ in range(n_steps):
            g = grad_f(x)
            v = self.gamma * v + self.lr * g
            x = x - v
            result.path.append(x.copy())
            result.losses.append(f(x))
            result.grad_norms.append(np.linalg.norm(g))
        
        return result


class NAG:
    """
    Nesterov Accelerated Gradient                  [L3, Slide 10]
    
    Improvement over Momentum: compute gradient at the "look-ahead" position.
    
    Update rule:
        v ← γ·v + η·∇C(p - γ·v)      ← gradient at predicted future point!
        p ← p - v
    
    "Look before you leap" — evaluates gradient where momentum
    WOULD take us, then corrects.
    
    Better convergence than Momentum, especially for RNNs.
    """
    
    def __init__(self, lr: float = 0.01, gamma: float = 0.9):
        self.lr = lr
        self.gamma = gamma
        self.name = f"NAG (γ={gamma})"
        self.color = "#7c3aed"
    
    def optimize(self, f, grad_f, x0, n_steps=500):
        x = x0.copy()
        v = np.zeros_like(x)
        result = OptimizerResult(name=self.name, color=self.color)
        result.path.append(x.copy())
        
        for _ in range(n_steps):
            # Look-ahead: gradient at (x - γ·v)
            g = grad_f(x - self.gamma * v)
            v = self.gamma * v + self.lr * g
            x = x - v
            result.path.append(x.copy())
            result.losses.append(f(x))
            result.grad_norms.append(np.linalg.norm(g))
        
        return result


class Adagrad:
    """
    Adagrad — Adaptive Gradient                    [L3, Slide 14]
    
    Per-parameter adaptive learning rates.
    
    Update rule:
        s ← s + (∇C)²                 ← accumulate squared gradients
        p ← p - η · ∇C / (√s + ε)    ← scale LR by inverse of accumulated gradient
    
    Parameters with large gradients → smaller effective LR
    Parameters with small gradients → larger effective LR
    
    Problem: s grows monotonically → learning rate → 0 (too aggressive)
    """
    
    def __init__(self, lr: float = 0.1, eps: float = 1e-8):
        self.lr = lr
        self.eps = eps
        self.name = f"Adagrad (η={lr})"
        self.color = "#059669"
    
    def optimize(self, f, grad_f, x0, n_steps=500):
        x = x0.copy()
        s = np.zeros_like(x)
        result = OptimizerResult(name=self.name, color=self.color)
        result.path.append(x.copy())
        
        for _ in range(n_steps):
            g = grad_f(x)
            s = s + g ** 2
            x = x - self.lr * g / (np.sqrt(s) + self.eps)
            result.path.append(x.copy())
            result.losses.append(f(x))
            result.grad_norms.append(np.linalg.norm(g))
        
        return result


class RMSprop:
    """
    RMSprop — Hinton (unpublished lecture notes)   [L3, Slide 16]
    
    Fixes Adagrad's aggressive LR decay with exponential moving average.
    
    Update rule:
        s ← β·s + (1-β)·(∇C)²        ← exponentially decaying average
        p ← p - η · ∇C / (√s + ε)
    
    Only recent gradients matter (controlled by β, typically 0.9).
    This prevents the learning rate from shrinking to zero.
    """
    
    def __init__(self, lr: float = 0.01, beta: float = 0.9, eps: float = 1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.name = f"RMSprop (β={beta})"
        self.color = "#ea580c"
    
    def optimize(self, f, grad_f, x0, n_steps=500):
        x = x0.copy()
        s = np.zeros_like(x)
        result = OptimizerResult(name=self.name, color=self.color)
        result.path.append(x.copy())
        
        for _ in range(n_steps):
            g = grad_f(x)
            s = self.beta * s + (1 - self.beta) * g ** 2
            x = x - self.lr * g / (np.sqrt(s) + self.eps)
            result.path.append(x.copy())
            result.losses.append(f(x))
            result.grad_norms.append(np.linalg.norm(g))
        
        return result


class Adam:
    """
    Adam — Adaptive Moment Estimation               [L3, Slide 18]
    
    Combines Momentum (1st moment) + RMSprop (2nd moment) + bias correction.
    
    Update rule:
        m ← β₁·m + (1-β₁)·∇C         ← 1st moment (mean of gradients)
        v ← β₂·v + (1-β₂)·(∇C)²      ← 2nd moment (variance of gradients)
        m̂ = m / (1 - β₁ᵗ)             ← bias correction
        v̂ = v / (1 - β₂ᵗ)
        p ← p - η · m̂ / (√v̂ + ε)
    
    Defaults: η=0.001, β₁=0.9, β₂=0.999, ε=1e-8
    "Standard in TensorFlow" — Lecture 3
    
    Bias correction is needed because m and v are initialized to 0,
    so they're biased towards 0 in the first few steps.
    """
    
    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.name = f"Adam (η={lr})"
        self.color = "#16a34a"
    
    def optimize(self, f, grad_f, x0, n_steps=500):
        x = x0.copy()
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        result = OptimizerResult(name=self.name, color=self.color)
        result.path.append(x.copy())
        
        for t in range(1, n_steps + 1):
            g = grad_f(x)
            
            # 1st moment (mean)
            m = self.beta1 * m + (1 - self.beta1) * g
            # 2nd moment (uncentered variance)
            v = self.beta2 * v + (1 - self.beta2) * g ** 2
            
            # Bias correction
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            
            # Update
            x = x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
            result.path.append(x.copy())
            result.losses.append(f(x))
            result.grad_norms.append(np.linalg.norm(g))
        
        return result


class Nadam:
    """
    Nadam — Nesterov Adam                          [L3, Slide 20]
    
    Adam + Nesterov look-ahead = Nadam.
    Combines the best of NAG and Adam.
    """
    
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.name = f"Nadam (η={lr})"
        self.color = "#be185d"
    
    def optimize(self, f, grad_f, x0, n_steps=500):
        x = x0.copy()
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        result = OptimizerResult(name=self.name, color=self.color)
        result.path.append(x.copy())
        
        for t in range(1, n_steps + 1):
            g = grad_f(x)
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * g ** 2
            
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            
            # Nesterov correction
            m_nesterov = self.beta1 * m_hat + (1 - self.beta1) * g / (1 - self.beta1 ** t)
            
            x = x - self.lr * m_nesterov / (np.sqrt(v_hat) + self.eps)
            result.path.append(x.copy())
            result.losses.append(f(x))
            result.grad_norms.append(np.linalg.norm(g))
        
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════
ALL_OPTIMIZERS = {
    "sgd":      SGD,
    "momentum": MomentumSGD,
    "nag":      NAG,
    "adagrad":  Adagrad,
    "rmsprop":  RMSprop,
    "adam":      Adam,
    "nadam":    Nadam,
}
