"""
Weight Initialization Methods
================================
Based on: Zemke, AML Lecture 3, Slides 9-14

Three strategies from numerical mathematics (Slide 9):
  1. Zero start    → FAILS (all neurons identical, no learning)
  2. Known values  → Transfer learning, pretrained weights
  3. Random start  → Xavier, He, Orthogonal, LeCun

Key insight: The VARIANCE of initial weights determines whether
signals and gradients explode or vanish through the network.

From Slide 13 (Glorot & Bengio [3], He et al. [4]):
┌───────────────┬──────────────────────────┬────────────────────────┐
│ Activation    │ Uniform on [-r, r]       │ Normal (σ)             │
├───────────────┼──────────────────────────┼────────────────────────┤
│ TanH (Xavier) │ r = √(6/(nᵢₙ + nₒᵤₜ))  │ σ = √(2/(nᵢₙ + nₒᵤₜ))│
│ ReLU (He)     │ r = √(12/(nᵢₙ + nₒᵤₜ)) │ σ = √(4/(nᵢₙ + nₒᵤₜ))│
└───────────────┴──────────────────────────┴────────────────────────┘

Simplified He (commonly used): σ = √(2/nᵢₙ)  [fan-in only]
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class InitInfo:
    """Metadata for an initialization method."""
    name: str
    short_name: str
    formula: str
    best_for: str
    color: str
    year: Optional[int]
    authors: str
    description: str


def zeros(shape: Tuple[int, int]) -> np.ndarray:
    """
    Zero Initialization  [L3, Slide 9]
    
    W = 0 → ALL neurons compute the same output → 
    ALL gradients are identical → symmetry never breaks.
    
    NEVER use this. Included only to demonstrate WHY it fails.
    """
    return np.zeros(shape)

ZEROS_INFO = InitInfo(
    name="Zero Initialization", short_name="Zeros",
    formula="W = 0", best_for="NOTHING (demonstration only)",
    color="#6b7280", year=None, authors="—",
    description="All neurons identical → network cannot learn"
)


def random_normal(shape: Tuple[int, int], std: float = 0.01) -> np.ndarray:
    """
    Naive Random Initialization
    
    W ~ N(0, σ²) with small σ.
    
    Problem: if σ too small → signals vanish in deep networks.
             if σ too large → signals explode.
    """
    return np.random.randn(*shape) * std

RANDOM_INFO = InitInfo(
    name="Random Normal (σ=0.01)", short_name="Random",
    formula="W ~ N(0, 0.01²)", best_for="Very shallow networks only",
    color="#a3a3a3", year=None, authors="—",
    description="Works for shallow nets, but signals vanish in deep ones"
)


def xavier_normal(shape: Tuple[int, int]) -> np.ndarray:
    """
    Xavier / Glorot Normal Initialization  [L3, Slide 13]
    
    σ = √(2 / (nᵢₙ + nₒᵤₜ))
    
    Derived by Glorot & Bengio (2010) to keep variance constant
    across layers when using TanH or Sigmoid activation.
    
    Goal: Var(output) = Var(input)  for each layer.
    """
    n_in, n_out = shape[1], shape[0]
    std = np.sqrt(2.0 / (n_in + n_out))
    return np.random.randn(*shape) * std

XAVIER_INFO = InitInfo(
    name="Xavier / Glorot Normal", short_name="Xavier",
    formula="σ = √(2/(nᵢₙ + nₒᵤₜ))", best_for="TanH, Sigmoid",
    color="#2563eb", year=2010, authors="Glorot & Bengio",
    description="Keeps variance constant for symmetric activations"
)


def xavier_uniform(shape: Tuple[int, int]) -> np.ndarray:
    """
    Xavier / Glorot Uniform Initialization  [L3, Slide 13]
    
    W ~ U[-r, r]  with r = √(6 / (nᵢₙ + nₒᵤₜ))
    """
    n_in, n_out = shape[1], shape[0]
    r = np.sqrt(6.0 / (n_in + n_out))
    return np.random.uniform(-r, r, size=shape)

XAVIER_UNIFORM_INFO = InitInfo(
    name="Xavier / Glorot Uniform", short_name="Xavier-U",
    formula="r = √(6/(nᵢₙ + nₒᵤₜ))", best_for="TanH, Sigmoid",
    color="#3b82f6", year=2010, authors="Glorot & Bengio",
    description="Uniform variant of Xavier"
)


def he_normal(shape: Tuple[int, int]) -> np.ndarray:
    """
    He / Kaiming Normal Initialization  [L3, Slide 13]
    
    σ = √(2 / nᵢₙ)
    
    Derived by He et al. (2015) specifically for ReLU.
    ReLU zeros out half the outputs → need 2× more variance.
    
    This is the DEFAULT for modern CNNs and FNNs with ReLU.
    """
    n_in = shape[1]
    std = np.sqrt(2.0 / n_in)
    return np.random.randn(*shape) * std

HE_INFO = InitInfo(
    name="He / Kaiming Normal", short_name="He",
    formula="σ = √(2/nᵢₙ)", best_for="ReLU, Leaky ReLU, ELU",
    color="#dc2626", year=2015, authors="He, Zhang, Ren & Sun",
    description="Default for ReLU networks — accounts for zero-out of negative inputs"
)


def he_uniform(shape: Tuple[int, int]) -> np.ndarray:
    """
    He / Kaiming Uniform Initialization
    
    W ~ U[-r, r]  with r = √(6 / nᵢₙ)
    """
    n_in = shape[1]
    r = np.sqrt(6.0 / n_in)
    return np.random.uniform(-r, r, size=shape)


def lecun_normal(shape: Tuple[int, int]) -> np.ndarray:
    """
    LeCun Normal Initialization
    
    σ = √(1 / nᵢₙ)
    
    Earlier variant by LeCun et al. (1998).
    Predecessor to Xavier (which adds nₒᵤₜ).
    """
    n_in = shape[1]
    std = np.sqrt(1.0 / n_in)
    return np.random.randn(*shape) * std

LECUN_INFO = InitInfo(
    name="LeCun Normal", short_name="LeCun",
    formula="σ = √(1/nᵢₙ)", best_for="SELU activation",
    color="#ea580c", year=1998, authors="LeCun et al.",
    description="Predecessor to Xavier, uses fan-in only"
)


def orthogonal(shape: Tuple[int, int], gain: float = 1.0) -> np.ndarray:
    """
    Orthogonal Initialization  [L3, Slide 5]
    
    Generate random matrix → compute SVD → use U or V.
    
    All singular values = 1 → perfectly preserves norms.
    Prevents both explosion and vanishing in linear networks.
    
    W = gain · Q,  where Q is orthogonal (from SVD)
    """
    n_rows, n_cols = shape
    A = np.random.randn(max(n_rows, n_cols), max(n_rows, n_cols))
    U, _, Vt = np.linalg.svd(A, full_matrices=True)
    return gain * U[:n_rows, :n_cols]

ORTHOGONAL_INFO = InitInfo(
    name="Orthogonal (SVD)", short_name="Orthogonal",
    formula="W = Q from SVD(random), all σᵢ = 1", best_for="RNNs, very deep nets",
    color="#16a34a", year=2013, authors="Saxe, McClelland & Ganguli",
    description="Perfectly preserves signal norms through layers"
)


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════
ALL_INITS = [
    ("zeros", zeros, ZEROS_INFO),
    ("random", random_normal, RANDOM_INFO),
    ("lecun", lecun_normal, LECUN_INFO),
    ("xavier", xavier_normal, XAVIER_INFO),
    ("he", he_normal, HE_INFO),
    ("orthogonal", orthogonal, ORTHOGONAL_INFO),
]
