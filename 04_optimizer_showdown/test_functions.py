"""
Test Functions for Optimization
=================================
Classic optimization surfaces to benchmark gradient descent variants.

Each function returns:
  - f(x): function value
  - grad_f(x): gradient
  - minimum: known global minimum location
  - name: for plotting
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass
class TestFunction:
    name: str
    f: Callable
    grad_f: Callable
    minimum: np.ndarray
    x0: np.ndarray        # starting point
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    description: str


# ═══════════════════════════════════════════════════════════════════════════
# 1. ROSENBROCK FUNCTION
#    f(x,y) = (a-x)² + b(y-x²)²    with a=1, b=100
#    Minimum at (1, 1)
#    Famous "banana" shape — long narrow valley
# ═══════════════════════════════════════════════════════════════════════════
def rosenbrock(x, a=1, b=100):
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

def rosenbrock_grad(x, a=1, b=100):
    dx = -2 * (a - x[0]) + 2 * b * (x[1] - x[0]**2) * (-2 * x[0])
    dy = 2 * b * (x[1] - x[0]**2)
    return np.array([dx, dy])

ROSENBROCK = TestFunction(
    name="Rosenbrock Function",
    f=rosenbrock, grad_f=rosenbrock_grad,
    minimum=np.array([1.0, 1.0]),
    x0=np.array([-1.5, 2.5]),
    xlim=(-2.5, 2.5), ylim=(-1.5, 3.5),
    description="f(x,y) = (1-x)² + 100(y-x²)²\nLong narrow valley — tests ravine navigation",
)


# ═══════════════════════════════════════════════════════════════════════════
# 2. BEALE FUNCTION
#    f(x,y) = (1.5-x+xy)² + (2.25-x+xy²)² + (2.625-x+xy³)²
#    Minimum at (3, 0.5)
#    Multiple flat regions and steep walls
# ═══════════════════════════════════════════════════════════════════════════
def beale(x):
    return ((1.5 - x[0] + x[0]*x[1])**2 +
            (2.25 - x[0] + x[0]*x[1]**2)**2 +
            (2.625 - x[0] + x[0]*x[1]**3)**2)

def beale_grad(x):
    t1 = 1.5 - x[0] + x[0]*x[1]
    t2 = 2.25 - x[0] + x[0]*x[1]**2
    t3 = 2.625 - x[0] + x[0]*x[1]**3
    dx = 2*t1*(-1+x[1]) + 2*t2*(-1+x[1]**2) + 2*t3*(-1+x[1]**3)
    dy = 2*t1*x[0] + 2*t2*2*x[0]*x[1] + 2*t3*3*x[0]*x[1]**2
    return np.array([dx, dy])

BEALE = TestFunction(
    name="Beale Function",
    f=beale, grad_f=beale_grad,
    minimum=np.array([3.0, 0.5]),
    x0=np.array([-1.0, -1.0]),
    xlim=(-4.5, 4.5), ylim=(-4.5, 4.5),
    description="f(x,y) = (1.5-x+xy)² + (2.25-x+xy²)² + (2.625-x+xy³)²\nFlat regions with steep walls",
)


# ═══════════════════════════════════════════════════════════════════════════
# 3. HIMMELBLAU FUNCTION
#    f(x,y) = (x²+y-11)² + (x+y²-7)²
#    Four identical minima: (3,2), (-2.81,3.13), (-3.78,-3.28), (3.58,-1.85)
# ═══════════════════════════════════════════════════════════════════════════
def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def himmelblau_grad(x):
    dx = 4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7)
    dy = 2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)
    return np.array([dx, dy])

HIMMELBLAU = TestFunction(
    name="Himmelblau Function",
    f=himmelblau, grad_f=himmelblau_grad,
    minimum=np.array([3.0, 2.0]),
    x0=np.array([-4.0, -3.0]),
    xlim=(-5, 5), ylim=(-5, 5),
    description="f(x,y) = (x²+y-11)² + (x+y²-7)²\nFour identical minima — which one does the optimizer find?",
)


# ═══════════════════════════════════════════════════════════════════════════
# 4. SADDLE POINT
#    f(x,y) = x² - y²
#    Saddle at (0, 0) — gradient = 0 but NOT a minimum!
# ═══════════════════════════════════════════════════════════════════════════
def saddle(x):
    return x[0]**2 - x[1]**2

def saddle_grad(x):
    return np.array([2*x[0], -2*x[1]])

SADDLE = TestFunction(
    name="Saddle Point",
    f=saddle, grad_f=saddle_grad,
    minimum=np.array([0.0, 0.0]),  # saddle, not minimum
    x0=np.array([0.5, 0.01]),
    xlim=(-2, 2), ylim=(-2, 2),
    description="f(x,y) = x² - y²\nSaddle point at origin — can the optimizer escape?",
)


# ═══════════════════════════════════════════════════════════════════════════
# 5. STYBLINSKI-TANG (high-dimensional friendly)
#    f(x) = ½ Σ (xᵢ⁴ - 16xᵢ² + 5xᵢ)
#    Min at (-2.903, -2.903)
# ═══════════════════════════════════════════════════════════════════════════
def styblinski_tang(x):
    return 0.5 * np.sum(x**4 - 16*x**2 + 5*x)

def styblinski_tang_grad(x):
    return 0.5 * (4*x**3 - 32*x + 5)

STYBLINSKI = TestFunction(
    name="Styblinski-Tang",
    f=styblinski_tang, grad_f=styblinski_tang_grad,
    minimum=np.array([-2.903534, -2.903534]),
    x0=np.array([4.0, 4.0]),
    xlim=(-5, 5), ylim=(-5, 5),
    description="f(x) = ½Σ(xᵢ⁴ - 16xᵢ² + 5xᵢ)\nMultiple local minima — tests global search",
)


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════
ALL_FUNCTIONS = {
    "rosenbrock":     ROSENBROCK,
    "beale":          BEALE,
    "himmelblau":     HIMMELBLAU,
    "saddle":         SADDLE,
    "styblinski":     STYBLINSKI,
}
