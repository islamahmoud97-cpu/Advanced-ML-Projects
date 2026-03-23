"""
Support Vector Machine — From Scratch
=======================================
Based on: Zemke, AML Lecture 1, Slides 8-14

The SVM finds the separating hyperplane wᵀx + b = 0 that MAXIMIZES
the margin between two classes.

    Margin = 2 / ||w||₂

Optimization problem (Lecture 1, Slide 11):
    min  ½||w||²
    s.t. yᵢ(wᵀxᵢ + b) ≥ 1   ∀i

Dual problem (via Lagrange multipliers αᵢ):
    max  Σαᵢ - ½ ΣΣ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
    s.t. 0 ≤ αᵢ ≤ C,  Σαᵢyᵢ = 0

Implementation: Simplified SMO (Sequential Minimal Optimization)
Based on Platt (1998) and the simplified version by Stanford CS229.
"""

import numpy as np
from typing import Optional, Callable
from kernels import get_kernel, linear_kernel, rbf_kernel


class SVM:
    """
    Support Vector Machine classifier.
    
    Parameters
    ----------
    kernel : str
        Kernel type: 'linear', 'polynomial', 'rbf', 'sigmoid'.
    C : float
        Regularization parameter. Large C → hard margin (may overfit).
        Small C → soft margin (allows misclassifications).
    gamma : float
        Parameter for RBF kernel (width of Gaussian).
    degree : int
        Degree for polynomial kernel.
    tol : float
        Numerical tolerance for KKT conditions.
    max_iter : int
        Maximum number of passes over training data.
    """
    
    def __init__(
        self,
        kernel: str = "linear",
        C: float = 1.0,
        gamma: float = 1.0,
        degree: int = 3,
        tol: float = 1e-3,
        max_iter: int = 1000,
    ):
        self.kernel_name = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.tol = tol
        self.max_iter = max_iter
        
        # Will be set during training
        self.alpha = None
        self.b = 0.0
        self.X_train = None
        self.y_train = None
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.support_vector_alphas_ = None
        self.n_support_ = 0
    
    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix K(X1, X2)."""
        if self.kernel_name == "linear":
            return linear_kernel(X1, X2)
        elif self.kernel_name == "polynomial":
            from kernels import polynomial_kernel
            return polynomial_kernel(X1, X2, degree=self.degree)
        elif self.kernel_name == "rbf":
            return rbf_kernel(X1, X2, gamma=self.gamma)
        elif self.kernel_name == "sigmoid":
            from kernels import sigmoid_kernel
            return sigmoid_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_name}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> "SVM":
        """
        Train the SVM using Simplified SMO.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
        y : ndarray, shape (n_samples,) with values {-1, +1}
        
        The SMO algorithm optimizes TWO Lagrange multipliers at a time:
        1. Pick αᵢ that violates KKT conditions
        2. Pick αⱼ randomly
        3. Optimize αᵢ, αⱼ jointly (closed-form solution)
        4. Update bias b
        """
        N, d = X.shape
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # Initialize Lagrange multipliers to zero
        self.alpha = np.zeros(N)
        self.b = 0.0
        
        # Precompute kernel matrix
        K = self._compute_kernel(X, X)
        
        # SMO main loop
        passes = 0
        while passes < self.max_iter:
            num_changed = 0
            
            for i in range(N):
                # Compute f(xᵢ) = Σ αⱼyⱼK(xⱼ,xᵢ) + b
                Ei = self._decision_function_cached(K, i) - y[i]
                
                # Check KKT violation
                if ((y[i] * Ei < -self.tol and self.alpha[i] < self.C) or
                    (y[i] * Ei > self.tol and self.alpha[i] > 0)):
                    
                    # Pick j ≠ i randomly
                    j = i
                    while j == i:
                        j = np.random.randint(0, N)
                    
                    Ej = self._decision_function_cached(K, j) - y[j]
                    
                    # Save old alphas
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    
                    # Compute bounds L and H
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta = 2K(xᵢ,xⱼ) - K(xᵢ,xᵢ) - K(xⱼ,xⱼ)
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    
                    # Update αⱼ
                    self.alpha[j] -= y[j] * (Ei - Ej) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Update αᵢ (from constraint: αᵢyᵢ + αⱼyⱼ = const)
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    
                    # Update bias b
                    b1 = (self.b - Ei 
                           - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i]
                           - y[j] * (self.alpha[j] - alpha_j_old) * K[i, j])
                    b2 = (self.b - Ej
                           - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j]
                           - y[j] * (self.alpha[j] - alpha_j_old) * K[j, j])
                    
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    num_changed += 1
            
            if num_changed == 0:
                passes += 1
            else:
                passes = 0
        
        # Extract support vectors (αᵢ > 0)
        sv_mask = self.alpha > 1e-7
        self.support_vectors_ = X[sv_mask]
        self.support_vector_labels_ = y[sv_mask]
        self.support_vector_alphas_ = self.alpha[sv_mask]
        self.n_support_ = np.sum(sv_mask)
        
        # For linear kernel: compute w explicitly
        if self.kernel_name == "linear":
            self.w = np.sum(
                (self.alpha * y)[:, np.newaxis] * X, axis=0
            )
        else:
            self.w = None
        
        if verbose:
            print(f"  Training complete: {self.n_support_} support vectors found")
            if self.kernel_name == "linear" and self.w is not None:
                margin = 2.0 / np.linalg.norm(self.w)
                print(f"  Margin = 2/||w|| = {margin:.4f}")
        
        return self
    
    def _decision_function_cached(self, K: np.ndarray, i: int) -> float:
        """Compute decision function using precomputed kernel matrix."""
        return np.sum(self.alpha * self.y_train * K[:, i]) + self.b
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute f(x) = Σ αᵢyᵢK(xᵢ, x) + b
        
        Returns signed distance to the hyperplane.
        """
        K = self._compute_kernel(self.X_train, X)
        return (self.alpha * self.y_train) @ K + self.b
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels {-1, +1}."""
        return np.sign(self.decision_function(X))
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        return np.mean(self.predict(X) == y)
    
    def get_margin(self) -> Optional[float]:
        """
        Return the margin width = 2/||w||₂ for linear kernel.
        [Lecture 1, Slide 11]
        """
        if self.w is not None:
            return 2.0 / np.linalg.norm(self.w)
        return None


class MultiClassSVM:
    """
    Multi-class SVM using One-vs-Rest strategy.
    
    For K classes, train K binary SVMs:
    - SVM_k: class k vs. all other classes
    - Predict: class with highest decision function value
    """
    
    def __init__(self, kernel="rbf", C=1.0, gamma=1.0, **kwargs):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.kwargs = kwargs
        self.classifiers = {}
        self.classes = None
    
    def fit(self, X, y, verbose=True):
        self.classes = np.unique(y)
        
        for cls in self.classes:
            if verbose:
                print(f"\n  Training SVM for class {cls} vs rest...")
            y_binary = np.where(y == cls, 1, -1).astype(float)
            svm = SVM(kernel=self.kernel, C=self.C, gamma=self.gamma, **self.kwargs)
            svm.fit(X, y_binary, verbose=verbose)
            self.classifiers[cls] = svm
        
        return self
    
    def predict(self, X):
        scores = np.zeros((len(X), len(self.classes)))
        for i, cls in enumerate(self.classes):
            scores[:, i] = self.classifiers[cls].decision_function(X)
        return self.classes[np.argmax(scores, axis=1)]
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
