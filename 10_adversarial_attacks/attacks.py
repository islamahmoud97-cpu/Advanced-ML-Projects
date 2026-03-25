"""
Adversarial Attack Methods
============================
Based on: Zemke, AML Lecture 7, Slides 30-45

"Close to almost every successfully classified image are so-called
 adversarial examples that a human cannot distinguish from the original,
 yet these are misclassified with high confidence." — Lecture 7

Attacks implemented:
  1. FGSM (Fast Gradient Sign Method)     — Goodfellow et al. 2014
  2. Targeted FGSM                         — attack towards specific class
  3. Iterative FGSM (I-FGSM / PGD)        — stronger iterative version
  4. Random Noise                          — baseline (non-adversarial)

All attacks work on our from-scratch FNN (no TensorFlow needed).
"""

import numpy as np
from typing import Optional


class SimpleNet:
    """
    Minimal FNN for adversarial attack demos.
    Architecture: input → Dense(128, ReLU) → Dense(10, Softmax)
    Trained on sklearn digits (8×8).
    """

    def __init__(self, input_dim=64, hidden=128, output=10, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(hidden, input_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((hidden, 1))
        self.W2 = np.random.randn(output, hidden) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros((output, 1))

    def forward(self, x):
        """x: (input_dim,) or (input_dim, 1)"""
        x = x.reshape(-1, 1)
        self.x = x
        self.z1 = self.W1 @ x + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.W2 @ self.a1 + self.b2
        # Softmax
        e = np.exp(self.z2 - np.max(self.z2))
        self.probs = e / np.sum(e)
        return self.probs.ravel()

    def predict(self, x):
        probs = self.forward(x)
        return np.argmax(probs)

    def compute_gradient_wrt_input(self, x, target_class):
        """
        Compute ∂L/∂x — gradient of loss w.r.t. INPUT image.
        
        This is the KEY to adversarial attacks:
        Normal training:  ∂L/∂W → update weights
        Adversarial:      ∂L/∂x → update INPUT image!
        
        [Lecture 7: "optimize the INPUT, not the weights"]
        """
        probs = self.forward(x)

        # Cross-entropy gradient at output
        delta2 = probs.reshape(-1, 1).copy()
        delta2[target_class] -= 1.0  # softmax + CE gradient

        # Backprop to hidden
        delta1 = self.W2.T @ delta2
        delta1 = delta1 * (self.z1 > 0).astype(float)  # ReLU derivative

        # Gradient w.r.t. input
        grad_x = self.W1.T @ delta1
        return grad_x.ravel()

    def train(self, X, y, epochs=100, lr=0.01, batch_size=32):
        """Quick training loop."""
        N = X.shape[0]
        for epoch in range(epochs):
            perm = np.random.permutation(N)
            for j in range(0, N, batch_size):
                batch_idx = perm[j:j + batch_size]
                for idx in batch_idx:
                    x = X[idx].reshape(-1, 1)
                    probs = self.forward(x)

                    # Output gradient (10, 1)
                    delta2 = self.probs.copy()  # already (10,1) from forward
                    delta2[y[idx]] -= 1.0

                    # Hidden gradient
                    delta1 = self.W2.T @ delta2
                    delta1 = delta1 * (self.z1 > 0).astype(float)

                    # Update
                    self.W2 -= lr / batch_size * (delta2 @ self.a1.T)
                    self.b2 -= lr / batch_size * delta2
                    self.W1 -= lr / batch_size * (delta1 @ x.T)
                    self.b1 -= lr / batch_size * delta1

            if (epoch + 1) % 20 == 0:
                correct = sum(self.predict(X[i]) == y[i] for i in range(N))
                print(f"    Epoch {epoch+1}/{epochs} — Acc: {correct/N*100:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
# ATTACK 1: FGSM  [Lecture 7, Slide 30 — Goodfellow et al. 2014]
# ═══════════════════════════════════════════════════════════════════════════
def fgsm_attack(model, x, true_label, epsilon=0.3):
    """
    Fast Gradient Sign Method (FGSM)
    
    [Lecture 7]: Goodfellow, Shlens, and Szegedy (2014)
    
    x_adv = x + ε · sign(∇_x L(θ, x, y))
    
    Key insight: Instead of optimizing weights to minimize loss,
    optimize the INPUT to MAXIMIZE loss. One gradient step is enough!
    
    Parameters
    ----------
    model : trained network
    x : original input image (flattened)
    true_label : correct class index
    epsilon : perturbation magnitude (smaller = less visible)
    
    Returns
    -------
    x_adv : adversarial image
    perturbation : the added noise
    """
    # Compute gradient of loss w.r.t. input
    grad = model.compute_gradient_wrt_input(x, true_label)

    # Sign of gradient — direction that INCREASES the loss most
    perturbation = epsilon * np.sign(grad)

    # Create adversarial example
    x_adv = np.clip(x + perturbation, 0, 1)

    return x_adv, perturbation


# ═══════════════════════════════════════════════════════════════════════════
# ATTACK 2: TARGETED FGSM
# ═══════════════════════════════════════════════════════════════════════════
def targeted_fgsm(model, x, target_label, epsilon=0.3):
    """
    Targeted FGSM — fool the network into predicting a SPECIFIC class.
    
    x_adv = x - ε · sign(∇_x L(θ, x, y_target))
    
    Note the MINUS: we MINIMIZE loss for the target class,
    making the network more confident it's the target.
    """
    grad = model.compute_gradient_wrt_input(x, target_label)
    perturbation = -epsilon * np.sign(grad)  # negative = minimize loss for target
    x_adv = np.clip(x + perturbation, 0, 1)
    return x_adv, perturbation


# ═══════════════════════════════════════════════════════════════════════════
# ATTACK 3: ITERATIVE FGSM (I-FGSM / PGD)
# ═══════════════════════════════════════════════════════════════════════════
def iterative_fgsm(model, x, true_label, epsilon=0.3, alpha=0.01, n_iter=40):
    """
    Iterative FGSM (I-FGSM) / Projected Gradient Descent (PGD)
    
    Repeat FGSM with small steps α, projecting back to ε-ball:
      x_{t+1} = clip(x_t + α · sign(∇_x L), x-ε, x+ε)
    
    Stronger than single-step FGSM but slower.
    """
    x_adv = x.copy()
    x_orig = x.copy()

    for _ in range(n_iter):
        grad = model.compute_gradient_wrt_input(x_adv, true_label)
        x_adv = x_adv + alpha * np.sign(grad)
        # Project back to ε-ball around original
        x_adv = np.clip(x_adv, x_orig - epsilon, x_orig + epsilon)
        x_adv = np.clip(x_adv, 0, 1)

    perturbation = x_adv - x_orig
    return x_adv, perturbation


# ═══════════════════════════════════════════════════════════════════════════
# ATTACK 4: RANDOM NOISE (baseline, non-adversarial)
# ═══════════════════════════════════════════════════════════════════════════
def random_noise_attack(model, x, epsilon=0.3):
    """
    Random uniform noise — baseline comparison.
    
    Shows that adversarial perturbations are NOT random noise.
    Random noise rarely fools the network even at high ε.
    """
    perturbation = np.random.uniform(-epsilon, epsilon, size=x.shape)
    x_adv = np.clip(x + perturbation, 0, 1)
    return x_adv, perturbation
