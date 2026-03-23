"""
Feedforward Neural Network from Scratch
=========================================
Based on: Zemke, Advanced Machine Learning, TUHH WS 2025/26
Lectures 1-3

Architecture: a₁:=x, zᵢ:=Wᵢ·aᵢ+bᵢ, aᵢ₊₁:=σ(zᵢ), f(x):=aₘ   [L1, Slide 31]
Training: Backpropagation + Minibatch SGD                          [L2, Slides 4-5]
Regularization: L2, Dropout                                        [L3, Slides 24-32]

This implementation uses ONLY NumPy — no PyTorch, no TensorFlow.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict

from activations import get_activation, softmax
from losses import get_loss
from optimizers import get_optimizer


class FeedforwardNet:
    """
    A fully-connected feedforward neural network.

    Parameters
    ----------
    layer_sizes : list of int
        Number of neurons per layer, e.g. [784, 128, 64, 10].
        n = (n₁, n₂, ..., nₘ)  where n₁ = input dim, nₘ = output dim.
    activation : str
        Hidden layer activation: 'relu', 'sigmoid', 'tanh', 'leaky_relu',
        'elu', 'swish', 'softplus'.
    output : str
        Output layer type: 'softmax' (classification) or 'linear' (regression).
    init_method : str
        Weight initialization: 'he', 'xavier', or 'orthogonal'.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "relu",
        output: str = "softmax",
        init_method: str = "he",
        seed: Optional[int] = None,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.layer_sizes = layer_sizes
        self.m = len(layer_sizes)  # number of layers
        self.output_type = output
        self.activation_name = activation
        self.act_fn, self.act_deriv = get_activation(activation)

        # Initialize weights and biases  [L3, Slides 2-5]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self._init_parameters(init_method)

        # Cache for forward/backward pass
        self.a: List[np.ndarray] = []  # activations
        self.z: List[np.ndarray] = []  # pre-activations

        # Training history
        self.history: Dict[str, list] = {"train_loss": [], "train_acc": []}

    # ==================================================================
    # Weight Initialization  [L3, Slides 2-5]
    # ==================================================================
    def _init_parameters(self, method: str):
        """
        Three initialization strategies (Lecture 3):
        - He/Kaiming:  σ = √(2/nᵢₙ)        → best for ReLU    [L3, Slide 4]
        - Xavier/Glorot: σ = √(2/(nᵢₙ+nₒᵤₜ)) → best for TanH  [L3, Slide 3]
        - Orthogonal:  W = U from SVD(random) → preserves norms [L3, Slide 5]
        """
        for i in range(self.m - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]

            if method == "he":
                # He (Kaiming) — for ReLU variants
                W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
            elif method == "xavier":
                # Xavier (Glorot) — for sigmoid/tanh
                W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / (n_in + n_out))
            elif method == "orthogonal":
                # Orthogonal — via SVD
                A = np.random.randn(n_out, n_in)
                U, _, Vt = np.linalg.svd(A, full_matrices=False)
                W = U if n_out <= n_in else Vt
            else:
                raise ValueError(f"Unknown init method: {method}")

            b = np.zeros((n_out, 1))
            self.weights.append(W)
            self.biases.append(b)

    # ==================================================================
    # Forward Pass  [L1, Slide 31]
    # ==================================================================
    def forward(self, X: np.ndarray, training: bool = False, dropout_rate: float = 0.0) -> np.ndarray:
        """
        Feedforward evaluation:
            a₁ := x
            zᵢ := Wᵢ · aᵢ + bᵢ          (affine map)
            aᵢ₊₁ := σ(zᵢ)               (activation)
            f(x) := aₘ                   (output)

        Parameters
        ----------
        X : ndarray, shape (n_features, n_samples)
            Input data (each column = one sample).
        training : bool
            If True, apply dropout masks.
        dropout_rate : float
            Probability of dropping a neuron [L3, Slide 30].

        Returns
        -------
        ndarray : Network output aₘ.
        """
        self.a = [X]
        self.z = []
        self.dropout_masks = []

        for i in range(self.m - 1):
            # zᵢ = Wᵢ · aᵢ + bᵢ
            zi = self.weights[i] @ self.a[i] + self.biases[i]
            self.z.append(zi)

            # Activation
            if i == self.m - 2:  # output layer
                if self.output_type == "softmax":
                    ai_next = softmax(zi)
                else:  # linear
                    ai_next = zi
            else:  # hidden layers
                ai_next = self.act_fn(zi)

                # Dropout [L3, Slide 30]: keep neurons with prob p ∈ [0.5, 0.8]
                if training and dropout_rate > 0:
                    mask = (np.random.rand(*ai_next.shape) > dropout_rate).astype(
                        np.float64
                    )
                    ai_next *= mask / (1.0 - dropout_rate)  # inverted dropout
                    self.dropout_masks.append(mask)
                else:
                    self.dropout_masks.append(None)

            self.a.append(ai_next)

        return self.a[-1]

    # ==================================================================
    # Backpropagation  [L2, Slides 4-5]
    # ==================================================================
    def backprop(
        self,
        y: np.ndarray,
        l2_lambda: float = 0.0,
        dropout_rate: float = 0.0,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backpropagation computes gradients of the cost function.

        δ_{m-1} = a'_{m-1}(z_{m-1}) ⊙ (aₘ - y)
        δᵢ = a'ᵢ(zᵢ) ⊙ (Wᵢ₊₁ᵀ · δᵢ₊₁)

        ∂C/∂Wᵢ = δᵢ · aᵢᵀ
        ∂C/∂bᵢ = δᵢ

        With L2 regularization [L3, Slide 24]:
        ∂C^λ/∂Wᵢ = ∂C/∂Wᵢ + λ · Wᵢ

        Returns
        -------
        dW, db : lists of gradients for each layer.
        """
        N = y.shape[1]  # minibatch size
        dW = []
        db = []

        # Output layer delta
        # For softmax + cross-entropy: δ = (aₘ - y) / N
        delta = (self.a[-1] - y) / N

        # Backward pass: from last layer to first
        for i in range(self.m - 2, -1, -1):
            # ∂C/∂Wᵢ = δᵢ · aᵢᵀ  (+L2 regularization)
            dW_i = delta @ self.a[i].T
            if l2_lambda > 0:
                dW_i += l2_lambda * self.weights[i]  # weight decay [L3, S.24]

            db_i = np.sum(delta, axis=1, keepdims=True)

            dW.insert(0, dW_i)
            db.insert(0, db_i)

            # Propagate delta to previous layer
            if i > 0:
                delta = self.weights[i].T @ delta
                delta *= self.act_deriv(self.z[i - 1])

                # Apply dropout mask if present
                if dropout_rate > 0 and self.dropout_masks[i - 1] is not None:
                    delta *= self.dropout_masks[i - 1] / (1.0 - dropout_rate)

        return dW, db

    # ==================================================================
    # Training Loop  [L2, Slides 17-20]
    # ==================================================================
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        optimizer: str = "adam",
        lr: float = 0.001,
        l2_lambda: float = 0.0,
        dropout_rate: float = 0.0,
        loss_fn: str = "cross_entropy",
        verbose: bool = True,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, list]:
        """
        Train the network using minibatch SGD.

        Parameters
        ----------
        X_train : shape (n_features, n_samples)
        y_train : shape (n_classes, n_samples) for classification
        epochs : number of passes through the full dataset
        batch_size : minibatch size [L2, Slide 17]
        optimizer : 'sgd', 'momentum', 'nag', 'rmsprop', 'adam'
        lr : learning rate η
        l2_lambda : L2 regularization strength λ [L3, Slide 24]
        dropout_rate : dropout probability [L3, Slide 30]
        loss_fn : 'cross_entropy', 'mse', or 'bce'
        verbose : print progress
        X_val, y_val : optional validation data
        """
        N = X_train.shape[1]
        loss_func, _ = get_loss(loss_fn)

        # Initialize optimizer
        opt = get_optimizer(optimizer, lr=lr)
        opt.initialize(self.weights, self.biases)

        self.history = {"train_loss": [], "train_acc": []}
        if X_val is not None:
            self.history["val_loss"] = []
            self.history["val_acc"] = []

        for epoch in range(epochs):
            # Shuffle training data
            perm = np.random.permutation(N)
            X_shuffled = X_train[:, perm]
            y_shuffled = y_train[:, perm]

            epoch_loss = 0.0
            n_batches = 0

            # Minibatch loop [L2, Slide 17]
            for j in range(0, N, batch_size):
                X_batch = X_shuffled[:, j : j + batch_size]
                y_batch = y_shuffled[:, j : j + batch_size]

                # Forward pass
                output = self.forward(X_batch, training=True, dropout_rate=dropout_rate)

                # Compute loss
                batch_loss = loss_func(output, y_batch)
                if l2_lambda > 0:
                    # Add L2 penalty: λ/2 · Σ||Wₗ||²_F  [L3, Slide 24]
                    l2_penalty = sum(np.sum(w**2) for w in self.weights)
                    batch_loss += (l2_lambda / 2.0) * l2_penalty

                epoch_loss += batch_loss
                n_batches += 1

                # Backpropagation
                dW, db = self.backprop(y_batch, l2_lambda, dropout_rate)

                # Parameter update
                opt.update(self.weights, self.biases, dW, db)

            # Record epoch metrics
            avg_loss = epoch_loss / n_batches
            self.history["train_loss"].append(avg_loss)

            # Training accuracy
            train_pred = self.forward(X_train, training=False)
            train_acc = np.mean(
                np.argmax(train_pred, axis=0) == np.argmax(y_train, axis=0)
            )
            self.history["train_acc"].append(train_acc)

            # Validation metrics
            if X_val is not None:
                val_pred = self.forward(X_val, training=False)
                val_loss = loss_func(val_pred, y_val)
                val_acc = np.mean(
                    np.argmax(val_pred, axis=0) == np.argmax(y_val, axis=0)
                )
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

            if verbose and (epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1):
                msg = f"Epoch {epoch+1:4d}/{epochs}  |  Loss: {avg_loss:.4f}  |  Acc: {train_acc:.4f}"
                if X_val is not None:
                    msg += f"  |  Val Loss: {val_loss:.4f}  |  Val Acc: {val_acc:.4f}"
                print(msg)

        return self.history

    # ==================================================================
    # Prediction & Evaluation
    # ==================================================================
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions (argmax of output)."""
        output = self.forward(X, training=False)
        return np.argmax(output, axis=0)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Return (loss, accuracy) on given data."""
        output = self.forward(X, training=False)
        loss_func, _ = get_loss("cross_entropy")
        loss = loss_func(output, y)
        acc = np.mean(np.argmax(output, axis=0) == np.argmax(y, axis=0))
        return loss, acc

    # ==================================================================
    # Representation
    # ==================================================================
    def summary(self):
        """Print network architecture summary."""
        print("=" * 60)
        print("FeedforwardNet Summary")
        print("=" * 60)
        total_params = 0
        for i in range(self.m - 1):
            n_params = self.weights[i].size + self.biases[i].size
            total_params += n_params
            act = self.activation_name if i < self.m - 2 else self.output_type
            print(
                f"  Layer {i+1}: {self.layer_sizes[i]:5d} → {self.layer_sizes[i+1]:5d}"
                f"  |  params: {n_params:8,d}  |  activation: {act}"
            )
        print("-" * 60)
        print(f"  Total parameters: {total_params:,d}")
        print("=" * 60)
