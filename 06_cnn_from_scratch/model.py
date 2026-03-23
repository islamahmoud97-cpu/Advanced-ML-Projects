"""
CNN Model — Sequential Architecture
======================================
Based on: Zemke, AML Lectures 4-6

CNN Architecture (Lecture 4, Slide 30):
  Conv → Activation → Pooling → (repeat) → Flatten → Dense layers

This class chains layers into a sequential model with:
  - Forward pass through all layers
  - Backward pass (backpropagation) through all layers in reverse
  - Adam optimizer for parameter updates
"""

import numpy as np
from typing import List, Dict


class CNN:
    """
    Sequential Convolutional Neural Network.
    
    Usage:
        model = CNN([
            Conv2D(1, 8, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2D(2, 2),
            Flatten(),
            Dense(8*14*14, 10),
            Softmax(),
        ])
        model.train(X_train, y_train, epochs=10)
    """
    
    def __init__(self, layers: list):
        self.layers = layers
        self.history = {"train_loss": [], "train_acc": []}
    
    def forward(self, X):
        """Forward pass through all layers."""
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def backward(self, y_true):
        """Backward pass through all layers in reverse."""
        grad = self.layers[-1].backward(y_true)  # Softmax+CE
        for layer in reversed(self.layers[:-1]):
            grad = layer.backward(grad)
    
    def _get_params(self):
        """Collect all trainable parameters."""
        all_params = []
        for layer in self.layers:
            all_params.extend(layer.params())
        return all_params
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 0.001,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """
        Train with Adam optimizer and minibatch SGD.
        
        Parameters
        ----------
        X_train : (N, C, H, W)
        y_train : (N, num_classes) one-hot encoded
        """
        N = X_train.shape[0]
        
        # Initialize Adam state
        params = self._get_params()
        m_states = [(np.zeros_like(w), np.zeros_like(w)) for w, _ in params]
        v_states = [(np.zeros_like(w), np.zeros_like(w)) for w, _ in params]
        t = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        self.history = {"train_loss": [], "train_acc": []}
        
        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(N)
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]
            
            epoch_loss = 0
            epoch_correct = 0
            n_batches = 0
            
            for j in range(0, N, batch_size):
                X_batch = X_shuffled[j:j+batch_size]
                y_batch = y_shuffled[j:j+batch_size]
                bs = X_batch.shape[0]
                
                # Forward
                output = self.forward(X_batch)
                
                # Loss (cross-entropy)
                loss = -np.sum(y_batch * np.log(output + 1e-12)) / bs
                epoch_loss += loss
                epoch_correct += np.sum(np.argmax(output, axis=1) == np.argmax(y_batch, axis=1))
                n_batches += 1
                
                # Backward
                self.backward(y_batch)
                
                # Adam update
                t += 1
                params = self._get_params()
                for idx, (w, dw) in enumerate(params):
                    # First moment
                    m_states[idx] = (
                        beta1 * m_states[idx][0] + (1 - beta1) * dw,
                        m_states[idx][1],
                    )
                    m_w = m_states[idx][0]
                    
                    # Second moment
                    v_states[idx] = (
                        v_states[idx][0],
                        beta2 * v_states[idx][1] + (1 - beta2) * dw**2,
                    )
                    v_w = v_states[idx][1]
                    
                    # Bias correction
                    m_hat = m_w / (1 - beta1**t)
                    v_hat = v_w / (1 - beta2**t)
                    
                    # Update
                    w -= lr * m_hat / (np.sqrt(v_hat) + eps)
            
            avg_loss = epoch_loss / n_batches
            accuracy = epoch_correct / N
            self.history["train_loss"].append(avg_loss)
            self.history["train_acc"].append(accuracy)
            
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"  Epoch {epoch+1:3d}/{epochs}  |  Loss: {avg_loss:.4f}  |  Acc: {accuracy:.4f}")
        
        return self.history
    
    def predict(self, X):
        """Return predicted class indices."""
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def evaluate(self, X, y):
        """Return (loss, accuracy)."""
        output = self.forward(X)
        loss = -np.sum(y * np.log(output + 1e-12)) / X.shape[0]
        acc = np.mean(np.argmax(output, axis=1) == np.argmax(y, axis=1))
        return loss, acc
    
    def summary(self):
        """Print architecture summary."""
        print("=" * 55)
        print("CNN Architecture Summary")
        print("=" * 55)
        total = 0
        for layer in self.layers:
            name = layer.__class__.__name__
            n_params = sum(w.size for w, _ in layer.params())
            total += n_params
            details = ""
            if hasattr(layer, 'W') and hasattr(layer, 'in_channels'):
                details = f"({layer.in_channels}→{layer.out_channels}, k={layer.kernel_size})"
            elif hasattr(layer, 'W') and hasattr(layer, 'b'):
                details = f"({layer.W.shape[1]}→{layer.W.shape[0]})"
            elif hasattr(layer, 'pool_size'):
                details = f"(size={layer.pool_size})"
            print(f"  {name:15s} {details:20s}  params: {n_params:,d}")
        print("-" * 55)
        print(f"  Total parameters: {total:,d}")
        print("=" * 55)
