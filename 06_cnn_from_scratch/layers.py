"""
CNN Layers — From Scratch
===========================
Based on: Zemke, AML Lectures 4-6

Implements all core CNN components:
  - Conv2D: 2D convolution via im2col + GEMM  [L4-L6]
  - MaxPool2D: Max pooling                     [L4, Slide 27]
  - AvgPool2D: Average pooling                 [L4, Slide 27]
  - Flatten: Reshape for dense layers
  - Dense: Fully connected layer
  - ReLU, Softmax: Activations

All layers follow a consistent interface:
  - forward(X)  → output
  - backward(dout) → gradient w.r.t. input
  - params() → list of (weights, gradients) for optimizer

Data format: NCHW (N=batch, C=channels, H=height, W=width) [L5]
"""

import numpy as np
from im2col import im2col, col2im


class Conv2D:
    """
    2D Convolution Layer via im2col
    
    Input:  (N, C_in, H, W)
    Output: (N, C_out, H_out, W_out)
    
    where H_out = floor((H + 2*padding - kernel_size) / stride) + 1
    [Lecture 4, Slide 24]
    
    Implementation uses im2col to convert convolution into matrix multiplication:
      output = W_reshaped @ im2col(input) + bias
    [Lecture 6, Slide 24]
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He initialization [L3]
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros((out_channels, 1))
        
        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
        # Cache for backward
        self._cache = None
    
    def forward(self, X):
        """
        Forward pass:
          1. im2col(X) → column matrix
          2. W_reshaped @ cols + b → output columns
          3. Reshape back to (N, C_out, H_out, W_out)
        """
        N, C, H, W = X.shape
        k = self.kernel_size
        
        out_h = (H + 2*self.padding - k) // self.stride + 1
        out_w = (W + 2*self.padding - k) // self.stride + 1
        
        # im2col: (C*k*k, N*out_h*out_w)
        cols = im2col(X, k, k, self.stride, self.padding)
        
        # Reshape weights: (C_out, C_in*k*k)
        W_col = self.W.reshape(self.out_channels, -1)
        
        # GEMM: (C_out, C_in*k*k) @ (C_in*k*k, N*out_h*out_w) + bias
        out = W_col @ cols + self.b
        
        # Reshape: (C_out, N*out_h*out_w) → (N, C_out, out_h, out_w)
        out = out.reshape(self.out_channels, N, out_h, out_w).transpose(1, 0, 2, 3)
        
        self._cache = (X, cols)
        return out
    
    def backward(self, dout):
        """
        Backpropagation through convolution [L4, Slide 32]
        
        ∂L/∂W = dout_reshaped @ cols.T
        ∂L/∂b = sum(dout)
        ∂L/∂X = col2im(W.T @ dout_reshaped)
        """
        X, cols = self._cache
        N = X.shape[0]
        k = self.kernel_size
        
        # Reshape dout: (N, C_out, out_h, out_w) → (C_out, N*out_h*out_w)
        dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(self.out_channels, -1)
        
        # Gradient w.r.t. weights
        W_col = self.W.reshape(self.out_channels, -1)
        self.dW = (dout_reshaped @ cols.T).reshape(self.W.shape)
        self.db = np.sum(dout_reshaped, axis=1, keepdims=True)
        
        # Gradient w.r.t. input
        dcols = W_col.T @ dout_reshaped
        dX = col2im(dcols, X.shape, k, k, self.stride, self.padding)
        
        return dX
    
    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


class MaxPool2D:
    """
    Max Pooling Layer [L4, Slide 27]
    
    Reduces spatial dimensions by taking the maximum in each window.
    Increases translation invariance.
    
    Backprop: gradient routes to the position of the max value (argmax).
    """
    
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self._cache = None
    
    def forward(self, X):
        N, C, H, W = X.shape
        p = self.pool_size
        s = self.stride
        
        out_h = (H - p) // s + 1
        out_w = (W - p) // s + 1
        
        out = np.zeros((N, C, out_h, out_w))
        self._mask = np.zeros_like(X)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start, w_start = i * s, j * s
                window = X[:, :, h_start:h_start+p, w_start:w_start+p]
                out[:, :, i, j] = np.max(window, axis=(2, 3))
                
                # Store argmax mask for backward
                max_val = out[:, :, i, j][:, :, np.newaxis, np.newaxis]
                mask = (window == max_val)
                self._mask[:, :, h_start:h_start+p, w_start:w_start+p] += mask
        
        self._cache = X
        return out
    
    def backward(self, dout):
        X = self._cache
        N, C, H, W = X.shape
        p = self.pool_size
        s = self.stride
        out_h = (H - p) // s + 1
        out_w = (W - p) // s + 1
        
        dX = np.zeros_like(X)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start, w_start = i * s, j * s
                window = X[:, :, h_start:h_start+p, w_start:w_start+p]
                max_val = np.max(window, axis=(2, 3), keepdims=True)
                mask = (window == max_val)
                dX[:, :, h_start:h_start+p, w_start:w_start+p] += \
                    mask * dout[:, :, i:i+1, j:j+1]
        
        return dX
    
    def params(self):
        return []


class Flatten:
    """Reshape (N, C, H, W) → (N, C*H*W) for dense layers."""
    
    def __init__(self):
        self._shape = None
    
    def forward(self, X):
        self._shape = X.shape
        return X.reshape(X.shape[0], -1)
    
    def backward(self, dout):
        return dout.reshape(self._shape)
    
    def params(self):
        return []


class Dense:
    """
    Fully Connected Layer
    z = W @ a + b  [Lecture 1]
    """
    
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(out_features, in_features) * scale
        self.b = np.zeros((out_features, 1))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._cache = None
    
    def forward(self, X):
        # X: (N, in_features) → transpose to (in_features, N)
        self._cache = X
        out = self.W @ X.T + self.b  # (out_features, N)
        return out.T  # (N, out_features)
    
    def backward(self, dout):
        X = self._cache
        N = X.shape[0]
        dout_T = dout.T  # (out_features, N)
        
        self.dW = dout_T @ X / N
        self.db = np.sum(dout_T, axis=1, keepdims=True) / N
        dX = (self.W.T @ dout_T).T  # (N, in_features)
        return dX
    
    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    """ReLU activation [L1]"""
    def __init__(self):
        self._mask = None
    
    def forward(self, X):
        self._mask = (X > 0)
        return X * self._mask
    
    def backward(self, dout):
        return dout * self._mask
    
    def params(self):
        return []


class Softmax:
    """Softmax + Cross-Entropy combined for numerical stability."""
    
    def __init__(self):
        self._cache = None
    
    def forward(self, X):
        # X: (N, num_classes)
        e = np.exp(X - np.max(X, axis=1, keepdims=True))
        self._cache = e / np.sum(e, axis=1, keepdims=True)
        return self._cache
    
    def backward(self, y_true):
        # Combined softmax + cross-entropy gradient: (pred - true)
        return self._cache - y_true
    
    def params(self):
        return []
