"""
im2col & col2im — Efficient Convolution via Matrix Multiplication
====================================================================
Based on: Zemke, AML Lecture 6, Slides 24-30

Key Insight (Lecture 6):
  "The matrix-matrix multiplication in BLAS Level 3 (GEMM) is fast.
   Can we use GEMM for convolution?"

Answer: YES! im2col rearranges image patches into columns so that
convolution becomes a single matrix multiplication.

  im2col: Image → Column matrix (for forward pass)
  col2im: Column matrix → Image (for backward pass)

This is how ALL major frameworks (PyTorch, TensorFlow, Caffe) implement
convolution internally.

Performance:
  - Naive convolution: 6 nested loops → O(N·C·H·W·K·K·M)
  - im2col + GEMM: 1 matrix multiply → leverages optimized BLAS
"""

import numpy as np


def im2col(X: np.ndarray, kernel_h: int, kernel_w: int,
           stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    Transform input images into column matrix for GEMM-based convolution.
    
    Each column = one flattened receptive field (patch) of the input.
    
    Parameters
    ----------
    X : ndarray, shape (N, C, H, W)
        Input images. N=batch, C=channels, H=height, W=width.
    kernel_h, kernel_w : int
        Kernel dimensions.
    stride : int
        Stride of convolution.
    padding : int
        Zero-padding added to input.
    
    Returns
    -------
    cols : ndarray, shape (C*kernel_h*kernel_w, N*out_h*out_w)
        Each column is a flattened receptive field.
    """
    N, C, H, W = X.shape
    
    # Apply padding
    if padding > 0:
        X_padded = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                          mode='constant', constant_values=0)
    else:
        X_padded = X
    
    H_pad, W_pad = X_padded.shape[2], X_padded.shape[3]
    
    # Output dimensions [L4: o = floor((i + 2p - k) / s) + 1]
    out_h = (H_pad - kernel_h) // stride + 1
    out_w = (W_pad - kernel_w) // stride + 1
    
    # Extract patches using stride tricks for efficiency
    # Shape: (N, C, out_h, out_w, kernel_h, kernel_w)
    shape = (N, C, out_h, out_w, kernel_h, kernel_w)
    strides_arr = X_padded.strides
    strides = (
        strides_arr[0],                    # batch
        strides_arr[1],                    # channel
        strides_arr[2] * stride,           # output row
        strides_arr[3] * stride,           # output col
        strides_arr[2],                    # kernel row
        strides_arr[3],                    # kernel col
    )
    
    patches = np.lib.stride_tricks.as_strided(X_padded, shape=shape, strides=strides)
    
    # Reshape to (C*kH*kW, N*out_h*out_w)
    cols = patches.transpose(1, 4, 5, 0, 2, 3).reshape(C * kernel_h * kernel_w, -1)
    
    return cols


def col2im(cols: np.ndarray, X_shape: tuple, kernel_h: int, kernel_w: int,
           stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    Inverse of im2col: reconstruct image from column matrix.
    Used in backpropagation to compute gradient w.r.t. input.
    
    Parameters
    ----------
    cols : ndarray, shape (C*kernel_h*kernel_w, N*out_h*out_w)
    X_shape : tuple (N, C, H, W) — original input shape
    
    Returns
    -------
    X : ndarray, shape (N, C, H, W) — reconstructed (gradient) image
    """
    N, C, H, W = X_shape
    
    H_pad = H + 2 * padding
    W_pad = W + 2 * padding
    out_h = (H_pad - kernel_h) // stride + 1
    out_w = (W_pad - kernel_w) // stride + 1
    
    X_padded = np.zeros((N, C, H_pad, W_pad))
    
    # Reshape cols back to patches
    cols_reshaped = cols.reshape(C, kernel_h, kernel_w, N, out_h, out_w)
    cols_reshaped = cols_reshaped.transpose(3, 0, 4, 5, 1, 2)  # (N, C, out_h, out_w, kH, kW)
    
    # Accumulate patches back into image (with overlap addition)
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            w_start = j * stride
            X_padded[:, :, h_start:h_start+kernel_h, w_start:w_start+kernel_w] += \
                cols_reshaped[:, :, i, j, :, :]
    
    # Remove padding
    if padding > 0:
        return X_padded[:, :, padding:-padding, padding:-padding]
    return X_padded
