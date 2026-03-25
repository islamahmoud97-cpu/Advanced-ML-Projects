"""
Three Convolution Methods — Naive, im2col, FFT
=================================================
Based on: Zemke, AML Lecture 6

Lecture 6 covers three fast convolution approaches:
  1. Separable kernels   — split 2D into two 1D convolutions
  2. FFT convolution     — Toeplitz → circulant → DFT multiplication
  3. im2col + GEMM       — reshape patches → matrix multiply (BLAS Level 3)
  4. Winograd            — minimal multiplication algorithm

This module implements ALL methods so we can benchmark them head-to-head.
"""

import numpy as np
import time
from typing import Tuple


# ═══════════════════════════════════════════════════════════════════════════
# METHOD 1: NAIVE CONVOLUTION  (6 nested loops — baseline)
# ═══════════════════════════════════════════════════════════════════════════
def conv2d_naive(image: np.ndarray, kernel: np.ndarray, padding: int = 0) -> np.ndarray:
    """
    Naive 2D convolution with explicit loops.
    
    Complexity: O(H_out × W_out × kH × kW) per image.
    This is SLOW but correct — used as reference.
    """
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    
    H, W = image.shape
    kH, kW = kernel.shape
    out_h = H - kH + 1
    out_w = W - kW + 1
    
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            output[i, j] = np.sum(image[i:i+kH, j:j+kW] * kernel)
    
    return output


# ═══════════════════════════════════════════════════════════════════════════
# METHOD 2: im2col + GEMM  [Lecture 6, Slides 24-30]
# ═══════════════════════════════════════════════════════════════════════════
def im2col_2d(image: np.ndarray, kH: int, kW: int, padding: int = 0) -> np.ndarray:
    """
    Extract all patches into columns.
    
    image: (H, W)
    Returns: (kH*kW, out_h*out_w)
    """
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    
    H, W = image.shape
    out_h = H - kH + 1
    out_w = W - kW + 1
    
    # Use stride tricks for efficiency
    shape = (out_h, out_w, kH, kW)
    strides = (image.strides[0], image.strides[1], image.strides[0], image.strides[1])
    patches = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)
    
    return patches.reshape(-1, kH * kW).T  # (kH*kW, out_h*out_w)


def conv2d_im2col(image: np.ndarray, kernel: np.ndarray, padding: int = 0) -> np.ndarray:
    """
    im2col convolution — converts to matrix multiplication.
    
    [Lecture 6]: "GEMM is fast. Can we use GEMM for convolution? YES!"
    
    Steps:
      1. im2col(image) → columns matrix
      2. kernel.flatten() @ columns → output
      3. Reshape to 2D
    """
    kH, kW = kernel.shape
    
    if padding > 0:
        image_padded = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    else:
        image_padded = image
    
    H, W = image_padded.shape
    out_h = H - kH + 1
    out_w = W - kW + 1
    
    cols = im2col_2d(image_padded, kH, kW, padding=0)  # (kH*kW, out_h*out_w)
    kernel_flat = kernel.reshape(1, -1)                   # (1, kH*kW)
    
    output = (kernel_flat @ cols).reshape(out_h, out_w)   # GEMM!
    return output


# ═══════════════════════════════════════════════════════════════════════════
# METHOD 3: FFT CONVOLUTION  [Lecture 6, Slides 12-23]
# ═══════════════════════════════════════════════════════════════════════════
def conv2d_fft(image: np.ndarray, kernel: np.ndarray, padding: int = 0) -> np.ndarray:
    """
    FFT-based convolution using the convolution theorem.
    
    [Lecture 6]: Toeplitz → circulant → diagonalized by DFT
    
    Convolution Theorem:
      conv(x, h) = IFFT(FFT(x) · FFT(h))
    
    Complexity: O(N·log(N)) instead of O(N·k)
    Faster than naive for LARGE kernels, slower for small kernels.
    """
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    
    H, W = image.shape
    kH, kW = kernel.shape
    
    # Output size (valid convolution)
    out_h = H - kH + 1
    out_w = W - kW + 1
    
    # FFT size (must be at least H+kH-1 for linear convolution)
    fft_h = H + kH - 1
    fft_w = W + kW - 1
    
    # Pad to next power of 2 for FFT efficiency
    fft_h = int(2 ** np.ceil(np.log2(fft_h)))
    fft_w = int(2 ** np.ceil(np.log2(fft_w)))
    
    # FFT of both, element-wise multiply, IFFT
    # Flip kernel for cross-correlation (DL convention)
    kernel_flipped = kernel[::-1, ::-1]
    Image_fft = np.fft.fft2(image, s=(fft_h, fft_w))
    Kernel_fft = np.fft.fft2(kernel_flipped, s=(fft_h, fft_w))
    
    result = np.real(np.fft.ifft2(Image_fft * Kernel_fft))
    
    # Extract valid region (cross-correlation = convolution with flipped kernel)
    # For cross-correlation, we need the center portion
    return result[kH-1:kH-1+out_h, kW-1:kW-1+out_w]


# ═══════════════════════════════════════════════════════════════════════════
# METHOD 4: WINOGRAD  [Lecture 6, Slides 31-40]
# ═══════════════════════════════════════════════════════════════════════════
def conv2d_winograd_f2x3(image: np.ndarray, kernel: np.ndarray, padding: int = 0) -> np.ndarray:
    """
    Winograd F(2,3) convolution for 3×3 kernels.
    
    [Lecture 6]: Minimal multiplication algorithm.
    
    For a 3×3 kernel on a 4×4 tile producing 2×2 output:
      - Naive: 2×2×3×3 = 36 multiplications per tile
      - Winograd F(2,3): only 16 multiplications per tile!
    
    Transforms:
      Input transform:   B^T · d · B
      Kernel transform:  G · g · G^T
      Output transform:  A^T · (element-wise multiply) · A
    
    Only works for 3×3 kernels (most common in modern CNNs).
    """
    if kernel.shape != (3, 3):
        # Fall back to im2col for non-3x3 kernels
        return conv2d_im2col(image, kernel, padding)
    
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    
    H, W = image.shape
    out_h = H - 2  # 3x3 kernel, valid
    out_w = W - 2
    
    # Winograd transformation matrices for F(2,3)
    # B^T: input transform (4×4)
    BT = np.array([
        [ 1,  0, -1,  0],
        [ 0,  1,  1,  0],
        [ 0, -1,  1,  0],
        [ 0,  1,  0, -1],
    ], dtype=np.float64)
    
    B = BT.T
    
    # G: kernel transform (4×3)
    G = np.array([
        [   1,    0,   0],
        [ 0.5,  0.5, 0.5],
        [ 0.5, -0.5, 0.5],
        [   0,    0,   1],
    ], dtype=np.float64)
    
    GT = G.T
    
    # A^T: output transform (2×4)
    AT = np.array([
        [1, 1,  1, 0],
        [0, 1, -1, -1],
    ], dtype=np.float64)
    
    A = AT.T
    
    # Transform kernel once (reused for all tiles)
    U = G @ kernel @ GT  # (4, 4)
    
    # Process in 4×4 tiles with 2-pixel stride (2×2 output per tile)
    # Pad if necessary
    n_tiles_h = (out_h + 1) // 2
    n_tiles_w = (out_w + 1) // 2
    
    pad_h = n_tiles_h * 2 + 2 - H
    pad_w = n_tiles_w * 2 + 2 - W
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, max(0, pad_h)), (0, max(0, pad_w))), mode='constant')
    
    output = np.zeros((n_tiles_h * 2, n_tiles_w * 2))
    
    for ti in range(n_tiles_h):
        for tj in range(n_tiles_w):
            # Extract 4×4 tile
            r = ti * 2
            c = tj * 2
            d = image[r:r+4, c:c+4]
            
            if d.shape != (4, 4):
                continue
            
            # Winograd: V = B^T · d · B
            V = BT @ d @ B
            
            # Element-wise multiply
            M = V * U
            
            # Output transform: Y = A^T · M · A
            Y = AT @ M @ A  # (2, 2)
            
            output[ti*2:ti*2+2, tj*2:tj*2+2] = Y
    
    return output[:out_h, :out_w]


# ═══════════════════════════════════════════════════════════════════════════
# METHOD 5: SEPARABLE CONVOLUTION  [Lecture 6, Slides 5-11]
# ═══════════════════════════════════════════════════════════════════════════
def conv2d_separable(image: np.ndarray, kernel: np.ndarray, padding: int = 0) -> np.ndarray:
    """
    Separable convolution — split 2D kernel into two 1D passes.
    
    [Lecture 6]: "2D: separable kernels"
    
    Only works if kernel = col_vector @ row_vector (rank 1).
    Gaussian blur is separable!
    
    Complexity: O(H·W·k) + O(H·W·k) = O(2·H·W·k)
    Instead of: O(H·W·k²)
    """
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    
    # Check if kernel is separable via SVD
    U, S, Vt = np.linalg.svd(kernel)
    
    if S[1] / S[0] > 1e-6:
        # Not separable, fall back to im2col
        return conv2d_im2col(image, kernel, padding=0)
    
    # Rank-1 decomposition: kernel ≈ σ₁ · u₁ · v₁^T
    col_kernel = (U[:, 0] * np.sqrt(S[0])).reshape(-1, 1)
    row_kernel = (Vt[0, :] * np.sqrt(S[0])).reshape(1, -1)
    
    H, W = image.shape
    kH, kW = kernel.shape
    
    # 1D convolution along columns (vertical)
    temp = np.zeros((H - kH + 1, W))
    for i in range(H - kH + 1):
        temp[i, :] = np.sum(image[i:i+kH, :] * col_kernel, axis=0)
    
    # 1D convolution along rows (horizontal)
    out_h = H - kH + 1
    out_w = W - kW + 1
    output = np.zeros((out_h, out_w))
    for j in range(out_w):
        output[:, j] = np.sum(temp[:, j:j+kW] * row_kernel, axis=1)
    
    return output


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════
ALL_METHODS = {
    "naive":     conv2d_naive,
    "im2col":    conv2d_im2col,
    "fft":       conv2d_fft,
    "winograd":  conv2d_winograd_f2x3,
    "separable": conv2d_separable,
}
