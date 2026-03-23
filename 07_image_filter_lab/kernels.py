"""
Image Kernels / Filters — Complete Collection
================================================
Based on: Zemke, AML Lecture 4, Slides 8-20

From Lecture 4:
  "There are natural kernels/filters:
   - derivative filter d = [-1, 1] / h
   - Laplacian d = [-1, 2, -1] / h²
   - averaging (low-pass) d = [1, 1, 1] / 3"

This module implements all classic 2D image kernels and the
convolution operation from scratch (no OpenCV, no scipy).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class KernelInfo:
    """Metadata for a filter kernel."""
    name: str
    kernel: np.ndarray
    category: str       # "edge", "blur", "sharpen", "emboss", "other"
    description: str
    color: str


# ═══════════════════════════════════════════════════════════════════════════
# Convolution Operation (from scratch, no scipy)
# ═══════════════════════════════════════════════════════════════════════════
def convolve2d(image: np.ndarray, kernel: np.ndarray, padding: str = "same") -> np.ndarray:
    """
    2D cross-correlation (what deep learning calls 'convolution').
    [Lecture 4, Slide 10-11]
    
    (x ⋆ d)ᵢ = Σ x_{i+k} · d_k
    
    Parameters
    ----------
    image : (H, W) grayscale image
    kernel : (kH, kW) filter kernel
    padding : "same" (output = input size) or "valid" (no padding)
    """
    kH, kW = kernel.shape
    
    if padding == "same":
        pH = kH // 2
        pW = kW // 2
        image_padded = np.pad(image, ((pH, pH), (pW, pW)), mode="constant", constant_values=0)
    else:
        image_padded = image
    
    H, W = image_padded.shape
    out_h = H - kH + 1
    out_w = W - kW + 1
    
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            output[i, j] = np.sum(image_padded[i:i+kH, j:j+kW] * kernel)
    
    return output


def apply_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply filter to potentially multi-channel image. Returns float result."""
    if image.ndim == 3:
        # Apply to each channel
        result = np.zeros_like(image, dtype=np.float64)
        for c in range(image.shape[2]):
            result[:, :, c] = convolve2d(image[:, :, c], kernel)
        return result
    else:
        return convolve2d(image, kernel)


# ═══════════════════════════════════════════════════════════════════════════
# IDENTITY
# ═══════════════════════════════════════════════════════════════════════════
IDENTITY = KernelInfo(
    name="Identity",
    kernel=np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=np.float64),
    category="other",
    description="Does nothing — output = input. Baseline reference.",
    color="#6b7280",
)


# ═══════════════════════════════════════════════════════════════════════════
# EDGE DETECTION  [Lecture 4: derivative filter d = [-1, 1] / h]
# ═══════════════════════════════════════════════════════════════════════════
SOBEL_X = KernelInfo(
    name="Sobel X (Horizontal Edges)",
    kernel=np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64),
    category="edge",
    description="Detects vertical edges (horizontal gradient). Combines Gaussian smoothing with differentiation.",
    color="#2563eb",
)

SOBEL_Y = KernelInfo(
    name="Sobel Y (Vertical Edges)",
    kernel=np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64),
    category="edge",
    description="Detects horizontal edges (vertical gradient).",
    color="#7c3aed",
)

PREWITT_X = KernelInfo(
    name="Prewitt X",
    kernel=np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float64),
    category="edge",
    description="Simpler version of Sobel (no Gaussian weighting).",
    color="#0ea5e9",
)

SCHARR_X = KernelInfo(
    name="Scharr X",
    kernel=np.array([
        [-3,  0,  3],
        [-10, 0, 10],
        [-3,  0,  3]
    ], dtype=np.float64),
    category="edge",
    description="More accurate rotational symmetry than Sobel.",
    color="#0d9488",
)


# ═══════════════════════════════════════════════════════════════════════════
# LAPLACIAN  [Lecture 4: d = [-1, 2, -1] / h² → 2D Laplacian]
# ═══════════════════════════════════════════════════════════════════════════
LAPLACIAN = KernelInfo(
    name="Laplacian (4-connected)",
    kernel=np.array([
        [ 0,  1,  0],
        [ 1, -4,  1],
        [ 0,  1,  0]
    ], dtype=np.float64),
    category="edge",
    description="2D Laplacian ∇² = ∂²/∂x² + ∂²/∂y². Detects ALL edges (isotropic). From Lecture 4: d = [-1, 2, -1]/h².",
    color="#dc2626",
)

LAPLACIAN_8 = KernelInfo(
    name="Laplacian (8-connected)",
    kernel=np.array([
        [ 1,  1,  1],
        [ 1, -8,  1],
        [ 1,  1,  1]
    ], dtype=np.float64),
    category="edge",
    description="Laplacian including diagonal neighbors. Stronger edge detection.",
    color="#b91c1c",
)


# ═══════════════════════════════════════════════════════════════════════════
# BLUR / SMOOTHING  [Lecture 4: averaging d = [1,1,1] / 3]
# ═══════════════════════════════════════════════════════════════════════════
BOX_BLUR_3 = KernelInfo(
    name="Box Blur 3×3",
    kernel=np.ones((3, 3), dtype=np.float64) / 9.0,
    category="blur",
    description="Simple averaging filter. From Lecture 4: d = [1,1,...,1]/n. Cheap low-pass filter.",
    color="#16a34a",
)

BOX_BLUR_5 = KernelInfo(
    name="Box Blur 5×5",
    kernel=np.ones((5, 5), dtype=np.float64) / 25.0,
    category="blur",
    description="Stronger box blur. More smoothing, more spatial information lost.",
    color="#15803d",
)


def make_gaussian_kernel(size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """
    Create Gaussian blur kernel.
    G(x,y) = (1/2πσ²) · exp(-(x²+y²) / 2σ²)
    """
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


GAUSSIAN_3 = KernelInfo(
    name="Gaussian Blur 3×3 (σ=1)",
    kernel=make_gaussian_kernel(3, 1.0),
    category="blur",
    description="Gaussian-weighted blur. Better than box blur (no ringing artifacts).",
    color="#22c55e",
)

GAUSSIAN_5 = KernelInfo(
    name="Gaussian Blur 5×5 (σ=1.5)",
    kernel=make_gaussian_kernel(5, 1.5),
    category="blur",
    description="Wider Gaussian blur for stronger smoothing.",
    color="#4ade80",
)


# ═══════════════════════════════════════════════════════════════════════════
# SHARPEN
# ═══════════════════════════════════════════════════════════════════════════
SHARPEN = KernelInfo(
    name="Sharpen",
    kernel=np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float64),
    category="sharpen",
    description="Identity + Laplacian = Sharpen. Enhances edges while keeping original image.",
    color="#ea580c",
)

UNSHARP_MASK = KernelInfo(
    name="Unsharp Mask",
    kernel=(1.0 / -256.0) * np.array([
        [ 1,  4,   6,  4,  1],
        [ 4, 16,  24, 16,  4],
        [ 6, 24,-476, 24,  6],
        [ 4, 16,  24, 16,  4],
        [ 1,  4,   6,  4,  1]
    ], dtype=np.float64),
    category="sharpen",
    description="Professional sharpening: original - blurred = sharp details.",
    color="#f97316",
)


# ═══════════════════════════════════════════════════════════════════════════
# EMBOSS
# ═══════════════════════════════════════════════════════════════════════════
EMBOSS = KernelInfo(
    name="Emboss",
    kernel=np.array([
        [-2, -1, 0],
        [-1,  1, 1],
        [ 0,  1, 2]
    ], dtype=np.float64),
    category="emboss",
    description="Creates a 3D raised/embossed effect.",
    color="#a855f7",
)


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════
ALL_KERNELS = [
    IDENTITY, SOBEL_X, SOBEL_Y, PREWITT_X, SCHARR_X,
    LAPLACIAN, LAPLACIAN_8,
    BOX_BLUR_3, BOX_BLUR_5, GAUSSIAN_3, GAUSSIAN_5,
    SHARPEN, UNSHARP_MASK, EMBOSS,
]

EDGE_KERNELS = [k for k in ALL_KERNELS if k.category == "edge"]
BLUR_KERNELS = [k for k in ALL_KERNELS if k.category == "blur"]
