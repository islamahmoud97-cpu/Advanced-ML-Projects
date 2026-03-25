"""
Fast Convolution Benchmark
============================
Based on: Zemke, AML Lecture 6

Races 5 convolution methods head-to-head:
  1. Naive (nested loops)         — O(N²k²) baseline
  2. im2col + GEMM               — O(N²k²) but uses BLAS [Slide 24]
  3. FFT (convolution theorem)    — O(N²·log N) [Slide 12]
  4. Winograd F(2,3)              — Minimal multiplications [Slide 31]
  5. Separable (rank-1 kernels)   — O(2N²k) for Gaussian [Slide 5]

Tests across:
  - Image sizes: 32, 64, 128, 256, 512
  - Kernel sizes: 3×3, 5×5, 7×7, 11×11, 15×15

Run: python benchmark.py
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os

from convolution_methods import (
    conv2d_naive, conv2d_im2col, conv2d_fft,
    conv2d_winograd_f2x3, conv2d_separable,
    ALL_METHODS,
)


def make_gaussian_kernel(size, sigma=1.0):
    """Gaussian kernel (separable!) for benchmarking."""
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()


def make_random_kernel(size):
    """Random kernel (not separable) for benchmarking."""
    return np.random.randn(size, size)


def benchmark_time(method_fn, image, kernel, n_runs=3):
    """Time a convolution method, return median time in ms."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = method_fn(image, kernel)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)
    return np.median(times), result


def verify_correctness():
    """Verify all methods produce the same output as naive."""
    np.random.seed(42)
    img = np.random.randn(32, 32)
    kernel_3x3 = np.random.randn(3, 3)
    kernel_5x5 = np.random.randn(5, 5)

    ref_3 = conv2d_naive(img, kernel_3x3)
    ref_5 = conv2d_naive(img, kernel_5x5)

    print("  Correctness Verification (vs Naive):")
    print("  " + "-" * 50)

    for name, fn in ALL_METHODS.items():
        if name == "naive":
            continue
        try:
            if name == "winograd":
                result = fn(img, kernel_3x3)
                err = np.max(np.abs(result - ref_3))
                print(f"  {name:12s} (3×3): max error = {err:.2e}  {'PASS' if err < 1e-6 else 'APPROX'}")
            elif name == "separable":
                gauss = make_gaussian_kernel(5)
                ref_g = conv2d_naive(img, gauss)
                result = fn(img, gauss)
                err = np.max(np.abs(result - ref_g))
                print(f"  {name:12s} (5×5 Gauss): max error = {err:.2e}  {'PASS' if err < 1e-6 else 'APPROX'}")
            else:
                result = fn(img, kernel_5x5)
                err = np.max(np.abs(result - ref_5))
                print(f"  {name:12s} (5×5): max error = {err:.2e}  {'PASS' if err < 0.01 else 'FAIL'}")
        except Exception as e:
            print(f"  {name:12s}: ERROR — {e}")

    print()


def run_image_size_benchmark():
    """Benchmark across different image sizes with fixed 3×3 kernel."""
    print("  Benchmark 1: Image Size (fixed 3×3 kernel)")
    print("  " + "-" * 50)

    image_sizes = [32, 64, 128, 256]
    kernel = np.random.randn(3, 3)
    methods = {
        "Naive": conv2d_naive,
        "im2col": conv2d_im2col,
        "FFT": conv2d_fft,
        "Winograd": conv2d_winograd_f2x3,
    }

    results = {name: [] for name in methods}

    for size in image_sizes:
        img = np.random.randn(size, size)
        print(f"  Image {size}×{size}: ", end="", flush=True)
        for name, fn in methods.items():
            n_runs = 5 if size <= 128 else 3
            t, _ = benchmark_time(fn, img, kernel, n_runs=n_runs)
            results[name].append(t)
            print(f"{name}={t:.1f}ms  ", end="", flush=True)
        print()

    return image_sizes, results


def run_kernel_size_benchmark():
    """Benchmark across different kernel sizes with fixed 128×128 image."""
    print("\n  Benchmark 2: Kernel Size (fixed 128×128 image)")
    print("  " + "-" * 50)

    kernel_sizes = [3, 5, 7, 11, 15]
    img = np.random.randn(128, 128)
    methods = {
        "Naive": conv2d_naive,
        "im2col": conv2d_im2col,
        "FFT": conv2d_fft,
    }

    results = {name: [] for name in methods}

    for ks in kernel_sizes:
        kernel = np.random.randn(ks, ks)
        print(f"  Kernel {ks}×{ks}: ", end="", flush=True)
        for name, fn in methods.items():
            t, _ = benchmark_time(fn, img, kernel, n_runs=5)
            results[name].append(t)
            print(f"{name}={t:.1f}ms  ", end="", flush=True)
        print()

    return kernel_sizes, results


def run_separable_benchmark():
    """Benchmark separable vs non-separable for Gaussian blur."""
    print("\n  Benchmark 3: Separable Gaussian Blur")
    print("  " + "-" * 50)

    kernel_sizes = [3, 5, 7, 11, 15, 21]
    img = np.random.randn(256, 256)

    results_sep = []
    results_im2col = []
    results_naive = []

    for ks in kernel_sizes:
        gauss = make_gaussian_kernel(ks, sigma=ks / 3.0)
        print(f"  Gaussian {ks}×{ks}: ", end="", flush=True)

        t_sep, _ = benchmark_time(conv2d_separable, img, gauss, n_runs=3)
        t_im2, _ = benchmark_time(conv2d_im2col, img, gauss, n_runs=3)
        t_naive, _ = benchmark_time(conv2d_naive, img, gauss, n_runs=2)

        results_sep.append(t_sep)
        results_im2col.append(t_im2)
        results_naive.append(t_naive)

        speedup = t_naive / t_sep if t_sep > 0 else 0
        print(f"Naive={t_naive:.0f}ms  im2col={t_im2:.0f}ms  Separable={t_sep:.0f}ms  (speedup: {speedup:.1f}×)")

    return kernel_sizes, results_naive, results_im2col, results_sep


def plot_all_results(img_sizes, img_results, ks_sizes, ks_results,
                     sep_sizes, sep_naive, sep_im2col, sep_sep):
    """Generate all benchmark plots."""
    os.makedirs("figures", exist_ok=True)

    colors = {
        "Naive": "#6b7280", "im2col": "#2563eb",
        "FFT": "#dc2626", "Winograd": "#16a34a",
        "Separable": "#ea580c",
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # ── 1) Image size benchmark ───────────────────────────────────
    ax = axes[0, 0]
    for name, times in img_results.items():
        ax.plot(img_sizes, times, "o-", color=colors[name], linewidth=2.5,
                markersize=8, label=name)
    ax.set_xlabel("Image Size (pixels)", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("Image Size Scaling — 3×3 Kernel", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # ── 2) Kernel size benchmark ──────────────────────────────────
    ax = axes[0, 1]
    for name, times in ks_results.items():
        ax.plot(ks_sizes, times, "s-", color=colors[name], linewidth=2.5,
                markersize=8, label=name)
    ax.set_xlabel("Kernel Size", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("Kernel Size Scaling — 128×128 Image", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # ── 3) Separable benchmark ────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(sep_sizes, sep_naive, "o-", color=colors["Naive"], linewidth=2.5, markersize=8, label="Naive")
    ax.plot(sep_sizes, sep_im2col, "s-", color=colors["im2col"], linewidth=2.5, markersize=8, label="im2col")
    ax.plot(sep_sizes, sep_sep, "^-", color=colors["Separable"], linewidth=2.5, markersize=8, label="Separable")
    ax.set_xlabel("Gaussian Kernel Size", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("Separable Gaussian Blur — 256×256 Image\nSeparable exploits rank-1 structure!",
                fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # ── 4) Speedup chart ──────────────────────────────────────────
    ax = axes[1, 1]
    speedups_im2col = [n / i if i > 0 else 0 for n, i in zip(sep_naive, sep_im2col)]
    speedups_sep = [n / s if s > 0 else 0 for n, s in zip(sep_naive, sep_sep)]

    x = np.arange(len(sep_sizes))
    w = 0.35
    ax.bar(x - w/2, speedups_im2col, w, color=colors["im2col"], alpha=0.8, label="im2col vs Naive")
    ax.bar(x + w/2, speedups_sep, w, color=colors["Separable"], alpha=0.8, label="Separable vs Naive")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}×{s}" for s in sep_sizes])
    ax.set_xlabel("Kernel Size", fontsize=12)
    ax.set_ylabel("Speedup (×)", fontsize=12)
    ax.set_title("Speedup over Naive Convolution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis="y")
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Fast Convolution Benchmark — Naive vs im2col vs FFT vs Winograd vs Separable\n"
        "[Lecture 6: FFT, im2col & Winograd]",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig("figures/benchmark.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  ✅ figures/benchmark.png")


def plot_complexity_theory(save_path=None):
    """Theoretical complexity comparison plot."""
    N = np.arange(16, 1025, 16)
    k_values = [3, 7, 15]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, k in zip(axes, k_values):
        naive = N**2 * k**2
        im2col = N**2 * k**2  # Same ops but BLAS-accelerated
        fft = N**2 * np.log2(N**2)
        winograd = N**2 * (k**2) * (4/9) if k == 3 else N**2 * k**2
        separable = 2 * N**2 * k

        ax.semilogy(N, naive, color="#6b7280", linewidth=2, label=f"Naive O(N²k²)")
        ax.semilogy(N, fft, color="#dc2626", linewidth=2, label=f"FFT O(N²log N)")
        ax.semilogy(N, separable, color="#ea580c", linewidth=2, linestyle="--",
                    label=f"Separable O(2N²k)")
        if k == 3:
            ax.semilogy(N, winograd, color="#16a34a", linewidth=2, linestyle="-.",
                        label=f"Winograd (~4/9 naive)")

        ax.set_xlabel("Image Size N", fontsize=11)
        ax.set_ylabel("Operations (log)", fontsize=11)
        ax.set_title(f"Kernel {k}×{k}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, which="both")

    fig.suptitle(
        "Theoretical Complexity — When Does Each Method Win?\n"
        "[FFT wins for large kernels, im2col wins for small kernels, Separable wins for Gaussian]",
        fontsize=15, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


def plot_method_summary(save_path=None):
    """Visual summary table of all methods."""
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.axis("off")

    headers = ["Method", "Complexity", "Best For", "Limitation", "Lecture Ref"]
    rows = [
        ["Naive", "O(N²k²)", "Reference only", "Very slow", "L4"],
        ["im2col + GEMM", "O(N²k²) via BLAS", "Small kernels (3×3, 5×5)", "High memory (patches)", "L6, Slide 24"],
        ["FFT", "O(N²·log N)", "Large kernels (11+)", "Overhead for small k", "L6, Slide 12"],
        ["Winograd F(2,3)", "~4/9 of naive (3×3)", "3×3 kernels only", "Only 3×3, numerical issues", "L6, Slide 31"],
        ["Separable", "O(2N²k)", "Gaussian, box blur", "Only rank-1 kernels", "L6, Slide 5"],
    ]

    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    row_colors = ["#6b7280", "#2563eb", "#dc2626", "#16a34a", "#ea580c"]
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#1e293b")
        cell.set_text_props(color="white", fontweight="bold")

    for i in range(len(rows)):
        table[i+1, 0].set_text_props(fontweight="bold", color=row_colors[i])
        for j in range(len(headers)):
            table[i+1, j].set_facecolor("#f8fafc" if i % 2 == 0 else "white")

    fig.suptitle("Fast Convolution Methods — Summary", fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("  Fast Convolution Benchmark")
    print("=" * 60)

    # Verify correctness
    print("\n")
    verify_correctness()

    # Run benchmarks
    img_sizes, img_results = run_image_size_benchmark()
    ks_sizes, ks_results = run_kernel_size_benchmark()
    sep_sizes, sep_naive, sep_im2col, sep_sep = run_separable_benchmark()

    # Generate plots
    print("\n  Generating plots...")
    plot_all_results(img_sizes, img_results, ks_sizes, ks_results,
                     sep_sizes, sep_naive, sep_im2col, sep_sep)
    plot_complexity_theory(save_path="figures/complexity.png")
    plot_method_summary(save_path="figures/summary_table.png")

    print("\n✨ All benchmarks complete!")
