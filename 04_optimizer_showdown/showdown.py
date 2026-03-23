"""
The Optimizer Showdown — Who Reaches the Minimum First?
=========================================================
Based on: Zemke, AML Lecture 3, Slides 6-22

This script races 7 optimizers against each other on 5 test functions:

Optimizers (Lecture 3):
  - SGD (vanilla)
  - Momentum (γ=0.9)
  - NAG (Nesterov Accelerated)
  - Adagrad (adaptive LR)
  - RMSprop (exponential decay)
  - Adam (momentum + RMSprop)
  - Nadam (Nesterov + Adam)

Test Functions:
  - Rosenbrock (narrow valley)
  - Beale (flat regions + walls)
  - Himmelblau (4 minima)
  - Saddle point (escape test)
  - Styblinski-Tang (local minima)

Run: python showdown.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

from optimizers import SGD, MomentumSGD, NAG, Adagrad, RMSprop, Adam, Nadam
from test_functions import ALL_FUNCTIONS, ROSENBROCK, BEALE, HIMMELBLAU, SADDLE, STYBLINSKI


def create_optimizers():
    """Create all optimizers with tuned learning rates."""
    return [
        SGD(lr=0.001),
        MomentumSGD(lr=0.001, gamma=0.9),
        NAG(lr=0.001, gamma=0.9),
        Adagrad(lr=0.1),
        RMSprop(lr=0.01, beta=0.9),
        Adam(lr=0.01, beta1=0.9, beta2=0.999),
        Nadam(lr=0.01),
    ]


def plot_contour_race(test_fn, optimizers, n_steps=500, save_path=None):
    """Plot all optimizer paths on the contour of a test function."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Contour
    x = np.linspace(test_fn.xlim[0], test_fn.xlim[1], 300)
    y = np.linspace(test_fn.ylim[0], test_fn.ylim[1], 300)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = test_fn.f(np.array([X[i, j], Y[i, j]]))
    
    # Clip extreme values for better visualization
    Z_clipped = np.clip(Z, Z.min(), np.percentile(Z, 98))
    
    try:
        ax.contourf(X, Y, Z_clipped, levels=40, cmap="Greys", alpha=0.3)
        ax.contour(X, Y, Z_clipped, levels=20, cmap="Greys", alpha=0.4, linewidths=0.5)
    except Exception:
        ax.contourf(X, Y, Z_clipped, levels=20, cmap="Greys", alpha=0.3)
    
    # Run each optimizer
    results = []
    for opt in optimizers:
        result = opt.optimize(test_fn.f, test_fn.grad_f, test_fn.x0.copy(), n_steps)
        results.append(result)
        
        path = np.array(result.path)
        # Clip path to visible region
        mask = ((path[:, 0] >= test_fn.xlim[0]) & (path[:, 0] <= test_fn.xlim[1]) &
                (path[:, 1] >= test_fn.ylim[0]) & (path[:, 1] <= test_fn.ylim[1]))
        path_vis = path[mask]
        
        if len(path_vis) > 1:
            ax.plot(path_vis[:, 0], path_vis[:, 1], color=result.color,
                    linewidth=1.8, alpha=0.8, label=result.name)
            ax.scatter(path_vis[0, 0], path_vis[0, 1], color=result.color,
                      s=80, marker="o", edgecolors="white", linewidths=1, zorder=5)
            ax.scatter(path_vis[-1, 0], path_vis[-1, 1], color=result.color,
                      s=100, marker="*", edgecolors="white", linewidths=1, zorder=5)
    
    # Mark true minimum
    ax.scatter(test_fn.minimum[0], test_fn.minimum[1], color="#dc2626",
              s=250, marker="*", edgecolors="white", linewidths=2, zorder=10,
              label="Global minimum")
    
    ax.set_title(f"{test_fn.name}\n{test_fn.description}",
                fontsize=14, fontweight="bold")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.15)
    ax.set_xlim(test_fn.xlim)
    ax.set_ylim(test_fn.ylim)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()
    
    return results


def plot_convergence_comparison(test_fn, results, save_path=None):
    """Plot loss curves and gradient norms for all optimizers."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1) Loss over iterations
    ax = axes[0]
    for r in results:
        if len(r.losses) > 0:
            ax.semilogy(r.losses, color=r.color, linewidth=2, alpha=0.8, label=r.name)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("f(x) — log scale", fontsize=12)
    ax.set_title("Convergence Speed", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, which="both")
    
    # 2) Distance to minimum
    ax = axes[1]
    for r in results:
        path = np.array(r.path)
        dist = np.linalg.norm(path - test_fn.minimum, axis=1)
        ax.semilogy(dist, color=r.color, linewidth=2, alpha=0.8, label=r.name)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("||x - x*|| — log scale", fontsize=12)
    ax.set_title("Distance to Minimum", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, which="both")
    
    # 3) Gradient norm
    ax = axes[2]
    for r in results:
        if len(r.grad_norms) > 0:
            # Smooth with moving average
            window = min(20, len(r.grad_norms) // 5 + 1)
            smoothed = np.convolve(r.grad_norms, np.ones(window)/window, mode='valid')
            ax.semilogy(smoothed, color=r.color, linewidth=2, alpha=0.8, label=r.name)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("||∇f|| — log scale", fontsize=12)
    ax.set_title("Gradient Magnitude", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, which="both")
    
    fig.suptitle(f"Optimizer Comparison on {test_fn.name} — [Lecture 3]",
                fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


def plot_grand_showdown(save_path=None):
    """
    The GRAND SHOWDOWN: All optimizers on all functions.
    5 rows (functions) × 1 column (contour with all paths)
    """
    functions = [ROSENBROCK, BEALE, HIMMELBLAU, SADDLE, STYBLINSKI]
    n_steps = 500
    
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    axes_flat = axes.ravel()
    
    summary = []
    
    for idx, test_fn in enumerate(functions):
        ax = axes_flat[idx]
        
        x = np.linspace(test_fn.xlim[0], test_fn.xlim[1], 200)
        y = np.linspace(test_fn.ylim[0], test_fn.ylim[1], 200)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = test_fn.f(np.array([X[i, j], Y[i, j]]))
        
        Z_clip = np.clip(Z, Z.min(), np.percentile(Z, 95))
        ax.contourf(X, Y, Z_clip, levels=25, cmap="Greys", alpha=0.25)
        ax.contour(X, Y, Z_clip, levels=15, cmap="Greys", alpha=0.3, linewidths=0.3)
        
        optimizers = create_optimizers()
        fn_results = {}
        
        for opt in optimizers:
            result = opt.optimize(test_fn.f, test_fn.grad_f, test_fn.x0.copy(), n_steps)
            path = np.array(result.path)
            
            mask = ((path[:, 0] >= test_fn.xlim[0]) & (path[:, 0] <= test_fn.xlim[1]) &
                    (path[:, 1] >= test_fn.ylim[0]) & (path[:, 1] <= test_fn.ylim[1]))
            path_vis = path[mask]
            
            if len(path_vis) > 1:
                ax.plot(path_vis[:, 0], path_vis[:, 1], color=result.color,
                        linewidth=1.5, alpha=0.8, label=result.name)
                ax.scatter(path_vis[-1, 0], path_vis[-1, 1], color=result.color,
                          s=60, marker="*", edgecolors="white", linewidths=0.5, zorder=5)
            
            final_dist = np.linalg.norm(path[-1] - test_fn.minimum)
            fn_results[result.name] = final_dist
        
        ax.scatter(test_fn.minimum[0], test_fn.minimum[1], color="#dc2626",
                  s=150, marker="*", edgecolors="white", linewidths=1.5, zorder=10)
        
        # Find winner
        winner = min(fn_results, key=fn_results.get)
        ax.set_title(f"{test_fn.name}\nWinner: {winner}", fontsize=12, fontweight="bold")
        ax.set_xlim(test_fn.xlim)
        ax.set_ylim(test_fn.ylim)
        ax.grid(True, alpha=0.1)
        
        summary.append((test_fn.name, winner, fn_results))
    
    # Last panel: legend + scoreboard
    ax_legend = axes_flat[5]
    ax_legend.axis("off")
    
    # Count wins
    win_counts = {}
    for _, winner, _ in summary:
        win_counts[winner] = win_counts.get(winner, 0) + 1
    
    sorted_wins = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)
    
    text = "SCOREBOARD\n" + "=" * 30 + "\n\n"
    for name, wins in sorted_wins:
        medals = "[1st]" if wins == max(w for _, w in sorted_wins) else "[2nd]" if wins >= 2 else "     "
        text += f"  {medals} {name}: {wins} win(s)\n"
    
    text += "\n" + "-" * 30 + "\n\n"
    for fn_name, winner, _ in summary:
        text += f"  {fn_name}: {winner}\n"
    
    ax_legend.text(0.1, 0.5, text, fontsize=13, fontfamily="monospace",
                  verticalalignment="center", transform=ax_legend.transAxes,
                  bbox=dict(boxstyle="round", facecolor="#f8fafc", edgecolor="#e2e8f0"))
    
    # Add global legend
    optimizers = create_optimizers()
    for opt in optimizers:
        result = opt.optimize(lambda x: 0, lambda x: np.zeros(2), np.zeros(2), 1)
        ax_legend.plot([], [], color=result.color, linewidth=3, label=result.name)
    ax_legend.legend(fontsize=11, loc="lower center", ncol=2)
    
    fig.suptitle(
        "The Grand Optimizer Showdown — 7 Optimizers × 5 Functions\n"
        "[Lecture 3: SGD → Momentum → NAG → Adagrad → RMSprop → Adam → Nadam]",
        fontsize=17, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()
    
    return summary


def plot_momentum_vs_sgd(save_path=None):
    """
    Deep dive: WHY momentum helps.
    Shows SGD oscillating in a ravine while Momentum smoothly descends.
    [Lecture 3, Slide 8: "oscillating slow convergence"]
    """
    # Elongated quadratic: f(x,y) = 50x² + y²  (condition number = 50)
    def elongated(x):
        return 50 * x[0]**2 + x[1]**2
    
    def elongated_grad(x):
        return np.array([100 * x[0], 2 * x[1]])
    
    x0 = np.array([1.0, 10.0])
    
    sgd = SGD(lr=0.005)
    momentum = MomentumSGD(lr=0.005, gamma=0.9)
    nag = NAG(lr=0.005, gamma=0.9)
    
    r_sgd = sgd.optimize(elongated, elongated_grad, x0, 100)
    r_mom = momentum.optimize(elongated, elongated_grad, x0, 100)
    r_nag = nag.optimize(elongated, elongated_grad, x0, 100)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Contour
    ax = axes[0]
    xx = np.linspace(-1.5, 1.5, 200)
    yy = np.linspace(-12, 12, 200)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = 50 * XX**2 + YY**2
    
    ax.contour(XX, YY, ZZ, levels=20, cmap="Greys", alpha=0.4)
    ax.contourf(XX, YY, ZZ, levels=20, cmap="Greys", alpha=0.1)
    
    for r in [r_sgd, r_mom, r_nag]:
        path = np.array(r.path)
        ax.plot(path[:, 0], path[:, 1], color=r.color, linewidth=2, alpha=0.8, label=r.name)
        ax.scatter(path[0, 0], path[0, 1], color=r.color, s=80, marker="o", edgecolors="white", zorder=5)
        ax.scatter(path[-1, 0], path[-1, 1], color=r.color, s=100, marker="*", edgecolors="white", zorder=5)
    
    ax.scatter(0, 0, color="#dc2626", s=200, marker="*", edgecolors="white", linewidths=2, zorder=10)
    ax.set_title("Ravine Problem: f(x,y) = 50x² + y²\nSGD oscillates, Momentum smooths it out",
                fontsize=13, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    
    # Loss curves
    ax = axes[1]
    ax.semilogy(r_sgd.losses, color=r_sgd.color, linewidth=2.5, label=r_sgd.name)
    ax.semilogy(r_mom.losses, color=r_mom.color, linewidth=2.5, label=r_mom.name)
    ax.semilogy(r_nag.losses, color=r_nag.color, linewidth=2.5, label=r_nag.name)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("f(x) — log scale", fontsize=12)
    ax.set_title("Convergence: SGD oscillates, Momentum converges faster",
                fontsize=13, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.2, which="both")
    
    fig.suptitle(
        "Why Momentum Helps — Lecture 3, Slide 8\n"
        "SGD oscillates across the ravine; Momentum accumulates velocity along it",
        fontsize=15, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    
    print("=" * 60)
    print("  🏁 THE OPTIMIZER SHOWDOWN")
    print("=" * 60)
    
    # 1) Rosenbrock detailed
    print("\n📊 1/4: Rosenbrock Function (detailed)...")
    opts = create_optimizers()
    results = plot_contour_race(ROSENBROCK, opts, n_steps=500,
                                save_path="figures/rosenbrock_race.png")
    plot_convergence_comparison(ROSENBROCK, results,
                               save_path="figures/rosenbrock_convergence.png")
    
    # 2) Momentum vs SGD deep dive
    print("\n📊 2/4: Momentum vs SGD deep dive...")
    plot_momentum_vs_sgd(save_path="figures/momentum_vs_sgd.png")
    
    # 3) Himmelblau
    print("\n📊 3/4: Himmelblau Function...")
    opts = create_optimizers()
    plot_contour_race(HIMMELBLAU, opts, n_steps=500,
                      save_path="figures/himmelblau_race.png")
    
    # 4) Grand Showdown
    print("\n📊 4/4: THE GRAND SHOWDOWN...")
    summary = plot_grand_showdown(save_path="figures/grand_showdown.png")
    
    print("\n" + "=" * 60)
    print("  🏆 RESULTS")
    print("=" * 60)
    for fn_name, winner, results_dict in summary:
        print(f"  {fn_name:20s} → Winner: {winner}")
    print("=" * 60)
    print("\n✨ All figures generated!")
