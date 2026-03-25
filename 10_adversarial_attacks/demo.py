"""
Adversarial Attack Demo — Fooling Neural Networks
====================================================
Based on: Zemke, AML Lecture 7, Slides 30-45

"In Szegedy et al. the authors found that close to almost every
 successfully classified image are adversarial examples that a human
 cannot distinguish from the original, yet these are misclassified
 with high confidence." — Lecture 7

This script:
  1. Trains a network on handwritten digits
  2. Attacks it with FGSM, Targeted FGSM, I-FGSM, and Random Noise
  3. Compares attack success rates across epsilon values
  4. Visualizes perturbations and misclassifications
  5. Demonstrates adversarial training as a defense

Run: python demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from attacks import (
    SimpleNet, fgsm_attack, targeted_fgsm,
    iterative_fgsm, random_noise_attack,
)


def load_data():
    """Load sklearn digits (8×8, 10 classes)."""
    from sklearn.datasets import load_digits
    digits = load_digits()
    X = digits.data / 16.0  # normalize to [0, 1]
    y = digits.target
    # Split
    n = int(0.8 * len(X))
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]
    return X[:n], y[:n], X[n:], y[n:]


def plot_fgsm_examples(model, X_test, y_test, save_path=None):
    """Show FGSM attacks at different epsilon values."""
    epsilons = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
    n_examples = 5

    # Find correctly classified examples
    correct_idx = [i for i in range(len(X_test)) if model.predict(X_test[i]) == y_test[i]]

    fig, axes = plt.subplots(n_examples, len(epsilons), figsize=(20, 16))

    for row in range(n_examples):
        idx = correct_idx[row]
        x = X_test[idx]
        true_label = y_test[idx]

        for col, eps in enumerate(epsilons):
            ax = axes[row, col]

            if eps == 0:
                x_show = x
                pred = model.predict(x)
                conf = model.forward(x)[pred]
            else:
                x_adv, _ = fgsm_attack(model, x, true_label, epsilon=eps)
                x_show = x_adv
                pred = model.predict(x_adv)
                conf = model.forward(x_adv)[pred]

            ax.imshow(x_show.reshape(8, 8), cmap="gray", vmin=0, vmax=1)

            color = "#16a34a" if pred == true_label else "#dc2626"
            symbol = "correct" if pred == true_label else "FOOLED!"
            ax.set_title(f"Pred: {pred} ({conf:.0%})\n{symbol}",
                        fontsize=10, fontweight="bold", color=color)
            ax.axis("off")

            if row == 0:
                axes[0, col].set_title(
                    f"ε = {eps}\nPred: {pred} ({conf:.0%})\n"
                    f"{'correct' if pred == true_label else 'FOOLED!'}",
                    fontsize=10, fontweight="bold",
                    color="#16a34a" if pred == true_label else "#dc2626")

        # Add true label on the left
        axes[row, 0].set_ylabel(f"True: {true_label}", fontsize=12, fontweight="bold")

    fig.suptitle(
        "FGSM Attack: Increasing ε → More Perturbation → More Fooling\n"
        "[Lecture 7: x_adv = x + ε · sign(∇_x L)]",
        fontsize=16, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


def plot_perturbation_anatomy(model, X_test, y_test, save_path=None):
    """Show: original + perturbation = adversarial."""
    idx = next(i for i in range(len(X_test)) if model.predict(X_test[i]) == y_test[i])
    x = X_test[idx]
    true_label = y_test[idx]

    eps = 0.25
    x_adv_fgsm, pert_fgsm = fgsm_attack(model, x, true_label, eps)
    x_adv_ifgsm, pert_ifgsm = iterative_fgsm(model, x, true_label, eps, alpha=0.02, n_iter=20)
    x_adv_rand, pert_rand = random_noise_attack(model, x, eps)

    attacks = [
        ("FGSM", x_adv_fgsm, pert_fgsm),
        ("I-FGSM (PGD)", x_adv_ifgsm, pert_ifgsm),
        ("Random Noise", x_adv_rand, pert_rand),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(18, 13))

    for row, (name, x_adv, pert) in enumerate(attacks):
        pred_orig = model.predict(x)
        conf_orig = model.forward(x)[pred_orig]
        pred_adv = model.predict(x_adv)
        conf_adv = model.forward(x_adv)[pred_adv]

        # Original
        axes[row, 0].imshow(x.reshape(8, 8), cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"Original\nPred: {pred_orig} ({conf_orig:.0%})",
                               fontsize=11, fontweight="bold", color="#16a34a")

        # Perturbation (amplified for visibility)
        axes[row, 1].imshow(pert.reshape(8, 8), cmap="RdBu_r",
                           vmin=-np.max(np.abs(pert)), vmax=np.max(np.abs(pert)))
        axes[row, 1].set_title(f"{name} Perturbation\n(ε={eps}, amplified)",
                               fontsize=11, fontweight="bold")

        # Adversarial
        color = "#dc2626" if pred_adv != true_label else "#16a34a"
        label = "FOOLED!" if pred_adv != true_label else "survived"
        axes[row, 2].imshow(x_adv.reshape(8, 8), cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title(f"Adversarial\nPred: {pred_adv} ({conf_adv:.0%}) {label}",
                               fontsize=11, fontweight="bold", color=color)

        # Confidence bars
        ax = axes[row, 3]
        probs = model.forward(x_adv)
        bar_colors = ["#dc2626" if i == pred_adv and pred_adv != true_label
                      else "#16a34a" if i == true_label
                      else "#d1d5db" for i in range(10)]
        ax.barh(range(10), probs, color=bar_colors, edgecolor="white")
        ax.set_yticks(range(10))
        ax.set_xlabel("Confidence")
        ax.set_title(f"Class Probabilities\n(green=true, red=predicted)", fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.2, axis="x")

    for ax in axes.ravel():
        if not ax.get_xlabel():
            ax.axis("off")

    fig.suptitle(
        f"Adversarial Attack Anatomy — True Label: {true_label}, ε = {eps}\n"
        "Original + Perturbation = Adversarial (looks same to humans, fools the network!)",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


def plot_epsilon_robustness(model, X_test, y_test, save_path=None):
    """Accuracy vs epsilon for all attack methods."""
    epsilons = np.arange(0, 0.55, 0.05)
    n_test = min(200, len(X_test))

    methods = {
        "FGSM": lambda x, yl, eps: fgsm_attack(model, x, yl, eps),
        "I-FGSM (10 steps)": lambda x, yl, eps: iterative_fgsm(model, x, yl, eps, alpha=0.02, n_iter=10),
        "Random Noise": lambda x, yl, eps: random_noise_attack(model, x, eps),
    }
    colors = {"FGSM": "#2563eb", "I-FGSM (10 steps)": "#dc2626", "Random Noise": "#6b7280"}

    results = {name: [] for name in methods}

    for eps in epsilons:
        for name, attack_fn in methods.items():
            correct = 0
            for i in range(n_test):
                if eps == 0:
                    pred = model.predict(X_test[i])
                else:
                    x_adv, _ = attack_fn(X_test[i], y_test[i], eps)
                    pred = model.predict(x_adv)
                if pred == y_test[i]:
                    correct += 1
            results[name].append(correct / n_test * 100)
        print(f"    ε={eps:.2f}: FGSM={results['FGSM'][-1]:.0f}%  "
              f"I-FGSM={results['I-FGSM (10 steps)'][-1]:.0f}%  "
              f"Random={results['Random Noise'][-1]:.0f}%")

    fig, ax = plt.subplots(figsize=(12, 7))
    for name, accs in results.items():
        ax.plot(epsilons, accs, "o-", color=colors[name], linewidth=2.5,
                markersize=7, label=name)

    ax.set_xlabel("Epsilon (ε) — Perturbation Magnitude", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(
        "Robustness Under Attack — Accuracy vs Epsilon\n"
        "Adversarial perturbations are FAR more effective than random noise!",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 105)

    # Annotations
    ax.annotate("Random noise barely hurts", xy=(0.35, results["Random Noise"][7]),
                xytext=(0.38, 85), fontsize=10, color="#6b7280",
                arrowprops=dict(arrowstyle="->", color="#6b7280"))
    ax.annotate("FGSM: one gradient step\ndrops accuracy fast!", xy=(0.2, results["FGSM"][4]),
                xytext=(0.25, 30), fontsize=10, color="#2563eb",
                arrowprops=dict(arrowstyle="->", color="#2563eb"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


def plot_targeted_attack(model, X_test, y_test, save_path=None):
    """Demonstrate targeted attack: force network to predict specific class."""
    # Find a "3" that's correctly classified
    idx = next(i for i in range(len(X_test))
               if y_test[i] == 3 and model.predict(X_test[i]) == 3)
    x = X_test[idx]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    # Attack towards each target class
    for target in range(10):
        ax = axes[target // 5, target % 5]

        x_adv, _ = targeted_fgsm(model, x, target_label=target, epsilon=0.4)
        pred = model.predict(x_adv)
        conf = model.forward(x_adv)[pred]

        ax.imshow(x_adv.reshape(8, 8), cmap="gray", vmin=0, vmax=1)
        success = "HIT" if pred == target else f"miss→{pred}"
        color = "#16a34a" if pred == target else "#dc2626"
        ax.set_title(f"Target: {target}\nPred: {pred} ({conf:.0%})\n{success}",
                    fontsize=11, fontweight="bold", color=color)
        ax.axis("off")

    fig.suptitle(
        f"Targeted FGSM — Forcing a '3' to be Classified as Each Digit (ε=0.4)\n"
        "x_adv = x − ε · sign(∇_x L(θ, x, y_target))",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


def plot_adversarial_training(X_train, y_train, X_test, y_test, save_path=None):
    """
    Defense: Adversarial Training [Lecture 7: "Learning from attacks"]
    
    Train on adversarial examples to make the network robust.
    """
    print("\n  Training standard model...")
    model_std = SimpleNet(seed=42)
    model_std.train(X_train, y_train, epochs=60, lr=0.01)

    print("\n  Training adversarially robust model...")
    model_robust = SimpleNet(seed=42)
    # First normal training
    model_robust.train(X_train, y_train, epochs=30, lr=0.01)
    # Then adversarial training
    for epoch in range(30):
        perm = np.random.permutation(len(X_train))
        for idx in perm[:200]:  # subset for speed
            x = X_train[idx]
            # Generate adversarial example
            x_adv, _ = fgsm_attack(model_robust, x, y_train[idx], epsilon=0.2)
            # Train on both original and adversarial
            model_robust.forward(x.reshape(-1, 1))
            delta2 = model_robust.probs.copy()
            delta2[y_train[idx]] -= 1.0
            delta1 = model_robust.W2.T @ delta2 * (model_robust.z1 > 0).astype(float)
            model_robust.W2 -= 0.005 * delta2 @ model_robust.a1.T
            model_robust.W1 -= 0.005 * delta1 @ x.reshape(-1, 1).T

            model_robust.forward(x_adv.reshape(-1, 1))
            delta2 = model_robust.probs.copy()
            delta2[y_train[idx]] -= 1.0
            delta1 = model_robust.W2.T @ delta2 * (model_robust.z1 > 0).astype(float)
            model_robust.W2 -= 0.005 * delta2 @ model_robust.a1.T
            model_robust.W1 -= 0.005 * delta1 @ x_adv.reshape(-1, 1).T

    # Compare robustness
    epsilons = np.arange(0, 0.45, 0.05)
    n = min(200, len(X_test))

    acc_std = []
    acc_robust = []

    for eps in epsilons:
        c_std, c_robust = 0, 0
        for i in range(n):
            if eps == 0:
                c_std += (model_std.predict(X_test[i]) == y_test[i])
                c_robust += (model_robust.predict(X_test[i]) == y_test[i])
            else:
                x_adv_s, _ = fgsm_attack(model_std, X_test[i], y_test[i], eps)
                x_adv_r, _ = fgsm_attack(model_robust, X_test[i], y_test[i], eps)
                c_std += (model_std.predict(x_adv_s) == y_test[i])
                c_robust += (model_robust.predict(x_adv_r) == y_test[i])
        acc_std.append(c_std / n * 100)
        acc_robust.append(c_robust / n * 100)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(epsilons, acc_std, "o-", color="#dc2626", linewidth=2.5, markersize=7,
            label="Standard Training")
    ax.plot(epsilons, acc_robust, "s-", color="#16a34a", linewidth=2.5, markersize=7,
            label="Adversarial Training")

    ax.set_xlabel("Epsilon (ε)", fontsize=13)
    ax.set_ylabel("Accuracy Under FGSM Attack (%)", fontsize=13)
    ax.set_title(
        "Defense: Adversarial Training Makes the Network Robust\n"
        "[Lecture 7: 'Learning from attacks' — train on adversarial examples]",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


if __name__ == "__main__":
    np.random.seed(42)
    os.makedirs("figures", exist_ok=True)

    print("=" * 60)
    print("  Adversarial Attack Demo — FGSM")
    print("=" * 60)

    # Load data
    print("\n  Loading data...")
    X_train, y_train, X_test, y_test = load_data()
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # Train model
    print("\n  Training network [64 → 128 → 10]...")
    model = SimpleNet(input_dim=64, hidden=128, output=10)
    model.train(X_train, y_train, epochs=100, lr=0.01)

    clean_acc = sum(model.predict(X_test[i]) == y_test[i] for i in range(len(X_test))) / len(X_test)
    print(f"\n  Clean accuracy: {clean_acc*100:.1f}%")

    # Plots
    print("\n  1/5: FGSM examples across epsilon values...")
    plot_fgsm_examples(model, X_test, y_test, save_path="figures/fgsm_examples.png")

    print("\n  2/5: Perturbation anatomy...")
    plot_perturbation_anatomy(model, X_test, y_test, save_path="figures/perturbation_anatomy.png")

    print("\n  3/5: Epsilon robustness curve...")
    plot_epsilon_robustness(model, X_test, y_test, save_path="figures/robustness.png")

    print("\n  4/5: Targeted attack...")
    plot_targeted_attack(model, X_test, y_test, save_path="figures/targeted_attack.png")

    print("\n  5/5: Adversarial training defense...")
    plot_adversarial_training(X_train, y_train, X_test, y_test,
                              save_path="figures/adversarial_training.png")

    print("\n✨ All demos complete!")
