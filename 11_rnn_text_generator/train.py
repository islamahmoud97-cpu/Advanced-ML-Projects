"""
RNN Text Generator — Character-Level Language Model
======================================================
Based on: Zemke, AML Lecture 8

This script:
  1. Trains a character-level RNN on a text corpus
  2. Generates new text by sampling from the model
  3. Shows how generated text improves during training
  4. Demonstrates temperature-controlled sampling
  5. Visualizes hidden state dynamics

No external data needed — uses built-in text corpus.

Run: python train.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from rnn import CharRNN


# ═══════════════════════════════════════════════════════════════════════════
# Built-in Training Corpus (no external files needed!)
# ═══════════════════════════════════════════════════════════════════════════
TRAINING_TEXT = """
Machine learning is a subset of artificial intelligence that provides systems
the ability to automatically learn and improve from experience without being
explicitly programmed. Machine learning focuses on the development of computer
programs that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples,
direct experience, or instruction, in order to look for patterns in data and
make better decisions in the future based on the examples that we provide.

Neural networks are computing systems inspired by biological neural networks
that constitute animal brains. An artificial neural network consists of a
collection of connected units or nodes called artificial neurons. Each
connection between neurons can transmit a signal to other neurons. A neuron
that receives a signal then processes it and can signal neurons connected to it.

Deep learning is part of a broader family of machine learning methods based on
artificial neural networks with representation learning. Learning can be
supervised, semi-supervised or unsupervised. Deep learning architectures such
as deep neural networks, recurrent neural networks, and convolutional neural
networks have been applied to fields including computer vision, speech
recognition, natural language processing, and machine translation.

Recurrent neural networks are a class of neural networks where connections
between nodes can create a cycle, allowing output from some nodes to affect
subsequent input to the same nodes. This allows the network to exhibit temporal
dynamic behavior. Unlike feedforward neural networks, recurrent neural networks
can use their internal state or memory to process variable length sequences of
inputs. This makes them applicable to tasks such as handwriting recognition,
speech recognition, and natural language processing.

The training of a recurrent neural network involves backpropagation through
time, where the network is unrolled through time and standard backpropagation
is applied to the unrolled network. However, this approach suffers from the
vanishing gradient problem, where gradients become very small as they are
propagated back through many time steps. This was solved by the introduction
of Long Short-Term Memory networks, which use gating mechanisms to control
the flow of information through the network.

Convolutional neural networks are a class of deep neural networks, most
commonly applied to analyzing visual imagery. They use a mathematical
operation called convolution instead of general matrix multiplication in at
least one of their layers. Convolutional networks were inspired by biological
processes in that the connectivity pattern between neurons resembles the
organization of the animal visual cortex.

The transformer architecture has revolutionized natural language processing.
Unlike recurrent neural networks, transformers process all positions in the
input sequence simultaneously using self-attention mechanisms. This parallel
processing makes transformers significantly faster to train than recurrent
models. The attention mechanism allows the model to focus on different parts
of the input sequence when producing each element of the output sequence.

Generative adversarial networks consist of two neural networks, a generator
and a discriminator, that are trained simultaneously through adversarial
training. The generator creates new data instances, while the discriminator
evaluates them for authenticity. The generator improves its ability to create
realistic data, while the discriminator improves its ability to distinguish
real data from generated data. This process continues until the generator
produces data that is indistinguishable from real data.

Reinforcement learning is an area of machine learning concerned with how
intelligent agents ought to take actions in an environment in order to
maximize the notion of cumulative reward. The agent learns a policy that
maps states to actions, aiming to maximize the expected sum of rewards over
time. Unlike supervised learning, reinforcement learning does not require
labeled input and output pairs and instead learns through interaction with
the environment using trial and error.

The field of machine learning continues to advance rapidly, with new
architectures, training methods, and applications being developed at an
unprecedented pace. From autonomous vehicles to medical diagnosis, from
natural language understanding to creative content generation, machine
learning is transforming virtually every aspect of modern technology and
society. Understanding the mathematical foundations of these methods is
essential for developing the next generation of intelligent systems.
""".strip()


def prepare_data(text):
    """Create character mappings."""
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    return chars, char_to_idx, idx_to_char, vocab_size


def plot_training_progress(loss_history, samples, save_path=None):
    """Show loss curve and how generated text improves over training."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Loss curve
    ax = axes[0]
    ax.plot(loss_history, color="#2563eb", linewidth=1, alpha=0.8)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Smoothed Loss", fontsize=12)
    ax.set_title("Training Loss — Character-Level RNN", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Show samples at different training stages
    ax = axes[1]
    ax.axis("off")
    text_content = "Generated Text Samples During Training\n" + "=" * 60 + "\n\n"
    for iteration, sample in samples[:6]:
        clean = sample[:120].replace('\n', ' ')
        text_content += f"Iteration {iteration:,d}:\n  \"{clean}...\"\n\n"

    ax.text(0.02, 0.98, text_content, fontsize=11, fontfamily="monospace",
            verticalalignment="top", transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="#f8fafc", edgecolor="#e2e8f0"))

    fig.suptitle(
        "RNN Text Generator — Learning to Write Character by Character\n"
        "[Lecture 8: Elman Network + BPTT]",
        fontsize=16, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


def plot_temperature_comparison(model, h, seed_idx, idx_to_char, save_path=None):
    """Show how temperature affects text generation."""
    temperatures = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0]

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis("off")

    text = "Temperature-Controlled Text Generation\n" + "=" * 60 + "\n\n"
    text += "Low temp → conservative/repetitive | High temp → creative/chaotic\n\n"

    for temp in temperatures:
        np.random.seed(42)
        sample_idx = model.sample(h, seed_idx, 150, temperature=temp)
        sample_text = ''.join(idx_to_char[i] for i in sample_idx)
        clean = sample_text[:130].replace('\n', ' ')

        label = ""
        if temp <= 0.3:
            label = "(very conservative)"
        elif temp <= 0.7:
            label = "(balanced)"
        elif temp <= 1.0:
            label = "(natural)"
        elif temp <= 1.5:
            label = "(creative)"
        else:
            label = "(chaotic)"

        text += f"T = {temp:.1f} {label}:\n  \"{clean}...\"\n\n"

    ax.text(0.02, 0.98, text, fontsize=11, fontfamily="monospace",
            verticalalignment="top", transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="#f8fafc", edgecolor="#e2e8f0"))

    fig.suptitle(
        "Temperature Controls Randomness in Text Generation\n"
        "softmax(y / T): T→0 = argmax (deterministic), T→∞ = uniform (random)",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


def plot_hidden_state_dynamics(model, data, char_to_idx, save_path=None):
    """Visualize how hidden states evolve as the RNN reads text."""
    # Process a sequence and record hidden states
    seq = data[:200]
    h = np.zeros((model.hidden_size, 1))
    states = []

    for ch in seq:
        x = np.zeros((model.vocab_size, 1))
        x[char_to_idx[ch]] = 1
        h = np.tanh(model.Whh @ h + model.Whx @ x + model.bh)
        states.append(h[:20, 0].copy())  # first 20 hidden units

    states = np.array(states)  # (T, 20)

    fig, axes = plt.subplots(2, 1, figsize=(18, 10))

    # Heatmap of hidden states
    ax = axes[0]
    im = ax.imshow(states.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("Character Position in Sequence", fontsize=12)
    ax.set_ylabel("Hidden Unit Index", fontsize=12)
    ax.set_title("Hidden State Dynamics — First 20 Units Over 200 Characters\n"
                 "The RNN's 'memory' evolves as it reads each character",
                 fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Activation (tanh)")

    # Mark spaces and newlines
    for i, ch in enumerate(seq):
        if ch == '\n':
            ax.axvline(x=i, color="#16a34a", alpha=0.3, linewidth=0.5)

    # Individual hidden units
    ax = axes[1]
    for unit in [0, 3, 7, 12, 18]:
        ax.plot(states[:, unit], linewidth=1.5, alpha=0.7, label=f"Unit {unit}")
    ax.set_xlabel("Character Position", fontsize=12)
    ax.set_ylabel("Activation", fontsize=12)
    ax.set_title("Individual Hidden Unit Trajectories", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, ncol=5)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


if __name__ == "__main__":
    np.random.seed(42)
    os.makedirs("figures", exist_ok=True)

    print("=" * 60)
    print("  RNN Text Generator — Character-Level Language Model")
    print("=" * 60)

    # Prepare data
    chars, char_to_idx, idx_to_char, vocab_size = prepare_data(TRAINING_TEXT)
    print(f"\n  Corpus: {len(TRAINING_TEXT):,} characters")
    print(f"  Vocabulary: {vocab_size} unique characters")
    print(f"  Characters: {''.join(chars[:30])}...")

    # Build model
    model = CharRNN(
        vocab_size=vocab_size,
        hidden_size=128,
        seq_length=25,
        learning_rate=0.01,
    )

    n_params = (model.Whx.size + model.Whh.size + model.Wyh.size +
                model.bh.size + model.by.size)
    print(f"\n  Model: {n_params:,} parameters")
    print(f"  Hidden size: {model.hidden_size}")
    print(f"  Sequence length: {model.seq_length}")

    # Train
    print("\n  Training...\n")
    start = time.time()
    samples = model.train(
        TRAINING_TEXT, char_to_idx, idx_to_char,
        n_iterations=15000,
        print_every=3000,
        sample_every=3000,
        sample_length=200,
    )
    elapsed = time.time() - start
    print(f"\n  Training time: {elapsed:.1f}s")

    # Generate final sample
    print("\n" + "=" * 60)
    print("  Final Generated Text (T=0.8):")
    print("=" * 60)
    h = np.zeros((model.hidden_size, 1))
    seed = char_to_idx[TRAINING_TEXT[0]]
    final_idx = model.sample(h, seed, 500, temperature=0.8)
    final_text = ''.join(idx_to_char[i] for i in final_idx)
    print(final_text)
    print("=" * 60)

    # Plots
    print("\n  Generating plots...")
    plot_training_progress(model.loss_history, samples,
                          save_path="figures/training_progress.png")

    h = np.zeros((model.hidden_size, 1))
    plot_temperature_comparison(model, h, seed, idx_to_char,
                               save_path="figures/temperature.png")

    plot_hidden_state_dynamics(model, TRAINING_TEXT, char_to_idx,
                              save_path="figures/hidden_states.png")

    print("\n✨ Done!")
