"""
Transformer from Scratch — Demo
==================================
Based on: Zemke, AML Lecture 12

This script:
  1. Trains a Transformer on sentiment classification
  2. Visualizes self-attention weights (what attends to what)
  3. Shows positional encoding patterns
  4. Demonstrates multi-head attention diversity
  5. Compares Transformer vs simple baseline

No external data — generates synthetic sentiment data.

Run: python train.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from transformer import (
    TransformerClassifier, scaled_dot_product_attention,
    positional_encoding, MultiHeadAttention,
)


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic Sentiment Dataset (no external data needed!)
# ═══════════════════════════════════════════════════════════════════════════
def create_sentiment_dataset(n_samples=2000, seq_len=12, seed=42):
    """
    Generate synthetic sentiment classification data.
    
    Positive sentences contain words like: good, great, love, excellent, amazing
    Negative sentences contain words like: bad, terrible, hate, awful, poor
    """
    np.random.seed(seed)
    
    # Simple vocabulary
    positive_words = ["good", "great", "love", "excellent", "amazing", "wonderful",
                      "fantastic", "happy", "best", "beautiful", "perfect", "brilliant"]
    negative_words = ["bad", "terrible", "hate", "awful", "poor", "horrible",
                      "worst", "ugly", "boring", "sad", "failed", "broken"]
    neutral_words = ["the", "a", "is", "was", "it", "this", "that", "very",
                     "movie", "food", "place", "day", "thing", "much", "really",
                     "quite", "so", "and", "but", "of"]
    
    all_words = ["<PAD>"] + positive_words + negative_words + neutral_words
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    vocab_size = len(all_words)
    
    X = []
    y = []
    sentences = []
    
    for _ in range(n_samples):
        label = np.random.randint(2)  # 0=negative, 1=positive
        
        # Build sentence
        words = []
        # 2-4 sentiment words
        n_sent = np.random.randint(2, 5)
        if label == 1:
            words.extend(np.random.choice(positive_words, n_sent))
        else:
            words.extend(np.random.choice(negative_words, n_sent))
        
        # Fill with neutral words
        while len(words) < seq_len:
            words.append(np.random.choice(neutral_words))
        
        # Shuffle (so model can't just look at position)
        np.random.shuffle(words)
        words = words[:seq_len]
        
        # Convert to indices
        token_ids = [word_to_idx[w] for w in words]
        
        X.append(token_ids)
        y.append(label)
        sentences.append(words)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, sentences, word_to_idx, idx_to_word, vocab_size


def plot_positional_encoding(save_path=None):
    """Visualize the sinusoidal positional encoding."""
    d_model = 64
    max_len = 50
    pe = positional_encoding(max_len, d_model)[0]  # (max_len, d_model)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Full PE heatmap
    ax = axes[0, 0]
    im = ax.imshow(pe[:30, :32], cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_xlabel("Embedding Dimension", fontsize=12)
    ax.set_ylabel("Position", fontsize=12)
    ax.set_title("Positional Encoding Matrix\nPE(pos, 2i) = sin(pos/10000^(2i/d))",
                fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax)
    
    # Individual dimensions
    ax = axes[0, 1]
    for dim in [0, 1, 4, 5, 10, 11, 20, 21]:
        ax.plot(pe[:30, dim], linewidth=1.5, alpha=0.7, label=f"dim {dim}")
    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Individual PE Dimensions\nLow dims = high frequency, High dims = low frequency",
                fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)
    
    # Similarity matrix (dot product between position vectors)
    ax = axes[1, 0]
    similarity = pe[:20] @ pe[:20].T
    im = ax.imshow(similarity, cmap="viridis", aspect="equal")
    ax.set_xlabel("Position j", fontsize=12)
    ax.set_ylabel("Position i", fontsize=12)
    ax.set_title("Position Similarity (dot product)\nNearby positions are more similar",
                fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax)
    
    # Relative distances
    ax = axes[1, 1]
    for ref_pos in [0, 5, 10, 15]:
        distances = [np.dot(pe[ref_pos], pe[j]) for j in range(20)]
        ax.plot(distances, "o-", markersize=4, linewidth=1.5,
                label=f"from pos {ref_pos}")
    ax.set_xlabel("Position j", fontsize=12)
    ax.set_ylabel("Similarity to reference", fontsize=12)
    ax.set_title("Similarity Decay with Distance\nAttention naturally focuses on nearby tokens",
                fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    
    fig.suptitle(
        "Sinusoidal Positional Encoding — How Transformers Know Token Order\n"
        "[Lecture 12: Transformers process all positions simultaneously → need position info]",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


def plot_attention_demo(save_path=None):
    """Demonstrate self-attention on a simple example."""
    np.random.seed(42)
    
    # Simple example: 5 tokens, d=8
    seq_len = 5
    d_k = 8
    tokens = ["The", "food", "was", "really", "great"]
    
    Q = np.random.randn(1, seq_len, d_k) * 0.5
    K = np.random.randn(1, seq_len, d_k) * 0.5
    V = np.random.randn(1, seq_len, d_k) * 0.5
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Attention weights
    ax = axes[0]
    im = ax.imshow(weights[0], cmap="Blues", vmin=0, vmax=weights[0].max())
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(tokens, fontsize=11)
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(tokens, fontsize=11)
    ax.set_xlabel("Key (attends to)", fontsize=12)
    ax.set_ylabel("Query (from)", fontsize=12)
    ax.set_title("Self-Attention Weights\nEach row sums to 1 (softmax)",
                fontsize=13, fontweight="bold")
    
    for i in range(seq_len):
        for j in range(seq_len):
            ax.text(j, i, f"{weights[0, i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if weights[0, i, j] > 0.3 else "black")
    plt.colorbar(im, ax=ax)
    
    # Scores before softmax
    ax = axes[1]
    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_k)
    im = ax.imshow(scores[0], cmap="RdBu_r")
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(tokens, fontsize=11)
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(tokens, fontsize=11)
    ax.set_title("Raw Scores (Q·Kᵀ/√d)\nBefore softmax",
                fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax)
    
    # The formula
    ax = axes[2]
    ax.axis("off")
    formula_text = (
        "Scaled Dot-Product Attention\n"
        "=" * 40 + "\n\n"
        "Attention(Q, K, V)\n"
        "  = softmax(Q · Kᵀ / sqrt(d_k)) · V\n\n"
        "Q = queries (what am I looking for?)\n"
        "K = keys    (what do I contain?)\n"
        "V = values  (what information to pass)\n\n"
        "1/sqrt(d_k) scaling prevents\n"
        "softmax saturation for large d_k\n\n"
        "[Lecture 12, Slide 11]"
    )
    ax.text(0.5, 0.5, formula_text, fontsize=13, fontfamily="monospace",
            ha="center", va="center", transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="#f0f9ff", edgecolor="#2563eb"))
    ax.set_title("The Formula", fontsize=13, fontweight="bold")
    
    fig.suptitle(
        "Self-Attention: Each Token Attends to Every Other Token\n"
        "[Lecture 12: attention(q, K, V) = Σ similarity(q, kⱼ) · vⱼ]",
        fontsize=16, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


def plot_training_results(model, history, X_test, y_test, sentences_test,
                          idx_to_word, save_path=None):
    """Plot training curves and attention on real examples."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Loss curve
    ax = axes[0, 0]
    ax.plot(history["loss"], color="#2563eb", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss", fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax = axes[0, 1]
    ax.plot([a * 100 for a in history["acc"]], color="#16a34a", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training Accuracy", fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    # Attention visualization on a positive example
    pos_idx = np.where(y_test == 1)[0][0]
    probs, attn = model.forward(X_test[pos_idx:pos_idx+1])
    pred = np.argmax(probs[0])
    words = sentences_test[pos_idx]
    
    ax = axes[1, 0]
    # Average attention across heads (layer 0)
    avg_attn = np.mean(attn[0][0], axis=0)  # average over heads
    im = ax.imshow(avg_attn, cmap="Blues")
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=9)
    label = "Positive" if pred == 1 else "Negative"
    ax.set_title(f"Attention Map — Predicted: {label} ({probs[0, pred]:.0%})\n"
                 f"(averaged across heads, layer 1)",
                fontsize=12, fontweight="bold",
                color="#16a34a" if pred == y_test[pos_idx] else "#dc2626")
    plt.colorbar(im, ax=ax)
    
    # Attention on a negative example
    neg_idx = np.where(y_test == 0)[0][0]
    probs_neg, attn_neg = model.forward(X_test[neg_idx:neg_idx+1])
    pred_neg = np.argmax(probs_neg[0])
    words_neg = sentences_test[neg_idx]
    
    ax = axes[1, 1]
    avg_attn_neg = np.mean(attn_neg[0][0], axis=0)
    im = ax.imshow(avg_attn_neg, cmap="Reds")
    ax.set_xticks(range(len(words_neg)))
    ax.set_xticklabels(words_neg, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(words_neg)))
    ax.set_yticklabels(words_neg, fontsize=9)
    label_neg = "Positive" if pred_neg == 1 else "Negative"
    ax.set_title(f"Attention Map — Predicted: {label_neg} ({probs_neg[0, pred_neg]:.0%})\n"
                 f"(averaged across heads, layer 1)",
                fontsize=12, fontweight="bold",
                color="#16a34a" if pred_neg == y_test[neg_idx] else "#dc2626")
    plt.colorbar(im, ax=ax)
    
    fig.suptitle(
        "Transformer Training Results — Sentiment Classification\n"
        "[Lecture 12: Attention & Transformer]",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


def plot_multihead_diversity(model, X_test, sentences_test, save_path=None):
    """Show how different attention heads learn different patterns."""
    idx = 0
    probs, attn_weights = model.forward(X_test[idx:idx+1])
    words = sentences_test[idx]
    
    n_heads = attn_weights[0].shape[1]
    fig, axes = plt.subplots(1, n_heads + 1, figsize=(4 * (n_heads + 1), 5))
    
    for h in range(n_heads):
        ax = axes[h]
        head_attn = attn_weights[0][0, h]  # (seq, seq)
        ax.imshow(head_attn, cmap="Blues")
        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=8)
        ax.set_title(f"Head {h+1}", fontsize=12, fontweight="bold")
    
    # Average
    ax = axes[-1]
    avg = np.mean(attn_weights[0][0], axis=0)
    ax.imshow(avg, cmap="Purples")
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=8)
    ax.set_title("Average", fontsize=12, fontweight="bold", color="#9333ea")
    
    fig.suptitle(
        "Multi-Head Attention — Each Head Learns Different Patterns\n"
        "[Lecture 12: 'Instead of single attention, use h parallel attention heads']",
        fontsize=15, fontweight="bold", y=1.05,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


if __name__ == "__main__":
    np.random.seed(42)
    os.makedirs("figures", exist_ok=True)
    
    print("=" * 60)
    print("  Transformer from Scratch")
    print("=" * 60)
    
    # ── 1) Positional Encoding ────────────────────────────────────
    print("\n  1/4: Positional encoding visualization...")
    plot_positional_encoding(save_path="figures/positional_encoding.png")
    
    # ── 2) Attention Demo ─────────────────────────────────────────
    print("\n  2/4: Self-attention demo...")
    plot_attention_demo(save_path="figures/attention_demo.png")
    
    # ── 3) Train on sentiment classification ──────────────────────
    print("\n  3/4: Training Transformer on sentiment classification...")
    X, y, sentences, w2i, i2w, vocab_size = create_sentiment_dataset(n_samples=2000, seq_len=12)
    
    # Split
    n_train = 1600
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    sentences_test = sentences[n_train:]
    
    print(f"  Vocab: {vocab_size} tokens | Train: {n_train} | Test: {len(y_test)}")
    
    model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        max_len=20,
        n_classes=2,
    )
    
    n_params = model.embedding.size + model.W_cls.size + model.b_cls.size
    for block in model.blocks:
        for p, _ in block.params():
            n_params += p.size
    print(f"  Parameters: {n_params:,}")
    
    start = time.time()
    history = model.train(X_train, y_train, epochs=30, batch_size=64, lr=0.005)
    elapsed = time.time() - start
    print(f"\n  Training time: {elapsed:.1f}s")
    
    # Evaluate
    preds = model.predict(X_test)
    test_acc = np.mean(preds == y_test)
    print(f"  Test Accuracy: {test_acc*100:.1f}%")
    
    # ── 4) Visualizations ─────────────────────────────────────────
    print("\n  4/4: Generating plots...")
    plot_training_results(model, history, X_test, y_test, sentences_test, i2w,
                         save_path="figures/training_results.png")
    plot_multihead_diversity(model, X_test, sentences_test,
                            save_path="figures/multihead.png")
    
    print("\n✨ Done!")
