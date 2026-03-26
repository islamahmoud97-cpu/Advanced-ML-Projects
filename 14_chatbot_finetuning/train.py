"""
Simple Chatbot — Pretrain → Fine-tune → Chat
================================================
Based on: Zemke, AML Lecture 13: LLM, Finetuning, Fast Inference

"Training of ChatGPT:
  1. Pre-training (self-supervised learning)
  2. Q & A (supervised learning)
  3. RLHF (reinforcement learning from human feedback)"
 — Lecture 13

This script demonstrates the SAME pipeline at miniature scale:

  Step 1: PRE-TRAIN on general ML text (self-supervised)
          → Model learns language structure, common words, patterns
  
  Step 2: FINE-TUNE on Q&A instruction pairs (supervised)
          → Model learns to follow the "Q: ... A: ..." format
  
  Step 3: CHAT — ask questions and get answers
          → Model generates responses using learned patterns

Key insight from Lecture 13:
  "First layers often are universal, only later layers are trained new"
  → Pre-training gives general knowledge, fine-tuning specializes it.

Run: python train.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from mini_lm import MiniLM


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Pre-training Corpus (general ML knowledge)
# ═══════════════════════════════════════════════════════════════════════════
PRETRAIN_TEXT = """
neural networks are computing systems inspired by biological neural networks.
deep learning is part of machine learning based on artificial neural networks.
a convolutional neural network uses convolution for image processing tasks.
recurrent neural networks process sequences of data using hidden states.
the transformer architecture uses self attention instead of recurrence.
gradient descent optimizes the weights by following the negative gradient.
backpropagation computes gradients by applying the chain rule backwards.
activation functions like relu and sigmoid introduce non linearity.
batch normalization stabilizes training by normalizing layer inputs.
dropout randomly deactivates neurons to prevent overfitting.
the learning rate controls how large each optimization step is.
adam optimizer combines momentum with adaptive learning rates.
cross entropy loss measures the difference between predicted and true distributions.
overfitting occurs when the model memorizes training data but fails on new data.
regularization techniques like weight decay prevent overfitting.
transfer learning reuses pretrained models for new tasks with less data.
fine tuning adapts a pretrained model to a specific downstream task.
attention mechanisms allow the model to focus on relevant parts of the input.
the softmax function converts logits into a probability distribution.
embeddings map discrete tokens into continuous vector representations.
""".strip().lower()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Fine-tuning Data (Q&A instruction pairs)
# ═══════════════════════════════════════════════════════════════════════════
QA_PAIRS = [
    ("Q: what is deep learning?\nA:", " deep learning uses neural networks with many layers to learn from data.\n"),
    ("Q: what is a neural network?\nA:", " a neural network is a computing system inspired by the brain.\n"),
    ("Q: what is backpropagation?\nA:", " backpropagation computes gradients using the chain rule.\n"),
    ("Q: what is gradient descent?\nA:", " gradient descent optimizes weights by following the gradient.\n"),
    ("Q: what is overfitting?\nA:", " overfitting is when the model memorizes data but fails on new data.\n"),
    ("Q: what is a transformer?\nA:", " the transformer uses self attention to process sequences.\n"),
    ("Q: what is attention?\nA:", " attention lets the model focus on relevant parts of the input.\n"),
    ("Q: what is relu?\nA:", " relu is an activation function that outputs max of zero and input.\n"),
    ("Q: what is dropout?\nA:", " dropout randomly deactivates neurons to prevent overfitting.\n"),
    ("Q: what is adam?\nA:", " adam is an optimizer combining momentum with adaptive learning rates.\n"),
    ("Q: what is fine tuning?\nA:", " fine tuning adapts a pretrained model to a new task.\n"),
    ("Q: what is transfer learning?\nA:", " transfer learning reuses pretrained models for new tasks.\n"),
    ("Q: what is a cnn?\nA:", " a cnn uses convolution layers for image processing.\n"),
    ("Q: what is an rnn?\nA:", " an rnn processes sequences using recurrent hidden states.\n"),
    ("Q: what is softmax?\nA:", " softmax converts logits into a probability distribution.\n"),
    ("Q: what is cross entropy?\nA:", " cross entropy measures the difference between distributions.\n"),
    ("Q: what is batch normalization?\nA:", " batch normalization normalizes layer inputs for stable training.\n"),
    ("Q: what is regularization?\nA:", " regularization prevents overfitting using techniques like weight decay.\n"),
    ("Q: what is an embedding?\nA:", " an embedding maps tokens into continuous vector space.\n"),
    ("Q: what is the learning rate?\nA:", " the learning rate controls the size of optimization steps.\n"),
]


def prepare_vocab(texts):
    """Build character vocabulary from all texts."""
    all_text = "".join(texts)
    chars = sorted(list(set(all_text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return chars, char_to_idx, idx_to_char, len(chars)


def text_to_ids(text, char_to_idx):
    return [char_to_idx.get(ch, 0) for ch in text]


def ids_to_text(ids, idx_to_char):
    return "".join(idx_to_char.get(i, "?") for i in ids)


def run_pipeline():
    """Execute the full Pretrain → Fine-tune → Chat pipeline."""
    np.random.seed(42)
    
    print("=" * 60)
    print("  Chatbot Pipeline: Pretrain → Fine-tune → Chat")
    print("  [Lecture 13: LLM, Finetuning, Fast Inference]")
    print("=" * 60)
    
    # ── Prepare vocabulary ────────────────────────────────────────
    all_texts = [PRETRAIN_TEXT] + [q + a for q, a in QA_PAIRS]
    chars, c2i, i2c, vocab_size = prepare_vocab(all_texts)
    
    print(f"\n  Vocabulary: {vocab_size} chars")
    print(f"  Pre-training corpus: {len(PRETRAIN_TEXT)} chars")
    print(f"  Fine-tuning pairs: {len(QA_PAIRS)} Q&A pairs")
    
    # ── Build model ───────────────────────────────────────────────
    model = MiniLM(
        vocab_size=vocab_size,
        d_model=48,
        d_ff=96,
        max_len=80,
        learning_rate=0.003,
    )
    
    n_params = sum(p.size for p in model._params)
    print(f"  Model parameters: {n_params:,}")
    
    # ══════════════════════════════════════════════════════════════
    # STEP 1: PRE-TRAINING  (self-supervised next-token prediction)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STEP 1: PRE-TRAINING (self-supervised)")
    print("  'Learn language structure from unlabeled text'")
    print("=" * 60)
    
    pretrain_ids = text_to_ids(PRETRAIN_TEXT, c2i)
    
    start = time.time()
    pretrain_losses = model.train_on_text(
        pretrain_ids, epochs=30, seq_length=40,
        verbose=True, label="Pretrain"
    )
    pretrain_time = time.time() - start
    
    # Save pretrained weights
    pretrained_weights = model.save_weights()
    
    # Generate BEFORE fine-tuning
    print("\n  Sample generation (BEFORE fine-tuning):")
    prompt = "Q: what is deep learning?\nA:"
    prompt_ids = text_to_ids(prompt, c2i)
    generated = model.generate(prompt_ids, max_tokens=60, temperature=0.8)
    print(f"  '{ids_to_text(generated, i2c)}'")
    
    # ══════════════════════════════════════════════════════════════
    # STEP 2: FINE-TUNING  (supervised on Q&A pairs)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STEP 2: FINE-TUNING (supervised Q&A)")
    print("  'Teach the model to follow instructions'")
    print("=" * 60)
    
    # Prepare fine-tuning data
    finetune_text = ""
    for q, a in QA_PAIRS:
        finetune_text += q + a
    
    # Repeat the Q&A data to give more training signal
    finetune_text = (finetune_text * 5).lower()
    finetune_ids = text_to_ids(finetune_text, c2i)
    
    start = time.time()
    finetune_losses = model.train_on_text(
        finetune_ids, epochs=40, seq_length=50,
        verbose=True, label="Finetune"
    )
    finetune_time = time.time() - start
    
    # Generate AFTER fine-tuning
    print("\n  Sample generation (AFTER fine-tuning):")
    generated_ft = model.generate(prompt_ids, max_tokens=60, temperature=0.7)
    print(f"  '{ids_to_text(generated_ft, i2c)}'")
    
    # ══════════════════════════════════════════════════════════════
    # STEP 3: CHATBOT — answer multiple questions
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STEP 3: CHATBOT DEMO")
    print("=" * 60)
    
    test_questions = [
        "what is a neural network?",
        "what is backpropagation?",
        "what is a transformer?",
        "what is fine tuning?",
        "what is dropout?",
        "what is overfitting?",
    ]
    
    responses = []
    for question in test_questions:
        prompt_text = f"Q: {question}\nA:"
        prompt_ids_q = text_to_ids(prompt_text.lower(), c2i)
        
        newline_id = c2i.get("\n", None)
        gen = model.generate(prompt_ids_q, max_tokens=80, temperature=0.6,
                            stop_token=newline_id)
        response = ids_to_text(gen, i2c)
        
        # Extract just the answer part
        if "A:" in response:
            answer = response.split("A:")[-1].split("\n")[0].strip()
        else:
            answer = response[len(prompt_text):].split("\n")[0].strip()
        
        responses.append((question, answer))
        print(f"\n  Q: {question}")
        print(f"  A: {answer}")
    
    return {
        "model": model,
        "pretrain_losses": pretrain_losses,
        "finetune_losses": finetune_losses,
        "pretrained_weights": pretrained_weights,
        "responses": responses,
        "c2i": c2i, "i2c": i2c,
        "pretrain_time": pretrain_time,
        "finetune_time": finetune_time,
    }


def plot_pipeline(results, save_path=None):
    """Visualize the pretrain → fine-tune pipeline."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # ── 1) Training loss: pretrain vs fine-tune ───────────────────
    ax = axes[0, 0]
    pt_losses = results["pretrain_losses"]
    ft_losses = results["finetune_losses"]
    
    all_losses = pt_losses + ft_losses
    ax.plot(range(len(pt_losses)), pt_losses, "o-", color="#2563eb",
            linewidth=2, markersize=4, label="Step 1: Pre-training")
    ax.plot(range(len(pt_losses), len(all_losses)), ft_losses, "s-",
            color="#dc2626", linewidth=2, markersize=4, label="Step 2: Fine-tuning")
    ax.axvline(x=len(pt_losses) - 0.5, color="gray", linestyle="--", alpha=0.5)
    ax.annotate("Switch to\nFine-tuning", xy=(len(pt_losses), max(all_losses) * 0.7),
                fontsize=11, ha="center", color="gray")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Pipeline: Pretrain → Fine-tune\n"
                 "[Lecture 13: Same pipeline as ChatGPT]",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # ── 2) Weight change analysis ─────────────────────────────────
    ax = axes[0, 1]
    model = results["model"]
    pre_w = results["pretrained_weights"]
    
    weight_names = ["embedding", "W_Q", "W_K", "W_V", "W1", "W2", "W_out"]
    changes = []
    for name in weight_names:
        pre = pre_w[name]
        post = getattr(model, name)
        change = np.mean(np.abs(post - pre)) / (np.mean(np.abs(pre)) + 1e-10)
        changes.append(change * 100)
    
    colors = ["#ea580c" if c > np.median(changes) else "#2563eb" for c in changes]
    bars = ax.barh(weight_names, changes, color=colors, alpha=0.8, edgecolor="white")
    ax.set_xlabel("Relative Change (%)", fontsize=12)
    ax.set_title("Weight Changes After Fine-tuning\n"
                 "'First layers universal, later layers retrained'",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")
    
    # ── 3) Q&A Results ────────────────────────────────────────────
    ax = axes[1, 0]
    ax.axis("off")
    
    text = "Chatbot Responses After Fine-tuning\n" + "=" * 45 + "\n\n"
    for q, a in results["responses"][:6]:
        text += f"Q: {q}\nA: {a}\n\n"
    
    ax.text(0.02, 0.98, text, fontsize=11, fontfamily="monospace",
            verticalalignment="top", transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="#f0fdf4", edgecolor="#16a34a"))
    ax.set_title("Step 3: Chatbot Demo", fontsize=13, fontweight="bold")
    
    # ── 4) Pipeline summary ───────────────────────────────────────
    ax = axes[1, 1]
    ax.axis("off")
    
    summary = (
        "The GPT/ChatGPT Training Pipeline\n"
        "=" * 40 + "\n\n"
        "1. PRE-TRAINING (self-supervised)\n"
        f"   Corpus: {len(results['pretrain_losses'])} epochs\n"
        f"   Time: {results['pretrain_time']:.1f}s\n"
        "   Task: Predict next character\n"
        "   Result: Learns language structure\n\n"
        "2. FINE-TUNING (supervised Q&A)\n"
        f"   Pairs: {len(QA_PAIRS)} Q&A pairs\n"
        f"   Time: {results['finetune_time']:.1f}s\n"
        "   Task: Follow instruction format\n"
        "   Result: Answers questions!\n\n"
        "3. (RLHF - not implemented)\n"
        "   Would use human preference data\n"
        "   to further align responses.\n\n"
        "[Lecture 13: Zemke, AML, TUHH]"
    )
    ax.text(0.5, 0.5, summary, fontsize=12, fontfamily="monospace",
            ha="center", va="center", transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="#eff6ff", edgecolor="#2563eb"))
    
    fig.suptitle(
        "Simple Chatbot — Pretrain → Fine-tune → Chat\n"
        "[Lecture 13: 'Training of ChatGPT: pre-training, Q&A, RLHF']",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  ✅ {save_path}")
    plt.close()


def plot_before_after(results, save_path=None):
    """Compare model output before and after fine-tuning."""
    model = results["model"]
    c2i, i2c = results["c2i"], results["i2c"]
    pre_w = results["pretrained_weights"]
    
    questions = [
        "what is deep learning?",
        "what is a transformer?",
        "what is dropout?",
    ]
    
    fig, axes = plt.subplots(len(questions), 1, figsize=(16, 4 * len(questions)))
    
    for idx, question in enumerate(questions):
        ax = axes[idx]
        ax.axis("off")
        
        prompt = f"Q: {question}\nA:"
        prompt_ids = text_to_ids(prompt.lower(), c2i)
        newline_id = c2i.get("\n", None)
        
        # Generate with FINE-TUNED model
        gen_ft = model.generate(prompt_ids, max_tokens=70, temperature=0.6,
                               stop_token=newline_id)
        resp_ft = ids_to_text(gen_ft, i2c)
        if "A:" in resp_ft:
            answer_ft = resp_ft.split("A:")[-1].split("\n")[0].strip()
        else:
            answer_ft = "..."
        
        # Generate with PRE-TRAINED model (load old weights)
        current_w = model.save_weights()
        model.load_weights(pre_w)
        gen_pt = model.generate(prompt_ids, max_tokens=70, temperature=0.6,
                               stop_token=newline_id)
        resp_pt = ids_to_text(gen_pt, i2c)
        if "A:" in resp_pt:
            answer_pt = resp_pt.split("A:")[-1].split("\n")[0].strip()
        else:
            answer_pt = resp_pt[len(prompt):].split("\n")[0].strip()
        model.load_weights(current_w)  # restore fine-tuned
        
        text = f"Q: {question}\n\n"
        text += f"BEFORE fine-tuning:  \"{answer_pt[:80]}...\"\n"
        text += f"AFTER fine-tuning:   \"{answer_ft[:80]}\"\n"
        
        ax.text(0.02, 0.5, text, fontsize=12, fontfamily="monospace",
                va="center", transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="#fefce8", edgecolor="#ca8a04"))
    
    fig.suptitle(
        "Before vs After Fine-tuning — Same Model, Different Training\n"
        "Pre-trained model generates random text; Fine-tuned model answers questions!",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ {save_path}")
    plt.close()


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    
    results = run_pipeline()
    
    print("\n\n  Generating plots...")
    plot_pipeline(results, save_path="figures/pipeline.png")
    plot_before_after(results, save_path="figures/before_after.png")
    
    print("\n✨ Done!")
