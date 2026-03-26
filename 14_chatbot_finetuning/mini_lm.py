"""
Mini Language Model — Pretrain & Fine-tune Pipeline
=====================================================
Based on: Zemke, AML Lecture 13: LLM, Finetuning, Fast Inference

"Training of ChatGPT: pre-training (self-supervised learning),
 Q & A (supervised learning), reinforcement learning from human
 feedback (RLHF)" — Lecture 13

This implements the SAME pipeline used by GPT/ChatGPT, but at miniature scale:
  1. PRE-TRAINING:  Self-supervised next-token prediction on general text
  2. FINE-TUNING:   Supervised training on instruction/answer pairs
  3. INFERENCE:     Generate responses to user queries

"first layers often are universal, only later layers are trained new"
— Lecture 7 (transfer learning principle)

Architecture: Character-level Transformer (simplified)
  - Token Embedding + Positional Encoding
  - Single self-attention layer
  - Feed-forward layer
  - Output projection → next token prediction

All implemented with ONLY NumPy.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class MiniLM:
    """
    Miniature Language Model for demonstrating Pretrain → Fine-tune.
    
    Architecture (simplified Transformer decoder):
      Token embedding (vocab → d_model)
      + Positional encoding
      → Self-attention (single head)
      → Feed-forward (d_model → d_ff → d_model)
      → Output projection (d_model → vocab)
    
    Parameters
    ----------
    vocab_size : number of unique characters
    d_model : model dimension
    d_ff : feed-forward hidden dimension
    max_len : maximum sequence length
    learning_rate : for Adam optimizer
    """
    
    def __init__(self, vocab_size: int, d_model: int = 64,
                 d_ff: int = 128, max_len: int = 128, learning_rate: float = 0.001):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.max_len = max_len
        self.lr = learning_rate
        
        # ── Token Embedding ───────────────────────────────────────
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02
        
        # ── Positional Encoding (sinusoidal, not learned) ─────────
        self.pe = np.zeros((max_len, d_model))
        pos = np.arange(max_len)[:, np.newaxis]
        div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        self.pe[:, 0::2] = np.sin(pos * div)
        self.pe[:, 1::2] = np.cos(pos * div)
        
        # ── Self-Attention (Q, K, V projections) ──────────────────
        scale = np.sqrt(2.0 / d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale
        
        # ── Feed-Forward Network ──────────────────────────────────
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros(d_model)
        
        # ── Output Projection ────────────────────────────────────
        self.W_out = np.random.randn(d_model, vocab_size) * np.sqrt(2.0 / d_model)
        self.b_out = np.zeros(vocab_size)
        
        # ── Adam state ────────────────────────────────────────────
        self._params = [
            self.embedding, self.W_Q, self.W_K, self.W_V, self.W_O,
            self.W1, self.b1, self.W2, self.b2, self.W_out, self.b_out,
        ]
        self.adam_m = [np.zeros_like(p) for p in self._params]
        self.adam_v = [np.zeros_like(p) for p in self._params]
        self.adam_t = 0
        
        self.history = {"loss": []}
    
    def _causal_mask(self, seq_len):
        """Create causal (autoregressive) mask — can only attend to past."""
        mask = np.tril(np.ones((seq_len, seq_len)))
        return mask
    
    def _softmax(self, x, axis=-1):
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)
    
    def forward(self, token_ids: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass: predict next token at each position.
        
        token_ids: (seq_len,) integer array
        Returns: logits (seq_len, vocab_size), cache
        """
        seq_len = len(token_ids)
        
        # Embedding + positional encoding
        x = self.embedding[token_ids] * np.sqrt(self.d_model)  # (seq, d_model)
        x = x + self.pe[:seq_len]
        
        # Self-attention (single head, causal)
        Q = x @ self.W_Q  # (seq, d_model)
        K = x @ self.W_K
        V = x @ self.W_V
        
        d_k = self.d_model
        scores = (Q @ K.T) / np.sqrt(d_k)  # (seq, seq)
        
        # Causal mask — can only attend to past tokens
        mask = self._causal_mask(seq_len)
        scores = np.where(mask == 0, -1e9, scores)
        
        attn_weights = self._softmax(scores, axis=-1)
        attn_out = attn_weights @ V  # (seq, d_model)
        attn_proj = attn_out @ self.W_O
        
        # Residual + (skip LayerNorm for simplicity)
        x = x + attn_proj
        
        # Feed-forward
        ff_hidden = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        ff_out = ff_hidden @ self.W2 + self.b2
        
        # Residual
        x = x + ff_out
        
        # Output logits
        logits = x @ self.W_out + self.b_out  # (seq, vocab)
        
        cache = {
            "token_ids": token_ids, "x_embed": self.embedding[token_ids],
            "Q": Q, "K": K, "V": V, "attn_weights": attn_weights,
            "attn_out": attn_out, "ff_hidden": ff_hidden,
            "x_final": x,
        }
        return logits, cache
    
    def compute_loss(self, logits, targets):
        """Cross-entropy loss for next-token prediction."""
        probs = self._softmax(logits)
        N = len(targets)
        loss = 0
        for t in range(N):
            loss -= np.log(probs[t, targets[t]] + 1e-12)
        return loss / N, probs
    
    def train_step(self, input_ids, target_ids):
        """One training step with simplified backprop."""
        logits, cache = self.forward(input_ids)
        loss, probs = self.compute_loss(logits, target_ids)
        
        N = len(target_ids)
        
        # Output gradient
        d_logits = probs.copy()
        for t in range(N):
            d_logits[t, target_ids[t]] -= 1.0
        d_logits /= N
        
        # Gradient w.r.t. output projection
        x_final = cache["x_final"]
        dW_out = x_final.T @ d_logits
        db_out = np.sum(d_logits, axis=0)
        
        # Gradient through to x_final
        dx = d_logits @ self.W_out.T
        
        # Through feed-forward (simplified)
        dff_out = dx  # residual
        dff_hidden = dff_out @ self.W2.T
        dff_hidden = dff_hidden * (cache["ff_hidden"] > 0).astype(float)  # ReLU
        
        x_pre_ff = cache["x_embed"] * np.sqrt(self.d_model) + self.pe[:N] + cache["attn_out"] @ self.W_O
        dW1 = x_pre_ff.T @ dff_hidden
        db1 = np.sum(dff_hidden, axis=0)
        dW2 = cache["ff_hidden"].T @ dff_out
        db2 = np.sum(dff_out, axis=0)
        
        # Simplified embedding gradient
        d_embed = dx.copy()
        for t in range(N):
            tok = input_ids[t]
            self.embedding[tok] -= self.lr * 0.1 * d_embed[t] * np.sqrt(self.d_model)
        
        # Adam update for main parameters
        grads = [
            np.zeros_like(self.embedding),  # embedding updated separately
            np.zeros_like(self.W_Q),  # simplified: skip attention grads
            np.zeros_like(self.W_K),
            np.zeros_like(self.W_V),
            np.zeros_like(self.W_O),
            dW1, db1, dW2, db2, dW_out, db_out,
        ]
        
        self.adam_t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        for idx in range(len(self._params)):
            g = np.clip(grads[idx], -1, 1)
            self.adam_m[idx] = beta1 * self.adam_m[idx] + (1 - beta1) * g
            self.adam_v[idx] = beta2 * self.adam_v[idx] + (1 - beta2) * g**2
            m_hat = self.adam_m[idx] / (1 - beta1**self.adam_t)
            v_hat = self.adam_v[idx] / (1 - beta2**self.adam_t)
            self._params[idx] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Sync back
        self.W1, self.b1 = self._params[5], self._params[6]
        self.W2, self.b2 = self._params[7], self._params[8]
        self.W_out, self.b_out = self._params[9], self._params[10]
        
        return loss
    
    def generate(self, prompt_ids: List[int], max_tokens: int = 100,
                 temperature: float = 0.8, stop_token: Optional[int] = None) -> List[int]:
        """
        Autoregressive generation: predict one token at a time.
        
        [Lecture 13: "Fast inference" — generate tokens sequentially]
        """
        generated = list(prompt_ids)
        
        for _ in range(max_tokens):
            # Use last max_len tokens as context
            context = np.array(generated[-self.max_len:])
            logits, _ = self.forward(context)
            
            # Get logits for the LAST position
            next_logits = logits[-1] / temperature
            probs = self._softmax(next_logits)
            
            # Sample
            next_token = np.random.choice(self.vocab_size, p=probs)
            generated.append(next_token)
            
            if stop_token is not None and next_token == stop_token:
                break
        
        return generated
    
    def train_on_text(self, text_ids: List[int], epochs: int = 10,
                      seq_length: int = 32, verbose: bool = True, label: str = ""):
        """
        Train on a text corpus with next-token prediction.
        
        This is the SELF-SUPERVISED pre-training step:
        Input:  [t₁, t₂, t₃, ..., tₙ₋₁]
        Target: [t₂, t₃, t₄, ..., tₙ  ]
        
        No labels needed — the text IS the supervision!
        """
        N = len(text_ids)
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0
            
            # Random starting positions
            starts = np.random.permutation(max(1, N - seq_length - 1))[:N // seq_length]
            
            for start in starts:
                end = min(start + seq_length, N - 1)
                if end - start < 5:
                    continue
                
                input_ids = np.array(text_ids[start:end])
                target_ids = np.array(text_ids[start + 1:end + 1])
                
                loss = self.train_step(input_ids, target_ids)
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            self.history["loss"].append(avg_loss)
            
            if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
                print(f"    [{label}] Epoch {epoch+1:3d}/{epochs}  |  Loss: {avg_loss:.4f}")
        
        return losses
    
    def save_weights(self) -> Dict:
        """Save model weights (for comparing before/after fine-tuning)."""
        return {
            "embedding": self.embedding.copy(),
            "W_Q": self.W_Q.copy(), "W_K": self.W_K.copy(),
            "W_V": self.W_V.copy(), "W_O": self.W_O.copy(),
            "W1": self.W1.copy(), "b1": self.b1.copy(),
            "W2": self.W2.copy(), "b2": self.b2.copy(),
            "W_out": self.W_out.copy(), "b_out": self.b_out.copy(),
        }
    
    def load_weights(self, weights: Dict):
        """Load saved weights."""
        self.embedding = weights["embedding"].copy()
        self.W_Q = weights["W_Q"].copy()
        self.W_K = weights["W_K"].copy()
        self.W_V = weights["W_V"].copy()
        self.W_O = weights["W_O"].copy()
        self.W1 = weights["W1"].copy()
        self.b1 = weights["b1"].copy()
        self.W2 = weights["W2"].copy()
        self.b2 = weights["b2"].copy()
        self.W_out = weights["W_out"].copy()
        self.b_out = weights["b_out"].copy()
        self._params = [
            self.embedding, self.W_Q, self.W_K, self.W_V, self.W_O,
            self.W1, self.b1, self.W2, self.b2, self.W_out, self.b_out,
        ]
