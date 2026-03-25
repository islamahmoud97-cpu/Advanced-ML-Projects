"""
Character-Level RNN — From Scratch
=====================================
Based on: Zemke, AML Lecture 8

"Particular simple RNN are the so-called Jordan and Elman networks:
 An Elman network couples the hidden state with the previous hidden
 state and is based on:
   h_t = σ_h(W_hh · h_{t-1} + W_hx · x_t + b_h)
   y_t = σ_y(W_yh · h_t + b_y)"
 — Lecture 8

This implements a character-level Elman RNN that:
  1. Reads text character by character
  2. Learns to predict the next character
  3. Generates new text by sampling from predictions

Training: Backpropagation Through Time (BPTT) [Lecture 8]
  - Unroll the RNN for T time steps
  - Backpropagate through the unrolled network
  - Gradients flow backwards through time

All implemented with ONLY NumPy.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class CharRNN:
    """
    Character-level Recurrent Neural Network (Elman Network).
    
    Architecture [Lecture 8]:
      h_t = tanh(W_hh · h_{t-1} + W_hx · x_t + b_h)    (hidden state)
      y_t = softmax(W_yh · h_t + b_y)                     (output / prediction)
    
    Where:
      x_t ∈ R^vocab_size  : one-hot encoded input character
      h_t ∈ R^hidden_size : hidden state (memory)
      y_t ∈ R^vocab_size  : probability of next character
    
    Parameters
    ----------
    vocab_size : int
        Number of unique characters in the text.
    hidden_size : int
        Number of hidden units (memory capacity).
    seq_length : int
        Number of time steps to unroll for BPTT.
    learning_rate : float
        Learning rate for Adagrad optimizer.
    """
    
    def __init__(self, vocab_size: int, hidden_size: int = 128,
                 seq_length: int = 25, learning_rate: float = 0.01):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lr = learning_rate
        
        # ── Weight Initialization (Xavier) [Lecture 3] ────────────
        scale_hx = np.sqrt(2.0 / (vocab_size + hidden_size))
        scale_hh = np.sqrt(2.0 / (hidden_size + hidden_size))
        scale_yh = np.sqrt(2.0 / (hidden_size + vocab_size))
        
        # W_hx: input → hidden
        self.Whx = np.random.randn(hidden_size, vocab_size) * scale_hx
        # W_hh: hidden → hidden (recurrent)
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale_hh
        # W_yh: hidden → output
        self.Wyh = np.random.randn(vocab_size, hidden_size) * scale_yh
        # Biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))
        
        # Adagrad memory (accumulated squared gradients)
        self.mWhx = np.zeros_like(self.Whx)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWyh = np.zeros_like(self.Wyh)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)
        
        # Training history
        self.loss_history = []
    
    def forward(self, inputs: List[int], h_prev: np.ndarray
               ) -> Tuple[Dict, np.ndarray]:
        """
        Forward pass through time steps.
        
        inputs: list of character indices [x_0, x_1, ..., x_T]
        h_prev: hidden state from previous sequence (hidden_size, 1)
        
        Returns cache (for backprop) and final hidden state.
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = h_prev.copy()
        
        for t in range(len(inputs)):
            # One-hot encode input
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            
            # h_t = tanh(W_hh · h_{t-1} + W_hx · x_t + b_h)  [Lecture 8]
            hs[t] = np.tanh(self.Whh @ hs[t-1] + self.Whx @ xs[t] + self.bh)
            
            # y_t = W_yh · h_t + b_y  (logits)
            ys[t] = self.Wyh @ hs[t] + self.by
            
            # p_t = softmax(y_t)  (probabilities)
            e = np.exp(ys[t] - np.max(ys[t]))
            ps[t] = e / np.sum(e)
        
        cache = {"xs": xs, "hs": hs, "ps": ps}
        return cache, hs[len(inputs) - 1]
    
    def loss(self, cache: Dict, targets: List[int]) -> float:
        """Cross-entropy loss: L = -Σ log(p_t[target_t])"""
        total_loss = 0
        for t in range(len(targets)):
            total_loss -= np.log(cache["ps"][t][targets[t], 0] + 1e-12)
        return total_loss
    
    def backward(self, cache: Dict, targets: List[int]
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Backpropagation Through Time (BPTT) [Lecture 8]
        
        "BPTT results in a second discrete dynamical system for the
         derivatives of the cost function with respect to the hidden states."
        
        Key: gradients flow BACKWARDS through time steps.
        Problem: for long sequences, gradients vanish or explode
        (→ solved by LSTM in Lecture 9).
        """
        xs, hs, ps = cache["xs"], cache["hs"], cache["ps"]
        T = len(targets)
        
        # Initialize gradients
        dWhx = np.zeros_like(self.Whx)
        dWhh = np.zeros_like(self.Whh)
        dWyh = np.zeros_like(self.Wyh)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        dh_next = np.zeros_like(hs[0])
        
        for t in reversed(range(T)):
            # Output gradient: dy = p_t - one_hot(target_t)
            dy = ps[t].copy()
            dy[targets[t]] -= 1
            
            # ∂L/∂W_yh = dy · h_t^T
            dWyh += dy @ hs[t].T
            dby += dy
            
            # ∂L/∂h_t = W_yh^T · dy + dh_next (from future time step)
            dh = self.Wyh.T @ dy + dh_next
            
            # Through tanh: dh_raw = (1 - h_t²) ⊙ dh
            dh_raw = (1 - hs[t] ** 2) * dh
            
            # ∂L/∂W_hx = dh_raw · x_t^T
            dWhx += dh_raw @ xs[t].T
            # ∂L/∂W_hh = dh_raw · h_{t-1}^T
            dWhh += dh_raw @ hs[t-1].T
            dbh += dh_raw
            
            # Pass gradient to previous time step
            dh_next = self.Whh.T @ dh_raw
        
        # Gradient clipping (prevent exploding gradients) [Lecture 8]
        for dparam in [dWhx, dWhh, dWyh, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        
        return dWhx, dWhh, dWyh, dbh, dby
    
    def update(self, dWhx, dWhh, dWyh, dbh, dby):
        """Adagrad update [Lecture 3: adaptive learning rates]."""
        for param, dparam, mem in [
            (self.Whx, dWhx, self.mWhx),
            (self.Whh, dWhh, self.mWhh),
            (self.Wyh, dWyh, self.mWyh),
            (self.bh, dbh, self.mbh),
            (self.by, dby, self.mby),
        ]:
            mem += dparam ** 2
            param -= self.lr * dparam / (np.sqrt(mem) + 1e-8)
    
    def sample(self, h: np.ndarray, seed_idx: int, n_chars: int,
               temperature: float = 1.0) -> List[int]:
        """
        Generate text by sampling from the network.
        
        Parameters
        ----------
        h : hidden state
        seed_idx : starting character index
        n_chars : number of characters to generate
        temperature : controls randomness (0.1=conservative, 2.0=creative)
        
        Returns: list of character indices
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_idx] = 1
        indices = []
        
        for _ in range(n_chars):
            h = np.tanh(self.Whh @ h + self.Whx @ x + self.bh)
            y = self.Wyh @ h + self.by
            
            # Temperature scaling
            y = y / temperature
            e = np.exp(y - np.max(y))
            p = e / np.sum(e)
            
            # Sample from distribution
            idx = np.random.choice(self.vocab_size, p=p.ravel())
            
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1
            indices.append(idx)
        
        return indices
    
    def train(self, data: str, char_to_idx: Dict, idx_to_char: Dict,
              n_iterations: int = 10000, print_every: int = 1000,
              sample_every: int = 2000, sample_length: int = 200):
        """
        Train the RNN on a text corpus.
        
        Parameters
        ----------
        data : str — the training text
        char_to_idx : dict mapping characters to indices
        idx_to_char : dict mapping indices to characters
        n_iterations : total training steps
        """
        n = 0  # position in data
        p = 0  # data pointer
        smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_length
        h_prev = np.zeros((self.hidden_size, 1))
        
        samples = []
        
        for iteration in range(n_iterations):
            # Reset if end of data
            if p + self.seq_length + 1 >= len(data) or iteration == 0:
                h_prev = np.zeros((self.hidden_size, 1))
                p = 0
            
            # Prepare input and target sequences
            inputs = [char_to_idx[ch] for ch in data[p:p + self.seq_length]]
            targets = [char_to_idx[ch] for ch in data[p + 1:p + self.seq_length + 1]]
            
            # Forward pass
            cache, h_prev = self.forward(inputs, h_prev)
            
            # Compute loss
            loss = self.loss(cache, targets)
            smooth_loss = 0.999 * smooth_loss + 0.001 * loss
            self.loss_history.append(smooth_loss)
            
            # Backward pass (BPTT)
            grads = self.backward(cache, targets)
            
            # Update parameters
            self.update(*grads)
            
            # Print progress
            if iteration % print_every == 0:
                print(f"  Iter {iteration:6d}  |  Loss: {smooth_loss:.4f}")
            
            # Sample text
            if iteration % sample_every == 0:
                sample_idx = self.sample(h_prev, inputs[0], sample_length, temperature=0.8)
                sample_text = ''.join(idx_to_char[i] for i in sample_idx)
                samples.append((iteration, sample_text))
                
                if iteration % print_every == 0:
                    print(f"  --- Sample ---")
                    print(f"  {sample_text[:100]}...")
                    print()
            
            p += self.seq_length
        
        return samples
