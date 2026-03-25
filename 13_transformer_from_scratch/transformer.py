"""
Transformer — From Scratch
==============================
Based on: Zemke, AML Lecture 12, Slides 7-25

"attention(q, K, V) = Σⱼ similarity(q, kⱼ) · vⱼ
 where q 'query', k 'key', v 'value'"  — Lecture 12

"Scaled dot-product: similarity(q, k) = qᵀk / √d"  — Lecture 12

Components implemented:
  1. Scaled Dot-Product Attention        [L12, Slide 11]
  2. Multi-Head Attention                 [L12, Slide 14]
  3. Positional Encoding (sinusoidal)     [L12, Slide 17]
  4. Layer Normalization                  
  5. Feed-Forward Network                
  6. Transformer Encoder Block            [L12, Slide 16]
  7. Full Transformer Encoder             

All implemented with ONLY NumPy — no PyTorch, no TensorFlow.
"""

import numpy as np
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# 1. SCALED DOT-PRODUCT ATTENTION  [Lecture 12, Slide 11]
# ═══════════════════════════════════════════════════════════════════════════
def scaled_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scaled Dot-Product Attention.
    
    [Lecture 12]: "similarity(q, k) = qᵀk / √d"
    
    Attention(Q, K, V) = softmax(Q · Kᵀ / √d_k) · V
    
    Parameters
    ----------
    Q : (batch, seq_len_q, d_k)  — queries
    K : (batch, seq_len_k, d_k)  — keys
    V : (batch, seq_len_k, d_v)  — values
    mask : optional (batch, seq_len_q, seq_len_k)
    
    Returns
    -------
    output : (batch, seq_len_q, d_v)
    attention_weights : (batch, seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]
    
    # Q · Kᵀ / √d_k
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    
    # Apply mask (for causal/padding)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    # Softmax over key dimension
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Weighted sum of values
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights


# ═══════════════════════════════════════════════════════════════════════════
# 2. MULTI-HEAD ATTENTION  [Lecture 12, Slide 14]
# ═══════════════════════════════════════════════════════════════════════════
class MultiHeadAttention:
    """
    Multi-Head Attention.
    
    [Lecture 12]: "Instead of single attention, use h parallel attention heads"
    
    MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) · Wᴼ
    where headᵢ = Attention(Q·Wᵢᵠ, K·Wᵢᴷ, V·Wᵢⱽ)
    
    Each head learns different attention patterns:
    - Head 1 might attend to adjacent tokens
    - Head 2 might attend to syntactic relationships
    - Head 3 might attend to semantic similarity
    """
    
    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Projection matrices
        scale = np.sqrt(2.0 / d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale
        
        # Gradients
        self.dW_Q = np.zeros_like(self.W_Q)
        self.dW_K = np.zeros_like(self.W_K)
        self.dW_V = np.zeros_like(self.W_V)
        self.dW_O = np.zeros_like(self.W_O)
        
        self._cache = None
    
    def _split_heads(self, x):
        """(batch, seq, d_model) → (batch, n_heads, seq, d_k)"""
        batch, seq, _ = x.shape
        x = x.reshape(batch, seq, self.n_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def _merge_heads(self, x):
        """(batch, n_heads, seq, d_k) → (batch, seq, d_model)"""
        batch, _, seq, _ = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch, seq, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        """
        Forward pass.
        Q, K, V: (batch, seq_len, d_model)
        """
        batch = Q.shape[0]
        
        # Linear projections
        Q_proj = Q @ self.W_Q  # (batch, seq, d_model)
        K_proj = K @ self.W_K
        V_proj = V @ self.W_V
        
        # Split into heads
        Q_heads = self._split_heads(Q_proj)  # (batch, n_heads, seq, d_k)
        K_heads = self._split_heads(K_proj)
        V_heads = self._split_heads(V_proj)
        
        # Expand mask for heads
        if mask is not None:
            mask = mask[:, np.newaxis, :, :]  # (batch, 1, seq_q, seq_k)
        
        # Attention per head
        attn_output, attn_weights = scaled_dot_product_attention(
            Q_heads.reshape(-1, Q_heads.shape[2], self.d_k),
            K_heads.reshape(-1, K_heads.shape[2], self.d_k),
            V_heads.reshape(-1, V_heads.shape[2], self.d_k),
        )
        attn_output = attn_output.reshape(batch, self.n_heads, -1, self.d_k)
        attn_weights = attn_weights.reshape(batch, self.n_heads, Q.shape[1], K.shape[1])
        
        # Merge heads and project
        concat = self._merge_heads(attn_output)  # (batch, seq, d_model)
        output = concat @ self.W_O
        
        self._cache = {
            "Q": Q, "K": K, "V": V,
            "Q_proj": Q_proj, "K_proj": K_proj, "V_proj": V_proj,
            "attn_weights": attn_weights, "concat": concat,
        }
        
        return output, attn_weights
    
    def params(self):
        return [
            (self.W_Q, self.dW_Q), (self.W_K, self.dW_K),
            (self.W_V, self.dW_V), (self.W_O, self.dW_O),
        ]


# ═══════════════════════════════════════════════════════════════════════════
# 3. POSITIONAL ENCODING  [Lecture 12, Slide 17]
# ═══════════════════════════════════════════════════════════════════════════
def positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    """
    Sinusoidal Positional Encoding.
    
    [Lecture 12]: "Transformers process all positions simultaneously
     → need positional information added to embeddings"
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Returns: (1, max_len, d_model)
    """
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe[np.newaxis, :, :]  # (1, max_len, d_model)


# ═══════════════════════════════════════════════════════════════════════════
# 4. LAYER NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════
class LayerNorm:
    """Layer Normalization — normalizes across features (not batch)."""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.gamma = np.ones((1, 1, d_model))
        self.beta = np.zeros((1, 1, d_model))
        self.eps = eps
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
    
    def forward(self, x):
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_norm + self.beta
    
    def params(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]


# ═══════════════════════════════════════════════════════════════════════════
# 5. FEED-FORWARD NETWORK
# ═══════════════════════════════════════════════════════════════════════════
class FeedForward:
    """
    Position-wise Feed-Forward Network.
    FFN(x) = ReLU(x · W₁ + b₁) · W₂ + b₂
    
    Typically d_ff = 4 × d_model.
    """
    
    def __init__(self, d_model: int, d_ff: int):
        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / d_ff)
        self.W1 = np.random.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros((1, 1, d_ff))
        self.W2 = np.random.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros((1, 1, d_model))
        
        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.b2)
        
        self._cache = None
    
    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        output = self.a1 @ self.W2 + self.b2
        self._cache = x
        return output
    
    def params(self):
        return [
            (self.W1, self.dW1), (self.b1, self.db1),
            (self.W2, self.dW2), (self.b2, self.db2),
        ]


# ═══════════════════════════════════════════════════════════════════════════
# 6. TRANSFORMER ENCODER BLOCK  [Lecture 12, Slide 16]
# ═══════════════════════════════════════════════════════════════════════════
class TransformerEncoderBlock:
    """
    Single Transformer Encoder Block.
    
    [Lecture 12]: The encoder consists of:
      1. Multi-Head Self-Attention + Residual + LayerNorm
      2. Feed-Forward Network + Residual + LayerNorm
    
    x → LayerNorm(x + MultiHeadAttention(x, x, x))
      → LayerNorm(x + FFN(x))
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # Self-Attention + Residual + LayerNorm
        attn_out, attn_weights = self.mha.forward(x, x, x, mask)
        x = self.norm1.forward(x + attn_out)  # residual connection
        
        # FFN + Residual + LayerNorm
        ffn_out = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_out)  # residual connection
        
        return x, attn_weights
    
    def params(self):
        return self.mha.params() + self.ffn.params() + self.norm1.params() + self.norm2.params()


# ═══════════════════════════════════════════════════════════════════════════
# 7. FULL TRANSFORMER ENCODER (for classification)
# ═══════════════════════════════════════════════════════════════════════════
class TransformerClassifier:
    """
    Transformer Encoder for sequence classification.
    
    Architecture:
      Input tokens → Embedding → + Positional Encoding
      → N × TransformerEncoderBlock
      → Mean pooling → Dense → Softmax
    
    Parameters
    ----------
    vocab_size : number of unique tokens
    d_model : embedding dimension (and model width)
    n_heads : number of attention heads
    n_layers : number of encoder blocks
    d_ff : feed-forward hidden dimension
    max_len : maximum sequence length
    n_classes : number of output classes
    """
    
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = 128, max_len: int = 50,
                 n_classes: int = 2):
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Token embedding
        self.embedding = np.random.randn(vocab_size, d_model) * 0.1
        self.d_embedding = np.zeros_like(self.embedding)
        
        # Positional encoding
        self.pe = positional_encoding(max_len, d_model)
        
        # Encoder blocks
        self.blocks = [
            TransformerEncoderBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]
        
        # Classification head
        scale = np.sqrt(2.0 / d_model)
        self.W_cls = np.random.randn(d_model, n_classes) * scale
        self.b_cls = np.zeros((1, n_classes))
        self.dW_cls = np.zeros_like(self.W_cls)
        self.db_cls = np.zeros_like(self.b_cls)
        
        # Training state
        self.history = {"loss": [], "acc": []}
    
    def forward(self, token_ids):
        """
        Forward pass.
        token_ids: (batch, seq_len) integer token indices
        """
        batch, seq_len = token_ids.shape
        
        # Embedding + positional encoding
        x = self.embedding[token_ids]  # (batch, seq, d_model)
        x = x * np.sqrt(self.d_model)  # scale embeddings
        x = x + self.pe[:, :seq_len, :]
        
        # Encoder blocks
        all_attn_weights = []
        for block in self.blocks:
            x, attn_w = block.forward(x)
            all_attn_weights.append(attn_w)
        
        # Mean pooling over sequence
        pooled = np.mean(x, axis=1)  # (batch, d_model)
        
        # Classification head
        logits = pooled @ self.W_cls + self.b_cls  # (batch, n_classes)
        
        # Softmax
        e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = e / np.sum(e, axis=-1, keepdims=True)
        
        self._cache = {
            "token_ids": token_ids, "x_final": x, "pooled": pooled,
            "probs": probs, "attn_weights": all_attn_weights,
        }
        return probs, all_attn_weights
    
    def _all_params(self):
        params = [(self.W_cls, self.dW_cls), (self.b_cls, self.db_cls)]
        for block in self.blocks:
            params.extend(block.params())
        return params
    
    def train(self, X, y, epochs=20, batch_size=32, lr=0.001, verbose=True):
        """
        Train with Adam optimizer.
        X: (n_samples, seq_len) integer token IDs
        y: (n_samples,) integer class labels
        """
        N = X.shape[0]
        n_classes = self.W_cls.shape[1]
        
        # Adam state
        params = self._all_params()
        m_list = [np.zeros_like(p) for p, _ in params]
        v_list = [np.zeros_like(p) for p, _ in params]
        t = 0
        
        for epoch in range(epochs):
            perm = np.random.permutation(N)
            epoch_loss = 0
            epoch_correct = 0
            n_batches = 0
            
            for j in range(0, N, batch_size):
                batch_idx = perm[j:j+batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                bs = len(batch_idx)
                
                # Forward
                probs, _ = self.forward(X_batch)
                
                # Loss (cross-entropy)
                y_onehot = np.zeros((bs, n_classes))
                y_onehot[np.arange(bs), y_batch] = 1
                loss = -np.sum(y_onehot * np.log(probs + 1e-12)) / bs
                epoch_loss += loss
                epoch_correct += np.sum(np.argmax(probs, axis=1) == y_batch)
                n_batches += 1
                
                # Simplified gradient (through classification head)
                d_logits = (probs - y_onehot) / bs
                pooled = self._cache["pooled"]
                
                self.dW_cls = pooled.T @ d_logits
                self.db_cls = np.sum(d_logits, axis=0, keepdims=True)
                
                # Adam update (classification head only for speed)
                t += 1
                for idx, (p, dp) in enumerate([(self.W_cls, self.dW_cls),
                                                (self.b_cls, self.db_cls)]):
                    g = np.clip(dp, -1, 1)
                    m_list[idx] = 0.9 * m_list[idx] + 0.1 * g
                    v_list[idx] = 0.999 * v_list[idx] + 0.001 * g**2
                    m_hat = m_list[idx] / (1 - 0.9**t)
                    v_hat = v_list[idx] / (1 - 0.999**t)
                    p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
                
                # Update embedding with gradient
                d_pooled = d_logits @ self.W_cls.T  # (batch, d_model)
                # Simplified: update embedding directly
                for b in range(bs):
                    for s in range(X_batch.shape[1]):
                        tok = X_batch[b, s]
                        self.embedding[tok] -= lr * 0.1 * d_pooled[b] / X_batch.shape[1]
            
            avg_loss = epoch_loss / n_batches
            accuracy = epoch_correct / N
            self.history["loss"].append(avg_loss)
            self.history["acc"].append(accuracy)
            
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"  Epoch {epoch+1:3d}/{epochs}  |  Loss: {avg_loss:.4f}  |  Acc: {accuracy:.4f}")
        
        return self.history
    
    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def get_attention_weights(self, X):
        """Get attention weights for visualization."""
        _, attn_weights = self.forward(X)
        return attn_weights
