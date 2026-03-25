"""
Variational Autoencoder (VAE) — From Scratch
===============================================
Based on: Zemke, AML Lecture 11, Slides 9-25

"Autoencoders are neural networks trained on given data that try to
 compute the data, i.e., f_autoencoder: R^n → R^n. Most feature a
 layer with lower dimension m << n that acts as a bottleneck."
 — Lecture 11

Architecture:
  Encoder:  x → μ, σ²     (map input to latent distribution parameters)
  Latent:   z = μ + σ·ε   (reparametrization trick, ε ~ N(0,1))
  Decoder:  z → x̂          (reconstruct input from latent code)

Loss:
  L = Reconstruction Loss + KL Divergence
    = ||x - x̂||² + KL(q(z|x) || p(z))
    = MSE + (-½ · Σ(1 + log(σ²) - μ² - σ²))

The KL term pushes the latent distribution towards N(0,1),
creating a smooth, continuous latent space that we can sample from.

All implemented with ONLY NumPy.
"""

import numpy as np
from typing import Tuple, Dict, List


class VAE:
    """
    Variational Autoencoder.
    
    Architecture:
      Encoder: input(784) → hidden(256) → ReLU → hidden(128) → ReLU → μ(latent), logσ²(latent)
      Decoder: z(latent) → hidden(128) → ReLU → hidden(256) → ReLU → output(784) → Sigmoid
    
    Parameters
    ----------
    input_dim : int — flattened input size (e.g., 784 for 28×28)
    hidden_dims : list — encoder hidden layer sizes
    latent_dim : int — dimension of latent space z
    """
    
    def __init__(self, input_dim: int = 64, hidden_dims: List[int] = [128, 64],
                 latent_dim: int = 2, learning_rate: float = 0.001):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.lr = learning_rate
        
        # ── Encoder Weights ───────────────────────────────────────
        dims = [input_dim] + hidden_dims
        self.enc_W = []
        self.enc_b = []
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            self.enc_W.append(np.random.randn(dims[i+1], dims[i]) * scale)
            self.enc_b.append(np.zeros((dims[i+1], 1)))
        
        # μ and log(σ²) layers
        last_hidden = hidden_dims[-1]
        scale_mu = np.sqrt(2.0 / last_hidden)
        self.W_mu = np.random.randn(latent_dim, last_hidden) * scale_mu
        self.b_mu = np.zeros((latent_dim, 1))
        self.W_logvar = np.random.randn(latent_dim, last_hidden) * scale_mu
        self.b_logvar = np.zeros((latent_dim, 1))
        
        # ── Decoder Weights ───────────────────────────────────────
        dec_dims = [latent_dim] + list(reversed(hidden_dims)) + [input_dim]
        self.dec_W = []
        self.dec_b = []
        for i in range(len(dec_dims) - 1):
            scale = np.sqrt(2.0 / dec_dims[i])
            self.dec_W.append(np.random.randn(dec_dims[i+1], dec_dims[i]) * scale)
            self.dec_b.append(np.zeros((dec_dims[i+1], 1)))
        
        # Adam state
        self._init_adam()
        self.t = 0
        
        # History
        self.history = {"loss": [], "recon_loss": [], "kl_loss": []}
    
    def _init_adam(self):
        """Initialize Adam optimizer state for all parameters."""
        self.adam_m = {}
        self.adam_v = {}
        for name, params in self._all_params().items():
            self.adam_m[name] = [np.zeros_like(p) for p in params]
            self.adam_v[name] = [np.zeros_like(p) for p in params]
    
    def _all_params(self):
        return {
            "enc_W": self.enc_W, "enc_b": self.enc_b,
            "W_mu": [self.W_mu], "b_mu": [self.b_mu],
            "W_logvar": [self.W_logvar], "b_logvar": [self.b_logvar],
            "dec_W": self.dec_W, "dec_b": self.dec_b,
        }
    
    @staticmethod
    def _relu(z):
        return np.maximum(0, z)
    
    @staticmethod
    def _relu_deriv(z):
        return (z > 0).astype(float)
    
    @staticmethod
    def _sigmoid(z):
        z_safe = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_safe))
    
    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Encoder: x → μ, log(σ²)
        
        x: (input_dim, batch_size)
        Returns: mu, logvar, cache
        """
        cache = {"enc_z": [], "enc_a": [x]}
        a = x
        
        for i in range(len(self.enc_W)):
            z = self.enc_W[i] @ a + self.enc_b[i]
            cache["enc_z"].append(z)
            a = self._relu(z)
            cache["enc_a"].append(a)
        
        # μ and log(σ²) — no activation (linear)
        mu = self.W_mu @ a + self.b_mu
        logvar = self.W_logvar @ a + self.b_logvar
        
        cache["last_hidden"] = a
        return mu, logvar, cache
    
    def reparametrize(self, mu: np.ndarray, logvar: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reparametrization Trick [Lecture 11]
        
        z = μ + σ · ε,   where ε ~ N(0, I)
        
        This trick allows gradients to flow through the sampling step.
        Without it, we can't backpropagate through random sampling!
        
        σ = exp(0.5 · log(σ²))
        """
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        z = mu + std * eps
        return z, eps
    
    def decode(self, z: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Decoder: z → x̂
        
        z: (latent_dim, batch_size)
        Returns: x_reconstructed, cache
        """
        cache = {"dec_z": [], "dec_a": [z]}
        a = z
        
        for i in range(len(self.dec_W)):
            zi = self.dec_W[i] @ a + self.dec_b[i]
            cache["dec_z"].append(zi)
            
            if i == len(self.dec_W) - 1:
                # Last layer: sigmoid (output in [0, 1])
                a = self._sigmoid(zi)
            else:
                a = self._relu(zi)
            cache["dec_a"].append(a)
        
        return a, cache
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Full forward pass: encode → sample → decode."""
        mu, logvar, enc_cache = self.encode(x)
        z, eps = self.reparametrize(mu, logvar)
        x_recon, dec_cache = self.decode(z)
        
        cache = {**enc_cache, **dec_cache, "mu": mu, "logvar": logvar,
                 "z": z, "eps": eps}
        return x_recon, mu, logvar, z, cache
    
    def loss(self, x: np.ndarray, x_recon: np.ndarray,
             mu: np.ndarray, logvar: np.ndarray) -> Tuple[float, float, float]:
        """
        VAE Loss = Reconstruction + KL Divergence  [Lecture 11]
        
        Reconstruction: MSE(x, x̂) = (1/N) · Σ||x - x̂||²
        
        KL Divergence: KL(q(z|x) || p(z))
          = -½ · Σ(1 + log(σ²) - μ² - σ²)
          
        This pushes q(z|x) towards the prior p(z) = N(0, I).
        """
        N = x.shape[1]
        
        # Reconstruction loss (MSE)
        recon_loss = np.sum((x - x_recon) ** 2) / N
        
        # KL divergence  [Lecture 11: Bregman divergence → KL]
        kl_loss = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar)) / N
        
        total_loss = recon_loss + kl_loss
        return total_loss, recon_loss, kl_loss
    
    def backward(self, x, x_recon, cache):
        """Backpropagation through the entire VAE."""
        N = x.shape[1]
        mu = cache["mu"]
        logvar = cache["logvar"]
        eps = cache["eps"]
        
        # ── Decoder backward ─────────────────────────────────────
        # d(recon_loss)/d(x_recon) = 2(x_recon - x) / N
        da = 2 * (x_recon - x) / N
        
        # Through sigmoid (last layer)
        sig = cache["dec_a"][-1]
        da = da * sig * (1 - sig)
        
        d_dec_W = []
        d_dec_b = []
        
        for i in range(len(self.dec_W) - 1, -1, -1):
            dW = da @ cache["dec_a"][i].T
            db = np.sum(da, axis=1, keepdims=True)
            d_dec_W.insert(0, dW)
            d_dec_b.insert(0, db)
            
            if i > 0:
                da = self.dec_W[i].T @ da
                da = da * self._relu_deriv(cache["dec_z"][i-1])
            else:
                da = self.dec_W[i].T @ da  # gradient w.r.t. z
        
        dz = da  # gradient w.r.t. latent z
        
        # ── Through reparametrization ─────────────────────────────
        # z = μ + σ·ε → dz/dμ = 1, dz/dσ = ε, dσ/dlogvar = 0.5·σ
        std = np.exp(0.5 * logvar)
        
        # d(total)/dμ = dz + d(KL)/dμ
        d_mu = dz + mu / N  # KL gradient: μ
        
        # d(total)/d(logvar) = dz·ε·0.5·σ + d(KL)/d(logvar)
        d_logvar = dz * eps * 0.5 * std + 0.5 * (np.exp(logvar) - 1) / N
        
        # ── μ and logvar layer gradients ──────────────────────────
        last_h = cache["last_hidden"]
        dW_mu = d_mu @ last_h.T
        db_mu = np.sum(d_mu, axis=1, keepdims=True)
        dW_logvar = d_logvar @ last_h.T
        db_logvar = np.sum(d_logvar, axis=1, keepdims=True)
        
        # Gradient through to encoder
        da_enc = self.W_mu.T @ d_mu + self.W_logvar.T @ d_logvar
        
        # ── Encoder backward ─────────────────────────────────────
        d_enc_W = []
        d_enc_b = []
        
        for i in range(len(self.enc_W) - 1, -1, -1):
            da_enc = da_enc * self._relu_deriv(cache["enc_z"][i])
            dW = da_enc @ cache["enc_a"][i].T
            db = np.sum(da_enc, axis=1, keepdims=True)
            d_enc_W.insert(0, dW)
            d_enc_b.insert(0, db)
            
            if i > 0:
                da_enc = self.enc_W[i].T @ da_enc
        
        return {
            "enc_W": d_enc_W, "enc_b": d_enc_b,
            "W_mu": [dW_mu], "b_mu": [db_mu],
            "W_logvar": [dW_logvar], "b_logvar": [db_logvar],
            "dec_W": d_dec_W, "dec_b": d_dec_b,
        }
    
    def _adam_update(self, grads):
        """Adam optimizer update."""
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        param_map = {
            "enc_W": self.enc_W, "enc_b": self.enc_b,
            "W_mu": [self.W_mu], "b_mu": [self.b_mu],
            "W_logvar": [self.W_logvar], "b_logvar": [self.b_logvar],
            "dec_W": self.dec_W, "dec_b": self.dec_b,
        }
        
        for name in grads:
            for idx in range(len(grads[name])):
                g = np.clip(grads[name][idx], -1, 1)
                self.adam_m[name][idx] = beta1 * self.adam_m[name][idx] + (1-beta1) * g
                self.adam_v[name][idx] = beta2 * self.adam_v[name][idx] + (1-beta2) * g**2
                m_hat = self.adam_m[name][idx] / (1 - beta1**self.t)
                v_hat = self.adam_v[name][idx] / (1 - beta2**self.t)
                param_map[name][idx] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Write back single params
        self.W_mu = param_map["W_mu"][0]
        self.b_mu = param_map["b_mu"][0]
        self.W_logvar = param_map["W_logvar"][0]
        self.b_logvar = param_map["b_logvar"][0]
    
    def train(self, X: np.ndarray, epochs: int = 50, batch_size: int = 64,
              verbose: bool = True):
        """
        Train the VAE.
        X: (n_samples, input_dim)
        """
        N = X.shape[0]
        
        for epoch in range(epochs):
            perm = np.random.permutation(N)
            epoch_loss, epoch_recon, epoch_kl = 0, 0, 0
            n_batches = 0
            
            for j in range(0, N, batch_size):
                batch = X[perm[j:j+batch_size]].T  # (input_dim, batch)
                
                x_recon, mu, logvar, z, cache = self.forward(batch)
                total, recon, kl = self.loss(batch, x_recon, mu, logvar)
                
                grads = self.backward(batch, x_recon, cache)
                self._adam_update(grads)
                
                epoch_loss += total
                epoch_recon += recon
                epoch_kl += kl
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon / n_batches
            avg_kl = epoch_kl / n_batches
            
            self.history["loss"].append(avg_loss)
            self.history["recon_loss"].append(avg_recon)
            self.history["kl_loss"].append(avg_kl)
            
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"  Epoch {epoch+1:3d}/{epochs}  |  Loss: {avg_loss:.4f}  "
                      f"|  Recon: {avg_recon:.4f}  |  KL: {avg_kl:.4f}")
        
        return self.history
    
    def generate(self, n_samples: int = 16) -> np.ndarray:
        """Generate new images by sampling z ~ N(0, I) and decoding."""
        z = np.random.randn(self.latent_dim, n_samples)
        x_gen, _ = self.decode(z)
        return x_gen.T  # (n_samples, input_dim)
    
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Encode and decode input (test reconstruction quality)."""
        x_recon, _, _, _, _ = self.forward(x.T)
        return x_recon.T
    
    def encode_to_latent(self, x: np.ndarray) -> np.ndarray:
        """Encode input to latent space (for visualization)."""
        mu, _, _ = self.encode(x.T)
        return mu.T  # (n_samples, latent_dim)
