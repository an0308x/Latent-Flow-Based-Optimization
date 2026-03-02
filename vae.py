"""
CHASE VAE Components: Compressor, Decompressor, and ESM2 Decoder.

Architecture based on ProtFlow (Kong et al., 2025) hourglass design:
- ESM2 encoder (frozen): sequence -> high-dim embeddings h ∈ R^{L×D}
- Compressor: h -> z ∈ R^{l×d}  (reduces both sequence length and feature dim)
- Decompressor: z -> h'          (symmetric reverse)
- Decoder: h' -> sequence logits (trained via cross-entropy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.norm(x)
        B, L, D = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b l (h d) -> b h l d', h=self.heads), qkv)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h l d -> b l (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, ff_mult: int = 4):
        super().__init__()
        self.attn = Attention(dim, heads)
        self.ff = FeedForward(dim, ff_mult)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x


# ---------------------------------------------------------------------------
# Compressor: ESM embeddings -> compact latent z
# ---------------------------------------------------------------------------

class Compressor(nn.Module):
    """
    Maps ESM2 embeddings h ∈ R^{L×D} to latent z ∈ R^{l×d}.

    Architecture (from ProtFlow / Kong et al. 2025):
      LayerNorm -> 2x TransformerBlock -> Conv1d (stride=c, downsample L->l) -> Linear(D->d)
      Then two parallel linear heads for mean and log-variance (β-VAE).

    Args:
        esm_dim:      D, dimension of ESM2 embeddings (320 for ESM2-8M)
        latent_dim:   d, compressed feature dimension
        compression:  c, stride factor; l = L // c
        n_layers:     number of transformer layers
        heads:        attention heads
    """

    def __init__(
        self,
        esm_dim: int = 320,
        latent_dim: int = 64,
        compression: int = 20,
        n_layers: int = 2,
        heads: int = 8,
    ):
        super().__init__()
        self.compression = compression

        self.input_norm = nn.LayerNorm(esm_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlock(esm_dim, heads) for _ in range(n_layers)]
        )
        # Downsample sequence length: L -> L // compression
        self.conv_down = nn.Conv1d(esm_dim, esm_dim, kernel_size=compression, stride=compression)
        # Project feature dim
        self.proj = nn.Linear(esm_dim, latent_dim)
        self.norm_out = nn.LayerNorm(latent_dim)

        # VAE heads
        self.to_mean = nn.Linear(latent_dim, latent_dim)
        self.to_logvar = nn.Linear(latent_dim, latent_dim)

    def forward(self, h: torch.Tensor):
        """
        Args:
            h: ESM2 embeddings, shape (B, L, D)
        Returns:
            mean, logvar: each (B, l, d) where l = L // compression
        """
        x = self.input_norm(h)
        x = self.transformer(x)

        # Conv expects (B, D, L)
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv_down(x)
        x = rearrange(x, 'b d l -> b l d')

        x = self.proj(x)
        x = self.norm_out(x)

        mean = self.to_mean(x)
        logvar = self.to_logvar(x)
        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean


# ---------------------------------------------------------------------------
# Decompressor: latent z -> ESM embedding space h'
# ---------------------------------------------------------------------------

class Decompressor(nn.Module):
    """
    Symmetric inverse of Compressor.
    Maps z ∈ R^{l×d} back to h' ∈ R^{L×D}.

    Architecture:
      Linear(d->D) -> LayerNorm -> InvConv1d (upsample l->L) -> 2x TransformerBlock -> LayerNorm
    """

    def __init__(
        self,
        esm_dim: int = 320,
        latent_dim: int = 64,
        compression: int = 20,
        n_layers: int = 2,
        heads: int = 8,
    ):
        super().__init__()
        self.compression = compression

        self.proj_in = nn.Linear(latent_dim, esm_dim)
        self.norm_in = nn.LayerNorm(esm_dim)
        # Upsample: l -> L using transposed conv
        self.conv_up = nn.ConvTranspose1d(esm_dim, esm_dim, kernel_size=compression, stride=compression)
        self.transformer = nn.Sequential(
            *[TransformerBlock(esm_dim, heads) for _ in range(n_layers)]
        )
        self.norm_out = nn.LayerNorm(esm_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: latent, shape (B, l, d)
        Returns:
            h': reconstructed embeddings, shape (B, L, D)
        """
        x = self.proj_in(z)
        x = self.norm_in(x)

        x = rearrange(x, 'b l d -> b d l')
        x = self.conv_up(x)
        x = rearrange(x, 'b d l -> b l d')

        x = self.transformer(x)
        x = self.norm_out(x)
        return x


# ---------------------------------------------------------------------------
# ESM2 Decoder: h' -> sequence logits
# ---------------------------------------------------------------------------

class ESM2Decoder(nn.Module):
    """
    Lightweight decoder that maps ESM2 embedding space -> amino acid logits.
    Trained with cross-entropy loss; frozen after Stage 1.

    Uses a simple linear projection with tanh non-linearity (as in ProtFlow).

    Args:
        esm_dim:   input embedding dimension (320 for ESM2-8M)
        vocab_size: number of tokens (33 for ESM2 vocabulary)
    """

    def __init__(self, esm_dim: int = 320, vocab_size: int = 33):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(esm_dim),
            nn.Linear(esm_dim, esm_dim),
            nn.Tanh(),
            nn.Linear(esm_dim, vocab_size),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: embeddings (B, L, D)
        Returns:
            logits (B, L, vocab_size)
        """
        return self.net(h)


# ---------------------------------------------------------------------------
# Full VAE wrapper
# ---------------------------------------------------------------------------

class ProteinVAE(nn.Module):
    """
    Full VAE pipeline used in CHASE:
      ESM2 (frozen) -> Compressor -> z -> Decompressor -> Decoder

    Training objectives (Eq. 2 in paper):
      L_VAE = L_MSE(h, h') + L_CE(x, x') + β * KL(q(z|x) || p(z))
    """

    def __init__(
        self,
        esm_model,                  # pretrained ESM2, used as frozen encoder
        esm_dim: int = 320,
        latent_dim: int = 64,
        compression: int = 20,
        vocab_size: int = 33,
        beta: float = 1e-4,
        n_transformer_layers: int = 2,
        n_attn_heads: int = 8,
    ):
        super().__init__()
        self.beta = beta
        self.esm = esm_model

        # Freeze ESM2 encoder
        for p in self.esm.parameters():
            p.requires_grad = False

        self.compressor = Compressor(esm_dim, latent_dim, compression, n_transformer_layers, n_attn_heads)
        self.decompressor = Decompressor(esm_dim, latent_dim, compression, n_transformer_layers, n_attn_heads)
        self.decoder = ESM2Decoder(esm_dim, vocab_size)

    def encode(self, tokens: torch.Tensor):
        """tokens -> mean, logvar, h"""
        with torch.no_grad():
            results = self.esm(tokens, repr_layers=[self.esm.num_layers])
            h = results["representations"][self.esm.num_layers]
        mean, logvar = self.compressor(h)
        return mean, logvar, h

    def decode(self, z: torch.Tensor):
        """z -> h', logits"""
        h_prime = self.decompressor(z)
        logits = self.decoder(h_prime)
        return h_prime, logits

    def forward(self, tokens: torch.Tensor):
        mean, logvar, h = self.encode(tokens)
        z = self.compressor.reparameterize(mean, logvar)
        h_prime, logits = self.decode(z)
        return logits, h, h_prime, mean, logvar

    def compute_loss(
        self,
        tokens: torch.Tensor,
        logits: torch.Tensor,
        h: torch.Tensor,
        h_prime: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ):
        """
        L_VAE = L_MSE(h, h') + L_CE(x, x') + β * KL

        Args:
            padding_mask: True where tokens are padding (to be excluded from CE loss)
        """
        # MSE on embeddings
        l_mse = F.mse_loss(h_prime, h.detach())

        # Cross-entropy on sequence reconstruction
        B, L, V = logits.shape
        ce_loss = F.cross_entropy(
            logits.reshape(B * L, V),
            tokens.reshape(B * L),
            ignore_index=1,  # ESM2 padding token index
        )

        # KL divergence
        kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

        loss = l_mse + ce_loss + self.beta * kl
        return loss, {"mse": l_mse.item(), "ce": ce_loss.item(), "kl": kl.item()}
