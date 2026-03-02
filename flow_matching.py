"""
Conditional Flow Matching Model for CHASE.

Implements:
- Linear interpolation probability path: z_t = (1-t)*z0 + t*z1
- Conditional Flow Matching (CFM) objective (Eq. 4)
- U-Net velocity field with fitness + time conditioning
- Classifier-Free Guidance (CFG) (Eq. 6): ε = (1+w)*v_cond - w*v_uncond
- ODE sampling via first-order Euler solver (Eq. 5)

References:
  Lipman et al. (2022) - Flow Matching for Generative Modeling
  Ho & Salimans (2022) - Classifier-Free Diffusion Guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


# ---------------------------------------------------------------------------
# Time and fitness conditioning utilities
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for continuous scalars (time / fitness)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: scalar tensor of shape (B,) or (B, 1)
        Returns:
            embeddings: (B, dim)
        """
        x = x.view(-1)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ConditioningMLP(nn.Module):
    """Projects time + fitness scalars into a shared conditioning vector."""

    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.time_emb = SinusoidalEmbedding(cond_dim)
        self.fitness_emb = SinusoidalEmbedding(cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: time values (B,)
            f: fitness values (B,); use zeros for unconditional
        Returns:
            cond: (B, hidden_dim)
        """
        t_emb = self.time_emb(t)
        f_emb = self.fitness_emb(f)
        return self.mlp(torch.cat([t_emb, f_emb], dim=-1))


# ---------------------------------------------------------------------------
# U-Net velocity field v_θ(z_t, t, f)
# ---------------------------------------------------------------------------

class ResidualBlock1D(nn.Module):
    """1D residual block with conditioning injection via AdaLN."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        # Conditioning scale and shift (AdaLN)
        self.cond_proj = nn.Linear(cond_dim, channels * 2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, C, L)
            cond: (B, cond_dim)
        """
        # AdaLN conditioning
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1)  # (B, C, 1)
        shift = shift.unsqueeze(-1)

        residual = x
        x = rearrange(x, 'b c l -> b l c')
        x = self.norm1(x)
        x = rearrange(x, 'b l c -> b c l')
        x = x * (1 + scale) + shift

        x = self.act(self.conv1(x))
        x = rearrange(x, 'b c l -> b l c')
        x = self.norm2(x)
        x = rearrange(x, 'b l c -> b c l')
        x = self.conv2(x)
        return x + residual


class AttentionBlock1D(nn.Module):
    """Self-attention over the sequence dimension."""

    def __init__(self, channels: int, heads: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.heads = heads
        self.head_dim = channels // heads
        self.to_qkv = nn.Linear(channels, channels * 3, bias=False)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L)"""
        x_in = x
        x = rearrange(x, 'b c l -> b l c')
        x = self.norm(x)
        B, L, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b l (h d) -> b h l d', h=self.heads), qkv)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.to_out(out)
        return rearrange(out, 'b l c -> b c l') + x_in


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, n_res: int = 2, use_attn: bool = True):
        super().__init__()
        self.res_blocks = nn.ModuleList([ResidualBlock1D(in_ch if i == 0 else out_ch, cond_dim) for i in range(n_res)])
        # Channel projection after first res block
        self.ch_proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.attn = AttentionBlock1D(out_ch) if use_attn else nn.Identity()
        self.downsample = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x, cond):
        for i, res in enumerate(self.res_blocks):
            if i == 0 and isinstance(self.ch_proj, nn.Conv1d):
                x = self.ch_proj(x)
            x = res(x, cond)
        x = self.attn(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cond_dim: int, n_res: int = 2, use_attn: bool = True):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_ch, in_ch, kernel_size=2, stride=2)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock1D(in_ch + skip_ch if i == 0 else out_ch, cond_dim) for i in range(n_res)]
        )
        self.ch_proj = nn.Conv1d(in_ch + skip_ch, out_ch, 1)
        self.attn = AttentionBlock1D(out_ch) if use_attn else nn.Identity()

    def forward(self, x, skip, cond):
        x = self.upsample(x)
        # Handle potential size mismatch from odd lengths
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1])
        x = torch.cat([x, skip], dim=1)
        for i, res in enumerate(self.res_blocks):
            if i == 0:
                x = self.ch_proj(x)
            x = res(x, cond)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.res1 = ResidualBlock1D(channels, cond_dim)
        self.attn = AttentionBlock1D(channels)
        self.res2 = ResidualBlock1D(channels, cond_dim)

    def forward(self, x, cond):
        x = self.res1(x, cond)
        x = self.attn(x)
        x = self.res2(x, cond)
        return x


class UNetVelocityField(nn.Module):
    """
    U-Net parameterization of the velocity field v_θ(z_t, t, f).

    Architecture follows the paper description:
    - 2 down-scaling blocks with attention
    - Middle block with attention
    - 2 up-scaling blocks with attention
    - Residual connections (skip connections)

    Args:
        latent_dim:      d, dimensionality of latent z
        base_channels:   U-Net base channel width
        cond_dim:        conditioning embedding dimension
        n_downup_blocks: number of down/up blocks (paper uses 2)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        base_channels: int = 128,
        cond_dim: int = 256,
        n_downup_blocks: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Conditioning
        self.conditioning = ConditioningMLP(cond_dim=64, hidden_dim=cond_dim)

        # Input projection: latent_dim -> base_channels
        self.input_proj = nn.Conv1d(latent_dim, base_channels, kernel_size=1)

        ch_mult = [1, 2]  # channel multipliers for each level
        channels = [base_channels * m for m in ch_mult]

        # Down blocks
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        for i, ch in enumerate(channels):
            self.down_blocks.append(DownBlock(in_ch, ch, cond_dim, use_attn=True))
            in_ch = ch

        # Middle
        self.middle = MiddleBlock(channels[-1], cond_dim)

        # Up blocks
        self.up_blocks = nn.ModuleList()
        for i, ch in enumerate(reversed(channels)):
            skip_ch = ch
            out_ch = base_channels if i == len(channels) - 1 else channels[-(i + 2)]
            self.up_blocks.append(UpBlock(in_ch, skip_ch, out_ch, cond_dim, use_attn=True))
            in_ch = out_ch

        # Output projection: base_channels -> latent_dim
        self.output_proj = nn.Sequential(
            nn.LayerNorm(base_channels),
            nn.Conv1d(base_channels, latent_dim, kernel_size=1),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        f: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_t: noisy latent (B, l, d)
            t:   time (B,) in [0, 1]
            f:   fitness condition (B,); zeros for unconditional
        Returns:
            velocity field (B, l, d)
        """
        cond = self.conditioning(t, f)  # (B, cond_dim)

        # (B, l, d) -> (B, d, l) for Conv1d
        x = rearrange(z_t, 'b l d -> b d l')
        x = self.input_proj(x)

        # Encoder path
        skips = []
        for down in self.down_blocks:
            x, skip = down(x, cond)
            skips.append(skip)

        # Middle
        x = self.middle(x, cond)

        # Decoder path
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip, cond)

        # Output
        x = rearrange(x, 'b c l -> b l c')
        x = self.output_proj[0](x)  # LayerNorm on last dim
        x = rearrange(x, 'b l c -> b c l')
        x = self.output_proj[1](x)

        return rearrange(x, 'b d l -> b l d')


# ---------------------------------------------------------------------------
# Conditional Flow Matching training and sampling
# ---------------------------------------------------------------------------

class ConditionalFlowMatcher(nn.Module):
    """
    Wraps the U-Net velocity field with:
    - CFM training objective (Eq. 4)
    - Classifier-Free Guidance sampling (Eq. 6)
    - Euler ODE solver (Eq. 5)
    """

    def __init__(self, velocity_field: UNetVelocityField, score_dropout: float = 0.0):
        super().__init__()
        self.v_theta = velocity_field
        self.score_dropout = score_dropout  # probability of dropping fitness condition

    def forward(
        self,
        z1: torch.Tensor,
        f: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CFM training loss.

        Eq. 3: z_t = (1-t)*z0 + t*z1
        Eq. 4: L_CFM = E[ ||v_θ(z_t, t, f) - (z1 - z0)||^2 ]

        Args:
            z1: data latents (B, l, d)
            f:  fitness values (B,)
        Returns:
            scalar loss
        """
        B = z1.shape[0]
        device = z1.device

        # Sample noise and time
        z0 = torch.randn_like(z1)
        t = torch.rand(B, device=device)

        # Linear interpolation
        t_expand = t.view(B, 1, 1)
        z_t = (1 - t_expand) * z0 + t_expand * z1

        # Target velocity: z1 - z0
        target = z1 - z0

        # Classifier-free guidance: randomly drop fitness condition
        if self.score_dropout > 0.0:
            drop_mask = torch.rand(B, device=device) < self.score_dropout
            f_in = f.clone()
            f_in[drop_mask] = 0.0  # unconditional = fitness 0
        else:
            f_in = f

        v_pred = self.v_theta(z_t, t, f_in)
        loss = F.mse_loss(v_pred, target)
        return loss

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        f: float,
        guidance_scale: float = 0.0,
        n_steps: int = 40,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Generate latents by integrating the learned velocity field.

        Eq. 5: z1 = z0 + ∫ v_θ(z_t, t, f) dt
        Eq. 6: ε = (1+w)*v_cond - w*v_uncond

        Args:
            shape:          (B, l, d) - shape of latent to generate
            f:              target fitness value
            guidance_scale: w in Eq. 6 (0 = no guidance)
            n_steps:        K Euler steps
            device:         torch device
        Returns:
            z1: generated latents (B, l, d)
        """
        if device is None:
            device = next(self.v_theta.parameters()).device

        B = shape[0]
        z = torch.randn(shape, device=device)
        dt = 1.0 / n_steps

        f_cond = torch.full((B,), f, device=device)
        f_uncond = torch.zeros(B, device=device)

        for k in range(n_steps):
            t = torch.full((B,), k * dt, device=device)

            v_cond = self.v_theta(z, t, f_cond)

            if guidance_scale != 0.0:
                v_uncond = self.v_theta(z, t, f_uncond)
                eps = (1 + guidance_scale) * v_cond - guidance_scale * v_uncond
            else:
                eps = v_cond

            z = z + dt * eps

        return z
