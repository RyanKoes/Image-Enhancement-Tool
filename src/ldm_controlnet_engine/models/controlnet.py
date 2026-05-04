"""ControlNet-style conditioning network.

This module implements a lightweight ControlNet encoder that mirrors the *down*
path of a diffusion UNet:
  - Takes a low-quality input image/latent `x_lq` and a timestep `t`.
  - Produces residual feature maps aligned to UNet feature resolutions.

Forward output:

	{
		"down": [r1, r2, r3, ...],  # residuals for UNet down/skip feature maps
		"mid": r_mid,               # residual for UNet middle feature map
	}

Key properties (per project requirements):
  - Matches UNet feature map resolutions by mirroring the downsampling schedule.
  - Uses zero-initialized conv layers for all residual outputs.
  - Mirrors UNet downsampling structure (diffusion UNet style), but lightweight.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int, *, max_period: int = 10_000) -> torch.Tensor:
	"""Create sinusoidal timestep embeddings.

	Args:
		timesteps: Tensor of shape (B,) containing integer or float timesteps.
		dim: Output embedding dimension.
		max_period: Controls minimum frequency of the embeddings.

	Returns:
		Tensor of shape (B, dim).
	"""

	if timesteps.ndim != 1:
		timesteps = timesteps.view(-1)

	half = dim // 2
	freqs = torch.exp(
		-math.log(float(max_period))
		* torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
		/ float(half)
	)
	args = timesteps.float()[:, None] * freqs[None]
	emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
	if dim % 2 == 1:
		emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
	return emb


def _group_norm(num_channels: int, *, max_groups: int = 32, eps: float = 1e-6) -> nn.GroupNorm:
	# Pick a group count that divides channels.
	groups = min(int(max_groups), int(num_channels))
	while groups > 1 and (num_channels % groups != 0):
		groups -= 1
	return nn.GroupNorm(num_groups=groups, num_channels=int(num_channels), eps=float(eps), affine=True)


class ZeroConv2d(nn.Module):
	"""1x1 conv initialized to output zeros.

	Used for ControlNet residual outputs so the conditioning starts as a no-op.
	"""

	def __init__(self, channels: int) -> None:
		super().__init__()
		self.conv = nn.Conv2d(int(channels), int(channels), kernel_size=1, stride=1, padding=0)
		nn.init.zeros_(self.conv.weight)
		nn.init.zeros_(self.conv.bias)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.conv(x)


class ResBlock(nn.Module):
	"""A diffusion-style residual block with timestep conditioning."""

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		*,
		time_emb_dim: int,
		dropout: float = 0.0,
	) -> None:
		super().__init__()
		self.in_channels = int(in_channels)
		self.out_channels = int(out_channels)

		self.in_norm = _group_norm(self.in_channels)
		self.in_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)

		self.time_proj = nn.Sequential(
			nn.SiLU(),
			nn.Linear(int(time_emb_dim), self.out_channels),
		)

		self.out_norm = _group_norm(self.out_channels)
		self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
		self.out_conv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)

		self.skip: nn.Module
		if self.in_channels == self.out_channels:
			self.skip = nn.Identity()
		else:
			self.skip = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

	def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
		h = self.in_conv(F.silu(self.in_norm(x)))
		# Add timestep embedding as a bias.
		emb = self.time_proj(time_emb).type_as(h)
		h = h + emb[:, :, None, None]
		h = self.out_conv(self.dropout(F.silu(self.out_norm(h))))
		return h + self.skip(x)


class Downsample(nn.Module):
	"""Downsample by 2x using a strided 3x3 conv (UNet-style)."""

	def __init__(self, channels: int) -> None:
		super().__init__()
		self.op = nn.Conv2d(int(channels), int(channels), kernel_size=3, stride=2, padding=1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.op(x)


class ControlNet(nn.Module):
	"""Lightweight ControlNet encoder that mirrors a diffusion UNet down path."""

	def __init__(
		self,
		*,
		in_channels: int = 3,
		model_channels: int = 64,
		channel_mult: tuple[int, ...] = (1, 2, 4, 8),
		num_res_blocks: int = 2,
		dropout: float = 0.0,
		time_embed_dim: int | None = None,
	) -> None:
		super().__init__()
		self.in_channels = int(in_channels)
		self.model_channels = int(model_channels)
		self.channel_mult = tuple(int(m) for m in channel_mult)
		self.num_res_blocks = int(num_res_blocks)

		if time_embed_dim is None:
			time_embed_dim = 4 * self.model_channels
		self.time_embed_dim = int(time_embed_dim)

		# Timestep embedding MLP.
		self.time_mlp = nn.Sequential(
			nn.Linear(self.model_channels, self.time_embed_dim),
			nn.SiLU(),
			nn.Linear(self.time_embed_dim, self.time_embed_dim),
		)

		# Mirror a UNet-style input/down path.
		self.input_blocks = nn.ModuleList()
		self.zero_convs = nn.ModuleList()

		ch = self.model_channels
		self.input_blocks.append(nn.Conv2d(self.in_channels, ch, kernel_size=3, padding=1))
		self.zero_convs.append(ZeroConv2d(ch))

		for level, mult in enumerate(self.channel_mult):
			out_ch = self.model_channels * int(mult)
			for _ in range(self.num_res_blocks):
				self.input_blocks.append(ResBlock(ch, out_ch, time_emb_dim=self.time_embed_dim, dropout=dropout))
				ch = out_ch
				self.zero_convs.append(ZeroConv2d(ch))

			if level != len(self.channel_mult) - 1:
				self.input_blocks.append(Downsample(ch))
				self.zero_convs.append(ZeroConv2d(ch))

		# Middle block (matches common diffusion UNet: ResBlock -> ResBlock).
		self.mid_block1 = ResBlock(ch, ch, time_emb_dim=self.time_embed_dim, dropout=dropout)
		self.mid_block2 = ResBlock(ch, ch, time_emb_dim=self.time_embed_dim, dropout=dropout)
		self.mid_zero = ZeroConv2d(ch)

	def forward(self, x_lq: torch.Tensor, t: torch.Tensor) -> dict[str, Any]:
		"""Compute ControlNet residuals.

		Args:
			x_lq: Input tensor of shape (B, C, H, W).
			t: Timestep tensor of shape (B,) (int or float).

		Returns:
			dict with keys:
			  - "down": list[Tensor] residuals aligned with UNet down/skip features
			  - "mid": Tensor residual aligned with UNet mid feature
		"""

		if t.ndim != 1:
			t = t.view(-1)

		# Build timestep embedding.
		temb = timestep_embedding(t, self.model_channels)
		temb = self.time_mlp(temb)

		h = x_lq
		downs: list[torch.Tensor] = []

		for block, zconv in zip(self.input_blocks, self.zero_convs):
			if isinstance(block, ResBlock):
				h = block(h, temb)
			else:
				h = block(h)
			downs.append(zconv(h))

		h = self.mid_block1(h, temb)
		h = self.mid_block2(h, temb)
		mid = self.mid_zero(h)

		return {"down": downs, "mid": mid}

