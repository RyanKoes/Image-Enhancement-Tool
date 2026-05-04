"""UNet wrapper for ControlNet conditioning.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .controlnet import ControlNet


class UNetWrapper(nn.Module):
    """A UNet wrapper that injects ControlNet conditioning.

    This module wraps a diffusion UNet and a ControlNet model. The forward pass
    first computes conditioning signals from the ControlNet and then passes them
    to the UNet.
    """

    def __init__(self, unet: nn.Module, controlnet: ControlNet) -> None:
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        *,
        x_lq: torch.Tensor,
    ) -> torch.Tensor:
        """Denoise a latent tensor with ControlNet conditioning.

        Args:
            z_t: Noisy latent tensor at timestep t.
            t: Timestep tensor.
            x_lq: Low-quality image/latent tensor for conditioning.

        Returns:
            The predicted noise (eps) from the UNet.
        """
        control = self.controlnet(x=x_lq, t=t)
        noise_pred = self.unet(
            z_t,
            t,
            down_block_additional_residuals=control["down"],
            mid_block_additional_residual=control["mid"],
        ).sample
        return noise_pred
