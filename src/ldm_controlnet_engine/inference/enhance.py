"""Inference pipeline for ControlNet-based image enhancement.

Designed to be called from a notebook for visual validation:

    from src.ldm_controlnet_engine.inference.enhance import enhance

    x_out = enhance(
        x_lq=x_lq,          # (1, 3, H, W) float tensor in [-1, 1]
        vae=vae,
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
    )
    # x_out: (1, 3, H, W) float tensor in [-1, 1]
"""

from __future__ import annotations

import torch
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel

from ..models.controlnet import ControlNet

VAE_SCALE_FACTOR = 0.18215


@torch.no_grad()
def enhance(
    x_lq: torch.Tensor,
    *,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    controlnet: ControlNet,
    scheduler: DDPMScheduler | DDIMScheduler,
    num_inference_steps: int = 50,
    encoder_hidden_states: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Denoise a low-quality image into a high-quality output.

    Args:
        x_lq: Low-quality input tensor of shape ``(B, 3, H, W)`` in ``[-1, 1]``.
        vae: Frozen VAE for encoding/decoding latents.
        unet: Frozen UNet (``UNet2DConditionModel``).
        controlnet: Trained ControlNet encoder.
        scheduler: Diffusion scheduler. For fast inference, prefer
            ``DDIMScheduler`` (50 steps); ``DDPMScheduler`` works but is slow.
        num_inference_steps: Number of denoising steps.
        encoder_hidden_states: Optional cross-attention context for the UNet.
            Defaults to a zero tensor of shape ``(B, 1, cross_attention_dim)``.
        generator: Optional ``torch.Generator`` for reproducible sampling.

    Returns:
        Enhanced image tensor of shape ``(B, 3, H, W)`` in ``[-1, 1]``.
    """
    device = x_lq.device
    dtype = x_lq.dtype
    batch_size = x_lq.shape[0]

    scheduler.set_timesteps(num_inference_steps, device=device)

    # Encode the LQ image to latent space.
    # z_lq serves two purposes: (1) determines the spatial size for noise
    # initialisation, (2) is the clean conditioning signal for ControlNet.
    z_lq = vae.encode(x_lq).latent_dist.sample(generator=generator) * VAE_SCALE_FACTOR
    z = torch.randn_like(z_lq, generator=generator)

    # Unconditional context (text-free enhancement).
    if encoder_hidden_states is None:
        cross_attn_dim: int = unet.config.cross_attention_dim  # type: ignore[attr-defined]
        encoder_hidden_states = torch.zeros(
            batch_size, 1, cross_attn_dim, device=device, dtype=dtype
        )

    controlnet.eval()

    for t in scheduler.timesteps:
        # Broadcast scalar timestep to batch dimension.
        t_batch = t.expand(batch_size).to(device)

        # ControlNet residuals conditioned on the LQ latent.
        control = controlnet(z_lq, t_batch)

        # UNet noise prediction with injected ControlNet residuals.
        noise_pred = unet(
            z,
            t,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=control["down"],
            mid_block_additional_residual=control["mid"],
        ).sample

        # Scheduler step: z_{t-1} = f(noise_pred, t, z_t)
        z = scheduler.step(noise_pred, t, z, generator=generator).prev_sample

    # Decode latent back to pixel space.
    x_out = vae.decode(z / VAE_SCALE_FACTOR).sample
    return x_out


def tensor_to_pil(x: torch.Tensor):
    """Convert a ``(B, 3, H, W)`` tensor in ``[-1, 1]`` to a list of PIL images."""
    from PIL import Image
    import numpy as np

    x = x.detach().cpu().float()
    x = (x * 0.5 + 0.5).clamp(0.0, 1.0)          # [-1,1] → [0,1]
    x = (x * 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(frame) for frame in x]
