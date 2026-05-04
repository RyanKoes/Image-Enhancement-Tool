"""Full forward pass combining VAE, ControlNet, and UNet for training.

Steps
-----
1. Encode the high-quality image to a latent with VAE.
2. Sample a random noise tensor and corrupt the latent at timestep ``t``.
3. Run the low-quality image through ControlNet to get residual feature maps.
4. Pass the noisy latent, timestep, and ControlNet residuals to the wrapped
   UNet to predict the noise.
5. Compute the MSE loss between predicted and actual noise.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.controlnet import ControlNet

# VAE latent scaling constant from the LDM paper.
VAE_SCALE_FACTOR = 0.18215


def forward_pass(
    *,
    vae: torch.nn.Module,
    controlnet: ControlNet,
    unet: torch.nn.Module,
    scheduler: object,
    x_hq: torch.Tensor,
    x_lq: torch.Tensor,
    t: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the training loss for one batch.

    Args:
        vae: Frozen VAE encoder/decoder (e.g. ``AutoencoderKL``).
        controlnet: ControlNet encoder module.
        unet: Raw diffusers UNet (e.g. ``UNet2DConditionModel``). Not wrapped.
        scheduler: Diffusion noise scheduler with an ``add_noise`` method.
        x_hq: High-quality images, shape ``(B, C, H, W)``.
        x_lq: Low-quality images, shape ``(B, C, H, W)``.
        t: Sampled timesteps, shape ``(B,)``.
        encoder_hidden_states: Optional text/context conditioning passed to
            ``UNet2DConditionModel``. Pass a zero tensor of shape
            ``(B, 1, cross_attention_dim)`` for unconditional training.

    Returns:
        Scalar MSE loss between predicted noise and true noise.
    """
    # Step 1: Encode both HQ and LQ images to latent space (frozen VAE).
    # The ControlNet must operate in latent space to match the SD UNet's
    # feature map resolutions and channel dimensions.
    with torch.no_grad():
        z_hq = vae.encode(x_hq).latent_dist.sample() * VAE_SCALE_FACTOR
        z_lq = vae.encode(x_lq).latent_dist.sample() * VAE_SCALE_FACTOR

    # Step 2: Sample random noise and corrupt the latent at timestep t.
    noise = torch.randn_like(z_hq)
    z_t = scheduler.add_noise(z_hq, noise, t)

    # Step 3: Compute ControlNet residuals from the LQ latent.
    # z_lq is the clean conditioning signal; z_t is the noisy target.
    control = controlnet(z_lq, t)

    # Step 4: Predict noise with UNet conditioned on ControlNet residuals.
    unet_kwargs: dict = dict(
        down_block_additional_residuals=control["down"],
        mid_block_additional_residual=control["mid"],
    )
    if encoder_hidden_states is not None:
        unet_kwargs["encoder_hidden_states"] = encoder_hidden_states
    noise_pred = unet(z_t, t, **unet_kwargs).sample

    # Step 5: Compute reconstruction loss in noise space.
    loss = F.mse_loss(noise_pred, noise)
    return loss
