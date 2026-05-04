"""Training engine for ControlNet-based image enhancement.

Designed to be imported and driven from a Jupyter notebook:

    from src.ldm_controlnet_engine.training.train_controlnet import (
        TrainConfig, build_dataloader, build_models, train,
    )

    cfg = TrainConfig(hq_root="data/kaggle/div2k_hr")
    vae, unet, scheduler, controlnet = build_models(cfg)
    dataloader = build_dataloader(cfg)
    train(cfg, vae=vae, unet=unet, scheduler=scheduler,
          controlnet=controlnet, dataloader=dataloader)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

from ..data.dataset import HQToLQDataset, HFStreamingDataset, PairTransform
from ..data.degradation import DegradationConfig, DegradationPipeline
from ..models.controlnet import ControlNet
from .forward_pass import forward_pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Data
    hq_root: str = "data/kaggle/div2k_hr"
    output_dir: str = "output/controlnet"

    # Pretrained base model
    base_model: str = "runwayml/stable-diffusion-v1-5"

    # ControlNet architecture
    # These must mirror the SD UNet's latent-space channel structure so that
    # ControlNet residuals are addition-compatible with UNet feature maps:
    #   SD 1.5 block_out_channels = (320, 640, 1280, 1280)
    #   VAE latent channels = 4
    controlnet_in_channels: int = 4   # VAE latent channels (not raw RGB)
    model_channels: int = 320         # matches SD UNet base channels
    channel_mult: tuple[int, ...] = (1, 2, 4, 4)  # → 320, 640, 1280, 1280
    num_res_blocks: int = 2
    dropout: float = 0.0

    # Training hyper-parameters
    crop_size: int = 256
    batch_size: int = 4
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    num_workers: int = 4
    seed: int | None = 42

    # Logging / checkpointing
    log_every: int = 50
    save_every: int = 1000


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_models(
    cfg: TrainConfig,
) -> tuple[AutoencoderKL, UNet2DConditionModel, DDPMScheduler, ControlNet]:
    """Load frozen VAE + UNet + scheduler and create a trainable ControlNet.

    Returns:
        (vae, unet, scheduler, controlnet)
    """
    logger.info(f"Loading pretrained models from '{cfg.base_model}' ...")

    vae: AutoencoderKL = AutoencoderKL.from_pretrained(cfg.base_model, subfolder="vae")
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(cfg.base_model, subfolder="unet")
    scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(cfg.base_model, subfolder="scheduler")

    vae.requires_grad_(False).eval()
    unet.requires_grad_(False).eval()

    controlnet = ControlNet(
        in_channels=cfg.controlnet_in_channels,
        model_channels=cfg.model_channels,
        channel_mult=tuple(cfg.channel_mult),
        num_res_blocks=cfg.num_res_blocks,
        dropout=cfg.dropout,
    )

    n_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
    logger.info(f"ControlNet trainable parameters: {n_params:,}")

    return vae, unet, scheduler, controlnet


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    cfg: TrainConfig,
    degradation: DegradationConfig | None = None,
) -> DataLoader:
    """Build the HQ→LQ paired dataloader from ``cfg``.

    Args:
        cfg: Training configuration.
        degradation: Optional custom degradation settings. When ``None`` the
            ``DegradationConfig`` defaults are used (mild blur + noise).
            Pass a custom instance to make the LQ images harder (e.g. more
            noise or stronger blur) without changing the rest of the config.
    """
    transform = PairTransform(
        crop_size=cfg.crop_size,
        random_crop=True,
        normalize=True,
    )
    dataset = HQToLQDataset(
        hq_root=cfg.hq_root,
        degradation=DegradationPipeline(config=degradation),
        transform=transform,
        seed=cfg.seed,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )


def build_hf_dataloader(
    cfg: TrainConfig,
    hf_dataset,
    *,
    hq_key: str = "hr",
    degradation: DegradationConfig | None = None,
    shuffle_buffer: int = 200,
) -> DataLoader:
    """Build a DataLoader that streams HQ images from a HuggingFace dataset.

    Args:
        cfg: Training configuration (crop_size, batch_size, etc.).
        hf_dataset: A HuggingFace ``IterableDataset`` loaded with
            ``streaming=True``, e.g.
            ``load_dataset("eugenesiow/Div2k", "bicubic_x2", split="train",
                           streaming=True)``.
        hq_key: Key in each HF example that contains the HQ PIL image.
            For ``eugenesiow/Div2k`` this is ``"hr"``.
        degradation: Optional degradation settings. Defaults to
            ``DegradationConfig()`` (mild blur + noise).
        shuffle_buffer: Size of the in-memory shuffle buffer. Set to 0 to
            disable shuffling (e.g. for validation).
    """
    transform = PairTransform(
        crop_size=cfg.crop_size,
        random_crop=True,
        normalize=True,
    )
    dataset = HFStreamingDataset(
        hf_dataset,
        hq_key=hq_key,
        degradation_config=degradation,
        transform=transform,
        shuffle_buffer=shuffle_buffer,
        seed=cfg.seed,
    )
    # IterableDataset does not support shuffle= or persistent_workers=.
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=0,          # HF streaming is not fork-safe
        pin_memory=True,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    cfg: TrainConfig,
    *,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    scheduler: DDPMScheduler,
    controlnet: ControlNet,
    dataloader: DataLoader,
    device: str | torch.device | None = None,
    on_step_end: Callable[[int, float], None] | None = None,
    resume_from: str | Path | None = None,
) -> ControlNet:
    """Run the full training loop.

    Args:
        cfg: Training configuration.
        vae: Frozen VAE (``AutoencoderKL``).
        unet: Frozen UNet (``UNet2DConditionModel``).
        scheduler: Noise scheduler (``DDPMScheduler``).
        controlnet: Trainable ControlNet to optimise.
        dataloader: Paired HQ/LQ dataloader.
        device: Target device. Defaults to CUDA if available, else CPU.
        on_step_end: Optional callback called after every optimizer step with
            ``(global_step, loss)`` — useful for notebook progress bars or
            custom metric tracking.
        resume_from: Optional path to a ``controlnet.pt`` state-dict to load
            into ``controlnet`` before training starts. Use this to fine-tune
            a previously trained ControlNet on a new dataset.

    Returns:
        The trained ``ControlNet`` (moved back to CPU, ``requires_grad`` intact).
    """
    if resume_from is not None:
        ckpt_path = Path(resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume_from checkpoint not found: {ckpt_path}")
        logger.info(f"Resuming ControlNet weights from: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        controlnet.load_state_dict(state)

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    vae = vae.to(device)
    unet = unet.to(device)
    controlnet = controlnet.to(device)

    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    cross_attn_dim: int = unet.config.cross_attention_dim  # type: ignore[attr-defined]

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_steps = math.ceil(
        len(dataloader) / cfg.gradient_accumulation_steps
    ) * cfg.num_epochs

    logger.info("***** Starting training *****")
    logger.info(f"  Device         = {device}")
    logger.info(f"  Dataset size   = {len(dataloader.dataset)}")
    logger.info(f"  Num epochs     = {cfg.num_epochs}")
    logger.info(f"  Batch size     = {cfg.batch_size}")
    logger.info(f"  Grad accum     = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total steps    = {total_steps}")

    global_step = 0
    accum_loss = 0.0
    accum_count = 0

    for epoch in range(cfg.num_epochs):
        controlnet.train()
        running_loss = 0.0

        for step, batch in enumerate(dataloader):
            x_hq: torch.Tensor = batch["hq"].to(device)
            x_lq: torch.Tensor = batch["lq"].to(device)

            t = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (x_hq.shape[0],),
                device=device,
            )

            encoder_hidden_states = torch.zeros(
                x_hq.shape[0], 1, cross_attn_dim,
                device=device, dtype=x_hq.dtype,
            )

            loss = forward_pass(
                vae=vae,
                controlnet=controlnet,
                unet=unet,
                scheduler=scheduler,
                x_hq=x_hq,
                x_lq=x_lq,
                t=t,
                encoder_hidden_states=encoder_hidden_states,
            )

            # Scale loss for gradient accumulation.
            (loss / cfg.gradient_accumulation_steps).backward()

            accum_loss += loss.detach().item()
            accum_count += 1
            running_loss += loss.detach().item()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(controlnet.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg.log_every == 0:
                    avg = accum_loss / max(1, accum_count)
                    logger.info(
                        f"epoch={epoch:04d}  step={global_step:07d}  "
                        f"loss={loss.item():.6f}  avg={avg:.6f}"
                    )
                    accum_loss = 0.0
                    accum_count = 0

                if on_step_end is not None:
                    on_step_end(global_step, loss.item())

                if global_step % cfg.save_every == 0:
                    _save_checkpoint(controlnet, output_dir, global_step)

        avg_epoch_loss = running_loss / max(1, len(dataloader))
        logger.info(f"Epoch {epoch:04d} complete — avg loss: {avg_epoch_loss:.6f}")

    # Final save.
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(controlnet.state_dict(), final_dir / "controlnet.pt")
    logger.info(f"Training complete. Final model saved to {final_dir}")

    return controlnet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(controlnet: ControlNet, output_dir: Path, step: int) -> None:
    ckpt_dir = output_dir / f"checkpoint-{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(controlnet.state_dict(), ckpt_dir / "controlnet.pt")
    logger.info(f"Saved checkpoint → {ckpt_dir}")
