"""Image degradation utilities for ControlNet training.

This module focuses on the two degradations requested for the dataset pipeline:
- Blur (Gaussian)
- Noise (additive Gaussian)

All functions operate on RGB PIL images and return RGB PIL images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from PIL import Image, ImageFilter

ImageLike: TypeAlias = Image.Image


def ensure_rgb(img: ImageLike) -> ImageLike:
	"""Ensure `img` is an RGB PIL image."""

	if img.mode != "RGB":
		return img.convert("RGB")
	return img


def gaussian_blur(img: ImageLike, *, radius: float) -> ImageLike:
	"""Apply a Gaussian blur with a given radius."""

	img = ensure_rgb(img)
	# Pillow treats radius<=0 as no-op; keep it explicit.
	if radius <= 0:
		return img
	return img.filter(ImageFilter.GaussianBlur(radius=float(radius)))


def add_gaussian_noise(
	img: ImageLike,
	*,
	sigma: float,
	rng: np.random.Generator | None = None,
) -> ImageLike:
	"""Add zero-mean Gaussian noise in pixel space.

	Args:
		sigma: Standard deviation in pixel units (0-255).
		rng: Optional numpy RNG for determinism.
	"""

	img = ensure_rgb(img)
	if sigma <= 0:
		return img

	if rng is None:
		rng = np.random.default_rng()

	arr = np.asarray(img).astype(np.float32)
	noise = rng.normal(loc=0.0, scale=float(sigma), size=arr.shape).astype(np.float32)
	arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
	return Image.fromarray(arr, mode="RGB")


@dataclass
class DegradationConfig:
	"""Parameter ranges for LQ generation."""

	blur_radius: tuple[float, float] = (0.2, 1.5)
	noise_sigma: tuple[float, float] = (0.0, 10.0)

	def sample_blur_radius(self, rng: np.random.Generator) -> float:
		lo, hi = self.blur_radius
		return float(rng.uniform(lo, hi))

	def sample_noise_sigma(self, rng: np.random.Generator) -> float:
		lo, hi = self.noise_sigma
		return float(rng.uniform(lo, hi))


class DegradationPipeline:
	"""Create a low-quality (LQ) image from a high-quality (HQ) image."""

	def __init__(self, config: DegradationConfig | None = None) -> None:
		self.config = config or DegradationConfig()

	def __call__(self, hq: ImageLike, *, rng: np.random.Generator | None = None) -> ImageLike:
		hq = ensure_rgb(hq)
		if rng is None:
			rng = np.random.default_rng()

		radius = self.config.sample_blur_radius(rng)
		sigma = self.config.sample_noise_sigma(rng)

		lq = gaussian_blur(hq, radius=radius)
		lq = add_gaussian_noise(lq, sigma=sigma, rng=rng)
		return lq
