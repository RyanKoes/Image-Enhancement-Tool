"""Dataset utilities for ControlNet training.

Implements the minimal pipeline requested:
- Load HQ image from disk
- Create LQ image via blur + noise (on-the-fly)

The default tensor convention matches common diffusion training:
- tensors are float32
- values are normalized from [0, 1] to [-1, 1] via mean=0.5, std=0.5
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, TypeAlias

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .degradation import DegradationConfig, DegradationPipeline

ImageLike: TypeAlias = Image.Image

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image_file(path: Path) -> bool:
	return path.is_file() and path.suffix.lower() in _IMG_EXTS


def list_images(root: Path, *, recursive: bool = True) -> list[Path]:
	"""List image files under `root` (stable, sorted)."""

	root = Path(root)
	if not root.exists():
		raise FileNotFoundError(f"HQ image root does not exist: {root}")
	if not root.is_dir():
		raise NotADirectoryError(f"HQ image root is not a directory: {root}")

	paths = [p for p in (root.rglob("*") if recursive else root.iterdir()) if _is_image_file(p)]
	paths = sorted(paths)
	if not paths:
		raise FileNotFoundError(f"No images found under: {root}")
	return paths


def pil_open_rgb(path: Path) -> ImageLike:
	"""Open an image file as RGB PIL Image."""

	with Image.open(path) as img:
		return img.convert("RGB")


def _clamp01(t: torch.Tensor) -> torch.Tensor:
	return t.clamp_(0.0, 1.0)


def _resize_bicubic(img: ImageLike, *, size_hw: tuple[int, int]) -> ImageLike:
	"""Resize to (H, W) using bicubic resampling."""

	h, w = size_hw
	return img.resize((int(w), int(h)), resample=Image.Resampling.BICUBIC)


def _crop(img: ImageLike, *, top: int, left: int, height: int, width: int) -> ImageLike:
	return img.crop((int(left), int(top), int(left + width), int(top + height)))


def _to_tensor(img: ImageLike) -> torch.Tensor:
	"""Convert RGB PIL image to float32 tensor in [0,1], shape (3,H,W)."""

	if img.mode != "RGB":
		img = img.convert("RGB")
	arr = np.asarray(img).astype(np.float32) / 255.0
	# (H,W,3) -> (3,H,W)
	t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
	return t


def _normalize(t: torch.Tensor, *, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
	mean_t = torch.tensor(mean, dtype=t.dtype, device=t.device)[:, None, None]
	std_t = torch.tensor(std, dtype=t.dtype, device=t.device)[:, None, None]
	return (t - mean_t) / std_t


@dataclass
class PairTransform:
	"""Paired transforms for (LQ, HQ).

	- Ensures sizes match.
	- Optional center or random crop.
	- Converts to float32 tensors.
	- Optional normalize to [-1, 1] (diffusion-friendly).
	"""

	crop_size: int | tuple[int, int] | None = None
	random_crop: bool = True
	normalize: bool = True
	mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
	std: tuple[float, float, float] = (0.5, 0.5, 0.5)

	def __call__(self, lq: ImageLike, hq: ImageLike, *, rng: np.random.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
		if rng is None:
			rng = np.random.default_rng()

		if lq.size != hq.size:
			hq_w, hq_h = hq.size
			lq = _resize_bicubic(lq, size_hw=(hq_h, hq_w))

		lq, hq = self._maybe_crop(lq, hq, rng=rng)

		lq_t = _to_tensor(lq)
		hq_t = _to_tensor(hq)

		_clamp01(lq_t)
		_clamp01(hq_t)

		if self.normalize:
			lq_t = _normalize(lq_t, mean=self.mean, std=self.std)
			hq_t = _normalize(hq_t, mean=self.mean, std=self.std)
		return lq_t, hq_t

	def _maybe_crop(self, lq: ImageLike, hq: ImageLike, *, rng: np.random.Generator) -> tuple[ImageLike, ImageLike]:
		if self.crop_size is None:
			return lq, hq

		if isinstance(self.crop_size, int):
			th = tw = self.crop_size
		else:
			th, tw = self.crop_size

		w, h = hq.size
		if w < tw or h < th:
			# Upscale both to at least crop size.
			scale = max(tw / max(1, w), th / max(1, h))
			new_w = int(math.ceil(w * scale))
			new_h = int(math.ceil(h * scale))
			hq = _resize_bicubic(hq, size_hw=(new_h, new_w))
			lq = _resize_bicubic(lq, size_hw=(new_h, new_w))
			w, h = new_w, new_h

		if self.random_crop:
			i = int(rng.integers(0, h - th + 1))
			j = int(rng.integers(0, w - tw + 1))
		else:
			i = (h - th) // 2
			j = (w - tw) // 2

		lq = _crop(lq, top=i, left=j, height=th, width=tw)
		hq = _crop(hq, top=i, left=j, height=th, width=tw)
		return lq, hq


class HQToLQDataset(Dataset[dict[str, object]]):
	"""HQ-only dataset that generates LQ on-the-fly using blur+noise."""

	def __init__(
		self,
		*,
		hq_root: str | Path,
		recursive: bool = True,
		degradation: DegradationPipeline | None = None,
		degradation_config: DegradationConfig | None = None,
		transform: PairTransform | Callable[[ImageLike, ImageLike], tuple[torch.Tensor, torch.Tensor]] | None = None,
		seed: int | None = None,
		return_pil: bool = False,
	) -> None:
		self.hq_root = Path(hq_root)
		self.paths = list_images(self.hq_root, recursive=recursive)

		if degradation is None:
			degradation = DegradationPipeline(config=degradation_config)
		self.degradation = degradation

		self.transform = transform or PairTransform()
		self.seed = seed
		self.return_pil = bool(return_pil)

	def __len__(self) -> int:
		return len(self.paths)

	def _rng_for_index(self, index: int) -> np.random.Generator:
		if self.seed is None:
			return np.random.default_rng()
		# Stable per-index RNG for reproducibility across epochs.
		seed = (int(self.seed) + int(index) * 10007) % 2**32
		return np.random.default_rng(seed)

	def __getitem__(self, index: int) -> dict[str, object]:
		path = self.paths[index]
		rng = self._rng_for_index(index)

		try:
			hq = pil_open_rgb(path)
		except Exception as e:  # noqa: BLE001
			raise RuntimeError(f"Failed to load HQ image: {path}") from e

		lq = self.degradation(hq, rng=rng)

		if self.return_pil:
			return {"lq": lq, "hq": hq, "path": str(path)}

		# Transform may optionally accept rng (PairTransform does).
		try:
			lq_t, hq_t = self.transform(lq, hq, rng=rng)  # type: ignore[misc]
		except TypeError:
			lq_t, hq_t = self.transform(lq, hq)  # type: ignore[misc]

		return {
			"lq": lq_t,
			"hq": hq_t,
			"path": str(path),
		}


def default_div2k_hq_root(workspace_root: str | Path) -> Path:
	"""Convenience helper for the repo's current DIV2K layout."""

	return Path(workspace_root) / "data" / "kaggle" / "div2k_hr"
