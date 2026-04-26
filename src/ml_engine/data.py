"""Data pipeline for CNN-based image enhancement.

This module is designed to be dataset-structure agnostic:
- Paired datasets: separate input (LQ) and target (HQ) folders.
- HQ-only datasets: only clean images; inputs are generated on-the-fly using a
  degradation pipeline (useful for denoising/deblurring/low-light style tasks).

Outputs follow a common convention for enhancement/super-resolution training:
- `lq`: model input tensor, float32
- `hq`: supervision tensor, float32
- `path`: source HQ path (string)
"""

from __future__ import annotations

import io
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, TypeAlias

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

ImageLike: TypeAlias = Image.Image


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image_file(path: Path) -> bool:
	return path.is_file() and path.suffix.lower() in _IMG_EXTS


def list_images(root: Path, recursive: bool = True) -> list[Path]:
	"""List image files under `root` (stable, sorted)."""

	root = Path(root)
	if not root.exists():
		raise FileNotFoundError(f"Image root does not exist: {root}")
	if not root.is_dir():
		raise NotADirectoryError(f"Image root is not a directory: {root}")

	if recursive:
		paths = [p for p in root.rglob("*") if _is_image_file(p)]
	else:
		paths = [p for p in root.iterdir() if _is_image_file(p)]
	return sorted(paths)


def _pil_open_rgb(path: Path) -> ImageLike:
	# Pillow image objects are lazy; convert to RGB to standardize channels.
	with Image.open(path) as img:
		return img.convert("RGB")


def _seed_worker(worker_id: int) -> None:
	"""Deterministic seeding for DataLoader workers."""

	worker_seed = torch.initial_seed() % 2**32
	random.seed(worker_seed)
	np.random.seed(worker_seed)


def _clamp01(t: torch.Tensor) -> torch.Tensor:
	return t.clamp_(0.0, 1.0)


@dataclass(frozen=True)
class Sample:
	"""A single training sample.

	For paired datasets, both paths are present.
	For HQ-only datasets, `lq_path` is None and LQ is generated on-the-fly.
	"""

	hq_path: Path
	lq_path: Path | None = None


class DegradationPipeline:
	"""Create a degraded input image from a clean image.

	The defaults are a practical "general enhancement" mix: slight blur,
	resampling artifacts, noise, and JPEG compression. You should adjust these
	to match your Kaggle dataset's degradation distribution.
	"""

	def __init__(
		self,
		*,
		p_blur: float = 0.25,
		blur_radius: tuple[float, float] = (0.2, 1.2),
		p_downsample: float = 0.35,
		downsample_scale: tuple[float, float] = (0.5, 1.0),
		p_noise: float = 0.35,
		noise_sigma: tuple[float, float] = (1.0, 10.0),
		p_jpeg: float = 0.35,
		jpeg_quality: tuple[int, int] = (35, 95),
		p_color: float = 0.20,
		brightness: tuple[float, float] = (0.9, 1.1),
		contrast: tuple[float, float] = (0.9, 1.1),
		saturation: tuple[float, float] = (0.9, 1.1),
		gamma: tuple[float, float] = (0.9, 1.1),
	) -> None:
		self.p_blur = float(p_blur)
		self.blur_radius = blur_radius
		self.p_downsample = float(p_downsample)
		self.downsample_scale = downsample_scale
		self.p_noise = float(p_noise)
		self.noise_sigma = noise_sigma
		self.p_jpeg = float(p_jpeg)
		self.jpeg_quality = jpeg_quality
		self.p_color = float(p_color)
		self.brightness = brightness
		self.contrast = contrast
		self.saturation = saturation
		self.gamma = gamma

	def __call__(self, img: ImageLike) -> ImageLike:
		if img.mode != "RGB":
			img = img.convert("RGB")

		img = self._maybe_color(img)
		img = self._maybe_blur(img)
		img = self._maybe_downsample(img)
		img = self._maybe_noise(img)
		img = self._maybe_jpeg(img)
		return img

	def _maybe_blur(self, img: ImageLike) -> ImageLike:
		if random.random() >= self.p_blur:
			return img
		radius = random.uniform(*self.blur_radius)
		return img.filter(ImageFilter.GaussianBlur(radius=radius))

	def _maybe_downsample(self, img: ImageLike) -> ImageLike:
		if random.random() >= self.p_downsample:
			return img

		w, h = img.size
		scale = random.uniform(*self.downsample_scale)
		# Keep at least 8 px to avoid edge cases.
		new_w = max(8, int(round(w * scale)))
		new_h = max(8, int(round(h * scale)))
		if new_w == w and new_h == h:
			return img

		img_small = TF.resize(img, [new_h, new_w], interpolation=InterpolationMode.BICUBIC)
		return TF.resize(img_small, [h, w], interpolation=InterpolationMode.BICUBIC)

	def _maybe_noise(self, img: ImageLike) -> ImageLike:
		if random.random() >= self.p_noise:
			return img

		sigma = random.uniform(*self.noise_sigma)
		arr = np.asarray(img).astype(np.float32)
		noise = np.random.normal(loc=0.0, scale=sigma, size=arr.shape).astype(np.float32)
		arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
		return Image.fromarray(arr, mode="RGB")

	def _maybe_jpeg(self, img: ImageLike) -> ImageLike:
		if random.random() >= self.p_jpeg:
			return img
		quality = int(random.randint(*self.jpeg_quality))
		buf = io.BytesIO()
		img.save(buf, format="JPEG", quality=quality, optimize=True)
		buf.seek(0)
		with Image.open(buf) as reloaded:
			return reloaded.convert("RGB")

	def _maybe_color(self, img: ImageLike) -> ImageLike:
		if random.random() >= self.p_color:
			return img
		b = random.uniform(*self.brightness)
		c = random.uniform(*self.contrast)
		s = random.uniform(*self.saturation)
		g = random.uniform(*self.gamma)

		img = TF.adjust_brightness(img, b)
		img = TF.adjust_contrast(img, c)
		img = TF.adjust_saturation(img, s)
		img = TF.adjust_gamma(img, g)
		return img


class TransformPipeline:
	"""Paired transforms for (LQ, HQ) images.

	Crops/augments are applied identically to both images.
	"""

	def __init__(
		self,
		*,
		crop_size: int | tuple[int, int] | None = 256,
		random_crop: bool = True,
		hflip: bool = True,
		vflip: bool = False,
		rotate90: bool = False,
		normalize: bool = True,
		mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
		std: tuple[float, float, float] = (0.5, 0.5, 0.5),
		match_size: bool = True,
	) -> None:
		self.crop_size = crop_size
		self.random_crop = bool(random_crop)
		self.hflip = bool(hflip)
		self.vflip = bool(vflip)
		self.rotate90 = bool(rotate90)
		self.normalize = bool(normalize)
		self.mean = mean
		self.std = std
		self.match_size = bool(match_size)

	def __call__(self, lq: ImageLike, hq: ImageLike) -> tuple[torch.Tensor, torch.Tensor]:
		if self.match_size and lq.size != hq.size:
			# Keep HQ as "truth" size; resize LQ to match.
			hq_w, hq_h = hq.size
			lq = TF.resize(lq, [hq_h, hq_w], interpolation=InterpolationMode.BICUBIC)

		lq, hq = self._maybe_crop(lq, hq)
		lq, hq = self._maybe_flip_rotate(lq, hq)

		lq_t = TF.to_tensor(lq)  # float32 in [0, 1]
		hq_t = TF.to_tensor(hq)

		_clamp01(lq_t)
		_clamp01(hq_t)

		if self.normalize:
			lq_t = TF.normalize(lq_t, mean=self.mean, std=self.std)
			hq_t = TF.normalize(hq_t, mean=self.mean, std=self.std)

		return lq_t, hq_t

	def _maybe_crop(self, lq: ImageLike, hq: ImageLike) -> tuple[ImageLike, ImageLike]:
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
			hq = TF.resize(hq, [new_h, new_w], interpolation=InterpolationMode.BICUBIC)
			lq = TF.resize(lq, [new_h, new_w], interpolation=InterpolationMode.BICUBIC)
			w, h = new_w, new_h

		if self.random_crop:
			i = random.randint(0, h - th)
			j = random.randint(0, w - tw)
		else:
			i = (h - th) // 2
			j = (w - tw) // 2

		hq = TF.crop(hq, top=i, left=j, height=th, width=tw)
		lq = TF.crop(lq, top=i, left=j, height=th, width=tw)
		return lq, hq

	def _maybe_flip_rotate(self, lq: ImageLike, hq: ImageLike) -> tuple[ImageLike, ImageLike]:
		if self.hflip and random.random() < 0.5:
			lq = TF.hflip(lq)
			hq = TF.hflip(hq)

		if self.vflip and random.random() < 0.5:
			lq = TF.vflip(lq)
			hq = TF.vflip(hq)

		if self.rotate90:
			k = random.randint(0, 3)
			if k:
				angle = 90 * k
				lq = TF.rotate(lq, angle=angle, interpolation=InterpolationMode.BILINEAR)
				hq = TF.rotate(hq, angle=angle, interpolation=InterpolationMode.BILINEAR)

		return lq, hq


class ImageDataset(Dataset[dict[str, object]]):
	"""Core dataset for image enhancement.

	The dataset yields dicts with keys: `lq`, `hq`, and `path`.
	"""

	def __init__(
		self,
		samples: Sequence[Sample],
		*,
		degradation: DegradationPipeline | None = None,
		transforms: TransformPipeline | None = None,
	) -> None:
		if len(samples) == 0:
			raise ValueError("ImageDataset received 0 samples")

		self.samples = list(samples)
		self.degradation = degradation
		self.transforms = transforms or TransformPipeline()

		# Validate: HQ must exist; LQ may be None.
		missing = [s.hq_path for s in self.samples if not s.hq_path.exists()]
		if missing:
			raise FileNotFoundError(f"Missing HQ image(s), first is: {missing[0]}")
		missing_lq = [s.lq_path for s in self.samples if s.lq_path is not None and not s.lq_path.exists()]
		if missing_lq:
			raise FileNotFoundError(f"Missing LQ image(s), first is: {missing_lq[0]}")

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int) -> dict[str, object]:
		sample = self.samples[idx]
		hq = _pil_open_rgb(sample.hq_path)

		if sample.lq_path is not None:
			lq = _pil_open_rgb(sample.lq_path)
		else:
			if self.degradation is None:
				raise RuntimeError(
					"HQ-only sample but no degradation pipeline was provided. "
					"Pass `degradation=DegradationPipeline(...)` to ImageDataset."
				)
			lq = self.degradation(hq.copy())

		lq_t, hq_t = self.transforms(lq, hq)

		return {
			"lq": lq_t,
			"hq": hq_t,
			"path": str(sample.hq_path),
		}

	@staticmethod
	def from_hq_dir(
		hq_dir: str | Path,
		*,
		recursive: bool = True,
		degradation: DegradationPipeline | None = None,
		transforms: TransformPipeline | None = None,
	) -> "ImageDataset":
		hq_paths = list_images(Path(hq_dir), recursive=recursive)
		samples = [Sample(hq_path=p, lq_path=None) for p in hq_paths]
		return ImageDataset(samples, degradation=degradation, transforms=transforms)

	@staticmethod
	def from_paired_dirs(
		lq_dir: str | Path,
		hq_dir: str | Path,
		*,
		recursive: bool = True,
		transforms: TransformPipeline | None = None,
		strict: bool = True,
	) -> "ImageDataset":
		lq_dir = Path(lq_dir)
		hq_dir = Path(hq_dir)

		lq_paths = list_images(lq_dir, recursive=recursive)
		hq_paths = list_images(hq_dir, recursive=recursive)

		# Pair by relative path first; fall back to filename.
		hq_by_rel: dict[str, Path] = {p.relative_to(hq_dir).as_posix(): p for p in hq_paths}
		hq_by_name: dict[str, Path] = {p.name: p for p in hq_paths}

		samples: list[Sample] = []
		missing: list[Path] = []
		for lq in lq_paths:
			rel = lq.relative_to(lq_dir).as_posix()
			hq = hq_by_rel.get(rel) or hq_by_name.get(lq.name)
			if hq is None:
				missing.append(lq)
				continue
			samples.append(Sample(hq_path=hq, lq_path=lq))

		if strict and missing:
			raise FileNotFoundError(
				"Could not find matching HQ images for some LQ images. "
				f"Example missing pair for: {missing[0]}"
			)

		if len(samples) == 0:
			raise ValueError(f"No paired samples found under {lq_dir} and {hq_dir}")

		return ImageDataset(samples, degradation=None, transforms=transforms)


def build_dataloader(
	*,
	root: str | Path,
	batch_size: int,
	num_workers: int = 4,
	train_split: Literal["train", "training"] = "train",
	val_split: Literal["val", "valid", "validation"] = "val",
	recursive: bool = True,
	paired: bool | None = None,
	lq_subdir: str | None = None,
	hq_subdir: str | None = None,
	val_fraction: float = 0.1,
	seed: int = 42,
	crop_size: int | tuple[int, int] | None = 256,
	normalize: bool = True,
	degradation: DegradationPipeline | None = None,
	pin_memory: bool | None = None,
	persistent_workers: bool | None = None,
	drop_last: bool = True,
) -> tuple[DataLoader, DataLoader]:
	"""Build PyTorch DataLoaders for training and validation.

	Supported dataset layouts (examples):

	1) Paired, explicit subdirs:
	   root/
		 train/lq/*.png
		 train/hq/*.png
		 val/lq/*.png
		 val/hq/*.png
	   -> set `paired=True`, `lq_subdir="lq"`, `hq_subdir="hq"`

	2) HQ-only:
	   root/train/*.jpg, root/val/*.jpg
	   -> set `paired=False` (or leave auto) and degradation is applied

	3) Single folder with random split:
	   root/images/*.jpg
	   -> when no train/val folders exist, we split by `val_fraction`
	"""

	root = Path(root)
	if pin_memory is None:
		pin_memory = torch.cuda.is_available()
	if persistent_workers is None:
		persistent_workers = num_workers > 0

	generator = torch.Generator()
	generator.manual_seed(int(seed))

	transforms = TransformPipeline(crop_size=crop_size, normalize=normalize)
	degradation = degradation or DegradationPipeline()

	def _dir_if_exists(p: Path) -> Path | None:
		return p if p.exists() and p.is_dir() else None

	# Prefer train/val directories if present.
	train_dir = _dir_if_exists(root / train_split)
	val_dir = _dir_if_exists(root / val_split)

	if train_dir is not None:
		# Determine paired vs HQ-only.
		if paired is None:
			paired = lq_subdir is not None and hq_subdir is not None

		if paired:
			if lq_subdir is None or hq_subdir is None:
				raise ValueError("For paired datasets, set `lq_subdir` and `hq_subdir`.")

			train_ds = ImageDataset.from_paired_dirs(
				train_dir / lq_subdir,
				train_dir / hq_subdir,
				recursive=recursive,
				transforms=transforms,
			)
			if val_dir is not None:
				val_ds = ImageDataset.from_paired_dirs(
					val_dir / lq_subdir,
					val_dir / hq_subdir,
					recursive=recursive,
					transforms=TransformPipeline(crop_size=crop_size, random_crop=False, normalize=normalize),
					strict=False,
				)
			else:
				# Split paired samples.
				base_samples = train_ds.samples
				n = len(base_samples)
				n_val = max(1, int(round(n * val_fraction)))
				indices = torch.randperm(n, generator=generator).tolist()
				val_idx = indices[:n_val]
				tr_idx = indices[n_val:]
				train_ds = ImageDataset([base_samples[i] for i in tr_idx], transforms=transforms)
				val_ds = ImageDataset(
					[base_samples[i] for i in val_idx],
					transforms=TransformPipeline(crop_size=crop_size, random_crop=False, normalize=normalize),
				)
		else:
			train_ds = ImageDataset.from_hq_dir(
				train_dir,
				recursive=recursive,
				degradation=degradation,
				transforms=transforms,
			)
			if val_dir is not None:
				val_ds = ImageDataset.from_hq_dir(
					val_dir,
					recursive=recursive,
					degradation=degradation,
					transforms=TransformPipeline(crop_size=crop_size, random_crop=False, normalize=normalize),
				)
			else:
				base_samples = train_ds.samples
				n = len(base_samples)
				n_val = max(1, int(round(n * val_fraction)))
				indices = torch.randperm(n, generator=generator).tolist()
				val_idx = indices[:n_val]
				tr_idx = indices[n_val:]
				train_ds = ImageDataset([base_samples[i] for i in tr_idx], degradation=degradation, transforms=transforms)
				val_ds = ImageDataset(
					[base_samples[i] for i in val_idx],
					degradation=degradation,
					transforms=TransformPipeline(crop_size=crop_size, random_crop=False, normalize=normalize),
				)
	else:
		# No explicit train folder. Split all images under root.
		all_images = list_images(root, recursive=recursive)
		if len(all_images) < 2:
			raise ValueError(
				"Not enough images found to split into train/val. "
				"Provide `root/train` and `root/val`, or add more images."
			)

		paired = False if paired is None else paired
		if paired:
			raise ValueError(
				"Paired=True requires explicit split directories (e.g., root/train and root/val) "
				"or explicit lq/hq subdirs."
			)

		n = len(all_images)
		n_val = max(1, int(round(n * val_fraction)))
		indices = torch.randperm(n, generator=generator).tolist()
		val_idx = set(indices[:n_val])
		train_imgs = [p for i, p in enumerate(all_images) if i not in val_idx]
		val_imgs = [p for i, p in enumerate(all_images) if i in val_idx]

		train_ds = ImageDataset(
			[Sample(hq_path=p) for p in train_imgs],
			degradation=degradation,
			transforms=transforms,
		)
		val_ds = ImageDataset(
			[Sample(hq_path=p) for p in val_imgs],
			degradation=degradation,
			transforms=TransformPipeline(crop_size=crop_size, random_crop=False, normalize=normalize),
		)

	train_loader = DataLoader(
		train_ds,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=bool(pin_memory),
		persistent_workers=bool(persistent_workers),
		drop_last=drop_last,
		worker_init_fn=_seed_worker if num_workers > 0 else None,
		generator=generator,
	)

	val_loader = DataLoader(
		val_ds,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=bool(pin_memory),
		persistent_workers=bool(persistent_workers),
		drop_last=False,
		worker_init_fn=_seed_worker if num_workers > 0 else None,
		generator=generator,
	)

	return train_loader, val_loader

