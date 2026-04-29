
"""Trainer utilities.

Name: Ryan Koes
Date: 2026-02-18

Provides a lightweight `Trainer` for PyTorch models with:
    - training + validation loops
    - optional Weights & Biases logging
    - optional learning-rate scheduling
    - early stopping and checkpoint save/resume

Metric computation uses `torchmetrics` when installed. Supported tasks are:
"multiclass", "binary", and "regression". A legacy task value "rnn" enables
simple padding support for variable-length 1D sequences.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from .config import MetricsConfig, ModelConfig, TrainerConfig


@torch.no_grad()
def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    eps: float = 1e-8,
) -> float:
    """Compute PSNR for image tensors.

    Assumes `pred` and `target` are in the same value range.
    For normalized tensors (e.g., [-1, 1]), pass a matching `data_range`.
    """
    pred = pred.float()
    target = target.float()
    mse = torch.mean((pred - target) ** 2).clamp_min(eps)
    return float((20.0 * torch.log10(torch.tensor(data_range)) - 10.0 * torch.log10(mse)).item())


def _unpack_image_batch(batch: object) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack a batch into (inputs, targets) for enhancement datasets.

    Supports:
    - dict batches: {"lq": ..., "hq": ...}
    - tuple/list batches: (inputs, targets, ...)
    """
    if isinstance(batch, dict):
        if "lq" not in batch or "hq" not in batch:
            raise KeyError("Expected dict batch with keys 'lq' and 'hq'")
        return batch["lq"], batch["hq"]

    if isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError("Expected batch like (inputs, targets, ...) with at least 2 items")
        return batch[0], batch[1]

    raise TypeError(f"Unsupported batch type for enhancement training: {type(batch)!r}")


@torch.no_grad()
def evaluate_image_to_image(
    *,
    model: nn.Module,
    loader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    max_batches: int | None = None,
    psnr_data_range: float = 2.0,
) -> dict[str, float]:
    """Evaluate an image-to-image model.

    Default PSNR data range is 2.0 because the notebook normalizes images to [-1, 1].
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    n = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = _unpack_image_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = loss_fn(pred, y)

        total_loss += float(loss.detach().item())
        total_psnr += psnr(pred, y, data_range=float(psnr_data_range))
        n += 1

    if n == 0:
        return {"loss": float("nan"), "psnr": float("nan")}
    return {"loss": total_loss / n, "psnr": total_psnr / n}


def train_image_to_image(
    *,
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = 2,
    lr: float = 2e-4,
    weight_decay: float = 0.0,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    grad_clip_norm: float | None = 1.0,
    psnr_data_range: float = 2.0,
    log_fn: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, list[float]]:
    """Train a simple image-to-image enhancement model.

    This is intentionally small and notebook-friendly.
    It works with the project's `ImageDataset` which yields dict batches.
    """

    model = model.to(device)
    if loss_fn is None:
        loss_fn = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
    }

    for epoch in range(int(epochs)):
        model.train()
        running_loss = 0.0
        n = 0

        for i, batch in enumerate(train_loader):
            if max_train_batches is not None and i >= max_train_batches:
                break

            x, y = _unpack_image_batch(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm is not None and float(grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()

            running_loss += float(loss.detach().item())
            n += 1

        train_loss = running_loss / max(1, n)
        val_metrics = evaluate_image_to_image(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            max_batches=max_val_batches,
            psnr_data_range=float(psnr_data_range),
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_psnr"].append(val_metrics["psnr"])

        if log_fn is not None:
            payload = {
                "epoch": float(epoch + 1),
                "train_loss": float(train_loss),
                "val_loss": float(val_metrics["loss"]),
                "val_psnr": float(val_metrics["psnr"]),
            }
            try:
                log_fn(payload)
            except Exception:
                # Logging must never crash training.
                pass

        if getattr(getattr(train_loader, "dataset", None), "__len__", None) is not None:
            pass
        if True:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | val_psnr={val_metrics['psnr']:.2f}"
            )

    return history

try:
    import torchmetrics
    from torchmetrics import MetricCollection

    METRIC_REGISTRY = {
        # Classification
        "acc": torchmetrics.Accuracy,
        "f1_macro": torchmetrics.F1Score,
        # Regression
        "mae": torchmetrics.MeanAbsoluteError,
        "mse": torchmetrics.MeanSquaredError,
        "r2": torchmetrics.R2Score,
    }
except ModuleNotFoundError:  # pragma: no cover
    torchmetrics = None  # type: ignore[assignment]
    MetricCollection = None  # type: ignore[assignment]
    METRIC_REGISTRY = {}


def _as_float_metric(value: torch.Tensor) -> float:
    """Convert a torchmetrics compute() output to a Python float.

    Some torchmetrics return a scalar tensor, others return a vector (e.g.,
    multioutput regression). We default to the mean for non-scalar outputs.
    """
    t = value.detach()
    if t.numel() == 1:
        return float(t.item())
    return float(t.mean().item())


def _is_wandb_hook(hook: object) -> bool:
    module_name = str(getattr(hook, "__module__", ""))
    if module_name.startswith("wandb"):
        return True

    name = str(
        getattr(hook, "__qualname__", "")
        or getattr(hook, "__name__", "")
        or ""
    ).lower()
    if "wandb" in name:
        return True
    try:
        return "wandb" in repr(hook).lower()
    except Exception:
        return False


def strip_wandb_hooks(model: nn.Module) -> None:
    """Remove lingering hooks registered by wandb.watch() from a model.

    Why this exists:
      - If a model is watched, then later the W&B run is finished, the model can
        retain forward hooks that attempt to log to wandb.run (which becomes None).

    This helper is safe to call even when W&B isn't initialized.
    """
    try:
        import wandb  # local import to avoid hard dependency during static analysis

        try:
            wandb.unwatch(model)
        except Exception:
            pass
    except Exception:
        pass

    for module in model.modules():
        for hook_dict_name in (
            "_forward_hooks",
            "_forward_pre_hooks",
            "_backward_hooks",
            "_backward_pre_hooks",
        ):
            hook_dict = getattr(module, hook_dict_name, None)
            if not isinstance(hook_dict, dict) or not hook_dict:
                continue
            for hook_id, hook in list(hook_dict.items()):
                if _is_wandb_hook(hook):
                    hook_dict.pop(hook_id, None)


class Trainer:
    """Train and validate a PyTorch model.

    This class wraps a standard training loop with optional W&B logging,
    checkpointing/resume, early stopping, and optional `torchmetrics` metrics.

    Args:
        model: The model to train.
        optimizer: Optimizer used to update model parameters.
        criterion: Loss function mapping (outputs, targets) -> loss tensor.
        config: Training configuration.
        run: Optional W&B run to log metrics to.
        metrics_config: Optional per-run metrics configuration. If provided, this
            overrides `config.metrics_config`.

    Notes:
        - `train_one_epoch()` and `validate()` are backward-compatible wrappers
          that return `(loss, acc)` tuples.
        - For W&B, the results dict and logged keys use `accuracy` (not `acc`).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        config: TrainerConfig = TrainerConfig(),
        run: Optional[wandb.Run] = None,
        metrics_config: Optional[MetricsConfig] = None,
    ) -> None:
        # `run=None` keeps this usable without any W&B setup.
        if metrics_config is not None:
            config.metrics_config = metrics_config

        self.model = model.to(config.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.run = run

        # Initialize state variables for checkpointing and early stopping
        self.start_epoch = 0                # Use for resuming training from checkpoint
        self.current_epoch = 0              # Tracks current epoch during training
        self.best_val_loss = float('inf')   # Default best validation loss
        self.patience_counter = 0           # How many epochs without improvement before stopping

        # Initialize learning rate scheduler if enabled
        self.scheduler = None
        if config.use_scheduler:
            self.scheduler = self._create_scheduler()

        if self.run is not None:
            wandb.watch(self.model, log="all", log_freq=10)


    @staticmethod
    def _rename_metric_key(key: str) -> str:
        # Public-facing output uses `accuracy` even if requested metric is `acc`.
        return "accuracy" if key == "acc" else key


    @staticmethod
    def _pad_1d_batch(
        sequences: List[torch.Tensor],
        *,
        pad_value: int,
    ) -> torch.Tensor:
        if len(sequences) == 0:
            raise ValueError("Cannot pad an empty batch.")
        if any(seq.ndim != 1 for seq in sequences):
            raise ValueError("RNN padding currently supports 1D tensors per sample.")
        max_len = max(int(seq.numel()) for seq in sequences)
        batch_size = len(sequences)
        out = sequences[0].new_full((batch_size, max_len), pad_value)
        for i, seq in enumerate(sequences):
            out[i, : seq.numel()] = seq
        return out


    def _maybe_pad_rnn_batch(self, batch, *, task: str):
        """Normalize batches to (inputs, targets) and optionally pad for legacy task='rnn'.

        This trainer historically assumes each DataLoader batch can be unpacked
        into exactly two values: (inputs, targets). In practice, some datasets
        and collate functions may yield extra fields (e.g., (inputs, targets, lengths)
        or (inputs, targets, metadata)). To keep training robust, we:

        - accept tuples/lists of length >= 2 and ignore any extras
        - accept list-of-samples where each sample is a tuple/list of length >= 2
        - for task='rnn', pad variable-length 1D inputs (and sequence targets)

        Returns:
            (inputs, targets)
        """

        task_norm = (task or "").strip().lower()

        # Case: DataLoader yields list of samples.
        # Each sample may be (x, y) or (x, y, ...). We ignore extras.
        if isinstance(batch, list) and batch and isinstance(batch[0], (tuple, list)):
            if len(batch[0]) < 2:
                raise ValueError(
                    "Batch samples must contain at least (inputs, targets)."
                )
            inputs_list = [sample[0] for sample in batch]
            targets_list = [sample[1] for sample in batch]

            if task_norm != "rnn":
                return inputs_list, targets_list

            inputs_list = [torch.as_tensor(x) for x in inputs_list]
            targets_list = [torch.as_tensor(y) for y in targets_list]

            inputs = self._pad_1d_batch(inputs_list, pad_value=0)
            if all(t.ndim == 0 for t in targets_list):
                targets = torch.stack(targets_list, dim=0)
            elif all(t.ndim == 1 for t in targets_list):
                targets = self._pad_1d_batch(targets_list, pad_value=-100)
            else:
                raise ValueError(
                    "RNN batches must have scalar targets or 1D sequence targets."
                )
            return inputs, targets

        # Case: batch is already a tuple/list. Accept (inputs, targets) or (inputs, targets, ...).
        if isinstance(batch, (tuple, list)):
            if len(batch) < 2:
                raise ValueError(
                    "Batch must contain at least (inputs, targets)."
                )
            inputs, targets = batch[0], batch[1]

            if task_norm != "rnn":
                return inputs, targets

            # If inputs is a list of variable-length tensors, pad it.
            if (
                isinstance(inputs, list)
                and inputs
                and isinstance(inputs[0], torch.Tensor)
            ):
                inputs = self._pad_1d_batch(
                    [torch.as_tensor(x) for x in inputs], pad_value=0
                )

            # Same for targets.
            if (
                isinstance(targets, list)
                and targets
                and isinstance(targets[0], torch.Tensor)
            ):
                targets_t = [torch.as_tensor(y) for y in targets]
                if all(t.ndim == 0 for t in targets_t):
                    targets = torch.stack(targets_t, dim=0)
                else:
                    targets = self._pad_1d_batch(targets_t, pad_value=-100)
            return inputs, targets

        # Unexpected batch type (e.g., dict) — fail loudly with context.
        raise ValueError(
            "Unsupported batch type. Expected (inputs, targets), (inputs, targets, ...), "
            "or list[(inputs, targets, ...)]. "
            f"Got type={type(batch)!r}."
        )


    def _wandb_safe_update_config(self, *, num_epochs: int, mc: MetricsConfig) -> None:
        """Update W&B run config, avoiding locked keys in sweeps."""
        if self.run is None:
            return

        config_payload = {
            "trainer_batch_size": self.config.trainer_batch_size,
            "evaluator_batch_size": self.config.evaluator_batch_size,
            "learning_rate": self.config.learning_rate,
            "device": str(self.config.device),
            "num_epochs": int(num_epochs),
            "weight_decay": self.config.weight_decay,
            "early_stopping_patience": self.config.early_stopping_patience,
            "early_stopping_min_delta": self.config.early_stopping_min_delta,
            "optimizer_name": self.config.optimizer_name,
            "momentum": self.config.momentum,
            "metrics": list(mc.metrics),
            "task": mc.task,
            "metrics_average": mc.average,
        }

        # In W&B sweeps, many config keys are locked. Attempting to update them
        # produces warnings like: "Config item 'x' was locked by 'sweep'".
        safe_payload = {k: v for k, v in config_payload.items() if k not in self.run.config}
        if safe_payload:
            self.run.config.update(safe_payload)


    def __enter__(self) -> "Trainer":
        return self


    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ) -> bool:
        # Always attempt to finish logging, even if an exception occurred.
        try:
            self.finish()
        finally:
            # Returning False means: do not suppress exceptions.
            return False


    def train_one_epoch(self, train_loader, *, task: Optional[str] = None) -> Tuple[float, float]:
        """Backward-compatible wrapper.

        Returns:
            (avg_loss, avg_acc)
        """
        mc = self._resolve_metrics_config(metrics=None, task=task)
        task_norm = mc.task
        wrapper_metrics = ["loss"] if task_norm == "regression" else ["loss", "acc"]
        metrics = self.train_one_epoch_metrics(
            train_loader, metrics=wrapper_metrics, task=mc.task
        )
        return metrics["loss"], metrics.get("acc", 0.0)


    def _metrics_update_inputs(
        self,
        *,
        task: str,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        task = str(task).strip().lower()
        if task == "regression":
            preds = outputs
            if preds.ndim >= 2 and preds.size(-1) == 1:
                preds = preds.squeeze(-1)
            t = targets
            if t.ndim >= 2 and t.size(-1) == 1:
                t = t.squeeze(-1)
            return preds, t.to(torch.float32)

        # Classification (binary / multiclass)
        if outputs.ndim == 1 or (outputs.ndim >= 2 and outputs.size(-1) == 1):
            preds = (outputs.squeeze(-1) > 0).to(torch.int64)
        else:
            preds = outputs.argmax(dim=1).to(torch.int64)
        return preds, targets.to(torch.int64)


    def _normalize_metrics(self, metrics: Optional[List[str]]) -> List[str]:
        """Normalize metric names and ensure required metrics are present."""
        if metrics is None:
            metrics = list(getattr(self.config, "metrics", []) or [])

        normalized: List[str] = []
        seen = set()

        def add(name: str) -> None:
            if name not in seen:
                normalized.append(name)
                seen.add(name)

        # Loss is always computed.
        add("loss")

        for raw in metrics:
            if raw is None:
                continue
            name = str(raw).strip().lower()
            if not name:
                continue
            if name in {"accuracy", "acc"}:
                add("acc")
            elif name in {"f1_macro", "macro_f1", "f1"}:
                add("f1_macro")
            elif name in {"mae"}:
                add("mae")
            elif name in {"mse"}:
                add("mse")
            elif name in {"r2"}:
                add("r2")
            elif name == "loss":
                add("loss")
            else:
                raise ValueError(
                    "Unknown metric: "
                    f"{raw}. Supported metrics: loss, acc, f1_macro, mae, mse, r2"
                )

        return normalized


    def _resolve_metrics_config(
        self,
        *,
        metrics: Optional[List[str]] = None,
        task: Optional[str] = None,
    ) -> MetricsConfig:
        """Resolve metrics/task/average for this call.

        Precedence:
          1) explicit args (`metrics`, `task`) if provided
          2) `self.config.metrics_config` if set
          3) legacy `self.config.task` + `self.config.metrics`
        """

        cfg_mc = getattr(self.config, "metrics_config", None)
        if isinstance(cfg_mc, MetricsConfig):
            mc = MetricsConfig(
                task=cfg_mc.task,
                metrics=list(cfg_mc.metrics),
                average=getattr(cfg_mc, "average", "macro"),
            )
        else:
            mc = MetricsConfig(
                task=getattr(self.config, "task", "multiclass"),
                metrics=list(getattr(self.config, "metrics", []) or []),
                average="macro",
            )

        if task is not None:
            mc.task = task
        if metrics is not None:
            mc.metrics = list(metrics)

        mc.task = str(mc.task).strip().lower()
        mc.average = str(mc.average).strip().lower() or "macro"

        # Treat RNN sequence classification as multiclass for metric purposes.
        # (Batch padding is handled separately and should not overload metric task.)
        if mc.task == "rnn":
            mc.task = "multiclass"
        task_alias = mc.task

        if task_alias not in {"multiclass", "binary", "regression"}:
            raise ValueError(
                f"Unknown task '{mc.task}'. Expected one of: multiclass, binary, regression."
            )

        mc.metrics = self._normalize_metrics(mc.metrics)

        allowed_for_task = {
            "regression": {"loss", "mae", "mse", "r2"},
            "binary": {"loss", "acc", "f1_macro"},
            "multiclass": {"loss", "acc", "f1_macro"},
        }[task_alias]
        invalid = [m for m in mc.metrics if m not in allowed_for_task]
        if invalid:
            raise ValueError(
                f"Invalid metrics for task='{mc.task}': {invalid}. "
                f"Allowed: {sorted(allowed_for_task)}"
            )

        return mc


    def _build_metric_collections(
        self,
        metrics: List[str],
        *,
        task: str,
        num_classes: Optional[int] = None,
        average: str = "macro",
    ) -> Tuple[Any, Any]:
        """Build torchmetrics MetricCollections for train/val.

        Notes:
            - For classification tasks ("binary", "multiclass"), we pass `task=...` and,
              for multiclass, also `num_classes` and `average`.
            - For regression, metrics must be constructed without `task` (and without
              classification-only kwargs), since regression metrics don't accept them.
        """

        if MetricCollection is None or not METRIC_REGISTRY:
            raise ImportError(
                "torchmetrics is required for _build_metric_collections but is not installed."
            )

        task = str(task).strip().lower()
        # Backwards-compat: older notebooks used task="rnn" to request RNN-style
        # batching/padding. Metrics still need a classification/regression task.
        if task == "rnn":
            task = "multiclass"

        if task not in {"multiclass", "binary", "regression"}:
            raise ValueError(
                f"Unknown task '{task}'. Expected one of: multiclass, binary, regression."
            )

        if task == "regression":
            kwargs: Dict[str, object] = {}
        else:
            kwargs = {"task": task}
            if task == "multiclass":
                if num_classes is None:
                    raise ValueError("num_classes is required when task='multiclass'.")
                kwargs["num_classes"] = int(num_classes)
                kwargs["average"] = average

        def _make_metric_objs() -> Dict[str, object]:
            objs: Dict[str, object] = {}
            for name in metrics:
                if name == "loss":
                    continue
                if name not in METRIC_REGISTRY:
                    raise ValueError(
                        "Unknown metric '"
                        f"{name}'. Supported: {', '.join(sorted(METRIC_REGISTRY.keys()))}"
                    )
                objs[name] = METRIC_REGISTRY[name](**kwargs)
            return objs

        train_collection = MetricCollection(_make_metric_objs(), prefix="train_").to(
            self.config.device
        )
        val_collection = MetricCollection(_make_metric_objs(), prefix="val_").to(
            self.config.device
        )
        return train_collection, val_collection


    def _update_confusion_matrix(
        self,
        confmat: torch.Tensor,
        preds: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """Update confusion matrix counts for one batch.

        Confusion matrix is indexed as [true, pred].
        """
        if preds.numel() == 0:
            return confmat

        preds = preds.view(-1).to(torch.int64)
        targets = targets.view(-1).to(torch.int64)

        valid = (targets >= 0) & (targets < num_classes)
        if valid.any():
            targets = targets[valid]
            preds = preds[valid]
        else:
            return confmat

        preds = preds.clamp(0, num_classes - 1)
        indices = targets * num_classes + preds
        bincount = torch.bincount(indices, minlength=num_classes * num_classes)
        return confmat + bincount.view(num_classes, num_classes)


    def _macro_f1_from_confusion_matrix(self, confmat: torch.Tensor) -> float:
        """Compute macro-F1 from a confusion matrix [true, pred]."""
        confmat = confmat.to(torch.float32)
        tp = torch.diag(confmat)
        fp = confmat.sum(dim=0) - tp
        fn = confmat.sum(dim=1) - tp
        eps = 1e-12
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        return float(f1.mean().item())


    def train_one_epoch_metrics(
        self,
        train_loader,
        metrics: Optional[List[str]] = None,
        *,
        task: str = "multiclass",
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training DataLoader.
            metrics: Metric names to compute ("loss" is always computed).
            task: One of "multiclass", "binary", "regression", or legacy "rnn".

        Returns:
            A dict of metric name -> value (e.g., {"loss": 0.42, "acc": 0.9}).
        """
        
        mc = self._resolve_metrics_config(metrics=metrics, task=task)
        metrics = mc.metrics
        task = mc.task

        need_acc = "acc" in metrics
        need_f1_macro = "f1_macro" in metrics

        train_mc = None
        need_torchmetrics = any(m != "loss" for m in metrics)

        self.model.train()
        total_loss = 0.0
        total_samples = 0
        num_classes: Optional[int] = None

        # Loop over the training data batches
        for batch in train_loader:
            inputs, targets = self._maybe_pad_rnn_batch(batch, task=task)
            inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            if need_torchmetrics and train_mc is None:
                # Infer num_classes on first batch when needed.
                if task == "multiclass" and num_classes is None:
                    if outputs.ndim != 2:
                        raise ValueError(
                            "multiclass metrics require model outputs shaped [batch, num_classes]."
                        )
                    num_classes = int(outputs.size(1))
                train_mc, _ = self._build_metric_collections(
                    metrics,
                    task=task,
                    num_classes=num_classes,
                    average=mc.average,
                )

            loss = self.criterion(outputs, targets)
            loss.backward()

            # Apply optional gradient clipping when configured on ModelConfig.
            clip_value = 0.0
            if hasattr(self.model, "config") and isinstance(self.model.config, ModelConfig):
                clip_value = float(getattr(self.model.config, "clip_grad_norm", 0.0))
            if clip_value > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=clip_value,
                )
            self.optimizer.step()

            if train_mc is not None:
                preds_for_metrics, targets_for_metrics = self._metrics_update_inputs(
                    task=task,
                    outputs=outputs.detach(),
                    targets=targets.detach(),
                )
                train_mc.update(preds_for_metrics, targets_for_metrics)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        if total_samples == 0:
            result: Dict[str, float] = {"loss": 0.0}
            if need_acc:
                result["acc"] = 0.0
            if need_f1_macro:
                result["f1_macro"] = 0.0
            return result

        avg_loss = total_loss / total_samples
        result: Dict[str, float] = {"loss": avg_loss}
        if train_mc is not None:
            computed = train_mc.compute()
            for k, v in computed.items():
                key = str(k)
                if key.startswith("train_"):
                    key = key[len("train_") :]
                result[key] = _as_float_metric(v)
        return result


    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create a learning-rate scheduler based on `self.config`.

        Returns:
            A PyTorch LR scheduler instance.

        Raises:
            ValueError: if `self.config.scheduler_type` is unknown.
        """
        scheduler_type = (self.config.scheduler_type or "").lower()

        if scheduler_type in {"step", "steplr"}:
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma,
            )

        if scheduler_type in {"exponential", "exponentiallr"}:
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config.scheduler_gamma,
            )

        if scheduler_type in {"cosine", "cosineannealing", "cosineannealinglr"}:
            # Default to one full cosine cycle over the configured training duration.
            t_max = max(1, int(self.config.num_epochs))
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
            )

        if scheduler_type in {"reduce_on_plateau", "plateau", "reducelronplateau"}:
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.scheduler_gamma,
                patience=self.config.scheduler_patience,
                min_lr=self.config.scheduler_min_lr,
            )

        raise ValueError(
            f"Unknown scheduler_type: {self.config.scheduler_type}. "
            "Expected one of: step, exponential, cosine, reduce_on_plateau."
        )


    def validate(self, val_loader, *, task: Optional[str] = None) -> Tuple[float, float]:
        """Backward-compatible wrapper.

        Returns:
            (avg_val_loss, avg_val_accuracy)
        """
        mc = self._resolve_metrics_config(metrics=None, task=task)
        task_norm = mc.task
        wrapper_metrics = ["loss"] if task_norm == "regression" else ["loss", "acc"]
        metrics = self.validate_metrics(val_loader, metrics=wrapper_metrics, task=mc.task)
        return metrics["loss"], metrics.get("acc", 0.0)


    def validate_metrics(
        self,
        val_loader,
        metrics: Optional[List[str]] = None,
        *,
        task: str = "multiclass",
    ) -> Dict[str, float]:
        """Validate for one epoch.

        Args:
            val_loader: Validation DataLoader.
            metrics: Metric names to compute ("loss" is always computed).
            task: One of "multiclass", "binary", "regression", or legacy "rnn".

        Returns:
            A dict of metric name -> value.
        """

        mc = self._resolve_metrics_config(metrics=metrics, task=task)
        metrics = mc.metrics
        task = mc.task

        need_acc = "acc" in metrics
        need_f1_macro = "f1_macro" in metrics

        val_mc = None
        need_torchmetrics = any(m != "loss" for m in metrics)

        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        num_classes: Optional[int] = None

        # Loop over the validation data batches without computing gradients
        with torch.no_grad():
            for batch in val_loader:
                X_batch, y_batch = self._maybe_pad_rnn_batch(batch, task=task)
                X_batch = X_batch.to(self.config.device)
                y_batch = y_batch.to(self.config.device)

                logits = self.model(X_batch)

                if need_torchmetrics and val_mc is None:
                    if task == "multiclass" and num_classes is None:
                        if logits.ndim != 2:
                            raise ValueError(
                                "multiclass metrics require model outputs shaped [batch, num_classes]."
                            )
                        num_classes = int(logits.size(1))
                    _, val_mc = self._build_metric_collections(
                        metrics,
                        task=task,
                        num_classes=num_classes,
                        average=mc.average,
                    )

                loss = self.criterion(logits, y_batch)

                if val_mc is not None:
                    preds_for_metrics, targets_for_metrics = self._metrics_update_inputs(
                        task=task,
                        outputs=logits.detach(),
                        targets=y_batch.detach(),
                    )
                    val_mc.update(preds_for_metrics, targets_for_metrics)

                batch_size = X_batch.size(0)
                running_loss += loss.item() * batch_size
                total_samples += batch_size

        if total_samples == 0:
            result: Dict[str, float] = {"loss": 0.0}
            if need_acc:
                result["acc"] = 0.0
            if need_f1_macro:
                result["f1_macro"] = 0.0
            return result

        avg_loss = running_loss / total_samples
        result: Dict[str, float] = {"loss": avg_loss}
        if val_mc is not None:
            computed = val_mc.compute()
            for k, v in computed.items():
                key = str(k)
                if key.startswith("val_"):
                    key = key[len("val_") :]
                result[key] = _as_float_metric(v)
        return result


    def fit(
        self,
        train_loader,
        val_loader,
        resume_from_last_checkpoint: bool = False,
        override_num_epochs: Optional[int] = None,
        metrics: Optional[List[str]] = None,
        task: Optional[str] = None,
    ) -> Dict[str, float]:
        """Train for multiple epochs, validating after each epoch.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            resume_from_last_checkpoint: If True, load the last checkpoint before
                starting.
            override_num_epochs: If set, overrides `config.num_epochs` for this run.
            metrics: Metric names to compute/log/return.
            task: If set, overrides the configured task.

        Returns:
            A dict containing `num_epochs` and the last epoch's metrics, with keys
            like `train_loss`, `val_loss`, `train_accuracy`, `val_accuracy`.
        """

        # Resume from checkpoint if specified
        if resume_from_last_checkpoint:
            self.load_checkpoint(retrieve_best=False)

        num_epochs = override_num_epochs if override_num_epochs is not None else self.config.num_epochs

        mc = self._resolve_metrics_config(metrics=metrics, task=task)
        task_value = mc.task
        normalized_metrics = mc.metrics

        self._wandb_safe_update_config(num_epochs=num_epochs, mc=mc)

        # Sanity check: Verify the batch sizes match the config supplied
        if hasattr(train_loader, 'batch_size') and train_loader.batch_size is not None:
            if train_loader.batch_size != self.config.trainer_batch_size:
                raise ValueError(f"Train loader batch size ({train_loader.batch_size}) does not match config ({self.config.trainer_batch_size})")

        last_epoch_metrics: Dict[str, float] = {}
        epochs_completed = self.start_epoch

        # Loop over the specified number of epochs, supporting resume-from-checkpoint
        # via `self.start_epoch`.
        if self.start_epoch >= num_epochs:
            if getattr(self.config, "verbose", True):
                print(
                    f"No epochs to run (start_epoch={self.start_epoch}, num_epochs={num_epochs})."
                )
        else:
            for self.current_epoch in range(self.start_epoch, num_epochs):
                train_metrics = self.train_one_epoch_metrics(
                    train_loader, metrics=normalized_metrics, task=task_value
                )
                val_metrics = self.validate_metrics(
                    val_loader, metrics=normalized_metrics, task=task_value
                )

                # Combine into a flat, prefixed dict
                epoch_metrics: Dict[str, float] = {
                    **{
                        f"train_{self._rename_metric_key(k)}": v
                        for k, v in train_metrics.items()
                    },
                    **{
                        f"val_{self._rename_metric_key(k)}": v
                        for k, v in val_metrics.items()
                    },
                }
                last_epoch_metrics = epoch_metrics

                # Pretty print common metrics; otherwise print the dict.
                if getattr(self.config, "verbose", True):
                    if (
                        "train_loss" in epoch_metrics
                        and "val_loss" in epoch_metrics
                        and "train_accuracy" in epoch_metrics
                        and "val_accuracy" in epoch_metrics
                    ):
                        print(
                            f"Epoch {self.current_epoch + 1}: "
                            f"Train Loss={epoch_metrics['train_loss']:.4f}, Train Accuracy={epoch_metrics['train_accuracy'] * 100:.2f}%, "
                            f"Val Loss={epoch_metrics['val_loss']:.4f}, Val Accuracy={epoch_metrics['val_accuracy'] * 100:.2f}%"
                        )
                    else:
                        metrics_str = ", ".join(
                            f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                            for k, v in sorted(epoch_metrics.items())
                        )
                        print(f"Epoch {self.current_epoch + 1}: {metrics_str}")

                epochs_completed = self.current_epoch + 1

                # Save the best model found so far (based on validation loss)
                val_loss = val_metrics["loss"]
                if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1

                # Step the learning rate scheduler
                if self.scheduler is not None:
                    if (self.config.scheduler_type or "").lower() == "reduce_on_plateau":
                        # ReduceLROnPlateau needs the validation loss
                        self.scheduler.step(val_loss)
                    else:
                        # Other schedulers just need to know an epoch completed
                        self.scheduler.step()
                    
                    # Log current learning rate to W&B
                    current_lr = self.optimizer.param_groups[0]['lr']
                    if self.run is not None:
                        self.run.log({"learning_rate": current_lr}, step=self.current_epoch)

                if self.run is not None:
                    self.run.log({"epoch": self.current_epoch, **epoch_metrics})

                # Periodic checkpoint saving (every N epochs)
                if (
                    hasattr(self.config, "checkpoint_save_interval")
                    and self.config.checkpoint_save_interval is not None
                    and self.config.checkpoint_save_interval > 0
                    and (self.current_epoch + 1) % self.config.checkpoint_save_interval == 0
                ):
                    self.save_checkpoint(is_best=False)

                # Early stopping: stop if validation loss hasn't improved for N epochs
                if (
                    self.config.early_stopping_patience is not None
                    and self.patience_counter >= self.config.early_stopping_patience
                ):
                    if getattr(self.config, "verbose", True):
                        print(
                            f"Early stopping at epoch {self.current_epoch + 1}: "
                            f"no improvement for {self.config.early_stopping_patience} epochs."
                        )
                    break

        # Stop the looging of metrics to Weights & Biases at the end of training
        self.finish()

        return {
            "num_epochs": epochs_completed,
            **last_epoch_metrics,
        }
    
    def finish(self):
        """Finalize W&B logging and make the Trainer safe to reuse.

        This is safe to call multiple times.
        """
        if self.run is None:
            return

        # Remove any lingering W&B hooks before finishing the run.
        strip_wandb_hooks(self.model)

        try:
            self.run.finish()
        finally:
            # Make finish() idempotent (safe to call again).
            self.run = None


    # Backwards-compatibility with notebook calls
    def finish_run(self):
        self.finish()

    def save_checkpoint(self, is_best: bool = False) -> None:
        """
        Save a checkpoint of the current model state, optimizer state, and training progress.
        Args:
            is_best (bool): If True, also saves a copy of the checkpoint as the best model.
        Returns:             
            None
        Raises:            
            ValueError: If checkpoint saving fails due to file system issues or invalid state.
        Notes:            
            - The checkpoint includes the model's state_dict, optimizer's state_dict, current epoch, best validation loss, and patience counter for early stopping.
            - Checkpoints are saved in the directory specified in the TrainerConfig, with filenames for the last checkpoint and best checkpoint.
            - The method ensures that the checkpoint directory exists and handles any exceptions that may arise during the saving process, providing informative error messages.
        """
        # Ensure the checkpoint directory exists
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        filepath = os.path.join(self.config.checkpoint_dir, self.config.checkpoint_last_filename)

        # Create the checkpoint dictionary containing model state, optimizer state, and training progress
        checkpoint = {
            # Model weights (the numbers)
            'model_state_dict': self.model.state_dict(),

            # Architecture specification (the blueprint)
            'model_architecture': self.model.get_architecture_config() if hasattr(self.model,
                                                                                    'get_architecture_config') else None,

            # Training state
            'trainer_config': asdict(self.config),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'epoch': self.current_epoch,
            'patience_counter': self.patience_counter,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }

        # Save the checkpoint to the specified filepath. If is_best is True, also save a copy as the best checkpoint.
        verbose = bool(getattr(self.config, "verbose", True))
        if is_best:
            best_path = os.path.join(
                self.config.checkpoint_dir,
                self.config.checkpoint_best_filename,
            )
            torch.save(checkpoint, best_path)
            if verbose:
                print(f"--> New best checkpoint saved: {best_path}")
                print(f"--> Also saving as last checkpoint: {filepath}")
        elif verbose:
            print(f"--> Saving checkpoint: {filepath}")
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, retrieve_best: bool = False) -> None:
        """
        Load a checkpoint to restore the model state, optimizer state, and training progress.
        Args:
            retrieve_best (bool): If True, loads the best model checkpoint instead of the last checkpoint.
        Returns:
            None
        Raises:
            FileNotFoundError: If the specified checkpoint file does not exist.
            ValueError: If the checkpoint file is invalid or cannot be loaded.
        Notes:
            - The method attempts to load the checkpoint from the specified filepath based on the retrieve_best flag.
            - It restores the model's state_dict, optimizer's state_dict, and training progress (current epoch, best validation loss, patience counter).
            - The method includes error handling to provide informative messages if the checkpoint file is missing or cannot be loaded, ensuring that users are aware of any issues during the loading process.
        """
        # Load the file using `torch.load`, doing error checking (paying attention to whether you're loading the last or the best checkpoint)
        if retrieve_best:
            filepath = os.path.join(self.config.checkpoint_dir, self.config.checkpoint_best_filename)
        else:
            filepath = os.path.join(self.config.checkpoint_dir, self.config.checkpoint_last_filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file does not exist: {filepath}")

        try:
            checkpoint = torch.load(filepath)
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint file: {filepath}. Error: {e}")

        # Apply the state dicts to the model and optimizer.
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Critical Detail:** You must update `self.start_epoch`. 
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.patience_counter = checkpoint['patience_counter']        