
from logging import config
import os
import sys
from typing import Callable, Optional, Tuple
import contextlib
import io
import inspect
from functools import partial

# Use package-relative imports so this module works whether imported as
# `src.my_engine.sweep` (common in notebooks) or as an installed package.
from .data import get_dataloaders
from .text import text_collate_fn
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Dataset

from .config import ConvBlockConfig, ModelConfig, TrainerConfig, ResidualBlockConfig
from .trainer import Trainer
from .utils import build_model, make_optimizer

def print_sweep_info(sweep_id: str) -> None:
    """
    Prints information about a Weights & Biases sweep, including the number of runs, expected run count, and current state.
    Args:
        sweep_id (str): The unique identifier of the sweep to query.
    Returns:
        None: This function prints the sweep information to the console and does not return any value.
    Notes:
        - The function uses the wandb API to access the sweep information based on the provided sweep_id.
        - It retrieves the sweep object and prints the total number of runs, the expected number of runs, and the current state of the sweep (e.g., "RUNNING", "FINISHED", "CANCELED").
    """
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    print(f"Sweep {sweep_id} has {len(sweep.runs)} runs")
    print(f"Sweep {sweep_id} expected {sweep.expected_run_count} runs")
    print(f"Sweep {sweep_id} current state is: {sweep.state}")

def terminate_sweep(sweep_id: str) -> bool:
    """
    Terminates a Weights & Biases sweep if it is currently running. If the sweep is already finished, it will not attempt to stop it.
    Args:
        sweep_id (str): The unique identifier of the sweep to terminate.
    Returns:
        bool: True if the sweep was successfully stopped or was already finished, False if there was an error stopping the sweep.
    Notes:
        - The function uses the wandb API to access the sweep information based on the provided sweep_id.
        - It checks the current state of the sweep. If the state is not "FINISHED", it attempts to stop the sweep using the wandb CLI command.
        - If the sweep is already finished, it simply returns True without attempting to stop it.
        - The function handles potential errors when trying to stop the sweep and returns False if it fails to stop it.
    """
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    print(f"Sweep {sweep_id} current state is: {sweep.state}")
    # If the sweep is not finished, attempt to stop it
    if sweep.state != "FINISHED":
        s = sweep.entity + '/' + sweep.project + '/' + sweep.name
        print(f"Stopping sweep {s}")
        exit_code = os.system('wandb sweep --stop ' + s)
        if exit_code != 0:
            print(f"Failed to stop sweep {s}")
            print(f"Exit code: {exit_code}")
            return False
        else:
            print(f"Sweep {s} stopped successfully")
            return True
    # Otherwise, the sweep is already finished, so we can just return True
    else:
        print(f"Sweep {sweep_id} is already finished")
        return True


def make_train_sweep(
    wandb_project_name: str,  # string passed to wandb.init
    datasets: Tuple[Dataset, Dataset],  # (train_dataset, val_dataset)
    device: torch.device,
    num_inputs: Optional[int] = None,
    num_outputs: Optional[int] = None,
    wandb_entity_name: Optional[str] = None,
    input_spec=None,
    wandb_name_prefix: Optional[str] = None,
) -> Callable[[], None]:
    """Factory that returns a W&B sweep training function.

    The returned function has no arguments (as required by `wandb.agent`) and
    captures the datasets/device/model sizes from this factory.

    Args:
        wandb_project_name: The name of the W&B project to log runs to.
        datasets: A tuple containing the training and validation datasets.
        device: The torch.device to use for training (e.g., torch.device('cuda')).
        num_inputs: The number of input features for the model.
        num_outputs: The number of output classes or targets for the model.
        wandb_entity_name: The W&B entity (user or team) to log runs under. If None, defaults to the current user.

    Returns:
        A callable function that can be passed to `wandb.agent` to execute the training loop for each run in the sweep. 
        This function will initialize a W&B run, read hyperparameters from `wandb.config`, create DataLoaders, build the model and optimizer, and train the model

    Notes:
        - The returned function will be called by W&B for each run in the sweep, and it will use the hyperparameters specified in the sweep configuration 
        (e.g., hidden_units, dropout, learning_rate, etc.) to configure the model and training process.
        - The function will log training metrics to W&B for each run, allowing you to track the performance of different hyperparameter combinations across the sweep.
    """

    train_dataset, val_dataset = datasets
    default_trainer_config = TrainerConfig()
    default_model_config = ModelConfig()

    default_cnn_conv_blocks = [
        ConvBlockConfig(out_channels=32, kernel_size=3, stride=1, padding=1, pool_size=2),
        ConvBlockConfig(out_channels=64, kernel_size=3, stride=1, padding=1, pool_size=2),
    ]

    if num_outputs is None:
        raise ValueError("make_train_sweep requires num_outputs (int).")

    # NOTE: Some model types (e.g., text models) don't require input_spec/num_inputs.
    # We validate this per-run after reading wandb.config.model_type.
    effective_input_spec = input_spec if input_spec is not None else num_inputs

    def _parse_hidden_units(wandb_cfg) -> list[int]:
        raw_hidden_units = getattr(wandb_cfg, "hidden_units", [128, 64])
        if isinstance(raw_hidden_units, int):
            return [raw_hidden_units]
        return list(raw_hidden_units)

    def _parse_dropout(wandb_cfg, hidden_units: list[int]) -> list[float]:
        raw_dropout = getattr(wandb_cfg, "dropout", [0.1, 0.2])
        if isinstance(raw_dropout, (int, float)):
            dropout = [float(raw_dropout)]
        else:
            dropout = [float(x) for x in list(raw_dropout)]

        # Broadcast a single dropout value across all hidden layers.
        if len(dropout) == 1 and len(hidden_units) > 1:
            dropout = dropout * len(hidden_units)
        return dropout

    def _parse_metrics(wandb_cfg) -> list[str]:
        raw_metrics = getattr(wandb_cfg, "metrics", default_trainer_config.metrics)
        if raw_metrics is None:
            metrics = list(default_trainer_config.metrics)
        elif isinstance(raw_metrics, str):
            metrics = [raw_metrics]
        elif isinstance(raw_metrics, (list, tuple, set)):
            metrics = list(raw_metrics)
        else:
            raise TypeError(
                "wandb.config.metrics must be a string or list/tuple/set of strings, "
                f"got {type(raw_metrics)}"
            )
        return [str(m) for m in metrics]

    def _parse_task(wandb_cfg, *, loss_name: str) -> str:
        raw_task = getattr(wandb_cfg, "task", None)
        if raw_task is not None:
            task = str(raw_task).strip().lower()
            if task == "rnn":
                # Backwards-compat: some notebooks used task="rnn" to mean
                # sequence classification. For metric purposes, treat it as multiclass.
                task = "multiclass"
            if task in {"multiclass", "binary", "regression"}:
                return task

        # Heuristic default: MSE implies regression; otherwise keep legacy default.
        if loss_name == "mse":
            return "regression"
        return str(getattr(default_trainer_config, "task", "multiclass")).strip().lower() or "multiclass"

    def _coerce_metrics_for_task(metrics: list[str], *, task: str) -> list[str]:
        task = str(task).strip().lower()
        if task != "regression":
            return metrics

        # If we're doing regression but metrics were left at classification defaults,
        # drop classification-only metrics and ensure we compute something useful.
        out: list[str] = []
        for m in metrics:
            m_norm = str(m).strip().lower()
            if m_norm in {"acc", "accuracy", "f1", "f1_macro", "macro_f1"}:
                continue
            out.append(m)

        # Ensure a regression metric exists (besides loss).
        has_reg_metric = any(
            str(m).strip().lower() in {"mae", "mse", "r2"} for m in out
        )
        if not has_reg_metric:
            out.append("mae")
        return out

    def _parse_filter_sizes(wandb_cfg) -> tuple[int, ...]:
        default_fs = getattr(default_model_config, "filter_sizes", (3, 4, 5))
        filter_sizes = getattr(wandb_cfg, "filter_sizes", default_fs)
        if not isinstance(filter_sizes, tuple):
            filter_sizes = tuple(filter_sizes)
        return filter_sizes

    def _normalize_loss_name(wandb_cfg) -> str:
        # Supported values: "cross_entropy" (default), "mse".
        # Aliases are accepted for convenience.
        loss_name = str(getattr(wandb_cfg, "loss_name", "cross_entropy")).strip().lower()
        loss_aliases = {
            "ce": "cross_entropy",
            "crossentropy": "cross_entropy",
            "cross-entropy": "cross_entropy",
        }
        return loss_aliases.get(loss_name, loss_name)

    def _make_run_name(
        *,
        model_type: str,
        trainer_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
        hidden_units: list[int],
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        num_filters: int,
        filter_sizes: tuple[int, ...],
        rnn_hidden_size: int,
        rnn_num_layers: int,
        bidirectional: bool,
        rnn_type: str,
        num_encoder_layers: int,
        dim_feedforward: int,
        wandb_name_prefix: Optional[str],
    ) -> str:
        if model_type == "textcnn":
            filter_sizes_str = "-".join(map(str, filter_sizes))
            name = (
                f"{model_type}_bs{trainer_batch_size}_lr{learning_rate:.5f}_nf{num_filters}_fs{filter_sizes_str}"
                f"_wd{weight_decay:.5f}"
            )
        elif model_type == "bow":
            name = (
                f"{model_type}_bs{trainer_batch_size}_lr{learning_rate:.5f}_vs{vocab_size}_ed{embedding_dim}"
                f"_wd{weight_decay:.5f}"
            )
        elif model_type in {"rnn", "text_rnn"}:
            name = (
                f"{model_type}_bs{trainer_batch_size}_lr{learning_rate:.5f}"
                f"_hs{rnn_hidden_size}_L{rnn_num_layers}"
                f"_bi{int(bidirectional)}_{rnn_type}_wd{weight_decay:.5f}"
            )
        elif model_type == "text_attn":
            name = (
                f"{model_type}_bs{trainer_batch_size}_lr{learning_rate:.5f}"
                f"_ed{embedding_dim}_nh{num_heads}_wd{weight_decay:.5f}"
            )
        elif model_type == "text_transformer":
            name = (
                f"{model_type}_bs{trainer_batch_size}_lr{learning_rate:.5f}"
                f"_ed{embedding_dim}_nh{num_heads}_nl{num_encoder_layers}"
                f"_dff{dim_feedforward}_wd{weight_decay:.5f}"
            )
        else:
            hidden_str = "x".join(map(str, hidden_units))
            name = (
                f"{model_type}_bs{trainer_batch_size}_lr{learning_rate:.5f}_h{hidden_str}_wd{weight_decay:.5f}"
                f"_m{momentum:.2f}"
            )

        if wandb_name_prefix is not None:
            name = f"{wandb_name_prefix}_{name}"
        return name

    def _build_trainer_config(
        *,
        trainer_batch_size: int,
        evaluator_batch_size: int,
        num_workers: int,
        pin_memory: bool,
        learning_rate: float,
        num_epochs: int,
        device: torch.device,
        optimizer_name: str,
        weight_decay: float,
        momentum: float,
        use_scheduler: bool,
        scheduler_type: str,
        scheduler_step_size: int,
        scheduler_gamma: float,
        scheduler_patience: int,
        scheduler_min_lr: float,
        early_stopping_patience: Optional[int],
        metrics: list[str],
        verbose: bool,
    ) -> TrainerConfig:
        trainer_config_kwargs = dict(
            trainer_batch_size=trainer_batch_size,
            evaluator_batch_size=evaluator_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            device=device,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            momentum=momentum,
            use_scheduler=use_scheduler,
            scheduler_type=scheduler_type,
            scheduler_step_size=scheduler_step_size,
            scheduler_gamma=scheduler_gamma,
            scheduler_patience=scheduler_patience,
            scheduler_min_lr=scheduler_min_lr,
            early_stopping_patience=early_stopping_patience,
            metrics=metrics,
        )

        # Backward/forward compatibility: only pass keys TrainerConfig accepts.
        try:
            accepted = set(inspect.signature(TrainerConfig).parameters.keys())
            trainer_config_kwargs = {k: v for k, v in trainer_config_kwargs.items() if k in accepted}
        except (TypeError, ValueError):
            # If signature introspection fails for any reason, fall back to best-effort kwargs.
            pass

        trainer_config = TrainerConfig(**trainer_config_kwargs)
        # Even if TrainerConfig doesn't declare it, we can still attach `verbose` dynamically
        # (dataclasses without slots allow this), and Trainer uses getattr(..., True).
        try:
            setattr(trainer_config, "verbose", verbose)
        except Exception:
            pass

        return trainer_config

    def _parse_conv_blocks(wandb_cfg, model_type: str) -> list[object]:
        if model_type != "cnn":
            return []

        raw_blocks = getattr(wandb_cfg, "conv_blocks", [])

        def _parse_conv_block(raw_block):
            # Already-parsed dataclass objects are allowed
            if isinstance(raw_block, (ConvBlockConfig, ResidualBlockConfig)):
                return raw_block

            if not isinstance(raw_block, dict):
                raise TypeError(
                    f"Each conv block must be a dict or block config object, got {type(raw_block)}"
                )

            block_data = dict(raw_block)  # copy so we can safely pop
            block_type = block_data.pop("block_type", "conv")  # backward-compatible default

            if block_type == "conv":
                return ConvBlockConfig(**block_data)
            if block_type == "residual":
                return ResidualBlockConfig(**block_data)

            raise ValueError(f"Unknown block_type '{block_type}' in conv_blocks")

        if isinstance(raw_blocks, (ConvBlockConfig, ResidualBlockConfig, dict)):
            raw_blocks = [raw_blocks]

        conv_blocks = []
        for b in list(raw_blocks):
            conv_blocks.append(_parse_conv_block(b))

        # Ensure CNN_Model receives a valid, non-empty conv_blocks list.
        if len(conv_blocks) == 0:
            conv_blocks = list(default_cnn_conv_blocks)

        return conv_blocks

    def _build_model_config(
        *,
        model_type: str,
        hidden_units: list[int],
        dropout: list[float],
        conv_blocks: list[object],
        in_channels: int,
        use_GAP: bool,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        padding_idx: int,
        freeze_embeddings: bool,
        max_seq_len: Optional[int],
        num_filters: int,
        filter_sizes: tuple[int, ...],
        rnn_hidden_size: int,
        rnn_num_layers: int,
        rnn_dropout: float,
        bidirectional: bool,
        rnn_type: str,
        clip_grad_norm: float,
        num_encoder_layers: int,
        dim_feedforward: int,
    ) -> ModelConfig:
        model_config_kwargs = dict(
            model_type=model_type,
            hidden_units=hidden_units,
            dropout=dropout,
            # CNN
            conv_blocks=conv_blocks,
            in_channels=in_channels,
            use_GAP=use_GAP,
            # NLP / embedding
            vocab_size=int(vocab_size),
            embedding_dim=int(embedding_dim),
            # --- AttentionClassifier support ---
            num_heads=int(num_heads),
            padding_idx=int(padding_idx),
            freeze_embeddings=bool(freeze_embeddings),
            max_seq_len=max_seq_len,
            # --- TransformerClassifier support ---
            num_encoder_layers=int(num_encoder_layers),
            dim_feedforward=int(dim_feedforward),
            # TextCNN1D
            num_filters=int(num_filters),
            filter_sizes=list(filter_sizes),
            # --- RNN support ---
            rnn_hidden_size=int(rnn_hidden_size),
            rnn_num_layers=int(rnn_num_layers),
            rnn_dropout=float(rnn_dropout),
            bidirectional=bool(bidirectional),
            rnn_type=str(rnn_type),
            clip_grad_norm=float(clip_grad_norm),
        )

        # Backward/forward compatibility: only pass keys ModelConfig accepts.
        try:
            accepted = set(inspect.signature(ModelConfig).parameters.keys())
            model_config_kwargs = {k: v for k, v in model_config_kwargs.items() if k in accepted}
        except (TypeError, ValueError):
            pass

        return ModelConfig(**model_config_kwargs)

    def _build_criterion(loss_name: str) -> nn.Module:
        if loss_name == "mse":
            return nn.MSELoss()
        if loss_name == "cross_entropy":
            return nn.CrossEntropyLoss()
        raise ValueError(
            f"Unsupported loss_name: {loss_name}. Supported: 'cross_entropy', 'mse'"
        )

    def train_sweep() -> None:
        # Step 1: Initialize a W&B run (reduce W&B's own console output in notebooks).
        os.environ.setdefault("WANDB_SILENT", "true")
        run = wandb.init(
            project=wandb_project_name,
            entity=wandb_entity_name,
            reinit=True,  # Allow multiple runs in the same process
            settings=wandb.Settings(x_stats_sampling_interval=2.0, console="off"),
        )

        # Step 2: Read hyperparameters from wandb.config
        config = wandb.config

        hidden_units = _parse_hidden_units(config)
        dropout = _parse_dropout(config, hidden_units)

        trainer_batch_size = int(getattr(config, "trainer_batch_size", 64))
        evaluator_batch_size = int(
            getattr(config, "evaluator_batch_size", default_trainer_config.evaluator_batch_size)
        )
        num_workers = int(getattr(config, "num_workers", default_trainer_config.num_workers))
        pin_memory = bool(getattr(config, "pin_memory", default_trainer_config.pin_memory))

        # DataLoader multiprocessing is fragile in interactive contexts (Jupyter/IPython)
        # and when executed via W&B agent threads. In these environments, worker
        # processes can fail with FileNotFoundError on '<stdin>' / missing __main__.__file__.
        # Fall back to single-process loading by default.
        try:
            import __main__ as _main
            main_file = getattr(_main, "__file__", None)
            is_interactive = (main_file is None) or ("ipykernel" in sys.modules)
        except Exception:
            is_interactive = True

        if is_interactive and num_workers != 0:
            num_workers = 0

        # pin_memory isn't supported on MPS (and spams warnings); disable by default.
        if getattr(device, "type", None) == "mps":
            pin_memory = False

        learning_rate = float(getattr(config, "learning_rate", 1e-3))
        optimizer_name = str(getattr(config, "optimizer_name", "adam"))
        weight_decay = float(getattr(config, "weight_decay", 0.0))
        momentum = float(getattr(config, "momentum", 0.9))

        model_type = str(getattr(config, "model_type", default_model_config.model_type)).lower()
        use_GAP = getattr(config, "use_GAP", default_model_config.use_GAP)
        in_channels = int(getattr(config, "in_channels", default_model_config.in_channels))

        # NLP/embedding
        vocab_size = getattr(config, "vocab_size", default_model_config.vocab_size)
        embedding_dim = getattr(config, "embedding_dim", default_model_config.embedding_dim)
        padding_idx = getattr(config, "padding_idx", default_model_config.padding_idx)
        freeze_embeddings = getattr(config, "freeze_embeddings", default_model_config.freeze_embeddings)
        max_seq_len = getattr(config, "max_seq_len", default_model_config.max_seq_len)

        # TextCNN1D
        num_filters = getattr(config, "num_filters", getattr(default_model_config, "num_filters", 100))
        filter_sizes = _parse_filter_sizes(config)

        # RNN
        rnn_hidden_size = getattr(config, "rnn_hidden_size", default_model_config.rnn_hidden_size)
        rnn_num_layers = getattr(config, "rnn_num_layers", default_model_config.rnn_num_layers)
        rnn_dropout = float(getattr(config, "rnn_dropout", getattr(default_model_config, "rnn_dropout", 0.0)))
        bidirectional = getattr(
            config,
            "bidirectional",
            getattr(config, "rnn_bidirectional", default_model_config.bidirectional),
        )
        rnn_type = getattr(config, "rnn_type", default_model_config.rnn_type)
        clip_grad_norm = getattr(config, "clip_grad_norm", default_model_config.clip_grad_norm)
        num_heads = getattr(config, "num_heads", default_model_config.num_heads)

        num_encoder_layers = getattr(config, "num_encoder_layers", default_model_config.num_encoder_layers)
        dim_feedforward = getattr(config, "dim_feedforward", default_model_config.dim_feedforward)

        # Loss / criterion selection
        loss_name = _normalize_loss_name(config)

        # Whether to print progress to stdout (notebook noise).
        verbose = bool(getattr(config, "verbose", False))
        if verbose:
            print(f"wandb.config: {config}")

        task = _parse_task(config, loss_name=loss_name)
        metrics = _coerce_metrics_for_task(_parse_metrics(config), task=task)

        # Optional LR scheduler settings (Trainer supports these via TrainerConfig)
        use_scheduler = bool(getattr(config, "use_scheduler", default_trainer_config.use_scheduler))
        scheduler_type = str(getattr(config, "scheduler_type", default_trainer_config.scheduler_type))
        scheduler_step_size = int(
            getattr(config, "scheduler_step_size", default_trainer_config.scheduler_step_size)
        )
        scheduler_gamma = float(getattr(config, "scheduler_gamma", default_trainer_config.scheduler_gamma))
        scheduler_patience = int(getattr(config, "scheduler_patience", default_trainer_config.scheduler_patience))
        scheduler_min_lr = float(getattr(config, "scheduler_min_lr", default_trainer_config.scheduler_min_lr))

        early_stopping_patience = getattr(
            config, "early_stopping_patience", default_trainer_config.early_stopping_patience
        )
        if isinstance(early_stopping_patience, str) and early_stopping_patience.strip().lower() in {
            "none",
            "null",
        }:
            early_stopping_patience = None
        if early_stopping_patience is not None:
            early_stopping_patience = int(early_stopping_patience)

        # Step 3: Name the run
        run.name = _make_run_name(
            model_type=model_type,
            trainer_batch_size=trainer_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            hidden_units=hidden_units,
            vocab_size=int(vocab_size),
            embedding_dim=int(embedding_dim),
            num_heads=int(num_heads),
            num_filters=int(num_filters),
            filter_sizes=filter_sizes,
            rnn_hidden_size=int(rnn_hidden_size),
            rnn_num_layers=int(rnn_num_layers),
            bidirectional=bool(bidirectional),
            rnn_type=str(rnn_type),
            num_encoder_layers=int(num_encoder_layers),
            dim_feedforward=int(dim_feedforward),
            wandb_name_prefix=wandb_name_prefix,
        )
        print(f"Run name set to: {run.name}")

        # Step 4: DataLoaders
        # `make_train_sweep(..., datasets=...)` historically expects Dataset objects,
        # but some notebooks pass (train_loader, val_loader) instead.
        # Support both to keep sweeps robust.
        if isinstance(train_dataset, DataLoader) or isinstance(val_dataset, DataLoader):
            if not isinstance(train_dataset, DataLoader) or not isinstance(val_dataset, DataLoader):
                raise TypeError(
                    "make_train_sweep expected both entries in `datasets` to be the same type. "
                    "Pass either (train_dataset, val_dataset) or (train_loader, val_loader)."
                )
            train_loader, val_loader = train_dataset, val_dataset
        else:
            # Validate that we were given actual Dataset objects.
            # Raw tensors/arrays/dataframes are iterable and will *appear* to work,
            # but then DataLoader yields only inputs (a single Tensor) which breaks
            # Trainer's (inputs, targets) contract.
            if not isinstance(train_dataset, Dataset) or not isinstance(val_dataset, Dataset):
                raise TypeError(
                    "make_train_sweep expected `datasets` to be either (train_loader, val_loader) "
                    "or (train_dataset, val_dataset), where each dataset yields (X, y). "
                    f"Got types: train={type(train_dataset)!r}, val={type(val_dataset)!r}. "
                    "If you have raw tensors/arrays, wrap them as TensorDataset(X, y) or use a "
                    "custom Dataset (e.g., TimeSeriesDataset/TextDataset)."
                )

            collate_fn = None
            if model_type in ("bow", "textcnn", "text_rnn", "text_attn", "text_transformer"):
                # Apply max_seq_len to all text models — for text_attn it prevents OOM errors,
                # for others it is a no-op when max_seq_len=None (the default).
                collate_fn = partial(
                    text_collate_fn,
                    max_seq_len=max_seq_len,
                    padding_value=padding_idx,
                )
            train_loader, val_loader = get_dataloaders(
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                train_batch_size=trainer_batch_size,
                eval_batch_size=evaluator_batch_size,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        # Step 5: Config objects
        trainer_config = _build_trainer_config(
            trainer_batch_size=trainer_batch_size,
            evaluator_batch_size=evaluator_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            learning_rate=learning_rate,
            num_epochs=int(getattr(config, "num_epochs", 10)),
            device=device,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            momentum=momentum,
            use_scheduler=use_scheduler,
            scheduler_type=scheduler_type,
            scheduler_step_size=scheduler_step_size,
            scheduler_gamma=scheduler_gamma,
            scheduler_patience=scheduler_patience,
            scheduler_min_lr=scheduler_min_lr,
            early_stopping_patience=early_stopping_patience,
            metrics=metrics,
            verbose=verbose,
        )
        # Ensure task is set (TrainerConfig supports this; older configs may ignore it).
        try:
            setattr(trainer_config, "task", task)
        except Exception:
            pass

        supported_model_types = {"mlp", "cnn", "bow", "textcnn", "text_rnn", "text_attn", "text_transformer", "rnn"}
        if model_type not in supported_model_types:
            raise ValueError(
                f"Unknown model_type: {model_type!r}. Supported types: {sorted(supported_model_types)}"
            )

        # Validate input_spec/num_inputs only for model types that actually use it.
        model_input_spec = effective_input_spec
        if model_type in {"mlp", "cnn", "rnn"}:
            if model_input_spec is None:
                raise ValueError(
                    "make_train_sweep requires either input_spec (for CNN: [H,W] / (H,W), for MLP: int) "
                    "or num_inputs (legacy MLP/RNN input size)."
                )
        else:
            # For text models, build_model ignores input_spec but still requires a non-None argument.
            if model_input_spec is None:
                model_input_spec = 1

        conv_blocks = _parse_conv_blocks(config, model_type)
        model_config = _build_model_config(
            model_type=model_type,
            hidden_units=hidden_units,
            dropout=dropout,
            conv_blocks=conv_blocks,
            in_channels=in_channels,
            use_GAP=use_GAP,
            vocab_size=int(vocab_size),
            embedding_dim=int(embedding_dim),
            num_heads=int(num_heads),
            padding_idx=int(padding_idx),
            freeze_embeddings=bool(freeze_embeddings),
            max_seq_len=max_seq_len,
            num_filters=int(num_filters),
            filter_sizes=filter_sizes,
            rnn_hidden_size=int(rnn_hidden_size),
            rnn_num_layers=int(rnn_num_layers),
            rnn_dropout=float(rnn_dropout),
            bidirectional=bool(bidirectional),
            rnn_type=str(rnn_type),
            clip_grad_norm=float(clip_grad_norm),
            # --- TransformerClassifier support ---
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
        )

        # Step 6: Build model / optimizer / criterion
        model = build_model(
            input_spec=model_input_spec,
            num_outputs=int(num_outputs),
            config=model_config,
        )
        optimizer = make_optimizer(model.parameters(), config=trainer_config)
        criterion = _build_criterion(loss_name)

        # Step 7: Train (log to this W&B run)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            config=trainer_config,
            run=run,
        )
        # If verbose is disabled, suppress stdout during training to keep notebooks clean.
        if verbose:
            results = trainer.fit(train_loader, val_loader)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                results = trainer.fit(train_loader, val_loader)

        # Step 8: Clean up
        trainer.finish_run()
        msg = f"✓ Run complete! Final val_loss: {results['val_loss']:.4f}"
        if "val_accuracy" in results:
            msg += f", val_accuracy: {results['val_accuracy']*100:.2f}%"
        if "val_f1_macro" in results:
            msg += f", val_f1_macro: {results['val_f1_macro']:.4f}"
        if "val_mae" in results:
            msg += f", val_mae: {results['val_mae']:.4f}"
        if "val_mse" in results:
            msg += f", val_mse: {results['val_mse']:.4f}"
        if "val_r2" in results:
            msg += f", val_r2: {results['val_r2']:.4f}"
        if verbose:
            print(msg)

    return train_sweep


def train_sweep() -> None:
    """Deprecated: use `make_train_sweep(...)` and pass the returned callable to W&B."""
    raise RuntimeError("Use make_train_sweep(...) to construct a sweep training function.")
