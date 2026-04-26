"""
config.py

This module contains configuration dataclasses useful for AI and ML projects.

Course: CSCI 357 - AI and Neural Networks
Author: Ryan Koes
Date: 02/18/2026

"""

from __future__ import annotations

from dataclasses import dataclass, field
import torch
from typing import List, Optional, Union


@dataclass
class ResidualBlockConfig:
    """Configuration for a simple 2D residual block.

    This is primarily used by the course CNN scaffolding.
    """

    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    num_layers: int = 2
    batch_norm: bool = False


@dataclass
class MetricsConfig:
    """Configuration for metric computation and interpretation.

    Attributes:
        task: What kind of ML task this run represents.
            Supported: "multiclass", "binary", "regression".

        metrics: Which metric names to compute/log.
            "loss" is always computed by the Trainer.

            Classification metrics:
              - "acc"
              - "f1_macro"

            Regression metrics:
              - "mae"
              - "mse"
              - "r2"

        average: Only relevant for multiclass classification metrics that require an
            averaging strategy (e.g., macro-F1).
    """

    task: str = "multiclass"
    metrics: List[str] = field(default_factory=lambda: ["loss", "acc"])
    average: str = "macro"

    # Alias for assignment/notebook compatibility.
    # Some code refers to the metric list as "names" instead of "metrics".
    names: Optional[List[str]] = None

    def __post_init__(self) -> None:
        # If `names` is provided, treat it as the source of truth.
        if self.names is not None:
            self.metrics = list(self.names)

@dataclass
class TrainerConfig:
    trainer_batch_size: int = 64
    evaluator_batch_size: int = 256
    learning_rate: float = 0.001
    device: torch.device = torch.device("cpu")
    num_epochs: int = 10
    weight_decay: float = 0.0
    early_stopping_patience: Optional[int] = 5          # Set to None to disable early stopping
    early_stopping_min_delta: float = 0.001
    optimizer_name: str = "adam"  # Options: "adam", "sgd", etc.
    momentum: float = 0.9  # Only used if optimizer_name is "sgd"
    # Checkpointing Settings
    checkpoint_dir: str = "./checkpoints"               # Directory where checkpoints will be saved
    checkpoint_last_filename: str = "last.pt"           # Filename for most recent checkpoint
    checkpoint_save_interval: int = 5                   # Save checkpoint every N epochs
    checkpoint_best_filename: str = "best_model.pt"     # Filename for the best model checkpoint
    # Learning Rate Scheduler Settings
    use_scheduler: bool = False                                # Enable/disable scheduling
    scheduler_type: str = "reduce_on_plateau"                  # Options: "step", "exponential", "cosine", "reduce_on_plateau"
    scheduler_step_size: int = 10                              # For StepLR: epochs between LR drops
    scheduler_gamma: float = 0.1                               # Factor to reduce LR
    scheduler_patience: int = 3                                # For ReduceLROnPlateau: epochs to wait before reducing
    scheduler_min_lr: float = 1e-6                             # Minimum learning rate (prevents it from going too low)

    # Metric configuration.
    # Preferred: use `metrics_config`.
    # Backwards-compatible: if `metrics_config` is None, Trainer falls back to
    # `task` and `metrics` below.
    metrics_config: Optional[MetricsConfig] = None

    # Legacy metric fields (kept for older notebooks and sweep code).
    # Supported tasks: "multiclass", "binary", "regression".
    task: str = "multiclass"
    # Supported metrics: "loss", "acc", "f1_macro", "mae", "mse", "r2".
    metrics: List[str] = field(default_factory=lambda: ["loss", "acc"])

    num_workers: int = 2
    pin_memory: bool = True
    verbose: bool = True

@dataclass
class ConvBlockConfig:
    """Configuration for a single [Conv2d -> ReLU -> MaxPool2d] block.

    Attributes:
        out_channels: Number of filters (output feature maps) for this block.
        kernel_size: Spatial size of the convolution kernel.
        stride: Stride of the convolution.
        padding: Zero-padding added to both sides of the input.
        pool_size: Kernel size for MaxPool2d. Set to 0 to skip pooling.
    """
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    pool_size: int = 2
    batch_norm: bool = False

@dataclass
class ModelConfig:
    """
    Configuration class for defining machine learning model parameters.

    Attributes:
        model_type (str): Type of neural network model to use. Defaults to "mlp" (Multi-Layer Perceptron).
        hidden_units (List[int]): Number of units in each hidden layer. Defaults to [128, 64],
            representing a two-layer network with 128 units in the first hidden layer and 64 in the second.
        dropout (List[float]): Dropout rates for regularization applied to each layer. Defaults to [0.1, 0.2],
            applying 10% dropout to the first layer and 20% to the second layer.

    Note:
        Lambda functions are used in the `default_factory` parameter because:
        - Dataclass fields with mutable default values (lists, dicts) require `default_factory`
          to create a new instance for each object initialization, preventing unintended sharing
          of mutable objects across instances.
        - Using `lambda: [128, 64]` ensures each ModelConfig instance gets its own independent
          list copy, avoiding bugs where modifying one instance's hidden_units would affect all instances.
        - This is a common pattern in Python dataclasses to safely handle mutable defaults.

    model_type: Which model to build. Supported values:
        - "mlp"     — MLP_Model (fully connected classifier/regressor)
        - "cnn"     — CNN_Model (2D convolutional network for images)
        - "bow"     — BagOfEmbeddings (embedding + mean pooling + classifier for text)
        - "textcnn" — TextCNN1D (1D convolutional text classifier with multiple filter sizes)
        - "rnn"     — RNNModel (vanilla RNN, LSTM or GRU)
        - "text_attn" — TextAttentionModel (RNN + attention for text classification)

    num_heads: Number of attention heads (used by text_attn). Must satisfy embed_dim % num_heads == 0.

    --- Transformer encoder fields ---
    num_encoder_layers: Number of stacked encoder layers (used by text_transformer).
    dim_feedforward: FFN hidden dimension (used by text_transformer). Typically 4 * embedding_dim.
    """
    model_type: str = "mlp"
    hidden_units: List[int] = field(default_factory=lambda: [128, 64])
    dropout: List[float] = field(default_factory=lambda: [0.1, 0.2])

    conv_blocks: List[Union[ConvBlockConfig, ResidualBlockConfig]] = field(default_factory=list)
    in_channels: int = 1   # 1 for grayscale, 3 for RGB

    use_GAP: bool = False

    vocab_size: int = 0
    embedding_dim: int = 100
    padding_idx: int = 0
    freeze_embeddings: bool = False

    filter_sizes: List[int] = field(default_factory=lambda: [3, 4, 5]) # The sizes of the filters to use in the TextCNN1D
    num_filters: int = 100 # The number of filters to use in the TextCNN1D model.

    # --- RNN fields ---
    rnn_hidden_size: int = 64           # Hidden state dimensionality for RNN layers
    rnn_num_layers: int = 1             # Number of stacked RNN layers
    bidirectional: bool = False         # If True, use bidirectional RNN
    rnn_type: str = "rnn"              # "rnn" for vanilla RNN, "lstm" for LSTM, "gru" for GRU
    rnn_dropout: float = 0.0            # Inter-layer dropout for stacked RNN/LSTM/GRU (applies when rnn_num_layers > 1)
    clip_grad_norm: float = 0.0        # Max gradient norm for clipping (0 = disabled)

    num_heads: int = 4             # Number of attention heads (only relevant for transformer-based models)
    max_seq_len: Optional[int] = None   # Truncate input sequences to at most this many tokens.
                                    # None = no truncation. Critical for text_attn, whose
                                    # attention score matrix scales as O(batch * L^2).

    # --- Transformer encoder fields ---
    num_encoder_layers: int = 2      # Number of stacked TransformerEncoderLayers
    dim_feedforward: int = 512       # Hidden dimension in the FFN sublayer (typically 4 * embedding_dim)