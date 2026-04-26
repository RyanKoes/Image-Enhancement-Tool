from dataclasses import asdict
import math

import torch
import torch.nn as nn
from .config import ModelConfig, ResidualBlockConfig


def _construct_fc_layers(
    *,
    start_layer_size: int,
    config: ModelConfig,
    num_outputs: int,
) -> nn.Sequential:
    """Construct a simple MLP head from `config.hidden_units` and `config.dropout`."""
    layers: list[nn.Module] = []
    prev_dim = int(start_layer_size)
    for hidden_dim, dropout_p in zip(config.hidden_units, config.dropout):
        hidden_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        if float(dropout_p) > 0.0:
            layers.append(nn.Dropout(float(dropout_p)))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, int(num_outputs)))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    """A basic residual block used by the course CNN scaffolding."""

    def __init__(self, in_channels: int, cfg: ResidualBlockConfig) -> None:
        super().__init__()
        out_channels = int(cfg.out_channels)
        k = int(cfg.kernel_size)
        p = int(cfg.padding)
        s = int(cfg.stride)

        layers: list[nn.Module] = []
        cur_in = int(in_channels)
        for i in range(max(1, int(getattr(cfg, "num_layers", 2)))):
            cur_stride = s if i == 0 else 1
            layers.append(nn.Conv2d(cur_in, out_channels, kernel_size=k, stride=cur_stride, padding=p))
            if bool(getattr(cfg, "batch_norm", False)):
                layers.append(nn.BatchNorm2d(out_channels))
            if i < max(1, int(getattr(cfg, "num_layers", 2))) - 1:
                layers.append(nn.ReLU())
            cur_in = out_channels
        self.net = nn.Sequential(*layers)

        self.proj: nn.Module | None = None
        if s != 1 or in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=s, padding=0)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.proj is None else self.proj(x)
        out = self.net(x)
        out = out + identity
        return self.act(out)


class EnhancementCNN(nn.Module):
    """A small residual CNN for image-to-image enhancement.

    This model maps an input image tensor to an output image tensor of the same
    shape (e.g., 3xHxW). It is intended as a simple baseline for denoising /
    deblurring / general enhancement tasks.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        out_channels: int = 3,
        features: int = 64,
        num_blocks: int = 8,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = bool(residual)

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        blocks: list[nn.Module] = []
        for _ in range(int(num_blocks)):
            blocks.append(_EnhanceResidualBlock(features))
        self.body = nn.Sequential(*blocks)

        self.tail = nn.Conv2d(features, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        feat = self.body(feat)
        out = self.tail(feat)
        return x + out if self.residual else out


class _EnhanceResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))

class CNN_Model(nn.Module):
    """Convolutional Neural Network following the [Conv2d -> ReLU -> MaxPool2d] x N motif.

    The model consists of:
      - A feature extractor: sequential conv blocks built from config.conv_blocks
      - A classifier head: Flatten -> Linear layers with dropout

    The flattened feature dimension is computed automatically via a dummy forward pass,
    so the model adapts to any input spatial size without manual calculation.
    """

    def __init__(self,
        input_height: int,
        input_width: int,
        num_outputs: int,
        config: ModelConfig) -> None:
        super().__init__()

        if config.model_type != "cnn":
            raise ValueError(f"Invalid model_type: {config.model_type}. Expected 'cnn'.")

        self.input_height = input_height
        self.input_width = input_width
        self.num_outputs = num_outputs
        self.config = config

        # --- Build the feature extractor ---
        # DONE: Build self.feature_extractor as an nn.Sequential.
        # Your list of layers in self.feature_extractor is determined by your config.conv_blocks.
        # Loop through config.conv_blocks. For each ConvBlockConfig, append:
        #   nn.Conv2d(parameters from block configg)
        #   nn.ReLU()
        #   nn.MaxPool2d(pool_size from block config) <-- only if block.pool_size > 0
        # Track current_in_channels: starts at config.in_channels, then becomes block.out_channels.
        conv_layers = []
        current_in_channels = config.in_channels
        for block_config in config.conv_blocks:
            # DONE: If block_config is a ResidualBlockConfig, append a ResidualBlock
            if isinstance(block_config, ResidualBlockConfig):
                conv_layers.append(ResidualBlock(current_in_channels, block_config))
            else:
                conv_layers.append(
                    nn.Conv2d(
                        current_in_channels,
                        block_config.out_channels,
                        block_config.kernel_size,
                        stride=block_config.stride,
                        padding=block_config.padding,
                    )
                )
                if block_config.batch_norm:
                    conv_layers.append(nn.BatchNorm2d(block_config.out_channels))
                conv_layers.append(nn.ReLU())
                if block_config.pool_size > 0:
                    conv_layers.append(nn.MaxPool2d(block_config.pool_size))
            current_in_channels = block_config.out_channels
        self.feature_extractor = nn.Sequential(*conv_layers)

        # --- Compute flattened feature dimension via dummy forward pass ---
        # DONE: Create a dummy tensor of shape (1, in_channels, input_height, input_width),
        # pass it through self.feature_extractor, and store the total number of elements
        # as self._flat_features.
        # Inside __init__, after building self.feature_extractor:
        if config.use_GAP:
            # Add Global Average Pooling (GAP)
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self._flat_features = current_in_channels
        else:
            self.gap = None
            # --- Compute flattened feature dimension via dummy forward pass ---
            with torch.no_grad():
                self.feature_extractor.eval()
                dummy = torch.zeros(1, config.in_channels, self.input_height, self.input_width)
                dummy_out = self.feature_extractor(dummy)
                self._flat_features = dummy_out.numel()
                self.feature_extractor.train()

        # --- Build the classifier head ---
        # DONE: Build self.classifier_head as an nn.Sequential.
        # First layer: nn.Linear(self._flat_features, config.hidden_units[0])
        # Then for each subsequent hidden unit / dropout pair: Linear -> ReLU -> Dropout
        # Final layer: nn.Linear(last_hidden, num_outputs)
        # (Follow the same pattern as MLP_Model for the linear layers.)
        classifier_layers = []
        prev_dim = self._flat_features
        for hidden_dim, dropout_p in zip(self.config.hidden_units, self.config.dropout):
            classifier_layers.append(nn.Linear(prev_dim, hidden_dim))
            classifier_layers.append(nn.ReLU())
            if dropout_p > 0.0:
                classifier_layers.append(nn.Dropout(dropout_p))
            prev_dim = hidden_dim
        classifier_layers.append(nn.Linear(prev_dim, self.num_outputs))

        self.classifier_head = _construct_fc_layers(
            start_layer_size=self._flat_features, 
            config=config, 
            num_outputs=num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_outputs).
        Notes:
            - Passes input through the feature extractor (conv blocks).
            - If GAP is enabled, applies global average pooling and squeezes spatial dimensions.
              Otherwise, flattens the feature maps into a 1D vector for the classifier head.
            - Finally, passes through the classifier head to get the output logits.
        """
        x = self.feature_extractor(x)

        # If GAP is enabled, apply global average pooling and squeeze spatial dimensions;
        # otherwise, flatten the feature maps into a 1D vector for the classifier head.
        if self.gap is not None:
            x = self.gap(x)                # (batch, channels, 1, 1)
            x = x.squeeze(-1).squeeze(-1)  # (batch, channels)
        else:
            x = torch.flatten(x, start_dim=1)

        x = self.classifier_head(x)
        return x

    def num_parameters(self) -> tuple[int, int]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

    def get_architecture_config(self) -> dict:
        """
        Get a dictionary representation of the model's architecture configuration, including the conv_blocks.
        Returns:
            dict: A dictionary containing the model's architecture configuration, including:
                - model_type: The type of model (e.g., "cnn").
                - input_height: The height of the input images.
                - input_width: The width of the input images.
                - num_outputs: The number of output classes or targets.
                - config: A dictionary representation of the ModelConfig used to build the model, with conv_blocks serialized to include block type information.
        Notes:
            - This method is useful for logging, checkpointing, or any scenario where you want to save or inspect the model's architecture configuration, especially when conv_blocks can contain different block types (ConvBlockConfig vs ResidualBlockConfig).
            - It converts the ModelConfig dataclass to a dictionary using asdict(), but for conv_blocks, it creates a custom serialization that includes the block type ("conv" or "residual") along with the block's parameters.   
        """

        def _serialize_config() -> dict:
            """
            Helper function to serialize the ModelConfig, especially the conv_blocks which can contain different block types.
            """
            config_dict = asdict(self.config)  # still use asdict for everything else
            config_dict['conv_blocks'] = [
                {'block_type': 'residual', **asdict(b)} if isinstance(b, ResidualBlockConfig)
                else {'block_type': 'conv', **asdict(b)}
                for b in self.config.conv_blocks
            ]
            return config_dict

        return {
            'model_type': self.config.model_type,
            'input_height': self.input_height,
            'input_width': self.input_width,
            'num_outputs': self.num_outputs,
            'config': _serialize_config(),
        }

    def __str__(self) -> str:
        """Return a readable architecture summary for this CNN.

        The summary includes:
        - `input_shape` (in_channels, height, width)
        - `conv_blocks`: a per-block description derived from `config.conv_blocks`, which may
          contain either `ConvBlockConfig` entries (standard conv blocks) or `ResidualBlockConfig`
          entries (residual blocks). For each block, the string includes the inferred
          `in_channels` (based on the previous block) plus the config fields relevant to that
          block type.
        - `classifier_head`: the Linear/ReLU/Dropout stack derived from `hidden_units` + `dropout`
        - Total and trainable parameter counts

        Returns:
            str: Human-readable summary of model architecture and size.
        """
        # DONE: Finish me!
        conv_block_strs: list[str] = []
        current_in_channels = self.config.in_channels
        for block in self.config.conv_blocks:
            if isinstance(block, ResidualBlockConfig):
                conv_block_strs.append(
                    "ResidualBlock("
                    f"in_channels={current_in_channels}, out_channels={block.out_channels}, stride={block.stride}"
                    ")"
                )
            else:
                conv_block_strs.append(
                    "ConvBlock("
                    f"in_channels={current_in_channels}, out_channels={block.out_channels}, "
                    f"kernel_size={block.kernel_size}, stride={block.stride}, padding={block.padding}, "
                    f"pool_size={block.pool_size}, batch_norm={block.batch_norm}"
                    ")"
                )
            current_in_channels = block.out_channels

        conv_blocks_summary = "\n    ".join(conv_block_strs) if conv_block_strs else "<none>"
        classifier_summary = []
        prev_dim = self._flat_features
        for hidden_dim, dropout_p in zip(self.config.hidden_units, self.config.dropout):
            classifier_summary.append(f"Linear({prev_dim} -> {hidden_dim})")
            classifier_summary.append(f"ReLU()")
            if dropout_p > 0.0:
                classifier_summary.append(f"Dropout(p={dropout_p})")
            prev_dim = hidden_dim
        classifier_summary.append(f"Linear({prev_dim} -> {self.num_outputs})")
        classifier_summary_str = "\n    ".join(classifier_summary)
        total_params, trainable_params = self.num_parameters()
        return (f"CNN_Model(\n"
                f"  input_shape=({self.config.in_channels}, {self.input_height}, {self.input_width}),\n"
                f"  conv_blocks=[\n    {conv_blocks_summary}\n  ],\n"
                f"  classifier_head=[\n    {classifier_summary_str}\n  ],\n"
                f"  total_params={total_params}, trainable_params={trainable_params}\n"
                f")")

    def __repr__(self) -> str:
        return self.__str__()