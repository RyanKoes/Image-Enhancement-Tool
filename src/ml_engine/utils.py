
"""
utils.py

This module contains a collection of helper utility functions that will be used throughout the course.
You will find reusable functions for metrics, data handling, and other tools to support labs, 
assignments, and projects.

Course: CSCI 357 - AI and Neural Networks
Author: Ryan Koes
Date: 02/18/2026

"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Iterable, List, Optional, Union
from .config import (
    ModelConfig,
    TrainerConfig,
    ConvBlockConfig,
    ResidualBlockConfig,
)
from .model import MLP_Model, CNN_Model, BagOfEmbeddings, RNNModel, TextCNN1D, TextRNNModel, AttentionClassifier, TransformerClassifier

def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate the accuracy given model logits and true labels.

    Args:
        logits (torch.Tensor): The raw output from the model (before softmax), of shape (batch_size, num_classes).
        labels (torch.Tensor): The true class labels, of shape (batch_size,).
    Returns:
        float: The accuracy as a value between 0 and 1.
    Notes:
        - The function computes the predicted class by taking the argmax of the logits along the class dimension.
        - It then compares the predicted classes to the true labels and calculates the proportion of correct predictions.
    """
    # DONE: Implement the accuracy_from_logits function.
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total


def build_model(
    input_spec: Union[int, List[int]],
    num_outputs: int,
    config: ModelConfig,
) -> nn.Module:
    """
    Factory function to build a model based on the provided configuration.
    Args:
        input_spec: Specification of the input size. For MLP, this is an int (flattened input size). For CNN, this is a tuple/list (height, width).
        num_outputs: The number of output classes or targets.
        config: ModelConfig object containing model architecture settings.

        - For "mlp": int, total number of input features (flattened)
        - For "cnn": (height, width) tuple or list specifying the 2D input image shape (excluding channels)
        - For "bow": int, total number of input features (flattened)
        - For "textcnn": (num_filters, filter_sizes) tuple or list specifying the 1D input sequence shape
        - For "rnn": int > 0, number of features per time step
        - For "text_rnn": unused (vocab_size/embedding_dim/etc. live in config)

    Returns:
        An instance of nn.Module representing the constructed model.
    Raises:
        ValueError: If config.model_type is unrecognized or if input_spec is invalid for the specified model type.
    Notes:
        - The function checks the model_type in the config and constructs either an MLP_Model or a CNN_Model accordingly.
        - For MLP, it expects input_spec to be an integer representing the size of the flattened input. For CNN, it expects input_spec to be a tuple or list containing the height and width of the input images.
    """
    if config.model_type == "mlp":
        if not isinstance(input_spec, int):
            raise ValueError("MLP requires input_spec as int (flattened input size).")
        return MLP_Model(num_inputs=input_spec, num_outputs=num_outputs, config=config)
    elif config.model_type == "cnn":
        if not isinstance(input_spec, (tuple, list)) or len(input_spec) != 2:
            raise ValueError("CNN requires input_spec as (height, width).")
        h, w = input_spec[0], input_spec[1]
        return CNN_Model(input_height=h, input_width=w, num_outputs=num_outputs, config=config)
    elif config.model_type == "bow":
        # DONE: Instantiate and return a BagOfEmbeddings model.
        #       input_spec is unused here; vocab_size lives in config.
        model = BagOfEmbeddings(num_outputs=num_outputs, config=config)
        return model
    elif config.model_type == "textcnn":
        return TextCNN1D(
            num_outputs=num_outputs,
            config=config,
        )
    elif config.model_type == "text_rnn":
        # input_spec is unused here; vocab_size / embedding_dim live in config.
        return TextRNNModel(num_outputs=num_outputs, config=config)
    elif config.model_type == "rnn":
        if not isinstance(input_spec, int) or input_spec <= 0:
            raise ValueError("RNN requires input_spec as int > 0 (input_size = features per time step).")
        return RNNModel(input_size=input_spec, num_outputs=num_outputs, config=config)
    elif config.model_type == "text_attn":
        return AttentionClassifier(num_outputs=num_outputs, config=config)
    elif config.model_type == "text_transformer":
        return TransformerClassifier(num_outputs=num_outputs, config=config)
    else:
        raise ValueError(
            f"Model type '{config.model_type}' not supported. Supported types: 'mlp', 'cnn', 'bow', 'textcnn', 'text_rnn', 'rnn'"
        )


def make_optimizer(
    params: Union[nn.Module, Iterable[torch.nn.Parameter]],
    config: TrainerConfig,
) -> torch.optim.Optimizer:
    # DONE: Implement the make_optimizer function.
    """
    Create an optimizer for the given model based on the provided configuration.

    Args:
        params (Union[nn.Module, Iterable[torch.nn.Parameter]]): Either a model (nn.Module)
            or an iterable of parameters (e.g., model.parameters()).
        config (TrainerConfig): Configuration object containing optimizer parameters such as learning_rate and weight_decay.

    Returns:
        torch.optim.Optimizer: An instance of a PyTorch optimizer configured according to the provided TrainerConfig.
    Notes:
        - The function should create an optimizer (e.g., Adam) that optimizes the parameters of the given model.
        - The learning rate and weight decay should be set according to the values specified in the TrainerConfig.
    """
    if isinstance(params, nn.Module):
        params = params.parameters()
    lr = config.learning_rate
    weight_decay = config.weight_decay
    momentum = config.momentum
    optimizer_name = config.optimizer_name.lower()

    if optimizer_name in {"sgd", "momentum"}:
        # Momentum is controlled via config.momentum (set to 0.0 for vanilla SGD)
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if optimizer_name == "adam":
        # DONE: Return Adam (adaptive learning rates per parameter)
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: torch.device = torch.device('cpu')
) -> nn.Module:
    """Reconstructs any model from a checkpoint file.

    This factory function inspects the checkpoint's model_architecture to
    determine the model type, then dispatches to the appropriate constructor.

    NOTE: This ONLY restores the model architecture and weights, not optimizer state or other metadata.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load onto (default: CPU)

    Returns:
        Reconstructed model with loaded weights

    Raises:
        ValueError: If model_type in checkpoint is unrecognized
        FileNotFoundError: if the checkpoint file does not exist
        KeyError: if the checkpoint is missing the model_architecture metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_architecture' not in checkpoint:
        raise KeyError("Checkpoint is missing 'model_architecture' metadata.")
    
    arch = checkpoint['model_architecture']
    model_type = arch.get("model_type")

    if model_type is None:
        model_class = arch.get("model_class")
        dict_class_to_type = {
            "MLP_Model": "mlp",
            "CNN_Model": "cnn",
            "BagOfEmbeddings": "bow",
            "TextCNN1D": "textcnn",
            "TextRNNModel": "text_rnn",
            "RNNModel": "rnn",
            "AttentionClassifier": "text_attn",
            "TransformerClassifier": "text_transformer",
        }
        model_type = dict_class_to_type.get(model_class)

    if model_type is None:
        raise ValueError(
            "Could not determine model type from checkpoint architecture metadata"
        )

    config = _rebuild_model_config(arch["config"])

    if model_type == "mlp":
        model = MLP_Model(
            num_inputs=arch["num_inputs"],
            num_outputs=arch["num_outputs"],
            config=config,
        )
    elif model_type == "cnn":
        model = CNN_Model(
            input_height=arch["input_height"],
            input_width=arch["input_width"],
            num_outputs=arch["num_outputs"],
            config=config,
        )
    elif model_type == "bow":
        model = BagOfEmbeddings(
            num_outputs=arch["num_outputs"],
            config=config,
        )
    elif model_type == "textcnn":
        model = TextCNN1D(
            num_outputs=arch["num_outputs"],
            config=config,
        )
    elif model_type == "text_rnn":
        model = TextRNNModel(
            num_outputs=arch["num_outputs"],
            config=config,
        )
    elif model_type == "rnn":
        model = RNNModel(
            input_size=arch["input_size"],
            num_outputs=arch["num_outputs"],
            config=config,
        )
    elif model_type == "text_attn":
        model = AttentionClassifier(
            num_outputs=arch["num_outputs"],
            config=config,
        )
    elif model_type == "text_transformer":
        model = TransformerClassifier(
            num_outputs=arch["num_outputs"],
            config=config,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model
    
def _rebuild_model_config(config_dict: dict) -> ModelConfig:
    """Rehydrate ModelConfig from checkpoint-safe dictionaries."""
    cfg = dict(config_dict)

    if "conv_blocks" in cfg and isinstance(cfg["conv_blocks"], list):
        rebuilt_blocks = []
        for block in cfg["conv_blocks"]:
            if isinstance(block, dict):
                block_type = block.get("block_type", "conv")
                block_payload = {k: v for k, v in block.items() if k != "block_type"}
                if block_type == "residual":
                    rebuilt_blocks.append(ResidualBlockConfig(**block_payload))
                else:
                    rebuilt_blocks.append(ConvBlockConfig(**block_payload))
            else:
                rebuilt_blocks.append(block)
        cfg["conv_blocks"] = rebuilt_blocks

    return ModelConfig(**cfg)

def make_lr_scheduler(optimizer: torch.optim.Optimizer, config: TrainerConfig) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Factory for learning rate schedulers.
    
    Args:
        optimizer: The optimizer to schedule
        config: Configuration containing scheduler settings
        
    Returns:
        Scheduler instance, or None if use_scheduler is False
        
    Raises:
        ValueError: If scheduler_type is unrecognized
    """
    # If scheduling is disabled, return None
    if not config.use_scheduler:
        return None
    
    # If scheduling is enabled, create the appropriate scheduler based on config.scheduler_type
    if config.scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma
        )
    elif config.scheduler_type == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler_gamma,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr
        )
    # Add more schedulers as needed...
    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")
    
def lr_range_test(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device = torch.device("cpu"),
    start_lr: float = 1e-6,
    end_lr: float = 1.0,
    num_iterations: int = 100,
) -> tuple[list[float], list[float]]:
    """Performs Leslie Smith's LR Range Test.

    Trains the model for num_iterations mini-batches, exponentially increasing the
    learning rate from start_lr to end_lr. Records the loss at each step.

    WARNING: This function modifies the model weights and optimizer state.
    You should create a fresh model/optimizer before calling this, or save and
    restore a checkpoint afterward.

    Args:
        model: The model to test.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer (will have its LR modified).
        device: Device to train on.
        start_lr: Starting learning rate.
        end_lr: Ending learning rate.
        num_iterations: Number of mini-batches to train.

    Returns:
        Tuple of (lrs, losses) -- lists of learning rates and corresponding losses.
    """
    # Implement the LR range test.
    # 1. Set the optimizer's LR to start_lr
    # 2. Compute the multiplicative factor: gamma = (end_lr / start_lr) ** (1 / num_iterations)
    # 3. Create a LambdaLR scheduler that multiplies LR by gamma each step:
    #    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: gamma ** step)
    # 4. Loop for num_iterations batches (cycle through train_loader if needed):
    #    a. Get the next batch, move to device
    #    b. Forward pass, compute loss
    #    c. Record current LR and loss
    #    d. Backward pass, optimizer step, scheduler step
    #    e. If loss > 4 * best_loss_so_far, stop early (diverging)
    # 5. Return (lrs, losses)

    # Move the model to the specified device and set it to training mode
    model = model.to(device)
    model.train()

    # Set the optimizer's learning rate to the starting value
    for pg in optimizer.param_groups:
        pg["lr"] = start_lr

    # Calculate the multiplicative factor for exponentially increasing the LR
    gamma = (end_lr / start_lr) ** (1 / num_iterations)

    # Create a LambdaLR scheduler that multiplies the LR by gamma each step
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: gamma ** step
    )

    lrs: list[float] = []     # Store the learning rates for each batch
    losses: list[float] = []  # Store the losses for each batch
    best_loss = float("inf")  # Track the smallest loss encountered
    loader_iter = iter(train_loader)  # Iterator for cycling through the train loader

    for _ in range(num_iterations):
        try:
            # Get the next batch from the data loader
            inputs, targets = next(loader_iter)
        except StopIteration:
            # Restart the iterator if we run out of data
            loader_iter = iter(train_loader)
            inputs, targets = next(loader_iter)

        # Move the data to the correct device
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero out the previous gradients
        optimizer.zero_grad()

        # Forward pass: compute outputs and loss
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss_val = loss.item()
        curr_lr = optimizer.param_groups[0]["lr"]

        # Record the current learning rate and loss
        lrs.append(curr_lr)
        losses.append(loss_val)

        # Update the best loss seen so far, and early stop if loss diverges
        if loss_val < best_loss:
            best_loss = loss_val
        elif loss_val > 4 * best_loss:
            break

        # Backward pass and optimizer/scheduler step
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Return recorded learning rates and corresponding losses
    return lrs, losses

def compute_saliency_map(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int = None,
    device: torch.device = None,
) -> tuple[np.ndarray, int]:
    """Compute a saliency map for a single image.

    The saliency map highlights which pixels most influence the model's
    prediction by computing |d(score_c) / d(input)|.

    Args:
        model: Trained model (will be set to eval mode).
        image_tensor: Single image tensor of shape (C, H, W). Do NOT include batch dim.
        target_class: Class index to compute saliency for.
                      If None, uses the model's predicted class (argmax).
        device: Device for computation. If None, uses the model's device.

    Returns:
        Tuple of (saliency_map as 2D numpy array normalized to [0, 1],
                  target_class index used).
    """
    # DONE: Set the model to evaluation mode for correct inference
    model.eval()

    # DONE: If device is not specified, use the model's device for computation
    if device is None:
        device = next(model.parameters()).device

    # DONE: Prepare the input image
    #   - Add batch dimension to image_tensor
    #   - Move image to the correct device
    #   - Ensure image.requires_grad is True
    image = image_tensor.unsqueeze(0).to(device)
    image.requires_grad_()

    model.zero_grad(set_to_none=True)

    # DONE: Forward pass through the model to get the output logits
    # (saliency requires gradients; enable them even if the caller used no_grad)
    with torch.enable_grad():
        logits = model(image)

    # DONE: Pick the target class to compute the saliency map for
    #   - If target_class is None, use model's predicted class (argmax)
    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    # DONE: Compute the score for the selected class
    #   - Backpropagate the gradient from this score
    score = logits[0, target_class]
    score.backward()

    # DONE: Extract the gradient (saliency) from image.grad
    #   - Take the absolute value
    saliency = image.grad.abs().squeeze()

    # DONE: For color images (multiple channels), take the max across channels
    if saliency.dim() == 3:
        saliency = saliency.max(dim=0)[0]

    # DONE: Normalize the saliency map to [0, 1]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    # DONE: Return the normalized saliency map and the target_class used
    return saliency.detach().cpu().numpy(), target_class

def denormalize_image(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    Denormalizes a tensor image using the given mean and standard deviation values.

    Args:
        tensor (torch.Tensor): The image tensor to be denormalized. Expected shape: (C, H, W).
        mean (tuple, optional): Mean values for each channel. Default is (0.5, 0.5, 0.5).
        std (tuple, optional): Standard deviation values for each channel. Default is (0.5, 0.5, 0.5).

    Returns:
        torch.Tensor: The denormalized image tensor, with values clipped to [0, 1].
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input must be a torch.Tensor")
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("Input tensor must have shape (3, H, W)")

    mean = torch.tensor(mean, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(3, 1, 1)

    denormalized = tensor * std + mean
    denormalized = torch.clamp(denormalized, 0.0, 1.0)
    return denormalized