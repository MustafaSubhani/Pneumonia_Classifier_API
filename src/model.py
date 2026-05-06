import logging
import os
import types

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


def get_resnet50() -> nn.Module:
    """Build ResNet50 with the project's custom FC head and layer1 frozen.

    FC head: Linear(2048, 512) -> ReLU -> BatchNorm1d(512) -> Dropout(0.5)
             -> Linear(512, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 2)
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    for param in model.layer1.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2),
    )
    return model


def load_model(config: types.ModuleType) -> nn.Module:
    """Load fine-tuned weights into the ResNet50 and return it in eval mode.

    Parameters
    ----------
    config:
        Project config module exposing MODEL_PATH, DEVICE, and MODEL_VERSION.

    Raises
    ------
    FileNotFoundError:
        If config.MODEL_PATH does not exist.
    RuntimeError:
        If the state-dict is incompatible with the architecture.
    """
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights not found at '{config.MODEL_PATH}'. "
            "Ensure the Models/ directory is present in the project root."
        )

    model = get_resnet50()

    state_dict = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.to(config.DEVICE)
    model.eval()

    logger.info("Model v%s loaded on device '%s'.", config.MODEL_VERSION, config.DEVICE)
    return model


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def frozen_layer_names(model: nn.Module) -> list[str]:
    """Return names of top-level child modules that have no trainable params."""
    return [
        name
        for name, module in model.named_children()
        if not any(p.requires_grad for p in module.parameters())
    ]
