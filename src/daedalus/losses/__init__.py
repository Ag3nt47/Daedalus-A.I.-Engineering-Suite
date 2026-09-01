"""Differentiable training objectives."""

from .loss_functions import CrossEntropyLoss, MSELoss, Reduction

__all__ = ["CrossEntropyLoss", "MSELoss", "Reduction"]
