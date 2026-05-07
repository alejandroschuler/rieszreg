"""Backend protocol and predictor-loader registry.

Concrete backends are provided by implementation packages (rieszboost, krrr).
"""

from .base import (
    Backend,
    FitResult,
    MomentBackend,
    Predictor,
    load_predictor,
    register_predictor_loader,
)

__all__ = [
    "Backend",
    "FitResult",
    "MomentBackend",
    "Predictor",
    "load_predictor",
    "register_predictor_loader",
]
