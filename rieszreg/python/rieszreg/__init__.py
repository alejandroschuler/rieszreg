"""rieszreg: shared abstractions for the Riesz-regression package family.

Implementation packages (rieszboost, krrr, ...) depend on rieszreg for the
estimand machinery, Bregman-Riesz losses, augmentation engine, Backend
Protocol, base diagnostics, and the sklearn-compatible `RieszEstimator`
orchestrator.

    from rieszreg import RieszEstimator, ATE, SquaredLoss
    from rieszboost.backends import XGBoostBackend
    est = RieszEstimator(estimand=ATE(), loss=SquaredLoss(), backend=XGBoostBackend())
    est.fit(X)
"""

from .augmentation import (
    AugmentedDataset,
    aug_grad_eta,
    aug_hess_eta,
    aug_loss_alpha,
    aug_loss_eta,
    build_augmented,
)
from .backends import (
    Backend,
    FitResult,
    MomentBackend,
    Predictor,
    load_predictor,
    register_predictor_loader,
)
from .diagnostics import Diagnostics, diagnose
from .estimands import (
    ATE,
    ATT,
    AdditiveShift,
    Estimand,
    LinearForm,
    LinearFormEstimand,
    LocalShift,
    StochasticIntervention,
    TSM,
    Tracer,
    estimand_from_spec,
    trace,
)
from .estimator import RieszEstimator
from .losses import (
    BernoulliLoss,
    BoundedSquaredLoss,
    KLLoss,
    Loss,
    LossSpec,
    SquaredLoss,
    loss_from_spec,
)
from .scoring import riesz_scorer

__all__ = [
    "ATE",
    "ATT",
    "AdditiveShift",
    "AugmentedDataset",
    "Backend",
    "aug_grad_eta",
    "aug_hess_eta",
    "aug_loss_alpha",
    "aug_loss_eta",
    "BernoulliLoss",
    "BoundedSquaredLoss",
    "Diagnostics",
    "Estimand",
    "FitResult",
    "KLLoss",
    "LinearForm",
    "LinearFormEstimand",
    "LocalShift",
    "Loss",
    "LossSpec",
    "MomentBackend",
    "Predictor",
    "RieszEstimator",
    "SquaredLoss",
    "StochasticIntervention",
    "TSM",
    "Tracer",
    "build_augmented",
    "diagnose",
    "estimand_from_spec",
    "load_predictor",
    "loss_from_spec",
    "register_predictor_loader",
    "riesz_scorer",
    "trace",
]
