"""RieszTreeRegressor — sklearn-compatible single-tree Riesz regressor.

Subclass of ``rieszreg.RieszEstimator`` defaulting to ``RieszTreeBackend``
and surfacing tree-specific hyperparameters on the constructor. Composes
with ``GridSearchCV``, ``cross_val_predict``, ``Pipeline``.
"""

from __future__ import annotations

from typing import Sequence

from rieszreg import Estimand, LossSpec, RieszEstimator, SquaredLoss

from .backend import RieszTreeBackend


class RieszTreeRegressor(RieszEstimator):
    """Single-tree Riesz regression.

    Fits the Riesz representer ``α₀`` of a linear functional by greedy splits
    on the augmented Bregman-Riesz loss. Each leaf stores the closed-form
    per-leaf optimum ``α_ℓ* = -C_ℓ / D_ℓ`` (loss-aware: projected to the
    loss's α-domain).

    Parameters
    ----------
    estimand : rieszreg.Estimand
        Carries ``feature_keys`` and the ``m(alpha)(z, y)`` operator.
    loss : rieszreg.LossSpec, default=None
        Bregman-Riesz loss. ``None`` resolves to ``SquaredLoss()``. Built-in
        support: ``SquaredLoss``, ``KLLoss``, ``BernoulliLoss``,
        ``BoundedSquaredLoss``.
    max_depth : int, default=8
        Maximum tree depth.
    min_samples_split : int, default=20
        Minimum count of original (D > 0) augmented rows in a node before
        considering a split.
    min_samples_leaf : int, default=10
        Minimum count of original rows in each child of a candidate split.
    max_leaves : int, default=31
        Cap for leafwise growth. Ignored when ``growth_policy="depthwise"``.
    growth_policy : {"depthwise", "leafwise"}, default="depthwise"
        Tree-growth strategy.
    pruning_alpha : float, default=0.0
        Cost-complexity penalty. ``0`` disables pruning.
    early_stopping_rounds : int or None, default=None
        Stop growing when held-out augmented loss has not improved for that
        many consecutive accepted splits. ``None`` disables.
    validation_fraction : float, default=0.1
        Held-out fraction the orchestrator splits off before augmentation
        when early stopping is enabled. Ignored when no holdout is needed.
    categorical_features : sequence of int, optional
        Column indices (into ``estimand.feature_keys``) treated as integer
        category labels; the splitter sorts levels by within-level α* and
        sweeps contiguous splits, per the standard CART convention.
    init : float or None
        α-space initialization, threaded through to ``RieszEstimator``.
    random_state : int, default=0
    """

    def __init__(
        self,
        estimand: Estimand,
        loss: LossSpec | None = None,
        max_depth: int = 8,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        max_leaves: int = 31,
        growth_policy: str = "depthwise",
        pruning_alpha: float = 0.0,
        early_stopping_rounds: int | None = None,
        validation_fraction: float = 0.1,
        categorical_features: Sequence[int] | None = None,
        init: float | None = None,
        random_state: int = 0,
    ):
        super().__init__(
            estimand=estimand,
            backend=None,
            loss=loss,
            init=init,
            random_state=random_state,
        )
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaves = max_leaves
        self.growth_policy = growth_policy
        self.pruning_alpha = pruning_alpha
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.categorical_features = categorical_features

    # ---- backend construction ----

    def _resolved_loss(self) -> LossSpec:
        return self.loss if self.loss is not None else SquaredLoss()

    def _resolved_backend(self) -> RieszTreeBackend:
        cat = (
            tuple(int(i) for i in self.categorical_features)
            if self.categorical_features is not None
            else ()
        )
        # Validation fraction is only consumed when a holdout is needed.
        val_frac = (
            float(self.validation_fraction)
            if self.early_stopping_rounds is not None
            else 0.0
        )
        return RieszTreeBackend(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaves=self.max_leaves,
            growth_policy=self.growth_policy,
            pruning_alpha=self.pruning_alpha,
            early_stopping_rounds=self.early_stopping_rounds,
            validation_fraction=val_frac,
            categorical_features=cat,
            random_state=self.random_state,
        )

    def fit(self, Z, y=None, eval_set=None, eval_y=None) -> "RieszTreeRegressor":
        """Fit the tree. After dispatch, we patch the predictor's
        ``feature_keys`` from the resolved estimand so save/load round-trips
        carry the column ordering."""
        super().fit(Z, y=y, eval_set=eval_set, eval_y=eval_y)
        self.predictor_.feature_keys = tuple(self.feature_keys_)
        return self

    # ---- save/load ----

    def _save_hyperparameters(self) -> dict:
        base = super()._save_hyperparameters()
        base.update(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaves=self.max_leaves,
            growth_policy=self.growth_policy,
            pruning_alpha=self.pruning_alpha,
            early_stopping_rounds=self.early_stopping_rounds,
            validation_fraction=self.validation_fraction,
            categorical_features=(
                list(int(i) for i in self.categorical_features)
                if self.categorical_features is not None
                else None
            ),
        )
        return base

    @classmethod
    def _construct_for_load(
        cls, *, estimand, loss, hyperparameters: dict
    ) -> "RieszTreeRegressor":
        cat = hyperparameters.get("categorical_features")
        return cls(
            estimand=estimand,
            loss=loss,
            max_depth=hyperparameters.get("max_depth", 8),
            min_samples_split=hyperparameters.get("min_samples_split", 20),
            min_samples_leaf=hyperparameters.get("min_samples_leaf", 10),
            max_leaves=hyperparameters.get("max_leaves", 31),
            growth_policy=hyperparameters.get("growth_policy", "depthwise"),
            pruning_alpha=hyperparameters.get("pruning_alpha", 0.0),
            early_stopping_rounds=hyperparameters.get("early_stopping_rounds"),
            validation_fraction=hyperparameters.get("validation_fraction", 0.1),
            categorical_features=tuple(int(i) for i in cat) if cat else None,
            init=hyperparameters.get("init"),
            random_state=hyperparameters.get("random_state", 0),
        )
