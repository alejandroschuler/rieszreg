"""AugForestRieszBackend satisfies Backend; AugForestPredictor exposes the loader."""

from __future__ import annotations

from forestriesz import AugForestPredictor, AugForestRieszBackend


def test_aug_backend_exposes_fit_augmented_only():
    backend = AugForestRieszBackend()
    assert callable(getattr(backend, "fit_augmented", None))
    # Augmentation-style: must NOT advertise fit_rows.
    assert not hasattr(backend, "fit_rows")


def test_aug_predictor_protocol_surface():
    assert AugForestPredictor.kind == "aug-forestriesz"
    for attr in ("predict_eta", "predict_alpha", "save"):
        assert callable(getattr(AugForestPredictor, attr)), attr


def test_aug_predictor_loader_registered():
    from rieszreg.backends.base import _PREDICTOR_LOADERS

    assert "aug-forestriesz" in _PREDICTOR_LOADERS
