from .estimands import ATE, TSM, AdditiveShift
from .tracer import LinearForm, Tracer, trace

__all__ = [
    "ATE",
    "AdditiveShift",
    "CrossFitResult",
    "Diagnostics",
    "LinearForm",
    "RieszBooster",
    "TSM",
    "Tracer",
    "build_augmented",
    "crossfit",
    "diagnose",
    "fit",
    "trace",
]


_LAZY = {
    "RieszBooster": ("engine", "RieszBooster"),
    "build_augmented": ("engine", "build_augmented"),
    "fit": ("engine", "fit"),
    "crossfit": ("crossfit", "crossfit"),
    "CrossFitResult": ("crossfit", "CrossFitResult"),
    "diagnose": ("diagnostics", "diagnose"),
    "Diagnostics": ("diagnostics", "Diagnostics"),
}


def __getattr__(name):
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        from importlib import import_module
        return getattr(import_module(f"{__name__}.{mod_name}"), attr)
    raise AttributeError(f"module 'rieszboost' has no attribute {name!r}")
