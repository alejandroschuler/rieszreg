"""Shared testing utilities: canonical DGPs, sklearn-conformance helpers,
reference-parity comparisons. Implementation packages import these and run
them against their concrete backend in CI."""

from . import conformance, dgps, parity

__all__ = ["conformance", "dgps", "parity"]
