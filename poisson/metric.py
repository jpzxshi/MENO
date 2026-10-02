"""Poisson scalar loss and metrics backed by the shared benchmark protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

try:
    from ..metrics_common import (
        SCALAR_EPSILON,
        ScalarFullCaseRelativeL2,
        ScalarRelativeMSESpec,
        summarize_scalar_full_case_metrics,
    )
except ImportError:
    from metrics_common import (
        SCALAR_EPSILON,
        ScalarFullCaseRelativeL2,
        ScalarRelativeMSESpec,
        summarize_scalar_full_case_metrics,
    )


_EPSILON = SCALAR_EPSILON


class RelativeMSESpec(ScalarRelativeMSESpec):
    """Poisson single-channel per-case relative-MSE training protocol."""


class FullCaseRelativeL2(ScalarFullCaseRelativeL2):
    """Accumulate Poisson full-case scalar error over query chunks."""


RELATIVE_MSE_LOSS = RelativeMSESpec()


def summarize_full_case_metrics(
    values: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    return summarize_scalar_full_case_metrics(values)


__all__ = [
    "FullCaseRelativeL2",
    "RELATIVE_MSE_LOSS",
    "RelativeMSESpec",
    "summarize_full_case_metrics",
]
