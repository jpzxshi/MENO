"""NASA-CRM output loss and metrics backed by the shared P-vector protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

try:
    from ..metrics_common import (
        PVectorFullCaseRelativeL2,
        PVectorRelativeMSESpec,
        summarize_p_vector_full_case_metrics,
    )
except ImportError:
    from metrics_common import (
        PVectorFullCaseRelativeL2,
        PVectorRelativeMSESpec,
        summarize_p_vector_full_case_metrics,
    )


class RelativeMSESpec(PVectorRelativeMSESpec):
    _output_name = "NASA"


class FullCaseRelativeL2(PVectorFullCaseRelativeL2):
    _output_name = "NASA"


RELATIVE_MSE_LOSS = RelativeMSESpec(
    names=("p", "vector"), groups=((0,), (1, 2, 3)), reduction="mean"
)


def summarize_full_case_metrics(
    values: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    return summarize_p_vector_full_case_metrics(values)


__all__ = [
    "FullCaseRelativeL2",
    "RELATIVE_MSE_LOSS",
    "RelativeMSESpec",
    "summarize_full_case_metrics",
]
