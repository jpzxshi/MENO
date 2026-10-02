"""Darcy scalar metrics and its benchmark relative-L2 training loss."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

try:
    from ..metrics_common import (
        SCALAR_EPSILON,
        ScalarFullCaseRelativeL2,
        ScalarRelativeMSESpec,
        summarize_scalar_full_case_metrics,
        validate_scalar_fields,
    )
except ImportError:
    from metrics_common import (
        SCALAR_EPSILON,
        ScalarFullCaseRelativeL2,
        ScalarRelativeMSESpec,
        summarize_scalar_full_case_metrics,
        validate_scalar_fields,
    )


class RelativeMSESpec(ScalarRelativeMSESpec):
    """Darcy single-channel per-case relative-MSE protocol."""


class FullCaseRelativeL2(ScalarFullCaseRelativeL2):
    """Accumulate Darcy scalar error over all query chunks of a case."""


RELATIVE_MSE_LOSS = RelativeMSESpec()


@dataclass(frozen=True)
class RelativeL2Spec:
    """Single-channel, per-case relative-L2 training protocol."""

    epsilon: float = SCALAR_EPSILON
    reduction: Literal["mean", "sum"] = "mean"

    def __post_init__(self) -> None:
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if self.reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        validate_scalar_fields(prediction, target)
        difference = prediction.float() - target.float()
        target32 = target.float()
        per_case_mse = difference.square().sum(dim=(1, 2)) / (
            target32.square().sum(dim=(1, 2)).clamp_min(self.epsilon)
        )
        per_case_l2 = torch.sqrt(per_case_mse.clamp_min(self.epsilon))
        loss = per_case_l2.mean() if self.reduction == "mean" else per_case_l2.sum()

        difference64 = prediction.detach().double() - target.detach().double()
        per_case_mse64 = difference64.square().sum(dim=(1, 2)) / (
            target.detach().double().square().sum(dim=(1, 2)).clamp_min(self.epsilon)
        )
        return loss, {
            "u_relative_mse": per_case_mse64.mean(),
            "u_relative_l2": torch.sqrt(per_case_mse64).mean(),
        }


RELATIVE_L2_LOSS = RelativeL2Spec()
RELATIVE_L2_SUM_LOSS = RelativeL2Spec(reduction="sum")


def summarize_full_case_metrics(
    values: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    return summarize_scalar_full_case_metrics(values)


__all__ = [
    "FullCaseRelativeL2",
    "RELATIVE_L2_LOSS",
    "RELATIVE_L2_SUM_LOSS",
    "RELATIVE_MSE_LOSS",
    "RelativeL2Spec",
    "RelativeMSESpec",
    "summarize_full_case_metrics",
]
