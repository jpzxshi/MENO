"""Shared loss and full-case metric implementations for formal benchmarks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch


SCALAR_EPSILON = 1.0e-30
_P_VECTOR_FIELDS = ("p", "vector", "magnitude")


def validate_p_vector_tensors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    output_name: str,
) -> None:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must have the same [B,Q,4] shape")
    if prediction.shape[-1] != 4:
        raise ValueError(f"{output_name} output must contain [P,Fx,Fy,Fz]")
    if prediction.shape[0] < 1 or prediction.shape[1] < 1:
        raise ValueError("prediction and target batch/query dimensions must be non-empty")


@dataclass(frozen=True)
class PVectorRelativeMSESpec:
    """Equal-weight relative-MSE over scalar P and a three-vector."""

    names: tuple[str, ...]
    groups: tuple[tuple[int, ...], ...]
    reduction: str = "mean"
    epsilon: float = 1.0e-30
    _output_name = "P-vector"

    def __post_init__(self) -> None:
        if len(self.names) != len(self.groups) or not self.groups:
            raise ValueError("loss names and groups must be non-empty and have equal length")
        if len(set(self.names)) != len(self.names):
            raise ValueError("loss names must be unique")
        if self.reduction not in ("mean", "sum"):
            raise ValueError("loss reduction must be 'mean' or 'sum'")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        validate_p_vector_tensors(
            prediction,
            target,
            output_name=self._output_name,
        )
        channels = prediction.shape[-1]
        difference = prediction.float() - target.float()
        target32 = target.float()
        per_case_relative_mse: list[torch.Tensor] = []
        for group in self.groups:
            index = list(group)
            if not index or min(index) < 0 or max(index) >= channels:
                raise ValueError(
                    f"loss group {group} is invalid for {channels} output channels"
                )
            numerator = difference[..., index].square().sum(dim=(1, 2))
            denominator = target32[..., index].square().sum(dim=(1, 2))
            per_case_relative_mse.append(
                numerator / denominator.clamp_min(self.epsilon)
            )

        group_means = torch.stack(per_case_relative_mse, dim=1).mean(dim=0)
        loss = group_means.mean() if self.reduction == "mean" else group_means.sum()
        metrics = {
            f"{name}_relative_mse": value
            for name, value in zip(self.names, group_means)
        }

        prediction64 = prediction.detach().double()
        target64 = target.detach().double()
        difference64 = prediction64 - target64
        p_relative_l2 = torch.sqrt(
            difference64[..., 0].square().sum(dim=1)
            / target64[..., 0].square().sum(dim=1).clamp_min(self.epsilon)
        )
        vector_relative_l2 = torch.sqrt(
            difference64[..., 1:].square().sum(dim=(1, 2))
            / target64[..., 1:].square().sum(dim=(1, 2)).clamp_min(self.epsilon)
        )
        prediction_magnitude = torch.linalg.vector_norm(prediction64[..., 1:], dim=2)
        target_magnitude = torch.linalg.vector_norm(target64[..., 1:], dim=2)
        magnitude_relative_l2 = torch.sqrt(
            (prediction_magnitude - target_magnitude).square().sum(dim=1)
            / target_magnitude.square().sum(dim=1).clamp_min(self.epsilon)
        )
        metrics.update(
            {
                "p_relative_l2": p_relative_l2.mean(),
                "vector_relative_l2": vector_relative_l2.mean(),
                "magnitude_relative_l2": magnitude_relative_l2.mean(),
            }
        )
        return loss, metrics


class PVectorFullCaseRelativeL2:
    """Accumulate P/vector/magnitude relative-L2 over query chunks."""

    _output_name = "P-vector"

    def __init__(self, batch_size: int, device: torch.device) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.epsilon = 1.0e-30
        self._updates = 0
        self._sums = {
            name: torch.zeros(self.batch_size, dtype=torch.float64, device=self.device)
            for name in (
                "p_num",
                "p_den",
                "vector_num",
                "vector_den",
                "magnitude_num",
                "magnitude_den",
            )
        }

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        validate_p_vector_tensors(
            prediction,
            target,
            output_name=self._output_name,
        )
        if prediction.shape[0] != self.batch_size:
            raise ValueError(
                f"prediction batch size {prediction.shape[0]} does not match "
                f"accumulator batch size {self.batch_size}"
            )
        if prediction.device != self.device or target.device != self.device:
            raise ValueError("prediction and target must be on the accumulator device")

        prediction64 = prediction.detach().double()
        target64 = target.detach().double()
        difference = (prediction.detach() - target.detach()).double()
        self._sums["p_num"] += difference[..., 0].square().sum(dim=1)
        self._sums["p_den"] += target64[..., 0].square().sum(dim=1)
        self._sums["vector_num"] += difference[..., 1:].square().sum(dim=(1, 2))
        self._sums["vector_den"] += target64[..., 1:].square().sum(dim=(1, 2))
        prediction_magnitude = torch.linalg.vector_norm(prediction64[..., 1:], dim=2)
        target_magnitude = torch.linalg.vector_norm(target64[..., 1:], dim=2)
        self._sums["magnitude_num"] += (
            (prediction_magnitude - target_magnitude).square().sum(dim=1)
        )
        self._sums["magnitude_den"] += target_magnitude.square().sum(dim=1)
        self._updates += 1

    def finalize(self) -> dict[str, torch.Tensor]:
        if self._updates == 0:
            raise RuntimeError("PVectorFullCaseRelativeL2 has not received any query chunks")
        return {
            field: torch.sqrt(
                self._sums[f"{field}_num"]
                / self._sums[f"{field}_den"].clamp_min(self.epsilon)
            )
            for field in _P_VECTOR_FIELDS
        }


def summarize_p_vector_full_case_metrics(
    values: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    summary: dict[str, float] = {}
    for field in _P_VECTOR_FIELDS:
        if field not in values:
            raise ValueError(f"full-case metrics are missing {field!r}")
        samples = np.asarray(values[field], dtype=np.float64)
        if samples.ndim != 1 or samples.size == 0:
            raise ValueError(f"full-case metric {field!r} must be a non-empty 1-D sequence")
        summary[f"case_{field}_relative_l2"] = float(np.mean(samples))
    return summary


def validate_scalar_fields(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    batch_size: int | None = None,
) -> None:
    if (
        prediction.shape != target.shape
        or prediction.ndim != 3
        or prediction.shape[-1] != 1
    ):
        raise ValueError("prediction and target must have the same [B,Q,1] shape")
    if prediction.shape[1] < 1:
        raise ValueError("prediction and target must contain at least one query point")
    if batch_size is not None and prediction.shape[0] != batch_size:
        raise ValueError(
            f"prediction batch={prediction.shape[0]}, expected={batch_size}"
        )


@dataclass(frozen=True)
class ScalarRelativeMSESpec:
    """Single-channel per-case relative-MSE training protocol."""

    epsilon: float = SCALAR_EPSILON

    def __post_init__(self) -> None:
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")

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
        relative_mse = per_case_mse.mean()
        difference64 = prediction.detach().double() - target.detach().double()
        per_case_l2 = torch.sqrt(
            difference64.square().sum(dim=(1, 2))
            / target.detach().double().square().sum(dim=(1, 2)).clamp_min(self.epsilon)
        )
        metrics = {
            "u_relative_mse": relative_mse,
            "u_relative_l2": per_case_l2.mean(),
        }
        return relative_mse, metrics


class ScalarFullCaseRelativeL2:
    """Accumulate scalar relative error over arbitrary query chunks."""

    def __init__(self, batch_size: int, device: torch.device) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self._numerator = torch.zeros(
            self.batch_size, dtype=torch.float64, device=self.device
        )
        self._denominator = torch.zeros_like(self._numerator)
        self._updates = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        validate_scalar_fields(prediction, target, batch_size=self.batch_size)
        if prediction.device != self.device or target.device != self.device:
            raise ValueError("prediction and target must be on the accumulator device")
        prediction64 = prediction.detach().double()
        target64 = target.detach().double()
        self._numerator += (prediction64 - target64).square().sum(dim=(1, 2))
        self._denominator += target64.square().sum(dim=(1, 2))
        self._updates += 1

    def finalize(self) -> dict[str, torch.Tensor]:
        if self._updates == 0:
            raise ValueError("ScalarFullCaseRelativeL2 has not received any query chunks")
        relative_mse = self._numerator / self._denominator.clamp_min(SCALAR_EPSILON)
        return {
            "relative_mse": relative_mse,
            "relative_l2": torch.sqrt(relative_mse),
        }


def summarize_scalar_full_case_metrics(
    values: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    missing = {"relative_mse", "relative_l2"} - values.keys()
    if missing:
        raise ValueError(f"full-case metrics is missing fields: {sorted(missing)}")
    relative_mse = np.asarray(values["relative_mse"], dtype=np.float64)
    relative_l2 = np.asarray(values["relative_l2"], dtype=np.float64)
    if (
        relative_mse.ndim != 1
        or relative_l2.ndim != 1
        or relative_mse.size == 0
        or relative_mse.shape != relative_l2.shape
    ):
        raise ValueError(
            "relative_mse and relative_l2 must be non-empty 1-D sequences "
            "with equal length"
        )
    return {
        "case_relative_mse": float(np.mean(relative_mse)),
        "case_relative_l2": float(np.mean(relative_l2)),
        "case_relative_l2_std": float(np.std(relative_l2)),
    }


__all__ = [
    "PVectorFullCaseRelativeL2",
    "PVectorRelativeMSESpec",
    "SCALAR_EPSILON",
    "ScalarFullCaseRelativeL2",
    "ScalarRelativeMSESpec",
    "summarize_p_vector_full_case_metrics",
    "summarize_scalar_full_case_metrics",
    "validate_p_vector_tensors",
    "validate_scalar_fields",
]
