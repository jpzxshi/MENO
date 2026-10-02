"""Small numerical helpers shared by the formal benchmark datasets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


COORDINATE_SCALE_SOURCES = ("prior", "train_bounds")
ESTIMATED_COORDINATE_MARGIN = 1.02


def derive_condition_moments(
    geometry_moments: np.ndarray,
    conditions: np.ndarray,
) -> np.ndarray:
    """Derive per-channel moments for sample-level constant conditions."""

    geometry_moments = np.asarray(geometry_moments, dtype=np.float32)
    conditions = np.asarray(conditions, dtype=np.float32)
    if geometry_moments.ndim < 2 or conditions.ndim != 1:
        raise ValueError("geometry_moments and conditions have invalid dimensions")
    return conditions.reshape((-1,) + (1,) * (geometry_moments.ndim - 1)) * (
        geometry_moments[0][None]
    )


def compose_function_scaling(
    condition_mean: np.ndarray,
    condition_std: np.ndarray,
    normal_channels: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose Local-function scaling while leaving unit normals unchanged."""

    if normal_channels < 0:
        raise ValueError("normal_channels must be non-negative")
    condition_mean = np.asarray(condition_mean, dtype=np.float32)
    condition_std = np.asarray(condition_std, dtype=np.float32)
    if condition_mean.ndim != 1 or condition_std.shape != condition_mean.shape:
        raise ValueError("condition_mean/std must be one-dimensional with equal shape")
    return (
        np.concatenate((np.zeros(normal_channels, np.float32), condition_mean)),
        np.concatenate((np.ones(normal_channels, np.float32), condition_std)),
    )


@dataclass
class ModelScaling:
    """Frozen training statistics and coordinate affine half-spans."""

    input_mean: np.ndarray
    input_std: np.ndarray
    function_mean: np.ndarray
    function_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    coordinate_scale: np.ndarray | None = None

    STATISTICAL_FIELDS = (
        "input_mean",
        "input_std",
        "function_mean",
        "function_std",
        "target_mean",
        "target_std",
    )

    @classmethod
    def identity(
        cls,
        mode: int,
        input_channels: int,
        output_channels: int,
        function_channels: int,
        dimension: int,
    ) -> "ModelScaling":
        if input_channels < 1 or output_channels < 1 or function_channels < 0:
            raise ValueError(
                "input/output channels must be positive and function_channels "
                "must be non-negative"
            )
        if dimension not in (2, 3):
            raise ValueError("dimension must be 2 or 3")
        tokens = mode**dimension
        return cls(
            input_mean=np.zeros((tokens, input_channels), np.float32),
            input_std=np.ones((tokens, input_channels), np.float32),
            target_mean=np.zeros(output_channels, np.float32),
            target_std=np.ones(output_channels, np.float32),
            function_mean=np.zeros(function_channels, np.float32),
            function_std=np.ones(function_channels, np.float32),
            coordinate_scale=np.ones(dimension, np.float32),
        )

    @classmethod
    def load(
        cls,
        filename: str,
        *,
        coordinate_scale: np.ndarray | None = None,
    ) -> "ModelScaling":
        """Load frozen statistics and attach the dataset coordinate half-span.

        Canonical data artifacts contain the six statistical fields; saved run
        scalings additionally contain ``coordinate_scale``.  A supplied dataset
        scale is checked against a stored run scale and is never guessed.
        """

        with np.load(filename, allow_pickle=False) as data:
            missing = [name for name in cls.STATISTICAL_FIELDS if name not in data]
            if missing:
                raise ValueError(f"scaling file is missing fields: {missing}")
            arrays = {
                name: np.asarray(data[name], np.float32).copy()
                for name in cls.STATISTICAL_FIELDS
            }
            stored_scale = (
                np.asarray(data["coordinate_scale"], np.float32).copy()
                if "coordinate_scale" in data
                else None
            )
        if coordinate_scale is not None:
            supplied = np.asarray(coordinate_scale, np.float32).copy()
            if stored_scale is not None and not np.array_equal(stored_scale, supplied):
                raise ValueError(
                    "stored coordinate_scale does not match the dataset transform: "
                    f"stored={stored_scale.tolist()}, supplied={supplied.tolist()}"
                )
            stored_scale = supplied
        return cls(**arrays, coordinate_scale=stored_scale)

    def save(self, filename: str) -> None:
        if self.coordinate_scale is None:
            raise ValueError("coordinate_scale is required before saving ModelScaling")
        np.savez(
            filename,
            **{
                name: np.asarray(getattr(self, name), np.float32)
                for name in self.STATISTICAL_FIELDS
            },
            coordinate_scale=np.asarray(self.coordinate_scale, np.float32),
        )


@dataclass(frozen=True)
class CoordinateTransform:
    """One dataset-wide affine map from physical coordinates to ``[-1, 1]``.

    ``raw_minimum`` and ``raw_maximum`` describe the union of complete raw
    training cases.  ``center`` and ``half_span`` define the actual map
    ``xi = (x - center) / half_span`` used by moment generation, Local queries,
    and physical MFE derivatives.  Estimated spans receive the fixed 2% safety
    margin; an explicit physical prior is used exactly as supplied.
    """

    center: np.ndarray
    half_span: np.ndarray
    raw_minimum: np.ndarray
    raw_maximum: np.ndarray
    source: Literal["prior", "train_bounds"]
    margin: float

    @property
    def minimum(self) -> np.ndarray:
        return self.center - self.half_span

    @property
    def maximum(self) -> np.ndarray:
        return self.center + self.half_span

    def normalize(self, coordinates: np.ndarray) -> np.ndarray:
        values = np.asarray(coordinates, dtype=np.float64)
        if values.shape[-1] != self.center.shape[0]:
            raise ValueError(
                "coordinate dimension mismatch: "
                f"actual={values.shape[-1]}, expected={self.center.shape[0]}"
            )
        return (values - self.center) / self.half_span

    def metadata(self) -> dict[str, object]:
        return {
            "source": self.source,
            "margin": self.margin,
            "center": self.center.tolist(),
            "half_span": self.half_span.tolist(),
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist(),
            "raw_training_minimum": self.raw_minimum.tolist(),
            "raw_training_maximum": self.raw_maximum.tolist(),
        }


def resolve_coordinate_transform(
    raw_minimum: np.ndarray | Sequence[float],
    raw_maximum: np.ndarray | Sequence[float],
    *,
    source: str,
    prior_half_span: float | Sequence[float] | np.ndarray | None,
    prior_center: Sequence[float] | np.ndarray | None = None,
) -> CoordinateTransform:
    """Resolve the single affine transform for one benchmark dataset.

    The raw bounds must come only from complete training cases.  ``prior``
    requires an explicit positive half-span and never applies a data-derived
    margin.  ``train_bounds`` rejects a stale prior and expands the empirical
    half-span around its midpoint by exactly 1.02.
    """

    minimum = np.asarray(raw_minimum, dtype=np.float64)
    maximum = np.asarray(raw_maximum, dtype=np.float64)
    if minimum.ndim != 1 or maximum.shape != minimum.shape or minimum.size < 1:
        raise ValueError("raw coordinate bounds must be matching non-empty vectors")
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise ValueError("raw coordinate bounds must be finite")
    if np.any(maximum <= minimum):
        raise ValueError("every raw coordinate span must be positive")
    if source not in COORDINATE_SCALE_SOURCES:
        raise ValueError(
            f"coordinate scale source={source!r}, expected {COORDINATE_SCALE_SOURCES}"
        )

    raw_center = 0.5 * (minimum + maximum)
    if source == "train_bounds":
        if prior_half_span is not None:
            raise ValueError("train_bounds coordinate scale must not receive a prior")
        center = raw_center
        half_span = (
            0.5 * (maximum - minimum) * ESTIMATED_COORDINATE_MARGIN
        )
        margin = ESTIMATED_COORDINATE_MARGIN
    else:
        if prior_half_span is None:
            raise ValueError("prior coordinate scale requires prior_half_span")
        supplied = np.asarray(prior_half_span, dtype=np.float64)
        if supplied.ndim == 0:
            supplied = np.full(minimum.shape, float(supplied), dtype=np.float64)
        if supplied.shape != minimum.shape:
            raise ValueError(
                "prior coordinate half-span shape mismatch: "
                f"actual={supplied.shape}, expected={minimum.shape}"
            )
        center = (
            raw_center
            if prior_center is None
            else np.asarray(prior_center, dtype=np.float64)
        )
        if center.shape != minimum.shape or not np.all(np.isfinite(center)):
            raise ValueError("prior coordinate center must be a finite matching vector")
        half_span = supplied
        margin = 1.0

    if not np.all(np.isfinite(half_span)) or np.any(half_span <= 0.0):
        raise ValueError("coordinate half-span must contain finite positive values")
    tolerance = 64.0 * np.finfo(np.float64).eps
    normalized_minimum = (minimum - center) / half_span
    normalized_maximum = (maximum - center) / half_span
    if np.any(normalized_minimum < -1.0 - tolerance) or np.any(
        normalized_maximum > 1.0 + tolerance
    ):
        raise ValueError(
            "resolved coordinate transform does not contain the raw training union: "
            f"normalized_min={normalized_minimum.tolist()}, "
            f"normalized_max={normalized_maximum.tolist()}"
        )
    return CoordinateTransform(
        center=np.ascontiguousarray(center),
        half_span=np.ascontiguousarray(half_span),
        raw_minimum=np.ascontiguousarray(minimum),
        raw_maximum=np.ascontiguousarray(maximum),
        source=source,
        margin=margin,
    )


def validate_stored_coordinate_transform(
    expected: CoordinateTransform,
    *,
    center: np.ndarray | Sequence[float],
    half_span: np.ndarray | Sequence[float],
    source: str,
    margin: float,
) -> None:
    """Reject a moment artifact generated under another coordinate policy."""

    stored_center = np.asarray(center, dtype=np.float64)
    stored_half_span = np.asarray(half_span, dtype=np.float64)
    mismatches: dict[str, object] = {}
    if source != expected.source:
        mismatches["source"] = {"stored": source, "expected": expected.source}
    if float(margin) != expected.margin:
        mismatches["margin"] = {"stored": float(margin), "expected": expected.margin}
    if stored_center.shape != expected.center.shape or not np.allclose(
        stored_center, expected.center, rtol=0.0, atol=1.0e-12
    ):
        mismatches["center"] = {
            "stored": stored_center.tolist(),
            "expected": expected.center.tolist(),
        }
    if stored_half_span.shape != expected.half_span.shape or not np.allclose(
        stored_half_span, expected.half_span, rtol=0.0, atol=1.0e-12
    ):
        mismatches["half_span"] = {
            "stored": stored_half_span.tolist(),
            "expected": expected.half_span.tolist(),
        }
    if mismatches:
        raise ValueError(
            "moment artifact coordinate transform does not match train.py; "
            f"regenerate data: {mismatches}"
        )


def validate_coordinate_transform_metadata(
    expected: CoordinateTransform,
    stored: Mapping[str, object],
    *,
    context: str,
) -> None:
    """Validate every serialized field of a dataset coordinate transform."""

    required = {
        "source",
        "margin",
        "center",
        "half_span",
        "minimum",
        "maximum",
        "raw_training_minimum",
        "raw_training_maximum",
    }
    if set(stored) != required:
        raise ValueError(
            f"{context} coordinate transform fields differ from the canonical schema"
        )
    validate_stored_coordinate_transform(
        expected,
        center=stored["center"],
        half_span=stored["half_span"],
        source=str(stored["source"]),
        margin=float(stored["margin"]),
    )
    expected_arrays = {
        "minimum": expected.minimum,
        "maximum": expected.maximum,
        "raw_training_minimum": expected.raw_minimum,
        "raw_training_maximum": expected.raw_maximum,
    }
    for name, wanted in expected_arrays.items():
        actual = np.asarray(stored[name], dtype=np.float64)
        if actual.shape != wanted.shape or not np.allclose(
            actual, wanted, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                f"{context} coordinate transform {name} does not match the "
                "resolved training transform"
            )


@dataclass(frozen=True)
class RaggedStatistics:
    """Population statistics for a sequence of variable-length cases.

    Every case receives one unit of probability mass, and points inside each
    case are treated uniformly.  This is the single canonical benchmark
    normalization protocol.
    """

    mean: np.ndarray
    std: np.ndarray
    case_count: int
    point_count: int


def stable_std(value: np.ndarray, epsilon: float = 1.0e-8) -> np.ndarray:
    """Return float64 scales, replacing values below ``epsilon`` with one."""

    value = np.asarray(value, dtype=np.float64)
    return np.where(value < epsilon, 1.0, value)


def ragged_channel_statistics(cases: Iterable[np.ndarray]) -> RaggedStatistics:
    """Compute finite float64 channel statistics without padding ragged cases.

    Every yielded case must be ``[N,C]`` with ``N > 0``.  Population variance
    (``ddof=0``) is used because normalization describes the complete frozen
    training population rather than an estimator of an unknown distribution.
    """

    first_moment: np.ndarray | None = None
    second_moment: np.ndarray | None = None
    case_count = 0
    point_count = 0

    for case_index, values in enumerate(cases):
        block = np.asarray(values, dtype=np.float64)
        if block.ndim != 2 or block.shape[0] < 1 or block.shape[1] < 1:
            raise ValueError(
                f"normalization case {case_index} shape={block.shape}, "
                "expected [N,C] with N,C > 0"
            )
        if not np.all(np.isfinite(block)):
            raise ValueError(f"normalization case {case_index} contains NaN/Inf")
        if first_moment is None:
            first_moment = np.zeros(block.shape[1], dtype=np.float64)
            second_moment = np.zeros_like(first_moment)
        elif block.shape[1] != first_moment.shape[0]:
            raise ValueError(
                f"normalization case {case_index} channels={block.shape[1]}, "
                f"expected={first_moment.shape[0]}"
            )

        assert second_moment is not None
        block_count = int(block.shape[0])
        block_first = block.mean(axis=0, dtype=np.float64)
        block_second = np.square(block).mean(axis=0, dtype=np.float64)
        case_count += 1
        first_moment += (block_first - first_moment) / case_count
        second_moment += (block_second - second_moment) / case_count
        point_count += block_count

    if first_moment is None or second_moment is None or case_count < 1:
        raise ValueError("normalization cases must not be empty")
    variance = np.maximum(second_moment - np.square(first_moment), 0.0)
    return RaggedStatistics(
        mean=first_moment,
        std=stable_std(np.sqrt(variance)),
        case_count=case_count,
        point_count=point_count,
    )


__all__ = [
    "COORDINATE_SCALE_SOURCES",
    "ESTIMATED_COORDINATE_MARGIN",
    "CoordinateTransform",
    "ModelScaling",
    "RaggedStatistics",
    "ragged_channel_statistics",
    "resolve_coordinate_transform",
    "stable_std",
    "validate_coordinate_transform_metadata",
    "validate_stored_coordinate_transform",
]
