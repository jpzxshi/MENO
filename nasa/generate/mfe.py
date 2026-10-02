"""Minimal NASA-CRM Legendre surface-moment primitives."""

from __future__ import annotations

import h5py
import numpy as np


CONDITION_FIELDS = (
    "Mach",
    "AlphaMean",
    "aileronInboard",
    "aileronOutboard",
    "elevator",
    "htp",
)
COORDINATE_FIELDS = ("CoordinateX", "CoordinateY", "CoordinateZ")
NORMAL_FIELDS = ("NormalX", "NormalY", "NormalZ")
TARGET_FIELDS = ("PressureCoefficient", "cfx", "cfy", "cfz")
GEOMETRY_MOMENT_FIELDS = ("surface", "normal_x", "normal_y", "normal_z")


def normalized_legendre_vander(values: np.ndarray, mode: int) -> np.ndarray:
    """Evaluate the orthonormal Legendre basis through degree ``mode - 1``."""

    if mode < 1:
        raise ValueError("mode must be positive")
    vandermonde = np.polynomial.legendre.legvander(
        np.asarray(values, dtype=np.float64), mode - 1
    )
    degree = np.arange(mode, dtype=np.float64)
    return vandermonde * np.sqrt((2.0 * degree + 1.0) / 2.0)


def normalize_coordinates(
    coordinates: np.ndarray,
    coordinate_center: np.ndarray,
    coordinate_half_span: np.ndarray,
    *,
    clip: bool,
) -> np.ndarray:
    """Map physical coordinates with the canonical center and half-span."""

    center = np.asarray(coordinate_center, np.float64)
    half_span = np.asarray(coordinate_half_span, np.float64)
    if (
        center.shape != (3,)
        or half_span.shape != (3,)
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(half_span))
        or np.any(half_span <= 0.0)
    ):
        raise ValueError("coordinate transform must contain finite [3] vectors")
    values = (np.asarray(coordinates) - center) / half_span
    return np.clip(values, -1.0, 1.0) if clip else values


def sample_keys(handle: h5py.File, maximum: int | None = None) -> list[str]:
    keys = sorted(handle.keys())
    if maximum is not None:
        if maximum < 1:
            raise ValueError("sample limits must be positive")
        keys = keys[:maximum]
    return keys


def read_sample(group: h5py.Group) -> dict[str, np.ndarray]:
    """Read and validate the point fields needed by the historical contraction."""

    fields = (
        *COORDINATE_FIELDS,
        *NORMAL_FIELDS,
        "Surface",
        *TARGET_FIELDS,
    )
    data = {name: np.asarray(group[name], np.float64) for name in fields}
    if len({values.shape for values in data.values()}) != 1:
        raise ValueError(f"inconsistent point-array shapes in {group.name}")
    if not all(np.all(np.isfinite(values)) for values in data.values()):
        raise ValueError(f"{group.name} contains NaN/Inf")
    if np.any(data["Surface"] <= 0.0):
        raise ValueError(f"{group.name}/Surface must be positive")
    return data


def tensor_legendre_surface_moments(
    sample: dict[str, np.ndarray],
    mode: int,
    coordinate_center: np.ndarray,
    coordinate_half_span: np.ndarray,
    chunk_size: int,
    clip_coordinates: bool,
    normalize_area: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the historical eight-field contraction and accumulation order."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    number_of_points = sample["Surface"].shape[0]
    moments = np.zeros((8, mode, mode, mode), dtype=np.float64)
    area_scale = float(sample["Surface"].sum()) if normalize_area else 1.0
    for start in range(0, number_of_points, chunk_size):
        stop = min(start + chunk_size, number_of_points)
        coordinates = np.column_stack(
            [sample[name][start:stop] for name in COORDINATE_FIELDS]
        )
        coordinates = normalize_coordinates(
            coordinates,
            coordinate_center,
            coordinate_half_span,
            clip=clip_coordinates,
        )
        legendre_x = normalized_legendre_vander(coordinates[:, 0], mode)
        legendre_y = normalized_legendre_vander(coordinates[:, 1], mode)
        legendre_z = normalized_legendre_vander(coordinates[:, 2], mode)
        fields = np.column_stack(
            [
                np.ones(stop - start, dtype=np.float64),
                *(sample[name][start:stop] for name in NORMAL_FIELDS),
                *(sample[name][start:stop] for name in TARGET_FIELDS),
            ]
        )
        weighted_fields = fields * (sample["Surface"][start:stop] / area_scale)[:, None]
        moments += np.einsum(
            "na,nb,nc,nf->fabc",
            legendre_x,
            legendre_y,
            legendre_z,
            weighted_fields,
            optimize=True,
        )
    return moments[:4], moments[4:]


__all__ = [
    "CONDITION_FIELDS",
    "COORDINATE_FIELDS",
    "GEOMETRY_MOMENT_FIELDS",
    "NORMAL_FIELDS",
    "TARGET_FIELDS",
    "read_sample",
    "sample_keys",
    "tensor_legendre_surface_moments",
]
