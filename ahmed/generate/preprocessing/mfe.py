"""Legendre moment operators for AhmedML preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


class Sliceable(Protocol):
    """Array-like object that supports shape inspection and slicing."""

    shape: tuple[int, ...]

    def __getitem__(self, key: object) -> np.ndarray: ...


@dataclass(frozen=True)
class CoordinateBounds:
    """Axis-aligned bounds used for affine coordinate normalization."""

    minimum: np.ndarray
    maximum: np.ndarray

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum, dtype=np.float64)
        maximum = np.asarray(self.maximum, dtype=np.float64)
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("coordinate bounds must both have shape [3]")
        if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
            raise ValueError("coordinate bounds must be finite")
        if np.any(maximum <= minimum):
            raise ValueError("every coordinate span must be positive")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def normalize(self, coordinates: np.ndarray, clip: bool = False) -> np.ndarray:
        """Map physical coordinates to the Legendre domain ``[-1, 1]^3``."""
        coordinates = np.asarray(coordinates, dtype=np.float64)
        if coordinates.shape[-1] != 3:
            raise ValueError("coordinates must have last dimension 3")
        normalized = (
            2.0 * (coordinates - self.minimum) / (self.maximum - self.minimum) - 1.0
        )
        return np.clip(normalized, -1.0, 1.0) if clip else normalized


def normalized_legendre_vander(x: np.ndarray, mode: int) -> np.ndarray:
    """Return an orthonormal Legendre Vandermonde matrix."""
    if mode < 1:
        raise ValueError("mode must be positive")
    x = np.asarray(x, dtype=np.float64)
    values = np.polynomial.legendre.legvander(x, mode - 1)
    degree = np.arange(mode, dtype=np.float64)
    return values * np.sqrt((2.0 * degree + 1.0) / 2.0)


def tensor_legendre_moments(
    coordinates: Sliceable,
    weights: Sliceable,
    fields: Sequence[Sliceable | None],
    bounds: CoordinateBounds,
    mode: int = 8,
    chunk_size: int = 65_536,
    normalize_measure: bool = False,
    clip_coordinates: bool = False,
) -> np.ndarray:
    """Integrate scalar channels against a tensor-product Legendre basis."""
    if mode < 1:
        raise ValueError("mode must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if len(coordinates.shape) != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape [N,3]")
    number_of_points = int(coordinates.shape[0])
    if weights.shape[0] != number_of_points:
        raise ValueError("weights and coordinates have different lengths")
    if not fields:
        raise ValueError("at least one field channel is required")
    for field in fields:
        if field is not None and field.shape[0] != number_of_points:
            raise ValueError("a field and coordinates have different lengths")

    measure = 1.0
    if normalize_measure:
        measure = 0.0
        for start in range(0, number_of_points, chunk_size):
            stop = min(start + chunk_size, number_of_points)
            measure += np.asarray(weights[start:stop], dtype=np.float64).sum()
        if not np.isfinite(measure) or measure <= 0.0:
            raise ValueError("the total integration measure must be positive")

    moments = np.zeros((len(fields), mode, mode, mode), dtype=np.float64)
    for start in range(0, number_of_points, chunk_size):
        stop = min(start + chunk_size, number_of_points)
        xyz = bounds.normalize(coordinates[start:stop], clip=clip_coordinates)
        lx = normalized_legendre_vander(xyz[:, 0], mode)
        ly = normalized_legendre_vander(xyz[:, 1], mode)
        lz = normalized_legendre_vander(xyz[:, 2], mode)

        field_block = np.empty((stop - start, len(fields)), dtype=np.float64)
        for channel, field in enumerate(fields):
            if field is None:
                field_block[:, channel] = 1.0
            else:
                values = np.asarray(field[start:stop], dtype=np.float64)
                if values.ndim != 1:
                    raise ValueError("each field channel must be scalar with shape [N]")
                field_block[:, channel] = values
        weight_block = np.asarray(weights[start:stop], dtype=np.float64)
        if np.any(weight_block < 0.0) or not np.all(np.isfinite(weight_block)):
            raise ValueError("integration weights must be finite and non-negative")
        weighted_fields = field_block * (weight_block / measure)[:, None]
        moments += np.einsum(
            "na,nb,nc,nf->fabc",
            lx,
            ly,
            lz,
            weighted_fields,
            optimize=True,
        )
    return moments


def fourier_encode_coordinates(
    normalized_coordinates: np.ndarray,
    frequencies: int = 8,
    include_coordinates: bool = True,
) -> np.ndarray:
    """Encode normalized 3-D coordinates with dyadic Fourier features."""
    if frequencies < 0:
        raise ValueError("frequencies must be non-negative")
    xyz = np.asarray(normalized_coordinates, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("normalized_coordinates must have shape [N,3]")
    parts: list[np.ndarray] = [xyz] if include_coordinates else []
    for exponent in range(frequencies):
        phase = np.float32(np.pi * (2**exponent)) * xyz
        parts.extend((np.sin(phase), np.cos(phase)))
    if not parts:
        return np.empty((xyz.shape[0], 0), dtype=np.float32)
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def surface_local_features(
    coordinates: np.ndarray,
    normals: np.ndarray,
    bounds: CoordinateBounds,
    frequencies: int = 8,
    clip_coordinates: bool = False,
) -> np.ndarray:
    """Build local surface features from encoded coordinates and normals."""
    normals = np.asarray(normals, dtype=np.float32)
    if normals.shape != np.asarray(coordinates).shape:
        raise ValueError("normals and coordinates must both have shape [N,3]")
    xyz = bounds.normalize(coordinates, clip=clip_coordinates).astype(np.float32)
    encoded = fourier_encode_coordinates(xyz, frequencies=frequencies)
    return np.concatenate((encoded, normals), axis=1)


def volume_local_features(
    coordinates: np.ndarray,
    bounds: CoordinateBounds,
    frequencies: int = 8,
    clip_coordinates: bool = False,
) -> np.ndarray:
    """Build local volume features from encoded coordinates."""
    xyz = bounds.normalize(coordinates, clip=clip_coordinates).astype(np.float32)
    return fourier_encode_coordinates(xyz, frequencies=frequencies)
