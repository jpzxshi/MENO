"""Legendre MFE encoding used to build Darcy moment artifacts from raw meshes."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import legvander


def standard_legendre(x: np.ndarray, mode: int) -> np.ndarray:
    """Return the orthonormal Legendre basis on ``[-1, 1]``."""

    x = np.asarray(x, dtype=np.float64)
    if mode < 1:
        raise ValueError("mode must be positive")
    tolerance = 2.0e-10
    if np.any(x < -1.0 - tolerance) or np.any(x > 1.0 + tolerance):
        raise ValueError("Darcy MFE coordinates must lie in [-1, 1]")
    return legvander(x, mode - 1) * np.sqrt(
        (2.0 * np.arange(mode, dtype=np.float64) + 1.0) / 2.0
    )


def encode_domain_and_permeability(
    physical_nodes: np.ndarray,
    normalized_nodes: np.ndarray,
    triangles: np.ndarray,
    permeability: np.ndarray,
    mode: int,
) -> np.ndarray:
    """Encode ``[domain, permeability]`` as ``[mode**2, 2]`` moments."""

    physical_nodes = np.asarray(physical_nodes, dtype=np.float64)
    normalized_nodes = np.asarray(normalized_nodes, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    permeability = np.asarray(permeability, dtype=np.float64)
    if mode < 1:
        raise ValueError("mode must be positive")
    if physical_nodes.ndim != 2 or physical_nodes.shape[1] != 2:
        raise ValueError(f"physical_nodes={physical_nodes.shape}, expected [N,2]")
    if normalized_nodes.shape != physical_nodes.shape:
        raise ValueError("normalized_nodes must match physical_nodes")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
        raise ValueError(f"triangles={triangles.shape}, expected [T,3]")
    if permeability.shape != (physical_nodes.shape[0],):
        raise ValueError("permeability must have one value per node")
    if triangles.min() < 0 or triangles.max() >= physical_nodes.shape[0]:
        raise ValueError("triangle index is out of bounds")
    if not (
        np.all(np.isfinite(physical_nodes))
        and np.all(np.isfinite(normalized_nodes))
        and np.all(np.isfinite(permeability))
    ):
        raise ValueError("raw Darcy arrays contain NaN/Inf")
    tolerance = 2.0e-10
    if np.any(normalized_nodes < -1.0 - tolerance) or np.any(
        normalized_nodes > 1.0 + tolerance
    ):
        raise ValueError("normalized Darcy coordinates must lie in [-1,1]^2")

    physical_vertices = physical_nodes[triangles]
    edge_1 = physical_vertices[:, 1] - physical_vertices[:, 0]
    edge_2 = physical_vertices[:, 2] - physical_vertices[:, 0]
    areas = 0.5 * np.abs(
        edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0]
    )
    if np.any(areas <= 0.0) or not np.all(np.isfinite(areas)):
        raise ValueError("raw Darcy mesh contains a degenerate triangle")

    barycentric = np.asarray(
        (
            (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
            (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
            (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
        ),
        dtype=np.float64,
    )
    normalized_vertices = normalized_nodes[triangles]
    quadrature = np.tensordot(
        barycentric, normalized_vertices, axes=([1], [1])
    ).transpose(1, 0, 2)
    phi_x = standard_legendre(quadrature[..., 0], mode)
    phi_y = standard_legendre(quadrature[..., 1], mode)
    quadrature_weights = areas[:, None] / 3.0
    domain = np.einsum(
        "tqi,tqj,tq->ij", phi_x, phi_y, quadrature_weights, optimize=True
    )
    permeability_quadrature = np.dot(permeability[triangles], barycentric.T)
    coefficient = np.einsum(
        "tqi,tqj,tq,tq->ij",
        phi_x,
        phi_y,
        permeability_quadrature,
        quadrature_weights,
        optimize=True,
    )
    encoded = np.stack(
        (domain.reshape(-1, order="C"), coefficient.reshape(-1, order="C")),
        axis=-1,
    )
    if not np.all(np.isfinite(encoded)):
        raise ValueError("Darcy moments contain NaN/Inf")
    return np.ascontiguousarray(encoded, dtype=np.float64)


__all__ = ["encode_domain_and_permeability", "standard_legendre"]
