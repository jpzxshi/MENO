"""Canonical Poisson native data contract, raw access, and normalization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Iterator

import numpy as np
from numpy.polynomial.legendre import legvander

try:
    from ..dataset_common import (
        CoordinateTransform,
        ModelScaling,
        ragged_channel_statistics,
        resolve_coordinate_transform,
        stable_std,
        validate_coordinate_transform_metadata,
    )
except ImportError:  # ``poisson`` imported as a top-level package.
    from dataset_common import (
        CoordinateTransform,
        ModelScaling,
        ragged_channel_statistics,
        resolve_coordinate_transform,
        stable_std,
        validate_coordinate_transform_metadata,
    )


RAW_DATA_FORMAT = "mfd-poisson-raw-npy-v1"
MOMENT_DATA_FORMAT = "mfd-poisson-moment"
DATA_FORMAT = "mfd-poisson-native"
NORMALIZATION_DATA_FORMAT = "mfd-poisson-normalization"
GLOBAL_MOMENT_CHANNELS = ("domain", "diffusion", "source", "boundary")
GLOBAL_MOMENT_CHANNEL_INDEX = {
    name: index for index, name in enumerate(GLOBAL_MOMENT_CHANNELS)
}
# Historical public name retained for manifest/generator compatibility.
INPUT_CHANNELS = GLOBAL_MOMENT_CHANNELS
LOCAL_FUNCTION_CHANNELS = ("f", "k", "g_reconstruction")
NORMALIZATION_NAME = "canonical"
NORMALIZATION_PROTOCOL = {"weighting": "uniform_per_case", "sampling": "full_raw"}
_RAW_FIELDS = ("points", "vertex", "line", "triangle", "k", "f", "g", "u")
SCALING_NAMES = (
    "input_mean",
    "input_std",
    "function_mean",
    "function_std",
    "target_mean",
    "target_std",
)
_COORDINATE_TRANSFORM_FIELDS = {
    "source",
    "margin",
    "center",
    "half_span",
    "minimum",
    "maximum",
    "raw_training_minimum",
    "raw_training_maximum",
}


def _sha256_file(filename: Path) -> str:
    digest = hashlib.sha256()
    with open(filename, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json(filename: Path) -> dict[str, Any]:
    if not filename.is_file():
        raise FileNotFoundError(f"missing Poisson manifest: {filename}")
    payload = json.loads(filename.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must contain a JSON object: {filename}")
    return payload


def _coordinate_transform_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = manifest.get("coordinate_transform")
    if not isinstance(payload, dict) or set(payload) != _COORDINATE_TRANSFORM_FIELDS:
        raise ValueError(
            "Poisson moment coordinate_transform schema mismatch; regenerate data"
        )
    for name in (
        "center",
        "half_span",
        "minimum",
        "maximum",
        "raw_training_minimum",
        "raw_training_maximum",
    ):
        values = np.asarray(payload[name], dtype=np.float64)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Poisson coordinate_transform.{name} must be finite [2]")
    if payload["source"] not in ("prior", "train_bounds"):
        raise ValueError("Poisson coordinate_transform.source is invalid")
    if not np.isfinite(float(payload["margin"])):
        raise ValueError("Poisson coordinate_transform.margin must be finite")
    return payload


def validate_coordinate_transform(
    manifest: Mapping[str, object],
    coordinate_transform: CoordinateTransform,
) -> None:
    """Reject a Local/scaling transform that differs from stored moments."""

    moment = manifest.get("moment")
    if not isinstance(moment, dict):
        raise ValueError("Poisson manifest lacks moment metadata")
    validate_coordinate_transform_metadata(
        coordinate_transform,
        _coordinate_transform_metadata(moment),
        context="Poisson",
    )


def _validate_raw_manifest(
    root: Path,
    manifest: dict[str, Any],
    *,
    verify_payload_hashes: bool,
) -> None:
    if manifest.get("format") != RAW_DATA_FORMAT:
        raise ValueError(
            f"Poisson raw format={manifest.get('format')!r}, "
            f"expected={RAW_DATA_FORMAT!r}"
        )
    if manifest.get("coordinate_domain") != [0.0, 1.0]:
        raise ValueError("Poisson raw coordinate_domain must be [0,1]")
    generator = manifest.get("generator")
    expected_generator = {
        "seed": 42,
        "mesh_size": 0.01,
        "radius_features": 1001,
        "field_features": 100,
        "samples_per_family": 5000,
    }
    if not isinstance(generator, dict) or any(
        generator.get(name) != value for name, value in expected_generator.items()
    ):
        raise ValueError("Poisson raw generator protocol mismatch")
    if generator.get("reference") != "poisson/generate/raw.py":
        raise ValueError("Poisson raw generator reference is not canonical")
    families = manifest.get("families")
    if not isinstance(families, dict) or tuple(families) != ("star", "annular"):
        raise ValueError("Poisson raw families must be ordered as star, annular")
    for family in ("star", "annular"):
        entry = families[family]
        if not isinstance(entry, dict) or int(entry.get("samples", -1)) != 5000:
            raise ValueError(f"invalid Poisson raw family metadata: {family}")
        cases = entry.get("cases")
        if not isinstance(cases, list) or len(cases) != 5000:
            raise ValueError(f"Poisson raw {family} must contain 5000 case entries")
        family_root = root / str(entry.get("directory", family))
        for raw_index, case in enumerate(cases):
            if not isinstance(case, dict) or int(case.get("raw_index", -1)) != raw_index:
                raise ValueError(f"Poisson raw {family} case ordering mismatch")
            filename = family_root / str(case.get("file"))
            if not filename.is_file():
                raise FileNotFoundError(f"missing Poisson raw case: {filename}")
            if filename.stat().st_size != int(case.get("bytes", -1)):
                raise ValueError(f"Poisson raw case size mismatch: {filename}")
            expected_hash = case.get("sha256")
            if not _is_sha256(expected_hash):
                raise ValueError(f"invalid Poisson raw SHA256 metadata: {family}/{raw_index}")
            if verify_payload_hashes and _sha256_file(filename) != expected_hash:
                raise ValueError(f"Poisson raw case SHA256 mismatch: {filename}")
    for split, expected in (("train", 9000), ("test", 1000)):
        entry = manifest.get("splits", {}).get(split)
        if not isinstance(entry, dict) or int(entry.get("samples", -1)) != expected:
            raise ValueError(f"invalid Poisson raw split metadata: {split}")
        blocks = entry.get("families")
        if not isinstance(blocks, list) or len(blocks) != 2:
            raise ValueError(
                f"Poisson raw split {split} must contain two family blocks"
            )
        count = 0
        for block in blocks:
            if not isinstance(block, dict) or block.get("family") not in families:
                raise ValueError(f"invalid family block in Poisson {split} split")
            start = int(block.get("start", -1))
            stop = int(block.get("stop", -1))
            if start < 0 or stop <= start or stop > 5000:
                raise ValueError(f"invalid raw range in Poisson {split} split: {block}")
            count += stop - start
        if count != expected:
            raise ValueError(
                f"Poisson {split} split ranges contain {count} cases, expected={expected}"
            )


def _validate_moment_manifest(
    root: Path, manifest: dict[str, Any], raw_manifest_file: Path
) -> None:
    if manifest.get("format") != MOMENT_DATA_FORMAT:
        raise ValueError(
            f"Poisson moment format={manifest.get('format')!r}, "
            f"expected={MOMENT_DATA_FORMAT!r}"
        )
    mode = int(manifest.get("mode", -1))
    if mode < 1:
        raise ValueError("Poisson moment mode must be positive")
    expected = {
        "raw_format": RAW_DATA_FORMAT,
        "dimension": 2,
        "basis": "orthonormal_tensor_legendre_2d",
        "basis_domain": [-1.0, 1.0],
        "flatten_order": "C: m=i*mode+j",
        "input_channels": list(INPUT_CHANNELS),
        "normalization": NORMALIZATION_PROTOCOL,
    }
    mismatches = {
        name: {"expected": value, "actual": manifest.get(name)}
        for name, value in expected.items()
        if manifest.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Poisson moment manifest protocol mismatch: {mismatches}")
    if manifest.get("raw_manifest_sha256") != _sha256_file(raw_manifest_file):
        raise ValueError(
            "Poisson moment manifest is not derived from this raw manifest"
        )
    _coordinate_transform_metadata(manifest)
    for split, expected_samples in (("train", 9000), ("test", 1000)):
        entry = manifest.get("splits", {}).get(split)
        if (
            not isinstance(entry, dict)
            or int(entry.get("samples", -1)) != expected_samples
        ):
            raise ValueError(f"invalid Poisson moment split metadata: {split}")
        filename = root / str(entry.get("moments"))
        if not filename.is_file():
            raise FileNotFoundError(f"missing Poisson moment array: {filename}")
        expected_hash = entry.get("sha256")
        if not _is_sha256(expected_hash):
            raise ValueError(f"invalid Poisson {split} moment SHA256 metadata")
        if _sha256_file(filename) != expected_hash:
            raise ValueError(f"Poisson {split} moment SHA256 mismatch: {filename}")
        array = np.load(filename, mmap_mode="r", allow_pickle=False)
        expected_shape = (expected_samples, mode**2, len(INPUT_CHANNELS))
        if array.shape != expected_shape or array.dtype != np.dtype(np.float32):
            raise ValueError(
                f"Poisson {split} moments={array.shape}/{array.dtype}, "
                f"expected={expected_shape}/float32"
            )


def load_manifest(
    data_dir: str | Path, *, verify_raw_hashes: bool = False
) -> tuple[Path, dict[str, Any]]:
    """Load the joined contract, always hashing moments and optionally full raw."""

    root = Path(data_dir).resolve()
    dataset_manifest = _read_json(root / "manifest.json")
    dataset_mode = int(dataset_manifest.get("mode", -1))
    if dataset_mode < 1:
        raise ValueError("Poisson dataset mode must be positive")
    expected_dataset_fields = {
        "format": DATA_FORMAT,
        "raw_manifest": "raw/data_manifest.json",
        "moment_manifest": "moment/data_manifest.json",
        "dimension": 2,
        "input_channels": list(INPUT_CHANNELS),
        "local_function_channels": list(LOCAL_FUNCTION_CHANNELS),
        "normalization": NORMALIZATION_PROTOCOL,
        "splits": {"train": {"samples": 9000}, "test": {"samples": 1000}},
    }
    dataset_mismatches = {
        name: {"expected": value, "actual": dataset_manifest.get(name)}
        for name, value in expected_dataset_fields.items()
        if dataset_manifest.get(name) != value
    }
    if dataset_mismatches:
        raise ValueError(f"Poisson dataset manifest mismatch: {dataset_mismatches}")
    raw_manifest_file = root / str(dataset_manifest["raw_manifest"])
    moment_manifest_file = root / str(dataset_manifest["moment_manifest"])
    if dataset_manifest.get("raw_manifest_sha256") != _sha256_file(raw_manifest_file):
        raise ValueError("Poisson dataset raw manifest SHA256 mismatch")
    if dataset_manifest.get("moment_manifest_sha256") != _sha256_file(
        moment_manifest_file
    ):
        raise ValueError("Poisson dataset moment manifest SHA256 mismatch")
    normalization_reference = dataset_manifest.get("normalization_manifest")
    normalization_hash = dataset_manifest.get("normalization_manifest_sha256")
    raw_root = raw_manifest_file.parent
    moment_root = moment_manifest_file.parent
    raw_manifest = _read_json(raw_manifest_file)
    moment_manifest = _read_json(moment_manifest_file)
    _validate_raw_manifest(
        raw_root,
        raw_manifest,
        verify_payload_hashes=bool(verify_raw_hashes),
    )
    _validate_moment_manifest(moment_root, moment_manifest, raw_manifest_file)
    if int(moment_manifest["mode"]) != dataset_mode:
        raise ValueError("Poisson dataset and moment modes differ")
    stored_transform = _coordinate_transform_metadata(moment_manifest)
    if dataset_manifest.get("coordinate_transform") != stored_transform:
        raise ValueError(
            "Poisson dataset/moment coordinate transforms differ; regenerate data"
        )
    joined = {
        "format": DATA_FORMAT,
        "raw": raw_manifest,
        "moment": moment_manifest,
        "mode": dataset_mode,
        "dimension": int(moment_manifest["dimension"]),
        "input_channels": list(INPUT_CHANNELS),
        "local_function_channels": list(LOCAL_FUNCTION_CHANNELS),
        "normalization": NORMALIZATION_PROTOCOL,
        "coordinate_transform": stored_transform,
        "splits": raw_manifest["splits"],
        "normalization_manifest": normalization_reference,
        "normalization_manifest_sha256": normalization_hash,
        "raw_manifest_sha256": dataset_manifest["raw_manifest_sha256"],
        "moment_manifest_sha256": dataset_manifest["moment_manifest_sha256"],
    }
    return root, joined


def verify_raw_payload_hashes(
    data_dir: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Run the expensive full-raw SHA256 preflight exactly where requested."""

    return load_manifest(data_dir, verify_raw_hashes=True)


def local_function_channels(manifest: dict[str, Any]) -> tuple[str, ...]:
    channels = tuple(manifest.get("local_function_channels", ()))
    if channels != LOCAL_FUNCTION_CHANNELS:
        raise ValueError(
            f"Poisson local channels={channels}, expected={LOCAL_FUNCTION_CHANNELS}"
        )
    return channels


def split_sample_count(data_dir: str | Path, split: str) -> int:
    _, manifest = load_manifest(data_dir)
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    return int(manifest["splits"][split]["samples"])


def raw_case_identity(
    raw_manifest: dict[str, Any], split: str, source_index: int
) -> tuple[str, int]:
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    index = int(source_index)
    samples = int(raw_manifest["splits"][split]["samples"])
    if index < 0 or index >= samples:
        raise IndexError(f"Poisson {split} source_index={index} outside [0,{samples})")
    offset = 0
    for block in raw_manifest["splits"][split]["families"]:
        start = int(block["start"])
        stop = int(block["stop"])
        count = stop - start
        if index < offset + count:
            return str(block["family"]), start + index - offset
        offset += count
    raise RuntimeError("Poisson raw split mapping is internally inconsistent")


def load_moment_array(
    data_root: Path, manifest: dict[str, Any], split: str
) -> np.ndarray:
    entry = manifest["moment"]["splits"][split]
    return np.load(
        data_root / "moment" / str(entry["moments"]),
        mmap_mode="r",
        allow_pickle=False,
    )


class PoissonRawStore:
    """Darcy-aligned per-case NPY reader with optional process-local caching."""

    def __init__(
        self,
        data_root: Path,
        raw_manifest: dict[str, Any],
        *,
        cache_cases: bool = False,
        access_mode: str = "auto",
    ) -> None:
        if access_mode not in ("auto", "npy"):
            raise ValueError("Poisson canonical access_mode must be auto or npy")
        self.root = Path(data_root).resolve() / "raw"
        self.manifest = raw_manifest
        self.cache_cases = bool(cache_cases)
        self.access_mode = "npy"
        self._case_cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_case_cache"] = {}
        return state

    def close(self) -> None:
        self._case_cache.clear()

    def case(
        self,
        family: str,
        raw_index: int,
        fields: tuple[str, ...] = ("points", "f", "k", "u"),
    ) -> dict[str, np.ndarray]:
        if family not in ("star", "annular"):
            raise ValueError(f"unknown Poisson family: {family!r}")
        if not fields or any(field not in _RAW_FIELDS for field in fields):
            raise ValueError(f"invalid Poisson raw field selection: {fields}")
        if "points" not in fields:
            raise ValueError("Poisson raw field selection must include points")
        entry = self.manifest["families"][family]
        index = int(raw_index)
        if index < 0 or index >= int(entry["samples"]):
            raise IndexError(f"Poisson raw index out of range: {family}/{index}")
        identity = (family, index)
        cached = self._case_cache.get(identity)
        if cached is not None and all(field in cached for field in fields):
            return {field: cached[field] for field in fields}

        case_entry = entry["cases"][index]
        filename = (
            self.root
            / str(entry.get("directory", family))
            / str(case_entry["file"])
        )
        packed = np.load(filename, mmap_mode="r", allow_pickle=False)
        if packed.shape != (1,):
            raise ValueError(f"Poisson split NPY must contain one record: {filename}")
        record = packed[0]
        result = {field: np.asarray(record[field]) for field in fields}
        points = result["points"]
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 1:
            raise ValueError(f"invalid Poisson raw points: {family}/{index} {points.shape}")
        node_count = int(points.shape[0])
        for field in (name for name in ("k", "f", "u") if name in result):
            if result[field].shape != (node_count,):
                raise ValueError(
                    f"invalid Poisson raw {field}: {family}/{index} {result[field].shape}"
                )
        if not all(np.all(np.isfinite(result[field])) for field in result):
            raise ValueError(f"Poisson raw case contains NaN/Inf: {family}/{index}")
        if self.cache_cases:
            destination = self._case_cache.setdefault(identity, {})
            for field, values in result.items():
                copied = np.array(values, copy=True, order="C")
                copied.setflags(write=False)
                destination[field] = copied
            return {field: destination[field] for field in fields}
        return result


def _fit_raw_training_coordinate_bounds(
    raw_dir: str | Path,
    raw_manifest: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one global coordinate box from every complete raw training case."""

    root = Path(raw_dir).resolve()
    manifest = (
        _read_json(root / "data_manifest.json")
        if raw_manifest is None
        else raw_manifest
    )
    _validate_raw_manifest(root, manifest, verify_payload_hashes=False)
    minimum = np.full(2, np.inf, dtype=np.float64)
    maximum = np.full(2, -np.inf, dtype=np.float64)
    for block in manifest["splits"]["train"]["families"]:
        family = str(block["family"])
        entry = manifest["families"][family]
        family_root = root / str(entry.get("directory", family))
        for raw_index in range(int(block["start"]), int(block["stop"])):
            case = entry["cases"][raw_index]
            record = np.load(
                family_root / str(case["file"]), mmap_mode="r", allow_pickle=False
            )[0]
            points = np.asarray(record["points"][:, :2], np.float64)
            if (
                points.ndim != 2
                or points.shape[0] < 1
                or points.shape[1] != 2
                or not np.all(np.isfinite(points))
            ):
                raise ValueError(
                    f"invalid Poisson training coordinates: {family}/{raw_index}"
                )
            minimum = np.minimum(minimum, points.min(axis=0))
            maximum = np.maximum(maximum, points.max(axis=0))
    if not np.all(np.isfinite(minimum)) or np.any(maximum <= minimum):
        raise ValueError("Poisson raw training coordinate bounds are invalid")
    return np.ascontiguousarray(minimum), np.ascontiguousarray(maximum)


def _resolve_dataset_coordinate_transform(
    raw_manifest: dict[str, Any],
    raw_minimum: np.ndarray,
    raw_maximum: np.ndarray,
    *,
    source: str,
    prior_half_span: float | Sequence[float] | np.ndarray | None,
) -> CoordinateTransform:
    """Resolve Poisson's affine map using the raw-domain prior when requested."""

    domain = np.asarray(raw_manifest["coordinate_domain"], dtype=np.float64)
    if domain.shape != (2,) or not np.all(np.isfinite(domain)) or domain[1] <= domain[0]:
        raise ValueError("Poisson raw coordinate_domain must be a finite interval")
    prior_center = np.full(2, 0.5 * float(domain.sum()), dtype=np.float64)
    return resolve_coordinate_transform(
        raw_minimum,
        raw_maximum,
        source=source,
        prior_half_span=prior_half_span,
        prior_center=prior_center,
    )


def fit_coordinate_transform(
    raw_dir: str | Path,
    *,
    source: str,
    prior_half_span: float | Sequence[float] | np.ndarray | None,
    raw_manifest: dict[str, Any] | None = None,
) -> CoordinateTransform:
    """Fit the configured affine map from every complete raw training case."""

    root = Path(raw_dir).resolve()
    manifest = (
        _read_json(root / "data_manifest.json")
        if raw_manifest is None
        else raw_manifest
    )
    raw_minimum, raw_maximum = _fit_raw_training_coordinate_bounds(root, manifest)
    return _resolve_dataset_coordinate_transform(
        manifest,
        raw_minimum,
        raw_maximum,
        source=source,
        prior_half_span=prior_half_span,
    )


def load_coordinate_transform(
    data_dir: str | Path,
    *,
    source: str,
    prior_half_span: float | Sequence[float] | np.ndarray | None,
) -> CoordinateTransform:
    """Load and strictly validate the moment artifact's affine coordinate map."""

    _, manifest = load_manifest(data_dir)
    stored = _coordinate_transform_metadata(manifest["moment"])
    expected = _resolve_dataset_coordinate_transform(
        manifest["raw"],
        np.asarray(stored["raw_training_minimum"], dtype=np.float64),
        np.asarray(stored["raw_training_maximum"], dtype=np.float64),
        source=source,
        prior_half_span=prior_half_span,
    )
    validate_coordinate_transform(manifest, expected)
    return expected


def standard_legendre(values: np.ndarray, mode: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return legvander(values, mode - 1) * np.sqrt(
        (2.0 * np.arange(mode, dtype=np.float64) + 1.0) / 2.0
    )


def reconstruct_standard_legendre_2d(
    coordinates: np.ndarray, coefficients: np.ndarray, mode: int
) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    matrix = np.asarray(coefficients, dtype=np.float64).reshape(mode, mode)
    phi_x = standard_legendre(coordinates[:, 0], mode)
    phi_y = standard_legendre(coordinates[:, 1], mode)
    # Avoid MKL's second OpenMP runtime after Torch has initialized its runtime.
    values = np.einsum("qi,qj,ij->q", phi_x, phi_y, matrix, optimize=False)
    return np.ascontiguousarray(values, dtype=np.float64)


def _normalization_cases(
    store: PoissonRawStore,
    raw_manifest: dict[str, Any],
    moments: np.ndarray,
    training_indices: np.ndarray,
    *,
    mode: int,
    coordinate_transform: CoordinateTransform,
) -> Iterator[np.ndarray]:
    for source_index in training_indices.tolist():
        family, raw_index = raw_case_identity(raw_manifest, "train", source_index)
        case = store.case(family, raw_index)
        selection = slice(None)
        raw_coordinates = np.asarray(case["points"][:, :2], dtype=np.float64)
        coordinates = np.ascontiguousarray(
            coordinate_transform.normalize(raw_coordinates), dtype=np.float32
        )
        boundary = reconstruct_standard_legendre_2d(
            coordinates, moments[source_index, :, 3], mode
        ).astype(np.float32, copy=False)
        yield np.ascontiguousarray(
            np.column_stack(
                (
                    np.asarray(case["f"][selection], np.float32),
                    np.asarray(case["k"][selection], np.float32),
                    boundary,
                    np.asarray(case["u"][selection], np.float32),
                )
            ),
            dtype=np.float32,
        )


def compute_statistical_scaling(
    data_dir: str | Path,
    training_indices: list[int],
    mode: int,
    *,
    coordinate_transform: CoordinateTransform,
) -> ModelScaling:
    """Compute canonical per-case scaling from complete raw training cases."""

    if not training_indices:
        raise ValueError("training_indices must not be empty")
    root, manifest = load_manifest(data_dir)
    validate_coordinate_transform(manifest, coordinate_transform)
    if int(manifest["mode"]) != int(mode):
        raise ValueError(f"Poisson mode={manifest['mode']}, requested={mode}")
    indices = np.asarray(training_indices, dtype=np.int64)
    train_count = int(manifest["splits"]["train"]["samples"])
    if indices.min() < 0 or indices.max() >= train_count:
        raise IndexError("training_indices outside the Poisson train split")
    moments = load_moment_array(root, manifest, "train")
    input_sum = np.zeros((mode**2, len(INPUT_CHANNELS)), dtype=np.float64)
    input_square_sum = np.zeros_like(input_sum)
    for start in range(0, indices.size, 64):
        block = np.asarray(moments[indices[start : start + 64]], dtype=np.float64)
        input_sum += block.sum(axis=0)
        input_square_sum += np.square(block).sum(axis=0)
    input_mean = input_sum / indices.size
    input_variance = np.maximum(
        input_square_sum / indices.size - np.square(input_mean), 0.0
    )

    store = PoissonRawStore(root, manifest["raw"])
    try:
        raw_statistics = ragged_channel_statistics(
            _normalization_cases(
                store,
                manifest["raw"],
                moments,
                indices,
                mode=mode,
                coordinate_transform=coordinate_transform,
            )
        )
    finally:
        store.close()
    return ModelScaling(
        input_mean=input_mean.astype(np.float32),
        input_std=stable_std(np.sqrt(input_variance)).astype(np.float32),
        function_mean=raw_statistics.mean[:3].astype(np.float32),
        function_std=raw_statistics.std[:3].astype(np.float32),
        target_mean=raw_statistics.mean[3:4].astype(np.float32),
        target_std=raw_statistics.std[3:4].astype(np.float32),
        coordinate_scale=np.asarray(coordinate_transform.half_span, np.float32),
    )


def load_precomputed_scaling(
    data_dir: str | Path,
    *,
    coordinate_transform: CoordinateTransform,
) -> ModelScaling:
    """Load a full-train raw-derived scaling artifact with strict lineage checks."""

    root, manifest = load_manifest(data_dir)
    validate_coordinate_transform(manifest, coordinate_transform)
    reference = manifest.get("normalization_manifest")
    if reference != "moment/normalization/manifest.json":
        raise FileNotFoundError(
            "Poisson dataset manifest does not declare the canonical normalization"
        )
    normalization_file = root / str(reference)
    declared_manifest_hash = manifest.get("normalization_manifest_sha256")
    if (
        not normalization_file.is_file()
        or declared_manifest_hash != _sha256_file(normalization_file)
    ):
        raise ValueError(
            "Poisson canonical normalization manifest is missing or has stale lineage"
        )
    normalization_root = normalization_file.parent
    normalization_manifest = _read_json(normalization_file)
    expected = {
        "format": NORMALIZATION_DATA_FORMAT,
        "raw_manifest_sha256": _sha256_file(root / "raw" / "data_manifest.json"),
        "moment_manifest_sha256": _sha256_file(root / "moment" / "data_manifest.json"),
        "training_split": "train",
        "training_cases": int(manifest["splits"]["train"]["samples"]),
        "normalization": NORMALIZATION_PROTOCOL,
    }
    mismatches = {
        name: {"expected": value, "actual": normalization_manifest.get(name)}
        for name, value in expected.items()
        if normalization_manifest.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Poisson normalization lineage mismatch: {mismatches}")
    entry = normalization_manifest.get("artifact")
    if not isinstance(entry, dict) or entry.get("name") != NORMALIZATION_NAME:
        raise ValueError("Poisson canonical normalization metadata mismatch")
    filename = normalization_root / str(entry.get("file"))
    if not filename.is_file():
        raise FileNotFoundError(f"missing Poisson normalization artifact: {filename}")
    if entry.get("sha256") != _sha256_file(filename):
        raise ValueError(f"Poisson normalization artifact SHA256 mismatch: {filename}")
    with np.load(filename, allow_pickle=False) as archive:
        if set(archive.files) != set(SCALING_NAMES):
            raise ValueError(f"Poisson scaling fields mismatch: {archive.files}")
        arrays = {
            name: np.asarray(archive[name], dtype=np.float32).copy()
            for name in SCALING_NAMES
        }
    mode = int(manifest["mode"])
    expected_shapes = {
        "input_mean": (mode**2, len(INPUT_CHANNELS)),
        "input_std": (mode**2, len(INPUT_CHANNELS)),
        "function_mean": (len(LOCAL_FUNCTION_CHANNELS),),
        "function_std": (len(LOCAL_FUNCTION_CHANNELS),),
        "target_mean": (1,),
        "target_std": (1,),
    }
    for name, expected_shape in expected_shapes.items():
        if arrays[name].shape != expected_shape or not np.all(np.isfinite(arrays[name])):
            raise ValueError(
                f"Poisson scaling {name}={arrays[name].shape}, expected={expected_shape}"
            )
        if name.endswith("_std") and np.any(arrays[name] <= 0.0):
            raise ValueError(f"Poisson scaling {name} must be positive")
    return ModelScaling(
        **arrays,
        coordinate_scale=np.asarray(coordinate_transform.half_span, np.float32),
    )


__all__ = [
    "DATA_FORMAT",
    "GLOBAL_MOMENT_CHANNELS",
    "GLOBAL_MOMENT_CHANNEL_INDEX",
    "INPUT_CHANNELS",
    "LOCAL_FUNCTION_CHANNELS",
    "MOMENT_DATA_FORMAT",
    "NORMALIZATION_NAME",
    "NORMALIZATION_PROTOCOL",
    "PoissonRawStore",
    "RAW_DATA_FORMAT",
    "SCALING_NAMES",
    "compute_statistical_scaling",
    "fit_coordinate_transform",
    "load_coordinate_transform",
    "load_manifest",
    "load_moment_array",
    "load_precomputed_scaling",
    "local_function_channels",
    "raw_case_identity",
    "reconstruct_standard_legendre_2d",
    "split_sample_count",
    "verify_raw_payload_hashes",
    "validate_coordinate_transform",
]
