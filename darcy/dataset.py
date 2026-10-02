"""Manifest and moment access for the native Darcy benchmark dataset.

The runtime never reads a prepared query pool.  This module owns the single
data manifest, split identities, generated moments, and the sole canonical
normalization.  Raw query loading lives in :mod:`darcy.native`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

try:
    from ..dataset_common import (
        CoordinateTransform,
        ModelScaling,
        resolve_coordinate_transform,
        validate_coordinate_transform_metadata,
    )
except ImportError:
    from dataset_common import (
        CoordinateTransform,
        ModelScaling,
        resolve_coordinate_transform,
        validate_coordinate_transform_metadata,
    )


DATA_FORMAT = "mfd-darcy-native"
MOMENT_FORMAT = "mfd-darcy-moment"
DEFAULT_MODE = 32
DIMENSION = 2
GLOBAL_MOMENT_CHANNELS = ("domain", "permeability")
GLOBAL_MOMENT_CHANNEL_INDEX = {
    name: index for index, name in enumerate(GLOBAL_MOMENT_CHANNELS)
}
# Historical public name retained for manifest/generator compatibility.
INPUT_CHANNELS = GLOBAL_MOMENT_CHANNELS
LOCAL_FUNCTION_CHANNELS = ("permeability",)
TARGET_CHANNELS = ("solution",)
RESOLUTION_SUBSETS = {
    "fine": "smooth_small_scale",
    "coarse": "smooth_large_scale",
}
RAW_SAMPLE_COUNTS = {
    "smooth_small_scale": 2000,
    "smooth_large_scale": 2000,
}
SPLIT_RANGES = {
    "train": (0, 500),
    "selection": (1600, 1700),
    "test": (1800, 2000),
}
NORMALIZATION_NAME = "canonical"
NORMALIZATION_PROTOCOL = {
    "weighting": "uniform_per_case",
    "sampling": "full_raw",
}
SPLIT_NAMES = ("train", "selection", "test")
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
    with filename.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_raw_inventory(
    raw_root: str | Path,
    sample_counts: Mapping[str, int] = RAW_SAMPLE_COUNTS,
) -> dict[str, dict[str, object]]:
    """Hash every declared raw triplet and measure native point-count ranges.

    This is intentionally an explicit, heavyweight operation. Dataset and
    manifest loading do not call it; generation validation and training
    preflight each opt in exactly once.
    """

    root = Path(raw_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"missing Darcy raw root: {root}")
    inventory: dict[str, dict[str, object]] = {}
    for subset, declared_count in sample_counts.items():
        count = int(declared_count)
        if count < 1:
            raise ValueError(f"Darcy raw sample count must be positive: {subset}")
        subset_root = (root / subset).resolve()
        if not subset_root.is_relative_to(root) or not subset_root.is_dir():
            raise FileNotFoundError(f"missing Darcy raw subset: {subset_root}")
        digest = hashlib.sha256()
        expected_names: set[str] = set()
        point_min: int | None = None
        point_max = 0
        for source_index in range(count):
            filenames = [
                subset_root / f"{prefix}_{source_index:05d}.npy"
                for prefix in ("nodes", "elements", "features")
            ]
            for filename in filenames:
                if not filename.is_file():
                    raise FileNotFoundError(filename)
                expected_names.add(filename.name)
                digest.update(filename.name.encode("utf-8"))
                digest.update(filename.stat().st_size.to_bytes(8, "little"))
                digest.update(bytes.fromhex(_sha256_file(filename)))
            nodes = np.load(filenames[0], mmap_mode="r", allow_pickle=False)
            if nodes.ndim != 2 or nodes.shape[0] < 1 or nodes.shape[1] != DIMENSION:
                raise ValueError(
                    f"invalid Darcy raw nodes: {subset}/{source_index}: {nodes.shape}"
                )
            points = int(nodes.shape[0])
            point_min = points if point_min is None else min(point_min, points)
            point_max = max(point_max, points)
        actual_names = {path.name for path in subset_root.glob("*.npy")}
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)[:10]
            extra = sorted(actual_names - expected_names)[:10]
            raise ValueError(
                f"Darcy raw inventory filenames changed for {subset}: "
                f"missing={missing}, extra={extra}"
            )
        inventory[subset] = {
            "path": f"raw/{subset}",
            "samples": count,
            "files_per_sample": ["nodes", "elements", "features"],
            "inventory_sha256": digest.hexdigest(),
            "point_count_min": point_min,
            "point_count_max": point_max,
        }
    return inventory


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Darcy artifact path escapes data root: {value!r}")
    return resolved


def _coordinate_transform_metadata(
    manifest: Mapping[str, object],
) -> dict[str, object]:
    metadata = manifest.get("coordinate_transform")
    if not isinstance(metadata, Mapping) or set(metadata) != _COORDINATE_TRANSFORM_FIELDS:
        raise ValueError(
            "Darcy coordinate_transform must use the canonical affine schema: "
            f"fields={sorted(_COORDINATE_TRANSFORM_FIELDS)}"
        )
    for name in (
        "center",
        "half_span",
        "minimum",
        "maximum",
        "raw_training_minimum",
        "raw_training_maximum",
    ):
        values = np.asarray(metadata[name], dtype=np.float64)
        if values.shape != (DIMENSION,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Darcy coordinate_transform.{name} must be finite [2]")
    source = str(metadata.get("source"))
    if source not in ("prior", "train_bounds"):
        raise ValueError("Darcy coordinate_transform.source is invalid")
    margin = float(metadata.get("margin", float("nan")))
    if not np.isfinite(margin) or margin < 1.0:
        raise ValueError("Darcy coordinate_transform.margin is invalid")
    return dict(metadata)


def validate_coordinate_transform(
    manifest: Mapping[str, object],
    coordinate_transform: CoordinateTransform,
) -> None:
    """Reject a Local/scaling transform that differs from stored moments."""

    validate_coordinate_transform_metadata(
        coordinate_transform,
        _coordinate_transform_metadata(manifest),
        context="Darcy",
    )


def _source_indices(component: Mapping[str, object]) -> tuple[int, ...]:
    selection = component.get("source_indices")
    if not isinstance(selection, Mapping) or selection.get("type") != "range":
        raise ValueError("Darcy split source_indices must be a range")
    start = int(selection.get("start", -1))
    stop = int(selection.get("stop", -1))
    step = int(selection.get("step", 0))
    if start < 0 or stop <= start or step != 1:
        raise ValueError(f"invalid Darcy source range [{start},{stop})/{step}")
    values = tuple(range(start, stop, step))
    if int(component.get("samples", -1)) != len(values):
        raise ValueError("Darcy component sample count does not match its range")
    return values


def load_data_manifest(data_root: str | Path) -> tuple[Path, dict[str, object]]:
    """Load and strictly validate the single Darcy runtime manifest."""

    root = Path(data_root).resolve()
    filename = root / "manifest.json"
    if not filename.is_file():
        raise FileNotFoundError(f"missing Darcy manifest: {filename}")
    manifest = json.loads(filename.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Darcy manifest must be a JSON object")
    mode = int(manifest.get("mode", -1))
    if mode < 1:
        raise ValueError("Darcy mode must be positive")
    expected_header = {
        "format": DATA_FORMAT,
        "dataset": "deformed_domain_darcy",
        "dimension": DIMENSION,
        "basis": "orthonormal_tensor_legendre_2d",
        "basis_domain": [-1.0, 1.0],
        "flatten_order": "C: m=i*mode+j",
        "input_channels": list(INPUT_CHANNELS),
        "local_function_channels": list(LOCAL_FUNCTION_CHANNELS),
        "target_channels": list(TARGET_CHANNELS),
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in expected_header.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Darcy manifest protocol mismatch: {mismatches}")
    _coordinate_transform_metadata(manifest)

    raw = manifest.get("raw")
    if not isinstance(raw, Mapping) or raw.get("root") != "raw":
        raise ValueError("Darcy raw root must be data/raw")
    subsets = raw.get("subsets")
    if not isinstance(subsets, Mapping) or set(subsets) != set(
        RESOLUTION_SUBSETS.values()
    ):
        raise ValueError("Darcy raw subsets must be the two smooth resolutions")
    raw_root = _resolve(root, str(raw["root"]))
    if not raw_root.is_dir():
        raise FileNotFoundError(f"missing Darcy raw root: {raw_root}")
    for subset, samples in RAW_SAMPLE_COUNTS.items():
        entry = subsets[subset]
        if not isinstance(entry, Mapping):
            raise ValueError(f"Darcy raw subset metadata is invalid: {subset}")
        expected = {
            "path": f"raw/{subset}",
            "samples": samples,
            "files_per_sample": ["nodes", "elements", "features"],
        }
        mismatches = {
            key: {"expected": value, "actual": entry.get(key)}
            for key, value in expected.items()
            if entry.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Darcy raw subset {subset} mismatch: {mismatches}")
        inventory_hash = str(entry.get("inventory_sha256", ""))
        if len(inventory_hash) != 64 or any(
            character not in "0123456789abcdef" for character in inventory_hash
        ):
            raise ValueError(f"Darcy raw subset {subset} inventory hash is invalid")
        if int(entry.get("point_count_min", 0)) < 1 or int(
            entry.get("point_count_max", 0)
        ) < int(entry.get("point_count_min", 0)):
            raise ValueError(f"Darcy raw subset {subset} point-count range is invalid")
        subset_root = _resolve(root, str(entry["path"]))
        if not subset_root.is_dir():
            raise FileNotFoundError(f"missing Darcy raw subset: {subset_root}")

    moment = manifest.get("moment")
    if not isinstance(moment, Mapping) or moment.get("format") != MOMENT_FORMAT:
        raise ValueError("Darcy moment protocol is invalid")
    if moment.get("root") != "moment":
        raise ValueError("Darcy moment root must be data/moment")

    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(SPLIT_NAMES):
        raise ValueError(f"Darcy splits must be exactly {SPLIT_NAMES}")
    identities: dict[str, set[tuple[str, int]]] = {}
    for split in SPLIT_NAMES:
        entry = splits[split]
        if not isinstance(entry, Mapping) or entry.get("ordering") != "fine_then_coarse":
            raise ValueError(f"Darcy {split} ordering must be fine_then_coarse")
        components = entry.get("components")
        if not isinstance(components, list) or len(components) != 2:
            raise ValueError(f"Darcy {split} must contain fine and coarse")
        if [component.get("resolution") for component in components] != [
            "fine",
            "coarse",
        ]:
            raise ValueError(f"Darcy {split} components are not fine then coarse")
        selected: set[tuple[str, int]] = set()
        for component in components:
            resolution = str(component.get("resolution"))
            subset = str(component.get("subset"))
            if RESOLUTION_SUBSETS.get(resolution) != subset:
                raise ValueError(f"Darcy {split}/{resolution} subset mismatch")
            source_indices = _source_indices(component)
            if source_indices != tuple(range(*SPLIT_RANGES[split])):
                raise ValueError(f"Darcy {split}/{resolution} source range changed")
            for source_index in source_indices:
                identity = (subset, source_index)
                if identity in selected:
                    raise ValueError(f"duplicate Darcy identity: {identity}")
                selected.add(identity)
        if int(entry.get("samples", -1)) != len(selected):
            raise ValueError(f"Darcy {split} sample count mismatch")
        identities[split] = selected
    for left, right in (
        ("train", "selection"),
        ("train", "test"),
        ("selection", "test"),
    ):
        overlap = identities[left] & identities[right]
        if overlap:
            raise ValueError(f"Darcy split leakage {left}/{right}: {sorted(overlap)[:5]}")

    normalization = manifest.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("Darcy manifest must declare one canonical normalization")
    expected_normalization = {
        "name": NORMALIZATION_NAME,
        **NORMALIZATION_PROTOCOL,
        "source": "raw",
        "source_split": "train",
        "input_weighting": "uniform_per_case",
        "input_sampling": "full_raw_physical_triangle_moments",
        "local_target_weighting": "uniform_per_case",
        "local_target_sampling": "full_raw",
    }
    mismatches = {
        key: {"expected": value, "actual": normalization.get(key)}
        for key, value in expected_normalization.items()
        if normalization.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Darcy normalization provenance mismatch: {mismatches}")
    scaling_file = _resolve(root, str(normalization.get("file", "")))
    if not scaling_file.is_file():
        raise FileNotFoundError(f"missing Darcy normalization: {scaling_file}")
    if _sha256_file(scaling_file) != normalization.get("sha256"):
        raise ValueError(f"Darcy normalization hash mismatch: {scaling_file}")
    expected_native_sampling = {
        "method": "uniform_over_raw_vertex_indices",
        "train": "without_replacement_if_Q_le_N_else_all_with_replacement",
        "selection": "full_raw",
        "test": "full_raw",
        "resampled_each_epoch": True,
    }
    if manifest.get("native_query_sampling") != expected_native_sampling:
        raise ValueError("Darcy native query sampling protocol changed")
    return root, manifest


def _fit_raw_training_coordinate_bounds(
    raw_root: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure physical coordinate bounds over every declared training mesh."""

    root = Path(raw_root).resolve()
    minimum = np.full(DIMENSION, np.inf, dtype=np.float64)
    maximum = np.full(DIMENSION, -np.inf, dtype=np.float64)
    for resolution in ("fine", "coarse"):
        subset = RESOLUTION_SUBSETS[resolution]
        subset_root = root / subset
        for source_index in range(*SPLIT_RANGES["train"]):
            filename = subset_root / f"nodes_{source_index:05d}.npy"
            if not filename.is_file():
                raise FileNotFoundError(filename)
            nodes = np.load(filename, mmap_mode="r", allow_pickle=False)
            if nodes.ndim != 2 or nodes.shape[1] != DIMENSION or nodes.shape[0] < 1:
                raise ValueError(
                    f"invalid Darcy training nodes: {subset}/{source_index}: "
                    f"{nodes.shape}"
                )
            values = np.asarray(nodes, dtype=np.float64)
            block_minimum = np.min(values, axis=0)
            block_maximum = np.max(values, axis=0)
            if not (
                np.all(np.isfinite(block_minimum))
                and np.all(np.isfinite(block_maximum))
            ):
                raise ValueError(
                    f"non-finite Darcy training nodes: {subset}/{source_index}"
                )
            minimum = np.minimum(minimum, block_minimum)
            maximum = np.maximum(maximum, block_maximum)
    return minimum, maximum


def _resolve_dataset_coordinate_transform(
    raw_minimum: Sequence[float],
    raw_maximum: Sequence[float],
    *,
    source: str,
    prior_half_span: Sequence[float] | None,
) -> CoordinateTransform:
    """Resolve Darcy's affine map with the physical origin as prior center."""

    return resolve_coordinate_transform(
        raw_minimum,
        raw_maximum,
        source=source,
        prior_half_span=prior_half_span,
        prior_center=(0.0, 0.0),
    )


def fit_coordinate_transform(
    raw_root: str | Path,
    *,
    source: str,
    prior_half_span: Sequence[float] | None,
) -> CoordinateTransform:
    """Fit the configured affine transform from the declared raw train split."""

    raw_minimum, raw_maximum = _fit_raw_training_coordinate_bounds(raw_root)
    return _resolve_dataset_coordinate_transform(
        raw_minimum,
        raw_maximum,
        source=source,
        prior_half_span=prior_half_span,
    )


def load_coordinate_transform(
    data_root: str | Path,
    *,
    source: str,
    prior_half_span: Sequence[float] | None,
) -> CoordinateTransform:
    """Re-resolve and strictly validate the stored raw-training affine map."""

    _, manifest = load_data_manifest(data_root)
    stored = _coordinate_transform_metadata(manifest)
    expected = _resolve_dataset_coordinate_transform(
        stored["raw_training_minimum"],
        stored["raw_training_maximum"],
        source=source,
        prior_half_span=prior_half_span,
    )
    validate_coordinate_transform(manifest, expected)
    return expected


def validate_raw_inventory(
    data_root: str | Path,
    manifest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Explicitly recompute and compare all 12,000 Darcy raw-file hashes."""

    if manifest is None:
        root, loaded = load_data_manifest(data_root)
        manifest = loaded
    else:
        root = Path(data_root).resolve()
    raw = manifest.get("raw")
    if not isinstance(raw, Mapping):
        raise ValueError("Darcy manifest lacks raw inventory metadata")
    expected = raw.get("subsets")
    if not isinstance(expected, Mapping):
        raise ValueError("Darcy manifest raw subsets are invalid")
    raw_root = _resolve(root, str(raw.get("root", "")))
    actual = compute_raw_inventory(raw_root)
    mismatches: dict[str, object] = {}
    for subset in RAW_SAMPLE_COUNTS:
        expected_entry = expected.get(subset)
        if not isinstance(expected_entry, Mapping):
            mismatches[subset] = "missing manifest entry"
            continue
        expected_dict = dict(expected_entry)
        if expected_dict != actual[subset]:
            mismatches[subset] = {
                key: {
                    "expected": expected_dict.get(key),
                    "actual": actual[subset].get(key),
                }
                for key in sorted(set(expected_dict) | set(actual[subset]))
                if expected_dict.get(key) != actual[subset].get(key)
            }
    if mismatches:
        raise ValueError(f"Darcy raw inventory does not match manifest: {mismatches}")
    return {
        "root": str(raw_root),
        "total_samples": sum(RAW_SAMPLE_COUNTS.values()),
        "total_files": 3 * sum(RAW_SAMPLE_COUNTS.values()),
        "subsets": actual,
    }


def load_scaling(
    data_root: str | Path,
    *,
    coordinate_transform: CoordinateTransform,
) -> tuple[ModelScaling, dict[str, object]]:
    """Load the sole raw-derived canonical normalization and its provenance."""

    root, manifest = load_data_manifest(data_root)
    mode = int(manifest["mode"])
    validate_coordinate_transform(manifest, coordinate_transform)
    metadata = dict(manifest["normalization"])
    filename = _resolve(root, str(metadata["file"]))
    with np.load(filename, allow_pickle=False) as archive:
        if set(archive.files) != set(SCALING_NAMES):
            raise ValueError(f"Darcy scaling fields mismatch: {archive.files}")
        arrays = {
            name: np.asarray(archive[name], dtype=np.float32).copy()
            for name in SCALING_NAMES
        }
    expected_shapes = {
        "input_mean": (mode**DIMENSION, len(INPUT_CHANNELS)),
        "input_std": (mode**DIMENSION, len(INPUT_CHANNELS)),
        "function_mean": (len(LOCAL_FUNCTION_CHANNELS),),
        "function_std": (len(LOCAL_FUNCTION_CHANNELS),),
        "target_mean": (len(TARGET_CHANNELS),),
        "target_std": (len(TARGET_CHANNELS),),
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected or not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"Darcy scaling {name}={arrays[name].shape}, expected={expected}")
        if name.endswith("_std") and np.any(arrays[name] <= 0.0):
            raise ValueError(f"Darcy scaling {name} must be positive")
    return ModelScaling(
        **arrays,
        coordinate_scale=np.asarray(coordinate_transform.half_span, np.float32),
    ), metadata


class DarcyMomentDataset:
    """Index generated moments using identities from the single manifest."""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        *,
        resolutions: Sequence[str] = ("fine", "coarse"),
    ) -> None:
        self.data_root, self.manifest = load_data_manifest(data_root)
        self.mode = int(self.manifest["mode"])
        if split not in SPLIT_NAMES:
            raise ValueError(f"Darcy split must be {SPLIT_NAMES}")
        requested = tuple(str(value) for value in resolutions)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("Darcy resolutions must be non-empty and unique")
        if any(value not in RESOLUTION_SUBSETS for value in requested):
            raise ValueError(f"Darcy resolution must be {tuple(RESOLUTION_SUBSETS)}")
        self.split = split
        self.resolutions = requested
        self.records: list[dict[str, object]] = []
        self._arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        components = self.manifest["splits"][split]["components"]
        for component in components:
            resolution = str(component["resolution"])
            if resolution not in requested:
                continue
            subset = str(component["subset"])
            moment_entry = component.get("moment")
            if not isinstance(moment_entry, Mapping):
                raise ValueError(f"Darcy {split}/{resolution} lacks moment metadata")
            component_root = _resolve(self.data_root, str(moment_entry["root"]))
            manifest_file = _resolve(self.data_root, str(moment_entry["manifest"]))
            component_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if not isinstance(component_manifest, dict):
                raise ValueError(
                    f"Darcy moment manifest is not an object: {manifest_file}"
                )
            if (
                component_manifest.get("format") != MOMENT_FORMAT
                or component_manifest.get("split") != split
                or component_manifest.get("resolution") != resolution
                or component_manifest.get("subset") != subset
                or int(component_manifest.get("mode", -1)) != self.mode
            ):
                raise ValueError(f"Darcy moment manifest mismatch: {manifest_file}")
            if component_manifest.get("coordinate_transform") != self.manifest.get(
                "coordinate_transform"
            ):
                raise ValueError(
                    f"Darcy moment coordinate transform mismatch: {manifest_file}"
                )
            if component_manifest.get("source_indices") != {
                "start": SPLIT_RANGES[split][0],
                "stop": SPLIT_RANGES[split][1],
                "step": 1,
            }:
                raise ValueError(f"Darcy moment source range mismatch: {manifest_file}")
            sources_file = component_root / "source_indices.npy"
            moments_file = component_root / "moments.npy"
            arrays = component_manifest.get("arrays")
            if not isinstance(arrays, Mapping):
                raise ValueError(f"Darcy moment array metadata is invalid: {manifest_file}")
            expected_arrays = {
                "source_indices": {
                    "file": "source_indices.npy",
                    "shape": [len(_source_indices(component))],
                    "dtype": "int64",
                },
                "moments": {
                    "file": "moments.npy",
                    "shape": [
                        len(_source_indices(component)),
                        self.mode**DIMENSION,
                        len(INPUT_CHANNELS),
                    ],
                    "dtype": "float32",
                },
            }
            if set(arrays) != set(expected_arrays):
                raise ValueError(f"Darcy moment arrays changed: {manifest_file}")
            for name, expected in expected_arrays.items():
                if not isinstance(arrays[name], Mapping) or any(
                    arrays[name].get(key) != value for key, value in expected.items()
                ):
                    raise ValueError(
                        f"Darcy moment {name} metadata mismatch: {manifest_file}"
                    )
            for name, filename in (
                ("source_indices", sources_file),
                ("moments", moments_file),
            ):
                if _sha256_file(filename) != arrays[name]["sha256"]:
                    raise ValueError(f"Darcy moment hash mismatch: {filename}")
            sources = np.load(sources_file, mmap_mode="r", allow_pickle=False)
            moments = np.load(moments_file, mmap_mode="r", allow_pickle=False)
            expected_sources = np.asarray(_source_indices(component), dtype=np.int64)
            if not np.array_equal(sources, expected_sources):
                raise ValueError(f"Darcy moment identities mismatch: {sources_file}")
            expected_shape = (len(sources), self.mode**DIMENSION, len(INPUT_CHANNELS))
            if moments.shape != expected_shape or moments.dtype != np.float32:
                raise ValueError(f"Darcy moments={moments.shape}/{moments.dtype}")
            key = f"{split}/{resolution}"
            self._arrays[key] = (sources, moments)
            for row, source_index in enumerate(sources):
                self.records.append(
                    {
                        "key": key,
                        "row": row,
                        "resolution": resolution,
                        "subset": subset,
                        "source_index": int(source_index),
                    }
                )

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_arrays"] = {}
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._reopen_arrays()

    def _reopen_arrays(self) -> None:
        if self._arrays:
            return
        for resolution in self.resolutions:
            component_root = self.data_root / "moment" / self.split / resolution
            self._arrays[f"{self.split}/{resolution}"] = (
                np.load(
                    component_root / "source_indices.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                ),
                np.load(
                    component_root / "moments.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                ),
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        import torch

        self._reopen_arrays()
        record = self.records[index]
        sources, moments = self._arrays[str(record["key"])]
        row = int(record["row"])
        source_index = int(record["source_index"])
        if int(sources[row]) != source_index:
            raise RuntimeError("Darcy moment row identity changed after validation")
        return {
            "sample": f"{record['subset']}_{source_index:05d}",
            "split": self.split,
            "resolution": str(record["resolution"]),
            "subset": str(record["subset"]),
            "index": torch.tensor(index, dtype=torch.int64),
            "source_index": torch.tensor(source_index, dtype=torch.int64),
            "moments": torch.from_numpy(
                np.array(moments[row], dtype=np.float32, order="C", copy=True)
            ),
        }


__all__ = [
    "DATA_FORMAT",
    "DIMENSION",
    "DarcyMomentDataset",
    "GLOBAL_MOMENT_CHANNELS",
    "GLOBAL_MOMENT_CHANNEL_INDEX",
    "INPUT_CHANNELS",
    "LOCAL_FUNCTION_CHANNELS",
    "DEFAULT_MODE",
    "NORMALIZATION_NAME",
    "NORMALIZATION_PROTOCOL",
    "RAW_SAMPLE_COUNTS",
    "RESOLUTION_SUBSETS",
    "SCALING_NAMES",
    "SPLIT_NAMES",
    "SPLIT_RANGES",
    "TARGET_CHANNELS",
    "compute_raw_inventory",
    "fit_coordinate_transform",
    "load_coordinate_transform",
    "load_data_manifest",
    "load_scaling",
    "validate_coordinate_transform",
    "validate_raw_inventory",
]
