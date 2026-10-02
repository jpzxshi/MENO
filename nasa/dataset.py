"""NASA-CRM raw-native dataset and training normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

try:
    from ..dataset_common import (
        CoordinateTransform,
        ModelScaling,
        compose_function_scaling,
        derive_condition_moments,
        ragged_channel_statistics,
        resolve_coordinate_transform,
        stable_std,
        validate_stored_coordinate_transform,
    )
    from ..query_sampling import normalize_query_limit, query_selection
except ImportError:
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from dataset_common import (
        CoordinateTransform,
        ModelScaling,
        compose_function_scaling,
        derive_condition_moments,
        ragged_channel_statistics,
        resolve_coordinate_transform,
        stable_std,
        validate_stored_coordinate_transform,
    )
    from query_sampling import normalize_query_limit, query_selection

from .native import (
    CONDITION_CHANNELS,
    COORDINATE_FIELDS,
    NASARawStore,
    NORMAL_FIELDS,
    TARGET_FIELDS,
    file_sha256,
    load_data_paths,
)


TARGET_CHANNELS = ("Cp", "Cfx", "Cfy", "Cfz")
EXPECTED_TRAIN_CASES = 105
EXPECTED_TEST_CASES = 44
EXPECTED_BASIS = "orthonormal_tensor_legendre_3d"
EXPECTED_FLATTEN_ORDER = "C: m=(a*mode+b)*mode+c"
GEOMETRY_MOMENT_CHANNELS = ("surface", "normal_x", "normal_y", "normal_z")
GLOBAL_MOMENT_CHANNELS = GEOMETRY_MOMENT_CHANNELS + CONDITION_CHANNELS
GLOBAL_MOMENT_CHANNEL_INDEX = {
    name: index for index, name in enumerate(GLOBAL_MOMENT_CHANNELS)
}
RAW_COORDINATE_FIELDS = COORDINATE_FIELDS
RAW_NORMAL_FIELDS = NORMAL_FIELDS
RAW_TARGET_FIELDS = TARGET_FIELDS

_COORDINATE_ATTRIBUTE_NAMES = (
    "raw_training_minimum",
    "raw_training_maximum",
    "coordinate_center",
    "coordinate_half_span",
    "coordinate_scale_source",
    "coordinate_scale_margin",
    "coordinate_min",
    "coordinate_max",
)


def _attribute_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def fit_coordinate_transform(
    raw_training_file: str | Path,
    *,
    source: str,
    prior_half_span: float | Sequence[float] | np.ndarray | None,
) -> CoordinateTransform:
    """Fit the affine map from the complete raw NASA training union.

    This is the only full-coordinate scan used by canonical generation.  The
    runtime loader below reconstructs the expected transform from the frozen
    raw-bound attributes instead of rescanning the multi-gigabyte source file.
    """

    filename = Path(raw_training_file).resolve()
    if not filename.exists():
        raise FileNotFoundError(filename)
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    case_count = 0
    if filename.is_dir():
        raw_manifest = json.loads(
            (filename / "data_manifest.json").read_text(encoding="utf-8")
        )
        split = raw_manifest["splits"]["train"]
        for case in split["cases"]:
            case_count += 1
            path = filename / str(split.get("directory", "train")) / str(case["file"])
            coordinates = np.load(path, mmap_mode="r", allow_pickle=False)[0][
                "coordinates"
            ]
            for start in range(0, int(coordinates.shape[0]), 262_144):
                block = np.asarray(coordinates[start : start + 262_144], np.float64)
                if not np.all(np.isfinite(block)):
                    raise ValueError(f"{path}:coordinates contains NaN/Inf")
                minimum = np.minimum(minimum, block.min(axis=0))
                maximum = np.maximum(maximum, block.max(axis=0))
    else:
        # Kept only for validating a legacy source before split.py publishes
        # the new canonical directory.
        with h5py.File(filename, "r") as handle:
            for key in sorted(handle.keys()):
                case_count += 1
                group = handle[key]
                for axis, name in enumerate(COORDINATE_FIELDS):
                    values = group.get(name)
                    if not isinstance(values, h5py.Dataset) or values.ndim != 1:
                        raise ValueError(f"{filename}:{key}/{name} must be a vector")
                    if values.shape[0] < 1:
                        raise ValueError(f"{filename}:{key}/{name} is empty")
                    for start in range(0, values.shape[0], 262_144):
                        block = np.asarray(values[start : start + 262_144], np.float64)
                        if not np.all(np.isfinite(block)):
                            raise ValueError(f"{filename}:{key}/{name} contains NaN/Inf")
                        minimum[axis] = min(minimum[axis], float(block.min()))
                        maximum[axis] = max(maximum[axis], float(block.max()))
    if case_count != EXPECTED_TRAIN_CASES:
        raise ValueError(
            f"NASA raw training cases={case_count}, expected={EXPECTED_TRAIN_CASES}"
        )
    return resolve_coordinate_transform(
        minimum,
        maximum,
        source=source,
        prior_half_span=prior_half_span,
    )


def load_coordinate_transform(
    data_dir: str | Path,
    *,
    source: str,
    prior_half_span: float | Sequence[float] | np.ndarray | None,
) -> CoordinateTransform:
    """Load and strictly validate the transform frozen into NASA moments."""

    paths = load_data_paths(data_dir)
    with h5py.File(paths.moment_file, "r") as moments:
        missing = [name for name in _COORDINATE_ATTRIBUTE_NAMES if name not in moments.attrs]
        if missing:
            raise ValueError(
                "NASA moment artifact uses an old/stale coordinate schema; "
                f"regenerate data (missing attributes: {missing})"
            )
        raw_minimum = np.asarray(moments.attrs["raw_training_minimum"], np.float64)
        raw_maximum = np.asarray(moments.attrs["raw_training_maximum"], np.float64)
        stored_center = np.asarray(moments.attrs["coordinate_center"], np.float64)
        stored_half_span = np.asarray(
            moments.attrs["coordinate_half_span"], np.float64
        )
        stored_source = _attribute_text(moments.attrs["coordinate_scale_source"])
        stored_margin = float(moments.attrs["coordinate_scale_margin"])
        stored_minimum = np.asarray(moments.attrs["coordinate_min"], np.float64)
        stored_maximum = np.asarray(moments.attrs["coordinate_max"], np.float64)

    expected = resolve_coordinate_transform(
        raw_minimum,
        raw_maximum,
        source=source,
        prior_half_span=prior_half_span,
    )
    validate_stored_coordinate_transform(
        expected,
        center=stored_center,
        half_span=stored_half_span,
        source=stored_source,
        margin=stored_margin,
    )
    if not np.allclose(stored_minimum, expected.minimum, rtol=0.0, atol=1.0e-12):
        raise ValueError("NASA moment coordinate_min differs from center-half_span")
    if not np.allclose(stored_maximum, expected.maximum, rtol=0.0, atol=1.0e-12):
        raise ValueError("NASA moment coordinate_max differs from center+half_span")
    manifest_transform = paths.manifest.get("moment", {}).get("coordinate_transform")
    if manifest_transform != expected.metadata():
        raise ValueError(
            "NASA manifest coordinate transform does not match train.py; "
            "regenerate data"
        )
    return expected


def _json_attribute(handle: h5py.Group | h5py.File, name: str) -> tuple[str, ...]:
    value = handle.attrs[name]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    parsed = json.loads(value)
    if not isinstance(parsed, list) or any(
        not isinstance(item, str) for item in parsed
    ):
        raise TypeError(f"{handle.name}:{name} must be a JSON string list")
    return tuple(parsed)


def validate_nasa_schema(
    data: str | Path,
    *,
    expected_mode: int,
    verify_payload_hashes: bool = False,
) -> dict[str, int]:
    """Validate the canonical raw/moment data."""

    path = Path(data).resolve()
    if not path.is_dir():
        raise ValueError(
            "NASA data must be the canonical directory containing manifest.json"
        )
    paths = load_data_paths(path)
    if verify_payload_hashes:
        for split in ("train", "test"):
            details = paths.raw_manifest["splits"][split]
            split_root = paths.raw_dir / str(details.get("directory", split))
            for case in details["cases"]:
                raw_file = split_root / str(case["file"])
                if file_sha256(raw_file) != case["sha256"]:
                    raise ValueError(f"NASA raw {split}/{case['key']} SHA256 differs")
        if file_sha256(paths.moment_file) != paths.manifest["moment"]["sha256"]:
            raise ValueError("NASA moment SHA256 differs from manifest")
    moment_file = paths.moment_file
    raw_store = NASARawStore(paths, cache_cases=False)
    try:
        keys = {split: raw_store.keys(split) for split in ("train", "test")}
        point_counts = [raw_store.point_count("train", key) for key in keys["train"]]
    finally:
        raw_store.close()

    if len(keys["train"]) != EXPECTED_TRAIN_CASES:
        raise ValueError(f"NASA train cases={len(keys['train'])}, expected 105")
    if len(keys["test"]) != EXPECTED_TEST_CASES:
        raise ValueError(f"NASA test cases={len(keys['test'])}, expected 44")
    with h5py.File(moment_file, "r") as moments:
        mode = int(moments.attrs.get("mode_per_axis", -1))
        if mode != expected_mode:
            raise ValueError(f"NASA mode={mode}, expected={expected_mode}")
        if moments.attrs.get("basis") != EXPECTED_BASIS:
            raise ValueError("NASA basis mismatch")
        if moments.attrs.get("flatten_order") != EXPECTED_FLATTEN_ORDER:
            raise ValueError("NASA flatten order mismatch")
        if (
            _json_attribute(moments, "geometry_moment_fields")
            != GEOMETRY_MOMENT_CHANNELS
        ):
            raise ValueError("NASA geometry moment channel order mismatch")
        for split in ("train", "test"):
            moment_keys = sorted(moments[split].keys())
            if moment_keys != keys[split]:
                raise ValueError(f"NASA raw/moment keys differ for {split}")
            for key in moment_keys:
                dataset = moments[split][key].get("geometry_moments")
                if not isinstance(dataset, h5py.Dataset) or dataset.shape != (
                    len(GEOMETRY_MOMENT_CHANNELS),
                    mode,
                    mode,
                    mode,
                ):
                    raise ValueError(f"invalid NASA moment {split}/{key}")
        missing_coordinate_attributes = [
            name for name in _COORDINATE_ATTRIBUTE_NAMES if name not in moments.attrs
        ]
        if missing_coordinate_attributes:
            raise ValueError(
                "NASA moment artifact uses an old/stale coordinate schema; "
                f"regenerate data (missing attributes: {missing_coordinate_attributes})"
            )
        coordinate_center = np.asarray(
            moments.attrs["coordinate_center"], np.float64
        )
        coordinate_half_span = np.asarray(
            moments.attrs["coordinate_half_span"], np.float64
        )
        if (
            coordinate_center.shape != (3,)
            or coordinate_half_span.shape != (3,)
            or not np.all(np.isfinite(coordinate_center))
            or not np.all(np.isfinite(coordinate_half_span))
            or np.any(coordinate_half_span <= 0.0)
        ):
            raise ValueError("NASA coordinate transform is invalid")
    return {
        "mode": expected_mode,
        "train_cases": len(keys["train"]),
        "test_cases": len(keys["test"]),
        "train_point_pool_min": min(point_counts) if point_counts else -1,
        "train_point_pool_max": max(point_counts) if point_counts else -1,
    }


def read_keys(data: str | Path, split: str) -> list[str]:
    paths = load_data_paths(data)
    store = NASARawStore(paths, cache_cases=False)
    try:
        return store.keys(split)
    finally:
        store.close()


class NASACRMSurfaceDataset:
    """Return moment tokens plus Local queries sampled from full raw surfaces."""

    def __init__(
        self,
        data: str | Path,
        split: str,
        keys: list[str],
        query_limit: int | None,
        random_queries: bool,
        coordinate_transform: CoordinateTransform,
        use_condition_moments: bool = True,
        query_seed: int = 42,
        cache_points: bool = True,
    ) -> None:
        self.split = split
        self.keys = list(keys)
        self.query_limit = normalize_query_limit(query_limit)
        self.random_queries = bool(random_queries)
        self.query_seed = int(query_seed)
        self.epoch = 0
        self.use_condition_moments = bool(use_condition_moments)
        self._handle: h5py.File | None = None
        self.paths = load_data_paths(data)
        self.filename = str(self.paths.moment_file)
        self.raw_store = NASARawStore(self.paths, cache_cases=cache_points)
        if not isinstance(coordinate_transform, CoordinateTransform):
            raise TypeError("coordinate_transform must be a CoordinateTransform")
        if coordinate_transform.center.shape != (3,):
            raise ValueError("NASA coordinate transform must be three-dimensional")
        self.coordinate_transform = coordinate_transform
        with h5py.File(self.filename, "r") as moments:
            stored_center = np.asarray(
                moments.attrs["coordinate_center"], np.float64
            )
            stored_half_span = np.asarray(
                moments.attrs["coordinate_half_span"], np.float64
            )
            if not np.allclose(
                stored_center, coordinate_transform.center, rtol=0.0, atol=1.0e-12
            ) or not np.allclose(
                stored_half_span,
                coordinate_transform.half_span,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError(
                    "Dataset coordinate transform differs from the moment artifact"
                )
            self.clip_coordinates = bool(
                moments.attrs.get("coordinates_clipped", False)
            )

    def __len__(self) -> int:
        return len(self.keys)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.filename, "r")
        return self._handle

    def __getitem__(self, index: int) -> dict[str, object]:
        import torch

        key = self.keys[index]
        moment_group = self._file()[self.split][key]
        moments = np.asarray(moment_group["geometry_moments"], np.float32)
        count = self.raw_store.point_count(self.split, key)
        conditions = self.raw_store.conditions(self.split, key)
        selection, pool_indices = query_selection(
            count,
            self.query_limit,
            random_queries=self.random_queries,
            seed=self.query_seed,
            epoch=self.epoch,
            sample_identity=("nasa", self.split, key),
        )
        physical, normals, targets, surface = self.raw_store.selected_case(
            self.split, key, selection
        )
        coordinates = self.coordinate_transform.normalize(physical)
        if self.clip_coordinates:
            np.clip(coordinates, -1.0, 1.0, out=coordinates)
        coordinates = coordinates.astype(np.float32)
        query_indices = pool_indices

        moment_tokens = moments.reshape(len(GEOMETRY_MOMENT_CHANNELS), -1).T
        if self.use_condition_moments:
            condition_moments = derive_condition_moments(moments, conditions)
            moment_tokens = np.concatenate(
                (
                    moment_tokens,
                    condition_moments.reshape(len(CONDITION_CHANNELS), -1).T,
                ),
                axis=-1,
            )
        functions = np.concatenate(
            (
                normals,
                np.broadcast_to(
                    conditions, (coordinates.shape[0], conditions.shape[0])
                ),
            ),
            axis=-1,
        )
        return {
            "sample": key,
            "moments": torch.from_numpy(
                np.ascontiguousarray(moment_tokens, dtype=np.float32)
            ),
            "coordinates": torch.from_numpy(
                np.ascontiguousarray(coordinates, dtype=np.float32)
            ),
            "functions": torch.from_numpy(
                np.ascontiguousarray(functions, dtype=np.float32)
            ),
            "normals": torch.from_numpy(
                np.array(normals, dtype=np.float32, copy=True, order="C")
            ),
            "targets": torch.from_numpy(
                np.array(targets, dtype=np.float32, copy=True, order="C")
            ),
            "surface": torch.from_numpy(
                np.array(surface, dtype=np.float32, copy=True, order="C")
            ),
            "query_indices": torch.from_numpy(
                np.ascontiguousarray(query_indices, dtype=np.int64)
            ),
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state


def compute_statistical_scaling(
    data: Path,
    training_keys: list[str],
    use_condition_moments: bool,
    *,
    coordinate_transform: CoordinateTransform,
) -> ModelScaling:
    """Fit the sole per-case/full-raw normalization on complete train cases."""

    path = Path(data).resolve()
    if not path.is_dir():
        raise ValueError("formal NASA scaling requires the canonical data directory")
    paths = load_data_paths(path)
    raw = NASARawStore(paths, cache_cases=False)
    geometry_rows: list[np.ndarray] = []
    condition_rows: list[np.ndarray] = []
    condition_moment_rows: list[np.ndarray] = []
    try:
        with h5py.File(paths.moment_file, "r") as moments:
            validate_stored_coordinate_transform(
                coordinate_transform,
                center=moments.attrs["coordinate_center"],
                half_span=moments.attrs["coordinate_half_span"],
                source=_attribute_text(moments.attrs["coordinate_scale_source"]),
                margin=float(moments.attrs["coordinate_scale_margin"]),
            )
            for key in training_keys:
                geometry = np.asarray(
                    moments[f"train/{key}/geometry_moments"], np.float32
                )
                conditions = raw.conditions("train", key)
                geometry_rows.append(
                    geometry.reshape(len(GEOMETRY_MOMENT_CHANNELS), -1)
                    .T.astype(np.float64)
                )
                condition_rows.append(conditions.astype(np.float64))
                condition_moment_rows.append(
                    derive_condition_moments(geometry, conditions)
                    .reshape(len(CONDITION_CHANNELS), -1)
                    .T.astype(np.float64)
                )
        target_stats = ragged_channel_statistics(
            raw.target_cases("train", training_keys)
        )
    finally:
        raw.close()
    geometry_values = np.stack(geometry_rows)
    condition_moments = np.stack(condition_moment_rows)
    conditions = np.stack(condition_rows)
    input_values = (
        np.concatenate((geometry_values, condition_moments), axis=-1)
        if use_condition_moments
        else geometry_values
    )
    condition_mean = conditions.mean(0)
    condition_std = stable_std(conditions.std(0))
    function_mean, function_std = compose_function_scaling(
        condition_mean, condition_std, normal_channels=3
    )
    return ModelScaling(
        input_mean=input_values.mean(0).astype(np.float32),
        input_std=stable_std(input_values.std(0)).astype(np.float32),
        function_mean=function_mean,
        function_std=function_std,
        target_mean=target_stats.mean.astype(np.float32),
        target_std=target_stats.std.astype(np.float32),
        coordinate_scale=coordinate_transform.half_span.astype(np.float32),
    )


__all__ = [
    "CONDITION_CHANNELS",
    "EXPECTED_TEST_CASES",
    "EXPECTED_TRAIN_CASES",
    "GEOMETRY_MOMENT_CHANNELS",
    "GLOBAL_MOMENT_CHANNELS",
    "GLOBAL_MOMENT_CHANNEL_INDEX",
    "NASACRMSurfaceDataset",
    "RAW_COORDINATE_FIELDS",
    "RAW_NORMAL_FIELDS",
    "RAW_TARGET_FIELDS",
    "TARGET_CHANNELS",
    "compute_statistical_scaling",
    "fit_coordinate_transform",
    "load_coordinate_transform",
    "read_keys",
    "validate_nasa_schema",
]
