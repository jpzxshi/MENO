"""AhmedML raw-native dataset and training normalization."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

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

from .native import AhmedRawStore, file_sha256, load_data_paths, run_group


GEOMETRY_MOMENT_CHANNELS = ("surface", "normal_x", "normal_y", "normal_z")
CONDITION_CHANNELS = (
    "body-length",
    "body-height",
    "body-width",
    "front-arc-diameter",
    "slant-angle-length",
    "slant-angle-height",
    "slant-surface-length",
    "slant-angle-degrees",
)
GLOBAL_MOMENT_CHANNELS = GEOMETRY_MOMENT_CHANNELS + CONDITION_CHANNELS
GLOBAL_MOMENT_CHANNEL_INDEX = {
    name: index for index, name in enumerate(GLOBAL_MOMENT_CHANNELS)
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    moment_channels: int
    normal_channels: int
    condition_channels: int
    scalar_name: str
    vector_name: str


TASKS = {
    "surface": TaskSpec(
        "surface",
        len(GEOMETRY_MOMENT_CHANNELS),
        3,
        len(CONDITION_CHANNELS),
        "Cp_static",
        "wall_shear",
    )
}

COORDINATE_ATTRIBUTE_NAMES = {
    "center": "coordinate_center",
    "half_span": "coordinate_half_span",
    "source": "coordinate_scale_source",
    "margin": "coordinate_scale_margin",
    "minimum": "coordinate_min",
    "maximum": "coordinate_max",
    "raw_minimum": "raw_training_minimum",
    "raw_maximum": "raw_training_maximum",
}


def _attribute_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def fit_coordinate_transform(
    raw_file: str | Path,
    training_runs: list[int],
    *,
    source: str,
    prior_half_span: tuple[float, ...] | list[float] | np.ndarray | None,
    chunk_size: int = 262_144,
) -> CoordinateTransform:
    """Fit the raw train-union bounds and resolve Ahmed's one affine map."""

    if not training_runs:
        raise ValueError("Ahmed coordinate fitting requires training runs")
    if chunk_size < 1:
        raise ValueError("coordinate fitting chunk_size must be positive")
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    source_path = Path(raw_file).resolve()
    if source_path.is_dir():
        raw_manifest = json.loads(
            (source_path / "data_manifest.json").read_text(encoding="utf-8")
        )
        cases = {int(case["run"]): case for case in raw_manifest["cases"]}
        for run in training_runs:
            case = cases.get(int(run))
            if case is None:
                raise KeyError(f"missing Ahmed raw run {run}")
            coordinates = np.load(
                source_path / str(case["file"]), mmap_mode="r", allow_pickle=False
            )[0]["coordinates"]
            if coordinates.ndim != 2 or coordinates.shape[1] != 3:
                raise ValueError(
                    f"run {run} coordinates shape={coordinates.shape}, expected [N,3]"
                )
            for start in range(0, int(coordinates.shape[0]), chunk_size):
                values = np.asarray(
                    coordinates[start : start + chunk_size], dtype=np.float64
                )
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"run {run} coordinates contain NaN/Inf")
                minimum = np.minimum(minimum, values.min(axis=0))
                maximum = np.maximum(maximum, values.max(axis=0))
    else:
        # Legacy HDF5 remains readable only while split.py is being validated.
        with h5py.File(source_path, "r") as raw:
            for run in training_runs:
                coordinates = raw[f"{run_group(run)}/coordinates"]
                if coordinates.ndim != 2 or coordinates.shape[1] != 3:
                    raise ValueError(
                        f"{coordinates.name} shape={coordinates.shape}, expected [N,3]"
                    )
                for start in range(0, int(coordinates.shape[0]), chunk_size):
                    values = np.asarray(
                        coordinates[start : start + chunk_size], dtype=np.float64
                    )
                    if not np.all(np.isfinite(values)):
                        raise ValueError(f"{coordinates.name} contains NaN/Inf")
                    minimum = np.minimum(minimum, values.min(axis=0))
                    maximum = np.maximum(maximum, values.max(axis=0))
    return resolve_coordinate_transform(
        minimum,
        maximum,
        source=source,
        prior_half_span=prior_half_span,
    )


def _stored_coordinate_transform(
    moments: h5py.File,
    *,
    source: str,
    prior_half_span: tuple[float, ...] | list[float] | np.ndarray | None,
) -> CoordinateTransform:
    missing = [
        name for name in COORDINATE_ATTRIBUTE_NAMES.values() if name not in moments.attrs
    ]
    if missing:
        raise ValueError(
            "Ahmed moment artifact predates the coordinate-transform schema; "
            f"regenerate data, missing attrs={missing}"
        )
    raw_minimum = np.asarray(
        moments.attrs[COORDINATE_ATTRIBUTE_NAMES["raw_minimum"]], np.float64
    )
    raw_maximum = np.asarray(
        moments.attrs[COORDINATE_ATTRIBUTE_NAMES["raw_maximum"]], np.float64
    )
    expected = resolve_coordinate_transform(
        raw_minimum,
        raw_maximum,
        source=source,
        prior_half_span=prior_half_span,
    )
    validate_stored_coordinate_transform(
        expected,
        center=moments.attrs[COORDINATE_ATTRIBUTE_NAMES["center"]],
        half_span=moments.attrs[COORDINATE_ATTRIBUTE_NAMES["half_span"]],
        source=_attribute_text(
            moments.attrs[COORDINATE_ATTRIBUTE_NAMES["source"]]
        ),
        margin=float(moments.attrs[COORDINATE_ATTRIBUTE_NAMES["margin"]]),
    )
    for label, actual, wanted in (
        (
            "minimum",
            moments.attrs[COORDINATE_ATTRIBUTE_NAMES["minimum"]],
            expected.minimum,
        ),
        (
            "maximum",
            moments.attrs[COORDINATE_ATTRIBUTE_NAMES["maximum"]],
            expected.maximum,
        ),
    ):
        values = np.asarray(actual, dtype=np.float64)
        if values.shape != wanted.shape or not np.allclose(
            values, wanted, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                f"Ahmed stored coordinate {label} differs from resolved transform: "
                f"stored={values.tolist()}, expected={wanted.tolist()}"
            )
    return expected


def load_coordinate_transform(
    data_dir: str | Path,
    *,
    source: str,
    prior_half_span: tuple[float, ...] | list[float] | np.ndarray | None,
) -> CoordinateTransform:
    """Load raw train bounds from moments and strictly validate the stored map."""

    paths = load_data_paths(data_dir)
    with h5py.File(paths.moment_file, "r") as moments:
        expected = _stored_coordinate_transform(
            moments,
            source=source,
            prior_half_span=prior_half_span,
        )
    manifest_transform = paths.manifest.get("moment", {}).get("coordinate_transform")
    if manifest_transform != expected.metadata():
        raise ValueError(
            "Ahmed manifest coordinate transform does not match train.py; "
            "regenerate data"
        )
    return expected


def read_splits(data_dir: str | Path) -> dict[str, list[int]]:
    paths = load_data_paths(data_dir)
    store = AhmedRawStore(paths, cache_cases=False)
    try:
        return {
            split: store.split_runs(split) for split in ("train", "validation", "test")
        }
    finally:
        store.close()


def validate_ahmed_schema(
    data_dir: str | Path,
    *,
    expected_mode: int,
    verify_payload_hashes: bool = False,
) -> dict[str, int]:
    paths = load_data_paths(data_dir)
    if verify_payload_hashes:
        for case in paths.raw_manifest["cases"]:
            path = paths.raw_dir / str(case["file"])
            if file_sha256(path) != case["sha256"]:
                raise ValueError(f"Ahmed raw run {case['run']} SHA256 differs")
        if file_sha256(paths.moment_file) != paths.manifest["moment"]["sha256"]:
            raise ValueError("Ahmed moment SHA256 differs from manifest")
    store = AhmedRawStore(paths, cache_cases=False)
    try:
        splits = {
            split: store.split_runs(split) for split in ("train", "validation", "test")
        }
        seen: set[int] = set()
        for split, expected in (("train", 400), ("validation", 50), ("test", 50)):
            runs = splits[split]
            if len(runs) != expected or seen.intersection(runs):
                raise ValueError(f"invalid Ahmed {split} split")
            seen.update(runs)
        if seen != set(range(1, 501)):
            raise ValueError("Ahmed splits must cover runs 1..500")
        point_counts = [store.point_count(run) for run in splits["train"]]
        if tuple(paths.raw_manifest.get("fields", {}).get("geometry_parameters", ())) != CONDITION_CHANNELS:
            raise ValueError("Ahmed condition channel order mismatch")
        for run in range(1, 501):
            if store.conditions(run).shape != (len(CONDITION_CHANNELS),):
                raise ValueError(f"invalid Ahmed condition channels for run {run}")
        with h5py.File(paths.moment_file, "r") as moments:
            mode = int(moments.attrs.get("mode_per_axis", -1))
            if mode != expected_mode:
                raise ValueError(f"Ahmed mode={mode}, expected={expected_mode}")
            fit_runs = {int(value) for value in moments["fit_runs"][...]}
            if fit_runs != set(splits["train"]):
                raise ValueError("Ahmed moment bounds include non-train runs")
            geometry_fields = moments.attrs.get("geometry_moment_fields")
            if geometry_fields is None or tuple(
                json.loads(_attribute_text(geometry_fields))
            ) != GEOMETRY_MOMENT_CHANNELS:
                raise ValueError("Ahmed geometry moment channel order mismatch")
            missing_coordinate_attrs = [
                name
                for name in COORDINATE_ATTRIBUTE_NAMES.values()
                if name not in moments.attrs
            ]
            if missing_coordinate_attrs:
                raise ValueError(
                    "Ahmed moment artifact uses an obsolete coordinate schema; "
                    f"missing attrs={missing_coordinate_attrs}"
                )
            for run in range(1, 501):
                dataset = moments.get(f"{run_group(run)}/geometry_moments")
                if not isinstance(dataset, h5py.Dataset) or dataset.shape != (
                    len(GEOMETRY_MOMENT_CHANNELS),
                    mode,
                    mode,
                    mode,
                ):
                    raise ValueError(f"invalid Ahmed moment run {run}")
    finally:
        store.close()
    return {
        "mode": expected_mode,
        "train_cases": 400,
        "validation_cases": 50,
        "test_cases": 50,
        "train_point_pool_min": min(point_counts),
        "train_point_pool_max": max(point_counts),
    }


def compute_statistical_scaling(
    data_dir: Path,
    training_runs: list[int],
    mode: int,
    *,
    coordinate_transform: CoordinateTransform,
) -> ModelScaling:
    """Fit the one canonical per-case scaling on complete raw train cases."""

    if not training_runs:
        raise ValueError("training_runs must not be empty")
    paths = load_data_paths(data_dir)
    raw = AhmedRawStore(paths, cache_cases=False)
    moment_rows: list[np.ndarray] = []
    condition_rows: list[np.ndarray] = []
    try:
        with h5py.File(paths.moment_file, "r") as moments:
            if int(moments.attrs["mode_per_axis"]) != mode:
                raise ValueError("Ahmed requested mode differs from moment file")
            validate_stored_coordinate_transform(
                coordinate_transform,
                center=moments.attrs[COORDINATE_ATTRIBUTE_NAMES["center"]],
                half_span=moments.attrs[
                    COORDINATE_ATTRIBUTE_NAMES["half_span"]
                ],
                source=_attribute_text(
                    moments.attrs[COORDINATE_ATTRIBUTE_NAMES["source"]]
                ),
                margin=float(
                    moments.attrs[COORDINATE_ATTRIBUTE_NAMES["margin"]]
                ),
            )
            for run in training_runs:
                geometry = np.asarray(
                    moments[f"{run_group(run)}/geometry_moments"], np.float32
                )
                conditions = raw.conditions(run)
                tokens = geometry.reshape(len(GEOMETRY_MOMENT_CHANNELS), -1).T
                condition_tokens = derive_condition_moments(
                    geometry, conditions
                ).reshape(len(CONDITION_CHANNELS), -1).T
                tokens = np.concatenate((tokens, condition_tokens), axis=-1)
                moment_rows.append(tokens.astype(np.float64))
                condition_rows.append(conditions.astype(np.float64))
        target_stats = ragged_channel_statistics(raw.target_cases(training_runs))
    finally:
        raw.close()
    tokens = np.stack(moment_rows)
    conditions = np.stack(condition_rows)
    condition_mean = conditions.mean(0)
    condition_std = stable_std(conditions.std(0))
    function_mean, function_std = compose_function_scaling(
        condition_mean, condition_std, normal_channels=3
    )
    return ModelScaling(
        input_mean=tokens.mean(0).astype(np.float32),
        input_std=stable_std(tokens.std(0)).astype(np.float32),
        function_mean=function_mean,
        function_std=function_std,
        target_mean=target_stats.mean.astype(np.float32),
        target_std=target_stats.std.astype(np.float32),
        coordinate_scale=coordinate_transform.half_span.astype(np.float32),
    )


class AhmedMLDataset:
    """Return Global moments with deterministic queries from native raw data."""

    def __init__(
        self,
        data_dir: str | Path,
        task: str,
        runs: list[int],
        mode: int,
        query_limit: int | None,
        random_queries: bool,
        coordinate_transform: CoordinateTransform,
        cache_points: bool = True,
        query_seed: int | None = None,
        use_condition_moments: bool = True,
    ) -> None:
        if task not in TASKS:
            raise ValueError(f"unsupported task: {task}")
        self.data_dir = Path(data_dir).resolve()
        self.task = task
        self.spec = TASKS[task]
        self.runs = list(runs)
        self.mode = int(mode)
        self.query_limit = normalize_query_limit(query_limit)
        self.random_queries = bool(random_queries)
        self.cache_points = bool(cache_points)
        self.query_seed = 42 if query_seed is None else int(query_seed)
        self.epoch = 0
        self.use_condition_moments = bool(use_condition_moments)
        if coordinate_transform.center.shape != (3,):
            raise ValueError("Ahmed coordinate transform must be three-dimensional")
        self.coordinate_transform = coordinate_transform
        self._moment_handle: h5py.File | None = None
        self._moment_cache: dict[int, np.ndarray] = {}
        self._condition_cache: dict[int, np.ndarray] = {}
        self.paths = load_data_paths(self.data_dir)
        self.raw_store = AhmedRawStore(self.paths, cache_cases=cache_points)
        with h5py.File(self.paths.moment_file, "r") as moments:
            validate_stored_coordinate_transform(
                coordinate_transform,
                center=moments.attrs[COORDINATE_ATTRIBUTE_NAMES["center"]],
                half_span=moments.attrs[
                    COORDINATE_ATTRIBUTE_NAMES["half_span"]
                ],
                source=_attribute_text(
                    moments.attrs[COORDINATE_ATTRIBUTE_NAMES["source"]]
                ),
                margin=float(
                    moments.attrs[COORDINATE_ATTRIBUTE_NAMES["margin"]]
                ),
            )

    def __len__(self) -> int:
        return len(self.runs)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _moments(self) -> h5py.File:
        assert self.paths is not None
        if self._moment_handle is None:
            self._moment_handle = h5py.File(self.paths.moment_file, "r")
        return self._moment_handle

    def _case(self, run: int) -> tuple[np.ndarray, np.ndarray, int]:
        moments = self._moment_cache.get(run)
        if moments is None:
            moments = np.asarray(
                self._moments()[f"{run_group(run)}/geometry_moments"], np.float32
            )
            self._moment_cache[run] = moments
        conditions = self._condition_cache.get(run)
        if conditions is None:
            conditions = self.raw_store.conditions(run)
            self._condition_cache[run] = conditions
        return (
            moments,
            conditions,
            self.raw_store.point_count(run),
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        import torch

        run = self.runs[index]
        moments, conditions, point_count = self._case(run)
        selection, indices = query_selection(
            point_count,
            self.query_limit,
            random_queries=self.random_queries,
            seed=self.query_seed,
            epoch=self.epoch,
            sample_identity=("ahmed", self.task, run),
        )
        physical, selected_normals, selected_targets = self.raw_store.selected_case(
            run, selection
        )
        selected_coordinates = self.coordinate_transform.normalize(physical).astype(
            np.float32
        )
        moment_tokens = moments.reshape(len(GEOMETRY_MOMENT_CHANNELS), -1).T
        if self.use_condition_moments:
            condition_tokens = (
                derive_condition_moments(moments, conditions)
                .reshape(len(CONDITION_CHANNELS), -1)
                .T
            )
            moment_tokens = np.concatenate((moment_tokens, condition_tokens), axis=-1)
        functions = np.concatenate(
            (
                selected_normals,
                np.broadcast_to(
                    conditions, (selected_coordinates.shape[0], conditions.shape[0])
                ),
            ),
            axis=-1,
        )
        return {
            "sample": f"run_{run}",
            "run": torch.tensor(run, dtype=torch.int64),
            "moments": torch.from_numpy(
                np.ascontiguousarray(moment_tokens, dtype=np.float32)
            ),
            "coordinates": torch.from_numpy(
                np.ascontiguousarray(selected_coordinates, dtype=np.float32)
            ),
            "functions": torch.from_numpy(
                np.ascontiguousarray(functions, dtype=np.float32)
            ),
            "normals": torch.from_numpy(
                np.array(selected_normals, dtype=np.float32, copy=True, order="C")
            ),
            "targets": torch.from_numpy(
                np.array(selected_targets, dtype=np.float32, copy=True, order="C")
            ),
            "query_indices": torch.from_numpy(
                np.ascontiguousarray(indices, dtype=np.int64)
            ),
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_moment_handle"] = None
        state["_moment_cache"] = {}
        state["_condition_cache"] = {}
        return state


__all__ = [
    "AhmedMLDataset",
    "CONDITION_CHANNELS",
    "GEOMETRY_MOMENT_CHANNELS",
    "GLOBAL_MOMENT_CHANNELS",
    "GLOBAL_MOMENT_CHANNEL_INDEX",
    "TASKS",
    "TaskSpec",
    "compute_statistical_scaling",
    "fit_coordinate_transform",
    "load_coordinate_transform",
    "read_splits",
    "validate_ahmed_schema",
]
