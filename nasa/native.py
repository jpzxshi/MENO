"""Lazy per-case NPY access for the canonical NASA-CRM point surfaces."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


FORMAT_ID = "mfd-nasa-native"
CONDITION_CHANNELS = (
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


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_values(
    dataset: np.ndarray,
    selection: slice | np.ndarray,
    *,
    block_size: int = 65_536,
) -> np.ndarray:
    """Gather random rows from a per-case NPY field with bounded scratch memory."""

    if isinstance(selection, slice):
        return np.asarray(dataset[selection], np.float32)
    indices = np.asarray(selection, np.int64)
    unique, inverse = np.unique(indices, return_inverse=True)
    if len(unique) <= 1_024:
        return np.asarray(dataset[unique], np.float32)[inverse]
    values = np.empty((len(unique), *dataset.shape[1:]), np.float32)
    block_ids = unique // block_size
    for block_id in np.unique(block_ids):
        positions = np.flatnonzero(block_ids == block_id)
        start = int(block_id) * block_size
        stop = min(start + block_size, int(dataset.shape[0]))
        block = np.asarray(dataset[start:stop], np.float32)
        values[positions] = block[unique[positions] - start]
    return values[inverse]


@dataclass(frozen=True)
class NASADataPaths:
    data_dir: Path
    manifest_path: Path
    moment_file: Path
    raw_dir: Path
    raw_manifest_path: Path
    raw_manifest: dict[str, Any]
    manifest: dict[str, Any]


def load_data_paths(data_dir: str | Path) -> NASADataPaths:
    """Resolve and structurally validate the canonical manifest paths."""

    root = Path(data_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT_ID:
        raise ValueError(
            f"NASA manifest format={payload.get('format')!r}, "
            f"expected {FORMAT_ID!r}"
        )
    raw = payload.get("raw")
    moment = payload.get("moment")
    if not isinstance(raw, dict) or not isinstance(moment, dict):
        raise TypeError("NASA manifest raw/moment entries must be objects")
    raw_dir = root / str(raw.get("directory", "raw"))
    raw_manifest_path = root / str(raw.get("manifest", "raw/data_manifest.json"))
    moment_file = root / str(moment["file"])
    if not raw_dir.is_dir():
        raise FileNotFoundError(raw_dir)
    for path in (raw_manifest_path, moment_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    if file_sha256(raw_manifest_path) != raw.get("manifest_sha256"):
        raise ValueError("NASA raw manifest SHA256 mismatch")
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    if raw_manifest.get("format") != "mfd-nasa-raw-npy-v1":
        raise ValueError("NASA raw NPY manifest format mismatch")
    for split in ("train", "test"):
        details = raw_manifest.get("splits", {}).get(split)
        if not isinstance(details, dict):
            raise ValueError(f"NASA raw manifest lacks split {split}")
        split_root = raw_dir / str(details.get("directory", split))
        cases = details.get("cases")
        if not isinstance(cases, list) or len(cases) != int(details.get("samples", -1)):
            raise ValueError(f"NASA raw manifest {split} case catalog mismatch")
        for case in cases:
            path = split_root / str(case.get("file"))
            if not path.is_file():
                raise FileNotFoundError(path)
    with h5py.File(moment_file, "r") as handle:
        if handle.attrs.get("format") != FORMAT_ID:
            raise ValueError(
                f"NASA moment format={handle.attrs.get('format')!r}, "
                f"expected {FORMAT_ID!r}"
            )
    return NASADataPaths(
        root,
        manifest_path,
        moment_file,
        raw_dir,
        raw_manifest_path,
        raw_manifest,
        payload,
    )


def normalization_file(paths: NASADataPaths) -> Path:
    """Resolve the sole canonical scaling when provenance and hash agree."""

    details = paths.manifest.get("normalization")
    if not isinstance(details, dict):
        raise ValueError("NASA canonical normalization artifact is missing")
    expected_relative = "moment/normalization/canonical.npz"
    if details.get("file") != expected_relative:
        raise ValueError("NASA canonical normalization path mismatch")
    if details.get("weighting") != "uniform_per_case":
        raise ValueError("NASA canonical normalization weighting mismatch")
    if details.get("sampling") != "full_raw":
        raise ValueError("NASA canonical normalization sampling mismatch")
    if details.get("source") != "train/full_raw+train/moment":
        raise ValueError("NASA canonical normalization source mismatch")
    expected_lineage = {
        "raw_train_sha256": paths.manifest["raw"]["manifest_sha256"],
        "moment_sha256": paths.manifest["moment"]["sha256"],
    }
    if details.get("lineage") != expected_lineage:
        raise ValueError("NASA canonical normalization lineage mismatch")
    path = paths.data_dir / expected_relative
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != details.get("sha256"):
        raise ValueError("NASA canonical normalization SHA256 mismatch")
    return path


class NASARawStore:
    """Process-local reader for one structured, mmap-capable NPY per case."""

    def __init__(self, paths: NASADataPaths, *, cache_cases: bool = True) -> None:
        self.paths = paths
        self.cache_cases = bool(cache_cases)
        self._cache: dict[tuple[str, str], tuple[np.ndarray, ...]] = {}
        self._cases = {
            split: {str(case["key"]): case for case in self.paths.raw_manifest["splits"][split]["cases"]}
            for split in ("train", "test")
        }

    def _case_path(self, split: str, key: str) -> Path:
        if split not in self._cases or key not in self._cases[split]:
            raise KeyError(f"unknown NASA case: {split}/{key}")
        details = self.paths.raw_manifest["splits"][split]
        return (
            self.paths.raw_dir
            / str(details.get("directory", split))
            / str(self._cases[split][key]["file"])
        )

    def _record(self, split: str, key: str) -> np.void:
        values = np.load(self._case_path(split, key), mmap_mode="r", allow_pickle=False)
        if values.shape != (1,):
            raise ValueError(f"NASA split NPY must contain one record: {split}/{key}")
        return values[0]

    def keys(self, split: str) -> list[str]:
        if split not in self._cases:
            raise ValueError(f"unsupported NASA split: {split!r}")
        return list(self._cases[split])

    def point_count(self, split: str, key: str) -> int:
        if split not in self._cases or key not in self._cases[split]:
            raise KeyError(f"unknown NASA case: {split}/{key}")
        return int(self._cases[split][key]["points"])

    def conditions(self, split: str, key: str) -> np.ndarray:
        return np.asarray(self._record(split, key)["conditions"], np.float32)

    def full_case(
        self, split: str, key: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return physical coordinates, normals, targets and surface weights."""

        identity = (split, key)
        cached = self._cache.get(identity)
        if cached is not None:
            return cached  # type: ignore[return-value]
        record = self._record(split, key)
        coordinates = np.asarray(record["coordinates"])
        normals = np.asarray(record["normals"])
        targets = np.asarray(record["targets"])
        surface = np.asarray(record["surface"])
        arrays = (coordinates, normals, targets, surface)
        if len({array.shape[0] for array in arrays}) != 1:
            raise ValueError(f"NASA raw fields differ in length for {split}/{key}")
        if self.cache_cases:
            arrays = tuple(np.array(array, copy=True, order="C") for array in arrays)
            self._cache[identity] = arrays
        return arrays

    def selected_case(
        self,
        split: str,
        key: str,
        selection: slice | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cached = self._cache.get((split, key))
        if cached is not None:
            return tuple(array[selection] for array in cached)  # type: ignore[return-value]
        if self.cache_cases:
            return tuple(
                array[selection] for array in self.full_case(split, key)
            )  # type: ignore[return-value]
        record = self._record(split, key)
        coordinates = _selected_values(record["coordinates"], selection)
        normals = _selected_values(record["normals"], selection)
        targets = _selected_values(record["targets"], selection)
        surface = _selected_values(record["surface"], selection)
        return coordinates, normals, targets, surface

    def target_cases(self, split: str, keys: list[str]):
        """Yield full raw targets case-by-case for ragged statistics."""

        for key in keys:
            yield np.asarray(self._record(split, key)["targets"], np.float32)

    def close(self) -> None:
        self._cache.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "CONDITION_CHANNELS",
    "COORDINATE_FIELDS",
    "FORMAT_ID",
    "NASADataPaths",
    "NASARawStore",
    "NORMAL_FIELDS",
    "TARGET_FIELDS",
    "file_sha256",
    "load_data_paths",
    "normalization_file",
]
