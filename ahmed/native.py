"""Lazy per-run NPY access for the canonical AhmedML surface dataset."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


FORMAT_ID = "mfd-ahmed-native"
EXPECTED_SPLITS = {"train": 400, "validation": 50, "test": 50}


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
    """Gather random rows from a per-run NPY field with bounded scratch memory."""

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


def run_group(run: int) -> str:
    return f"runs/{int(run):06d}"


@dataclass(frozen=True)
class AhmedDataPaths:
    data_dir: Path
    manifest_path: Path
    raw_dir: Path
    raw_manifest_path: Path
    raw_manifest: dict[str, Any]
    moment_file: Path
    manifest: dict[str, Any]


def load_data_paths(data_dir: str | Path) -> AhmedDataPaths:
    root = Path(data_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT_ID:
        raise ValueError(
            f"Ahmed manifest format={payload.get('format')!r}, "
            f"expected {FORMAT_ID!r}"
        )
    normalization = payload.get("normalization")
    if normalization is not None:
        if not isinstance(normalization, dict):
            raise TypeError("Ahmed normalization manifest entry must be an object")
        if normalization.get("weighting") != "uniform_per_case":
            raise ValueError("Ahmed normalization weighting must be uniform_per_case")
        if normalization.get("sampling") != "full_raw":
            raise ValueError("Ahmed normalization sampling must be full_raw")
    raw_details = payload["raw"]
    raw_dir = root / str(raw_details.get("directory", "raw"))
    raw_manifest_path = root / str(raw_details.get("manifest", "raw/data_manifest.json"))
    moment_file = root / str(payload["moment"]["file"])
    if not raw_dir.is_dir():
        raise FileNotFoundError(raw_dir)
    for path in (raw_manifest_path, moment_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    if file_sha256(raw_manifest_path) != raw_details.get("manifest_sha256"):
        raise ValueError("Ahmed raw manifest SHA256 mismatch")
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    if raw_manifest.get("format") != "mfd-ahmed-surface-raw-npy-v1":
        raise ValueError("Ahmed raw NPY manifest format mismatch")
    cases = raw_manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 500:
        raise ValueError("Ahmed raw NPY manifest must contain 500 cases")
    for case in cases:
        path = raw_dir / str(case.get("file"))
        if not path.is_file():
            raise FileNotFoundError(path)
    with h5py.File(moment_file, "r") as handle:
        if handle.attrs.get("format") != FORMAT_ID:
            raise ValueError(
                f"Ahmed moment format={handle.attrs.get('format')!r}, expected={FORMAT_ID!r}"
            )
    return AhmedDataPaths(
        root,
        manifest_path,
        raw_dir,
        raw_manifest_path,
        raw_manifest,
        moment_file,
        payload,
    )


def normalization_file(paths: AhmedDataPaths) -> Path:
    """Resolve the one canonical frozen scaling with strict lineage checks."""

    details = paths.manifest.get("normalization")
    if not isinstance(details, dict):
        raise ValueError("Ahmed canonical normalization artifact is missing")
    expected_relative = "moment/normalization/canonical.npz"
    if details.get("file") != expected_relative:
        raise ValueError("Ahmed canonical normalization path mismatch")
    if details.get("weighting") != "uniform_per_case":
        raise ValueError("Ahmed canonical normalization weighting mismatch")
    if details.get("sampling") != "full_raw":
        raise ValueError("Ahmed canonical normalization sampling mismatch")
    if details.get("source") != "train/full_raw+train/moment":
        raise ValueError("Ahmed canonical normalization source mismatch")
    expected_lineage = {
        "raw_sha256": paths.manifest["raw"]["manifest_sha256"],
        "moment_sha256": paths.manifest["moment"]["sha256"],
    }
    if details.get("lineage") != expected_lineage:
        raise ValueError("Ahmed canonical normalization lineage mismatch")
    path = paths.data_dir / expected_relative
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != details.get("sha256"):
        raise ValueError("Ahmed canonical normalization SHA256 mismatch")
    return path


class AhmedRawStore:
    """Process-local reader for one structured, mmap-capable NPY per run."""

    def __init__(self, paths: AhmedDataPaths, *, cache_cases: bool = True) -> None:
        self.paths = paths
        self.cache_cases = bool(cache_cases)
        self._cache: dict[int, tuple[np.ndarray, ...]] = {}
        self._cases = {int(case["run"]): case for case in paths.raw_manifest["cases"]}

    def _record(self, run: int) -> np.void:
        case = self._cases.get(int(run))
        if case is None:
            raise KeyError(f"unknown Ahmed run: {run}")
        values = np.load(
            self.paths.raw_dir / str(case["file"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        if values.shape != (1,):
            raise ValueError(f"Ahmed split NPY must contain one record: run={run}")
        return values[0]

    def split_runs(self, split: str) -> list[int]:
        if split not in EXPECTED_SPLITS:
            raise ValueError(f"unsupported Ahmed split: {split!r}")
        return [int(value) for value in self.paths.raw_manifest["splits"][split]]

    def point_count(self, run: int) -> int:
        case = self._cases.get(int(run))
        if case is None:
            raise KeyError(f"unknown Ahmed run: {run}")
        return int(case["points"])

    def conditions(self, run: int) -> np.ndarray:
        return np.asarray(self._record(run)["geometry_parameters"], np.float32)

    def full_case(
        self, run: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cached = self._cache.get(int(run))
        if cached is not None:
            return cached  # type: ignore[return-value]
        record = self._record(run)
        arrays = (
            np.asarray(record["coordinates"]),
            np.asarray(record["normals"]),
            np.asarray(record["targets"]),
            np.asarray(record["measure"]),
        )
        if len({array.shape[0] for array in arrays}) != 1:
            raise ValueError(f"Ahmed raw fields differ in length for run {run}")
        if self.cache_cases:
            arrays = tuple(np.array(array, copy=True, order="C") for array in arrays)
            self._cache[int(run)] = arrays
        return arrays

    def selected_case(
        self, run: int, selection: slice | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._cache.get(int(run))
        if cached is not None:
            return tuple(array[selection] for array in cached[:3])  # type: ignore[return-value]
        if self.cache_cases:
            return tuple(
                array[selection] for array in self.full_case(run)[:3]
            )  # type: ignore[return-value]
        record = self._record(run)
        return tuple(
            _selected_values(record[name], selection)
            for name in ("coordinates", "normals", "targets")
        )  # type: ignore[return-value]

    def target_cases(self, runs: list[int]):
        for run in runs:
            yield np.asarray(self._record(run)["targets"], np.float32)

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
    "AhmedDataPaths",
    "AhmedRawStore",
    "EXPECTED_SPLITS",
    "FORMAT_ID",
    "file_sha256",
    "load_data_paths",
    "normalization_file",
    "run_group",
]
