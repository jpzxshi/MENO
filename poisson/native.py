"""Direct, deterministic query sampling from the complete Poisson raw data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import (
    LOCAL_FUNCTION_CHANNELS,
    PoissonRawStore,
    load_manifest,
    load_moment_array,
    raw_case_identity,
    reconstruct_standard_legendre_2d,
    validate_coordinate_transform,
)

try:
    from ..dataset_common import CoordinateTransform
    from ..query_sampling import normalize_query_limit, query_selection
except ImportError:  # ``poisson`` imported as a top-level package.
    from dataset_common import CoordinateTransform
    from query_sampling import normalize_query_limit, query_selection


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"
NATIVE_QUERY_SAMPLING_PROTOCOL = "direct-raw-uniform"


class NativePoissonQueryDataset(Dataset):
    """Pair raw Local queries with the exactly derived affine Global moments."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        indices: list[int],
        query_limit: int | None,
        random_queries: bool,
        query_seed: int = 42,
        *,
        coordinate_transform: CoordinateTransform,
        cache_cases: bool = False,
        raw_access_mode: str = "auto",
    ) -> None:
        if split not in ("train", "test"):
            raise ValueError("split must be train or test")
        self.root, self.manifest = load_manifest(data_dir)
        validate_coordinate_transform(self.manifest, coordinate_transform)
        self.split = split
        self.indices = [int(value) for value in indices]
        self.query_limit = normalize_query_limit(query_limit)
        self.random_queries = bool(random_queries)
        self.query_seed = int(query_seed)
        self.cache_cases = bool(cache_cases)
        self.raw_access_mode = raw_access_mode
        self.coordinate_transform = coordinate_transform
        self.epoch = 0
        self.mode = int(self.manifest["mode"])
        self.moments: np.ndarray | None = load_moment_array(
            self.root, self.manifest, split
        )
        self.raw_store = PoissonRawStore(
            self.root,
            self.manifest["raw"],
            cache_cases=self.cache_cases,
            access_mode=self.raw_access_mode,
        )
        sample_count = int(self.manifest["splits"][split]["samples"])
        if self.indices and (
            min(self.indices) < 0 or max(self.indices) >= sample_count
        ):
            raise IndexError(f"Poisson {split} indices outside [0,{sample_count})")

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["moments"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._ensure_moments_open()

    def _ensure_moments_open(self) -> None:
        if self.moments is None:
            self.moments = load_moment_array(self.root, self.manifest, self.split)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        self._ensure_moments_open()
        assert self.moments is not None
        source_index = self.indices[index]
        family, raw_index = raw_case_identity(
            self.manifest["raw"], self.split, source_index
        )
        case = self.raw_store.case(family, raw_index)
        node_count = int(case["points"].shape[0])
        selection, query_indices = query_selection(
            node_count,
            self.query_limit,
            random_queries=self.random_queries,
            seed=self.query_seed,
            epoch=self.epoch,
            sample_identity=("poisson", self.split, family, raw_index),
        )

        raw_coordinates = np.asarray(case["points"][selection, :2], dtype=np.float64)
        coordinates = np.ascontiguousarray(
            self.coordinate_transform.normalize(raw_coordinates), dtype=np.float32
        )
        moment = np.array(
            self.moments[source_index], dtype=np.float32, order="C", copy=True
        )
        boundary = reconstruct_standard_legendre_2d(
            coordinates, moment[:, 3], self.mode
        ).astype(np.float32, copy=False)
        functions = np.ascontiguousarray(
            np.column_stack(
                (
                    np.asarray(case["f"][selection], dtype=np.float32),
                    np.asarray(case["k"][selection], dtype=np.float32),
                    boundary,
                )
            ),
            dtype=np.float32,
        )
        if functions.shape[1] != len(LOCAL_FUNCTION_CHANNELS):
            raise RuntimeError("Poisson Local function channel construction failed")
        targets = np.ascontiguousarray(
            np.asarray(case["u"][selection], dtype=np.float32)[:, None]
        )
        return {
            "sample": f"{self.split}_{source_index:04d}",
            "family": family,
            "raw_index": torch.tensor(raw_index, dtype=torch.int64),
            "index": torch.tensor(source_index, dtype=torch.int64),
            "moments": torch.from_numpy(moment),
            "coordinates": torch.from_numpy(coordinates),
            "functions": torch.from_numpy(functions),
            "targets": torch.from_numpy(targets),
            "query_indices": torch.from_numpy(query_indices),
        }


PoissonDataset = NativePoissonQueryDataset


__all__ = [
    "DEFAULT_DATA_ROOT",
    "NATIVE_QUERY_SAMPLING_PROTOCOL",
    "NativePoissonQueryDataset",
    "PoissonDataset",
]
