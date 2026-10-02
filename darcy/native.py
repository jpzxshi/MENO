"""Native-node query access for the manifest-driven Darcy benchmark."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from ..dataset_common import CoordinateTransform
    from ..query_sampling import sample_uniform_query_indices
except ImportError:  # Support ``darcy`` as a top-level package.
    from dataset_common import CoordinateTransform
    from query_sampling import sample_uniform_query_indices

from .dataset import DarcyMomentDataset, validate_coordinate_transform


class NativeDarcyDataset(Dataset):
    """Join generated moments to raw vertices without a prepared query pool.

    Training supplies ``query_count`` and ``random_queries=True``. Its draw is
    a deterministic function of seed, epoch, source identity, and dataset row,
    and preserves the historical Darcy replacement rule: no replacement when
    ``Q <= N`` and one all-with-replacement draw when ``Q > N``. Selection and
    test omit ``query_count`` and therefore expose every raw node of each case.
    Validated node/feature arrays are cached after first access by default; the
    process-local cache is omitted from DataLoader worker serialization.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        *,
        coordinate_transform: CoordinateTransform,
        resolutions: Sequence[str] = ("fine", "coarse"),
        query_count: int | None = None,
        random_queries: bool = False,
        seed: int = 42,
        cache_cases: bool = True,
    ) -> None:
        if query_count is not None and int(query_count) < 1:
            raise ValueError("Darcy query_count must be positive or None")
        if random_queries and query_count is None:
            raise ValueError("random Darcy queries require query_count")
        self.base = DarcyMomentDataset(
            data_root,
            split,
            resolutions=resolutions,
        )
        validate_coordinate_transform(self.base.manifest, coordinate_transform)
        self.raw_root = self.base.data_root / "raw"
        if not self.raw_root.is_dir():
            raise FileNotFoundError(f"missing Darcy raw root: {self.raw_root}")
        self.split = split
        self.resolutions = tuple(resolutions)
        self.query_count = None if query_count is None else int(query_count)
        self.random_queries = bool(random_queries)
        self.seed = int(seed)
        self.cache_cases = bool(cache_cases)
        self.coordinate_transform = coordinate_transform
        self.epoch = 0
        self._case_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

        missing: list[str] = []
        for record in self.base.records:
            subset = str(record["subset"])
            source_index = int(record["source_index"])
            case_root = self.raw_root / subset
            for prefix in ("nodes", "elements", "features"):
                filename = case_root / f"{prefix}_{source_index:05d}.npy"
                if not filename.is_file():
                    missing.append(str(filename))
                    if len(missing) == 10:
                        break
            if len(missing) == 10:
                break
        if missing:
            raise FileNotFoundError(
                "missing Darcy raw case files: " + ", ".join(missing)
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.base)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_case_cache"] = {}
        return state

    def close(self) -> None:
        """Release process-local raw case arrays without touching moment maps."""

        self._case_cache.clear()

    def _raw_case(
        self,
        subset: str,
        source_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        identity = (subset, int(source_index))
        cached = self._case_cache.get(identity)
        if cached is not None:
            return cached

        case_root = self.raw_root / subset
        nodes = np.asarray(
            np.load(
                case_root / f"nodes_{source_index:05d}.npy",
                allow_pickle=False,
            ),
            dtype=np.float64,
        )
        features = np.asarray(
            np.load(
                case_root / f"features_{source_index:05d}.npy",
                allow_pickle=False,
            ),
            dtype=np.float64,
        )
        if nodes.ndim != 2 or nodes.shape[1] != 2 or nodes.shape[0] < 1:
            raise ValueError(
                f"invalid Darcy nodes: {subset}/{source_index}: {nodes.shape}"
            )
        if features.shape != (nodes.shape[0], 2):
            raise ValueError(
                f"invalid Darcy features: {subset}/{source_index}: {features.shape}"
            )
        if not (np.all(np.isfinite(nodes)) and np.all(np.isfinite(features))):
            raise ValueError(f"non-finite Darcy raw case: {subset}/{source_index}")
        if self.cache_cases:
            nodes.setflags(write=False)
            features.setflags(write=False)
            self._case_cache[identity] = (nodes, features)
        return nodes, features

    def _selection(
        self,
        *,
        index: int,
        source_index: int,
        node_count: int,
    ) -> np.ndarray:
        if self.query_count is None:
            return np.arange(node_count, dtype=np.int64)
        if self.random_queries:
            # The seed is a stable function of run, epoch and case identity.
            sequence = np.random.SeedSequence(
                [self.seed, self.epoch, source_index, index]
            )
            return sample_uniform_query_indices(
                np.random.default_rng(sequence),
                node_count,
                self.query_count,
            )
        if self.query_count > node_count:
            raise ValueError(
                "fixed Darcy queries cannot synthesize points when Q exceeds N"
            )
        return np.arange(self.query_count, dtype=np.int64)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.base[index]
        subset = str(sample["subset"])
        source_index = int(sample["source_index"])
        nodes, features = self._raw_case(subset, source_index)
        selection = self._selection(
            index=index,
            source_index=source_index,
            node_count=len(nodes),
        )
        return {
            **sample,
            "coordinates": torch.from_numpy(
                np.ascontiguousarray(
                    self.coordinate_transform.normalize(nodes[selection]),
                    dtype=np.float32,
                )
            ),
            "functions": torch.from_numpy(
                np.ascontiguousarray(features[selection, 0:1], dtype=np.float32)
            ),
            "targets": torch.from_numpy(
                np.ascontiguousarray(features[selection, 1:2], dtype=np.float32)
            ),
            "query_indices": torch.from_numpy(selection),
        }


def combine_resolution_metrics(
    fine: Mapping[str, float | str], coarse: Mapping[str, float | str]
) -> dict[str, float | str]:
    """Combine resolution metrics with equal weight for every native case."""

    fine_samples = int(float(fine["samples"]))
    coarse_samples = int(float(coarse["samples"]))
    if fine_samples < 1 or coarse_samples < 1:
        raise ValueError("fine/coarse metrics must both contain samples")
    total = fine_samples + coarse_samples

    def weighted(name: str) -> float:
        return (
            fine_samples * float(fine[name]) + coarse_samples * float(coarse[name])
        ) / total

    mean = weighted("case_relative_l2")
    second_moment = (
        fine_samples
        * (
            float(fine["case_relative_l2_std"]) ** 2
            + float(fine["case_relative_l2"]) ** 2
        )
        + coarse_samples
        * (
            float(coarse["case_relative_l2_std"]) ** 2
            + float(coarse["case_relative_l2"]) ** 2
        )
    ) / total
    return {
        "loss_protocol": str(fine["loss_protocol"]),
        "metric_protocol": str(fine["metric_protocol"]),
        "samples": float(total),
        "case_relative_mse": weighted("case_relative_mse"),
        "case_relative_l2": mean,
        "case_relative_l2_std": math.sqrt(max(second_moment - mean**2, 0.0)),
        "elapsed_seconds": float(fine["elapsed_seconds"])
        + float(coarse["elapsed_seconds"]),
    }


__all__ = ["NativeDarcyDataset", "combine_resolution_metrics"]
