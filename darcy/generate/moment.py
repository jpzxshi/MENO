"""Build the complete affine Darcy moment/split/normalization package from raw.

The output contains no prepared query pool. Training queries are always read
from raw nodes at runtime, and one canonical normalization uses every raw
training node with equal case weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.lib.format import open_memmap

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_common import CoordinateTransform, ragged_channel_statistics, stable_std
from darcy.dataset import (
    DATA_FORMAT,
    MOMENT_FORMAT,
    NORMALIZATION_NAME,
    NORMALIZATION_PROTOCOL,
    DarcyMomentDataset,
    compute_raw_inventory,
    fit_coordinate_transform,
    load_coordinate_transform,
    load_data_manifest,
    load_scaling,
    validate_raw_inventory,
)

try:
    from .mfe import encode_domain_and_permeability
except ImportError:
    from mfe import encode_domain_and_permeability


DEFAULT_MODE = 32
INPUT_CHANNELS = ("domain", "permeability")
FUNCTION_CHANNELS = ("permeability",)
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


def _sha256_file(filename: Path) -> str:
    digest = hashlib.sha256()
    with filename.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _training_coordinate_policy() -> tuple[str, tuple[float, ...] | None]:
    """Return the fixed paper coordinate policy without importing PyTorch."""

    return "prior", (2.0, 2.0)


def _load_raw(raw_root: Path, subset: str, source_index: int):
    root = raw_root / subset
    nodes = np.asarray(
        np.load(root / f"nodes_{source_index:05d}.npy", allow_pickle=False),
        dtype=np.float64,
    )
    elements = np.asarray(
        np.load(root / f"elements_{source_index:05d}.npy", allow_pickle=False),
        dtype=np.int64,
    )
    features = np.asarray(
        np.load(root / f"features_{source_index:05d}.npy", allow_pickle=False),
        dtype=np.float64,
    )
    if nodes.ndim != 2 or nodes.shape[1] != 2 or nodes.shape[0] < 3:
        raise ValueError(f"invalid nodes: {subset}/{source_index}: {nodes.shape}")
    if elements.ndim != 2 or elements.shape[1] != 4 or not len(elements):
        raise ValueError(f"invalid elements: {subset}/{source_index}: {elements.shape}")
    if not np.all(elements[:, 0] == 2):
        raise ValueError(f"element dimension must be 2: {subset}/{source_index}")
    if features.shape != (nodes.shape[0], 2):
        raise ValueError(f"invalid features: {subset}/{source_index}: {features.shape}")
    triangles = elements[:, 1:4]
    if triangles.min() < 0 or triangles.max() >= len(nodes):
        raise ValueError(f"triangle index out of bounds: {subset}/{source_index}")
    if not (np.all(np.isfinite(nodes)) and np.all(np.isfinite(features))):
        raise ValueError(f"non-finite raw case: {subset}/{source_index}")
    return nodes, elements, features


def _encode(
    payload: tuple[
        str,
        str,
        int,
        int,
        tuple[float, float],
        tuple[float, float],
    ]
) -> tuple[int, np.ndarray]:
    raw_root_text, subset, source_index, mode, center, half_span = payload
    nodes, elements, features = _load_raw(Path(raw_root_text), subset, source_index)
    normalized_nodes = (nodes - np.asarray(center, np.float64)) / np.asarray(
        half_span, np.float64
    )
    moments = encode_domain_and_permeability(
        physical_nodes=nodes,
        normalized_nodes=normalized_nodes,
        triangles=elements[:, 1:4],
        permeability=features[:, 0],
        mode=mode,
    )
    return source_index, np.ascontiguousarray(moments, dtype=np.float32)


def _component(split: str, resolution: str) -> dict[str, object]:
    start, stop = SPLIT_RANGES[split]
    subset = RESOLUTION_SUBSETS[resolution]
    return {
        "resolution": resolution,
        "subset": subset,
        "source_indices": {"type": "range", "start": start, "stop": stop, "step": 1},
        "samples": stop - start,
        "moment": {
            "root": f"moment/{split}/{resolution}",
            "manifest": f"moment/{split}/{resolution}/manifest.json",
        },
    }


def _build_component(
    raw_root: Path,
    moment_root: Path,
    split: str,
    resolution: str,
    mode: int,
    workers: int,
    coordinate_transform: CoordinateTransform,
) -> tuple[np.ndarray, dict[str, object]]:
    start, stop = SPLIT_RANGES[split]
    source_indices = tuple(range(start, stop))
    subset = RESOLUTION_SUBSETS[resolution]
    output = moment_root / split / resolution
    output.mkdir(parents=True)
    source_array = open_memmap(
        output / "source_indices.npy",
        mode="w+",
        dtype=np.int64,
        shape=(len(source_indices),),
    )
    moments_array = open_memmap(
        output / "moments.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(source_indices), mode**2, len(INPUT_CHANNELS)),
    )
    center = tuple(float(value) for value in coordinate_transform.center)
    half_span = tuple(float(value) for value in coordinate_transform.half_span)
    payloads = [
        (str(raw_root), subset, index, mode, center, half_span)
        for index in source_indices
    ]
    iterator: Iterable[tuple[int, np.ndarray]]
    executor: ProcessPoolExecutor | None = None
    if workers > 1:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_encode, payloads, chunksize=4)
    else:
        iterator = map(_encode, payloads)
    row_for_source = {value: row for row, value in enumerate(source_indices)}
    try:
        for completed, (source_index, moments) in enumerate(iterator, start=1):
            row = row_for_source[source_index]
            source_array[row] = source_index
            moments_array[row] = moments
            if completed % 50 == 0 or completed == len(source_indices):
                print(
                    f"[{split}/{resolution}] moments {completed}/{len(source_indices)}",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown()
    source_array.flush()
    moments_array.flush()
    moments_copy = np.asarray(moments_array, dtype=np.float32).copy()
    manifest = {
        "format": MOMENT_FORMAT,
        "split": split,
        "resolution": resolution,
        "subset": subset,
        "mode": mode,
        "samples": len(source_indices),
        "source_indices": {"start": start, "stop": stop, "step": 1},
        "coordinate_transform": coordinate_transform.metadata(),
        "input_channels": list(INPUT_CHANNELS),
        "arrays": {
            "source_indices": {
                "file": "source_indices.npy",
                "shape": list(source_array.shape),
                "dtype": str(source_array.dtype),
            },
            "moments": {
                "file": "moments.npy",
                "shape": list(moments_array.shape),
                "dtype": str(moments_array.dtype),
            },
        },
    }
    del source_array, moments_array
    manifest["arrays"]["source_indices"]["sha256"] = _sha256_file(
        output / "source_indices.npy"
    )
    manifest["arrays"]["moments"]["sha256"] = _sha256_file(output / "moments.npy")
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return moments_copy, manifest


def _moment_statistics(moments: Iterable[np.ndarray], mode: int):
    count = 0
    mean = np.zeros((mode**2, len(INPUT_CHANNELS)), dtype=np.float64)
    m2 = np.zeros_like(mean)
    for values in moments:
        block = np.asarray(values, dtype=np.float64)
        if block.shape != mean.shape or not np.all(np.isfinite(block)):
            raise ValueError("invalid training moments for normalization")
        count += 1
        delta = block - mean
        mean += delta / count
        m2 += delta * (block - mean)
    if not count:
        raise ValueError("training moments cannot be empty")
    return mean, stable_std(np.sqrt(np.maximum(m2 / count, 0.0)))


def _raw_training_cases(raw_root: Path):
    # Match the benchmark ordering: fine first, then coarse.
    for resolution in ("fine", "coarse"):
        subset = RESOLUTION_SUBSETS[resolution]
        for source_index in range(*SPLIT_RANGES["train"]):
            _, _, features = _load_raw(raw_root, subset, source_index)
            yield subset, source_index, features


def _canonical_scaling(
    raw_root: Path,
    train_moments: list[np.ndarray],
    mode: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    input_mean, input_std = _moment_statistics(train_moments, mode)
    cases = (features for _, _, features in _raw_training_cases(raw_root))
    statistics = ragged_channel_statistics(cases)
    arrays = {
        "input_mean": input_mean.astype(np.float32),
        "input_std": input_std.astype(np.float32),
        "function_mean": statistics.mean[0:1].astype(np.float32),
        "function_std": statistics.std[0:1].astype(np.float32),
        "target_mean": statistics.mean[1:2].astype(np.float32),
        "target_std": statistics.std[1:2].astype(np.float32),
    }
    metadata = {
        "source": "raw",
        "source_split": "train",
        **NORMALIZATION_PROTOCOL,
        "input_weighting": "uniform_per_case",
        "input_sampling": "full_raw_physical_triangle_moments",
        "local_target_weighting": "uniform_per_case",
        "local_target_sampling": "full_raw",
        "case_count": statistics.case_count,
        "sampled_point_count": statistics.point_count,
        "population_variance": True,
    }
    return arrays, metadata


def build(raw_root: Path, output_root: Path, *, mode: int, workers: int) -> Path:
    if mode < 2:
        raise ValueError("mode must be at least 2")
    if workers < 1:
        raise ValueError("workers must be positive")
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {output_root}")
    staging = output_root.with_name(output_root.name + ".tmp")
    if staging.exists():
        raise FileExistsError(f"staging already exists: {staging}")
    coordinate_source, coordinate_prior = _training_coordinate_policy()
    coordinate_transform = fit_coordinate_transform(
        raw_root,
        source=coordinate_source,
        prior_half_span=coordinate_prior,
    )
    staging.mkdir(parents=True)
    try:
        moment_root = staging / "moment"
        train_moments: list[np.ndarray] = []
        for split in ("train", "selection", "test"):
            for resolution in ("fine", "coarse"):
                values, _ = _build_component(
                    raw_root,
                    moment_root,
                    split,
                    resolution,
                    mode,
                    workers,
                    coordinate_transform,
                )
                if split == "train":
                    train_moments.extend(values)

        normalization_root = moment_root / "normalization"
        normalization_root.mkdir()
        arrays, normalization_metadata = _canonical_scaling(
            raw_root, train_moments, mode
        )
        filename = normalization_root / f"{NORMALIZATION_NAME}.npz"
        np.savez(filename, **arrays)
        normalization = {
            "name": NORMALIZATION_NAME,
            "file": f"moment/normalization/{NORMALIZATION_NAME}.npz",
            "sha256": _sha256_file(filename),
            **normalization_metadata,
        }
        print(
            f"[canonical] function={arrays['function_mean'].tolist()}/"
            f"{arrays['function_std'].tolist()} target="
            f"{arrays['target_mean'].tolist()}/{arrays['target_std'].tolist()}",
            flush=True,
        )

        splits = {
            split: {
                "samples": 2 * (SPLIT_RANGES[split][1] - SPLIT_RANGES[split][0]),
                "ordering": "fine_then_coarse",
                "components": [
                    _component(split, resolution) for resolution in ("fine", "coarse")
                ],
            }
            for split in ("train", "selection", "test")
        }
        manifest = {
            "format": DATA_FORMAT,
            "dataset": "deformed_domain_darcy",
            "mode": mode,
            "dimension": 2,
            "basis": "orthonormal_tensor_legendre_2d",
            "basis_domain": [-1.0, 1.0],
            "flatten_order": "C: m=i*mode+j",
            "input_channels": list(INPUT_CHANNELS),
            "local_function_channels": list(FUNCTION_CHANNELS),
            "target_channels": list(TARGET_CHANNELS),
            "coordinate_transform": coordinate_transform.metadata(),
            "raw": {"root": "raw", "subsets": compute_raw_inventory(raw_root)},
            "moment": {"root": "moment", "format": MOMENT_FORMAT},
            "splits": splits,
            "normalization": normalization,
            "native_query_sampling": {
                "method": "uniform_over_raw_vertex_indices",
                "train": "without_replacement_if_Q_le_N_else_all_with_replacement",
                "selection": "full_raw",
                "test": "full_raw",
                "resampled_each_epoch": True,
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_root)
        return output_root / "manifest.json"
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def validate_existing_data(data_root: str | Path) -> dict[str, object]:
    """Read-only validation of the active manifest, moments, scaling, and raw data."""

    root, manifest = load_data_manifest(data_root)
    coordinate_source, coordinate_prior = _training_coordinate_policy()
    coordinate_transform = load_coordinate_transform(
        root,
        source=coordinate_source,
        prior_half_span=coordinate_prior,
    )
    moment_report: dict[str, object] = {}
    for split in ("train", "selection", "test"):
        dataset = DarcyMomentDataset(root, split)
        moment_report[split] = {
            "samples": len(dataset),
            "resolutions": list(dataset.resolutions),
            "artifact_hashes": "validated",
        }
    scaling, metadata = load_scaling(root, coordinate_transform=coordinate_transform)
    scaling_report = {
        "file": metadata["file"],
        "sha256": metadata["sha256"],
        "weighting": metadata["weighting"],
        "sampling": metadata["sampling"],
        "shapes": {
            name: list(np.asarray(getattr(scaling, name)).shape)
            for name in scaling.__dataclass_fields__
        },
        "artifact_hash": "validated",
    }
    raw_report = validate_raw_inventory(root, manifest)
    return {
        "status": "valid",
        "data_root": str(root),
        "manifest": {
            "file": str(root / "manifest.json"),
            "sha256": _sha256_file(root / "manifest.json"),
            "protocol": manifest["format"],
        },
        "moments": moment_report,
        "normalization": scaling_report,
        "raw_inventory": raw_report,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    generate_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=generate_root.parent / "data" / "raw"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=generate_root.parent / "data",
        help="existing data root checked by --validate-only",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="read-only validation, including all 12,000 raw file hashes",
    )
    parser.add_argument("--mode", type=int, default=DEFAULT_MODE)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1))
    )
    args = parser.parse_args(argv)
    if args.validate_only and args.output_dir is not None:
        parser.error("--validate-only cannot be combined with --output-dir")
    if not args.validate_only and args.output_dir is None:
        parser.error("--output-dir is required unless --validate-only is used")
    if args.mode < 2:
        parser.error("--mode must be at least 2")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.validate_only:
        report = validate_existing_data(args.data_root)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return
    assert args.output_dir is not None
    manifest = build(
        args.raw_dir.resolve(),
        args.output_dir.resolve(),
        mode=args.mode,
        workers=args.workers,
    )
    print(f"manifest={manifest}", flush=True)


if __name__ == "__main__":
    main()


__all__ = ["build", "parse_args", "validate_existing_data"]
