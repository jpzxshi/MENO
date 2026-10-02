"""Regenerate AhmedML moment/scaling artifacts from the canonical raw store."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from ..native import FORMAT_ID
from .preprocessing.mfe import CoordinateBounds, tensor_legendre_moments


DEFAULT_MODE = 16
BASIS = "orthonormal_tensor_legendre_3d"
FLATTEN_ORDER = "C: m=(a*mode+b)*mode+c"
EXPECTED_RUNS = tuple(range(1, 501))
EXPECTED_SPLITS = {"train": 400, "validation": 50, "test": 50}
NORMALIZATION_WEIGHTING = "uniform_per_case"
NORMALIZATION_SAMPLING = "full_raw"
QUERY_SAMPLING = "dynamic_uniform_native"
RAW_ARRAYS = {
    "coordinates": (None, 3),
    "normals": (None, 3),
    "measure": (None,),
    "targets": (None, 4),
    "geometry_parameters": (8,),
}


def _coordinate_policy() -> tuple[str, tuple[float, ...] | None]:
    """Return the fixed paper coordinate policy without importing PyTorch."""

    return "train_bounds", None


class _DatasetColumn:
    def __init__(self, dataset: np.ndarray, column: int) -> None:
        self.dataset = dataset
        self.column = int(column)
        self.shape = (int(dataset.shape[0]),)

    def __getitem__(self, key: object) -> np.ndarray:
        return self.dataset[key, self.column]


def _run_group(run: int) -> str:
    return f"runs/{int(run):06d}"


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _validate_shape(name: str, dataset: np.ndarray, count: int) -> None:
    expected = RAW_ARRAYS[name]
    resolved = tuple(count if value is None else value for value in expected)
    if dataset.shape != resolved or not np.issubdtype(dataset.dtype, np.number):
        raise ValueError(f"Ahmed {name} shape={dataset.shape}, expected={resolved}")


def describe_raw(path: Path) -> dict[str, object]:
    """Validate and describe the per-run NPY raw catalog."""

    path = path.resolve()
    manifest_file = path / "data_manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(manifest_file)
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    if payload.get("format") != "mfd-ahmed-surface-raw-npy-v1":
        raise ValueError("Ahmed raw NPY manifest format mismatch")
    seen: set[int] = set()
    for split, expected in EXPECTED_SPLITS.items():
        runs = [int(value) for value in payload["splits"][split]]
        if len(runs) != expected or len(set(runs)) != expected:
            raise ValueError(f"Ahmed {split} split must contain {expected} runs")
        if seen.intersection(runs):
            raise ValueError("Ahmed split overlap")
        seen.update(runs)
    if seen != set(EXPECTED_RUNS):
        raise ValueError("Ahmed splits do not cover runs 1..500")
    cases = payload.get("cases")
    if not isinstance(cases, list) or [int(case["run"]) for case in cases] != list(
        EXPECTED_RUNS
    ):
        raise ValueError("Ahmed raw case catalog must contain ordered runs 1..500")
    counts: list[int] = []
    for case in cases:
        filename = path / str(case["file"])
        if not filename.is_file() or filename.stat().st_size != int(case["bytes"]):
            raise ValueError(f"Ahmed raw case file mismatch: {filename}")
        record = np.load(filename, mmap_mode="r", allow_pickle=False)[0]
        count = int(record["coordinates"].shape[0])
        if count != int(case["points"]):
            raise ValueError(f"Ahmed raw point count mismatch: {filename}")
        for name in RAW_ARRAYS:
            _validate_shape(name, record[name], count)
        counts.append(count)
    return {
        "directory": "raw",
        "manifest": "raw/data_manifest.json",
        "manifest_sha256": _sha256(manifest_file),
        "format": payload["format"],
        "source": payload["source"],
        "runs": len(EXPECTED_RUNS),
        "point_count_min": min(counts),
        "point_count_max": max(counts),
        "point_count_total": sum(counts),
    }


def _moment_worker(
    arguments: tuple[str, int, np.ndarray, np.ndarray, int, int]
) -> tuple[int, np.ndarray]:
    raw_name, run, minimum, maximum, chunk_size, mode = arguments
    bounds = CoordinateBounds(minimum, maximum)
    raw_dir = Path(raw_name)
    record = np.load(
        raw_dir / f"features_{run:05d}.npy", mmap_mode="r", allow_pickle=False
    )[0]
    normals = record["normals"]
    moments = tensor_legendre_moments(
        coordinates=record["coordinates"],
        weights=record["measure"],
        fields=(None, *(_DatasetColumn(normals, axis) for axis in range(3))),
        bounds=bounds,
        mode=mode,
        chunk_size=chunk_size,
        normalize_measure=False,
        clip_coordinates=False,
    ).astype(np.float32)
    return run, moments


def build_moments(
    raw_file: Path,
    output: Path,
    *,
    overwrite: bool,
    workers: int,
    chunk_size: int,
    mode: int,
) -> dict[str, object]:
    """Fit train-only bounds and recompute all 500 geometry moments."""

    raw_file = raw_file.resolve()
    output = output.resolve()
    if not raw_file.is_dir():
        raise FileNotFoundError(raw_file)
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")
    if workers < 1 or chunk_size < 1 or mode < 2:
        raise ValueError(
            "workers and chunk_size must be positive, and mode must be at least 2"
        )
    raw_manifest = json.loads(
        (raw_file / "data_manifest.json").read_text(encoding="utf-8")
    )
    splits = {
        split: [int(value) for value in raw_manifest["splits"][split]]
        for split in EXPECTED_SPLITS
    }
    from ahmed.dataset import fit_coordinate_transform

    coordinate_source, coordinate_prior = _coordinate_policy()
    transform = fit_coordinate_transform(
        raw_file,
        splits["train"],
        source=coordinate_source,
        prior_half_span=coordinate_prior,
    )
    bounds = CoordinateBounds(transform.minimum, transform.maximum)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    jobs = [
        (
            str(raw_file),
            run,
            bounds.minimum,
            bounds.maximum,
            chunk_size,
            mode,
        )
        for run in EXPECTED_RUNS
    ]
    try:
        with h5py.File(temporary, "w") as destination:
            destination.attrs["format"] = FORMAT_ID
            destination.attrs["dataset"] = "AhmedML"
            destination.attrs["task"] = "surface"
            destination.attrs["mode_per_axis"] = mode
            destination.attrs["number_of_basis_functions"] = mode**3
            destination.attrs["basis"] = BASIS
            destination.attrs["flatten_order"] = FLATTEN_ORDER
            destination.attrs["geometry_moment_fields"] = json.dumps(
                ["surface", "normal_x", "normal_y", "normal_z"]
            )
            destination.attrs["coordinate_min"] = bounds.minimum
            destination.attrs["coordinate_max"] = bounds.maximum
            destination.attrs["coordinate_center"] = transform.center
            destination.attrs["coordinate_half_span"] = transform.half_span
            destination.attrs["coordinate_scale_source"] = transform.source
            destination.attrs["coordinate_scale_margin"] = transform.margin
            destination.attrs["raw_training_minimum"] = transform.raw_minimum
            destination.attrs["raw_training_maximum"] = transform.raw_maximum
            destination.attrs["coordinates_clipped"] = False
            destination.attrs["measure_normalized"] = False
            destination.attrs["moments_are_projection_coefficients"] = False
            destination.attrs["moment_source"] = "full_raw"
            destination.create_dataset(
                "fit_runs", data=np.asarray(splits["train"], dtype=np.int32)
            )
            run_root = destination.create_group("runs")
            if workers == 1:
                iterator = map(_moment_worker, jobs)
                executor = None
            else:
                executor = ProcessPoolExecutor(max_workers=workers)
                iterator = executor.map(_moment_worker, jobs)
            try:
                for index, (run, values) in enumerate(iterator, start=1):
                    print(f"[moment] {index}/500 run_{run}", flush=True)
                    group = run_root.create_group(f"{run:06d}")
                    group.create_dataset(
                        "geometry_moments",
                        data=values,
                        compression="lzf",
                        shuffle=True,
                    )
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    return {
        "file": f"moment/{output.name}",
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "mode": mode,
        "basis": BASIS,
        "source": "full_raw",
        "fit_split": "train",
        "coordinate_transform": transform.metadata(),
    }


def validate_canonical(
    data_dir: Path, *, full_values: bool = False
) -> dict[str, object]:
    data_dir = data_dir.resolve()
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT_ID:
        raise ValueError("Ahmed manifest format mismatch")
    raw_path = data_dir / str(manifest["raw"].get("directory", "raw"))
    raw_manifest_path = data_dir / str(manifest["raw"]["manifest"])
    moment_path = data_dir / manifest["moment"]["file"]
    mode = int(manifest["moment"]["mode"])
    if not raw_path.is_dir() or not raw_manifest_path.is_file():
        raise FileNotFoundError(raw_path)
    if _sha256(raw_manifest_path) != manifest["raw"]["manifest_sha256"]:
        raise ValueError("Ahmed raw manifest SHA256 mismatch")
    if (
        not moment_path.is_file()
        or _sha256(moment_path) != manifest["moment"]["sha256"]
    ):
        raise ValueError(f"SHA256 mismatch: {moment_path}")
    raw = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    cases = {int(case["run"]): case for case in raw["cases"]}
    with h5py.File(moment_path, "r") as moments:
        if moments.attrs.get("format") != FORMAT_ID:
            raise ValueError("Ahmed moment format mismatch")
        if int(moments.attrs.get("mode_per_axis", -1)) != mode:
            raise ValueError("Ahmed moment mode differs from manifest")
        split_runs: set[int] = set()
        for split, expected in EXPECTED_SPLITS.items():
            runs = [int(value) for value in raw["splits"][split]]
            if len(runs) != expected:
                raise ValueError(f"Ahmed {split} count mismatch")
            if split_runs.intersection(runs):
                raise ValueError("Ahmed split overlap")
            split_runs.update(runs)
        if split_runs != set(EXPECTED_RUNS):
            raise ValueError("Ahmed splits do not cover 1..500")
        for run in EXPECTED_RUNS:
            filename = raw_path / str(cases[run]["file"])
            if not filename.is_file() or filename.stat().st_size != int(
                cases[run]["bytes"]
            ):
                raise ValueError(f"Ahmed raw case mismatch: {filename}")
            record = np.load(filename, mmap_mode="r", allow_pickle=False)[0]
            count = int(record["coordinates"].shape[0])
            for name in RAW_ARRAYS:
                _validate_shape(name, record[name], count)
                if full_values:
                    for start in range(0, record[name].shape[0], 262_144):
                        values = np.asarray(record[name][start : start + 262_144])
                        if not np.all(np.isfinite(values)):
                            raise ValueError(f"{filename}:{name} contains NaN/Inf")
            moment = moments[f"{_run_group(run)}/geometry_moments"]
            if moment.shape != (4, mode, mode, mode):
                raise ValueError(f"invalid moment shape for run {run}")
            if full_values and not np.all(np.isfinite(moment[...])):
                raise ValueError(f"run {run} moment contains NaN/Inf")
        fit_runs = {int(value) for value in moments["fit_runs"][...]}
        train_runs = {int(value) for value in raw["splits"]["train"]}
        if fit_runs != train_runs:
            raise ValueError(
                "Ahmed moment coordinate bounds were not fit on train only"
            )
    from ahmed.dataset import load_coordinate_transform

    coordinate_source, coordinate_prior = _coordinate_policy()
    transform = load_coordinate_transform(
        data_dir,
        source=coordinate_source,
        prior_half_span=coordinate_prior,
    )
    if manifest["moment"].get("coordinate_transform") != transform.metadata():
        raise ValueError("Ahmed manifest coordinate transform differs from moments")
    normalization = manifest.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("Ahmed manifest must contain canonical normalization")
    expected_lineage = {
        "raw_sha256": manifest["raw"]["manifest_sha256"],
        "moment_sha256": manifest["moment"]["sha256"],
    }
    expected_arrays = {
        "input_mean",
        "input_std",
        "function_mean",
        "function_std",
        "target_mean",
        "target_std",
    }
    details = normalization
    if details.get("weighting") != NORMALIZATION_WEIGHTING:
        raise ValueError("Ahmed normalization weighting mismatch")
    if details.get("sampling") != NORMALIZATION_SAMPLING:
        raise ValueError("Ahmed normalization sampling mismatch")
    if details.get("source") != "train/full_raw+train/moment":
        raise ValueError("Ahmed normalization source mismatch")
    if details.get("lineage") != expected_lineage:
        raise ValueError("Ahmed normalization lineage mismatch")
    expected_relative = "moment/normalization/canonical.npz"
    if details.get("file") != expected_relative:
        raise ValueError("Ahmed normalization path mismatch")
    path = data_dir / expected_relative
    if not path.is_file():
        raise FileNotFoundError(path)
    if _sha256(path) != details.get("sha256"):
        raise ValueError("Ahmed normalization SHA256 mismatch")
    with np.load(path) as arrays:
        actual_arrays = set(arrays.files)
        if actual_arrays != expected_arrays:
            raise ValueError("Ahmed normalization must contain exactly six fields")
        for name in actual_arrays:
            if not np.all(np.isfinite(np.asarray(arrays[name]))):
                raise ValueError(f"Ahmed normalization/{name} contains NaN/Inf")
        for name in ("input_std", "function_std", "target_std"):
            if np.any(np.asarray(arrays[name]) <= 0.0):
                raise ValueError(f"Ahmed normalization/{name} is not positive")
        shapes = {name: np.asarray(arrays[name]).shape for name in expected_arrays}
        expected_shapes = {
            "input_mean": (mode**3, 12),
            "input_std": (mode**3, 12),
            "function_mean": (11,),
            "function_std": (11,),
            "target_mean": (4,),
            "target_std": (4,),
        }
        if shapes != expected_shapes:
            raise ValueError(f"Ahmed normalization shapes={shapes}")
    return manifest


def build_normalizations(data_dir: Path) -> dict[str, object]:
    """Compute the one canonical six-field scaling from full raw train cases."""

    from dataset_common import ModelScaling
    from ahmed.dataset import (
        compute_statistical_scaling,
        load_coordinate_transform,
        read_splits,
    )

    root = data_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    mode = int(manifest["moment"]["mode"])
    lineage = {
        "raw_sha256": manifest["raw"]["manifest_sha256"],
        "moment_sha256": manifest["moment"]["sha256"],
    }
    train_runs = read_splits(root)["train"]
    coordinate_source, coordinate_prior = _coordinate_policy()
    transform = load_coordinate_transform(
        root,
        source=coordinate_source,
        prior_half_span=coordinate_prior,
    )
    output_dir = root / "moment" / "normalization"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[normalization] canonical", flush=True)
    scaling = compute_statistical_scaling(
        root,
        train_runs,
        mode,
        coordinate_transform=transform,
    )
    path = output_dir / "canonical.npz"
    temporary = path.with_name(f"{path.stem}.tmp-{os.getpid()}.npz")
    np.savez(
        temporary,
        **{
            name: np.asarray(getattr(scaling, name), np.float32)
            for name in ModelScaling.STATISTICAL_FIELDS
        },
    )
    os.replace(temporary, path)
    return {
        "file": f"moment/normalization/{path.name}",
        "sha256": _sha256(path),
        "weighting": NORMALIZATION_WEIGHTING,
        "sampling": NORMALIZATION_SAMPLING,
        "source": "train/full_raw+train/moment",
        "lineage": lineage,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    case_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--moment-output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=DEFAULT_MODE,
        help="Legendre moment order",
    )
    parser.add_argument(
        "--manifest", type=Path, default=case_dir / "data" / "manifest.json"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=65_536)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--full-values", action="store_true")
    parser.add_argument("--normalization-only", action="store_true")
    args = parser.parse_args(argv)
    if args.mode < 2:
        parser.error("--mode must be at least 2")
    if args.moment_output is None:
        args.moment_output = (
            case_dir / "data" / "moment" / f"surface_legendre_mode{args.mode}.h5"
        )
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.validate_only:
        payload = validate_canonical(
            args.manifest.resolve().parent, full_values=args.full_values
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if args.normalization_only:
        manifest = args.manifest.resolve()
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["normalization"] = build_normalizations(manifest.parent)
        temporary = manifest.with_name(f"{manifest.name}.tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, manifest)
        print(f"Updated {manifest}")
        return
    data_dir = args.manifest.resolve().parent
    raw_path = data_dir / "raw"
    raw_details = describe_raw(raw_path)
    moment_details = build_moments(
        raw_path,
        args.moment_output,
        overwrite=args.overwrite,
        workers=args.workers,
        chunk_size=args.chunk_size,
        mode=args.mode,
    )
    raw_manifest = json.loads(
        (raw_path / "data_manifest.json").read_text(encoding="utf-8")
    )
    splits = {
        split: [int(value) for value in raw_manifest["splits"][split]]
        for split in EXPECTED_SPLITS
    }
    payload = {
        "format": FORMAT_ID,
        "dataset": "AhmedML",
        "task": "surface",
        "raw": raw_details,
        "moment": moment_details,
        "splits": splits,
        "split_strategy": "deterministic_random_80_10_10",
        "split_seed": 42,
        "query_sampling": QUERY_SAMPLING,
    }
    manifest = args.manifest.resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists() and not args.overwrite:
        raise FileExistsError(f"manifest exists; pass --overwrite: {manifest}")
    temporary = manifest.with_name(f"{manifest.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, manifest)
    payload["normalization"] = build_normalizations(manifest.parent)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, manifest)
    print(f"Read {raw_path}")
    print(f"Wrote {args.moment_output.resolve()}")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
