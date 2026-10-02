"""Build and validate the canonical NASA-CRM raw/moment dataset layout.

The canonical layout keeps one structured NPY per official train/test sample
under ``data/raw`` and writes deterministic Global MFE moments under
``data/moment``. Local queries are sampled from the per-case NPY at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from ..dataset import fit_coordinate_transform, load_coordinate_transform
from ..native import FORMAT_ID
from .mfe import (
    CONDITION_FIELDS,
    COORDINATE_FIELDS,
    GEOMETRY_MOMENT_FIELDS,
    NORMAL_FIELDS,
    TARGET_FIELDS,
    tensor_legendre_surface_moments,
)


DEFAULT_MODE = 8
BASIS = "orthonormal_tensor_legendre_3d"
FLATTEN_ORDER = "C: m=(a*mode+b)*mode+c"
EXPECTED_CASES = {"train": 105, "test": 44}
NORMALIZATION_WEIGHTING = "uniform_per_case"
NORMALIZATION_SAMPLING = "full_raw"
QUERY_SAMPLING = "dynamic_uniform_native"


def _coordinate_policy() -> tuple[str, object]:
    """Return the fixed paper coordinate policy without importing PyTorch."""

    return "train_bounds", None


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _json_attr(values: tuple[str, ...]) -> str:
    return json.dumps(values, ensure_ascii=False)


def _raw_inventory(raw_dir: Path) -> dict[str, dict[str, object]]:
    manifest_file = raw_dir / "data_manifest.json"
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    if payload.get("format") != "mfd-nasa-raw-npy-v1":
        raise ValueError("NASA raw split manifest format mismatch")
    inventory: dict[str, object] = {
        "directory": "raw",
        "manifest": "raw/data_manifest.json",
        "manifest_sha256": _sha256(manifest_file),
        "format": payload["format"],
        "source": payload["source"],
    }
    for split, expected in EXPECTED_CASES.items():
        details = payload["splits"][split]
        cases = details["cases"]
        if len(cases) != expected:
            raise ValueError(f"NASA {split} cases={len(cases)}, expected={expected}")
        split_root = raw_dir / str(details.get("directory", split))
        for case in cases:
            filename = split_root / str(case["file"])
            if not filename.is_file() or filename.stat().st_size != int(case["bytes"]):
                raise ValueError(f"NASA split case file mismatch: {filename}")
        inventory[split] = {
            "samples": expected,
            "point_count_min": int(details["point_count_min"]),
            "point_count_max": int(details["point_count_max"]),
            "point_count_total": int(details["point_count_total"]),
            "first_case": str(cases[0]["key"]),
            "last_case": str(cases[-1]["key"]),
        }
    return inventory  # type: ignore[return-value]


def _raw_sample(record: np.void) -> dict[str, np.ndarray]:
    coordinates = record["coordinates"]
    normals = record["normals"]
    targets = record["targets"]
    return {
        **{name: coordinates[:, axis] for axis, name in enumerate(COORDINATE_FIELDS)},
        **{name: normals[:, axis] for axis, name in enumerate(NORMAL_FIELDS)},
        "Surface": record["surface"],
        **{name: targets[:, axis] for axis, name in enumerate(TARGET_FIELDS)},
    }


def build_moments(
    raw_dir: Path,
    output: Path,
    *,
    mode: int,
    overwrite: bool,
    chunk_size: int,
) -> dict[str, object]:
    """Recompute moments from full raw surfaces."""

    raw_dir = raw_dir.resolve()
    output = output.resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")
    if mode < 2 or chunk_size < 1:
        raise ValueError("mode must be at least 2 and chunk_size must be positive")
    inventory = _raw_inventory(raw_dir)
    coordinate_source, coordinate_prior = _coordinate_policy()
    coordinate_transform = fit_coordinate_transform(
        raw_dir,
        source=coordinate_source,
        prior_half_span=coordinate_prior,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with h5py.File(temporary, "w") as destination:
            destination.attrs["format"] = FORMAT_ID
            destination.attrs["dataset"] = "NASA-CRM"
            destination.attrs["mode_per_axis"] = mode
            destination.attrs["number_of_basis_functions"] = mode**3
            destination.attrs["basis"] = BASIS
            destination.attrs["flatten_order"] = FLATTEN_ORDER
            destination.attrs["geometry_moment_fields"] = _json_attr(
                GEOMETRY_MOMENT_FIELDS
            )
            destination.attrs["condition_fields"] = _json_attr(CONDITION_FIELDS)
            destination.attrs["coordinate_fields"] = _json_attr(COORDINATE_FIELDS)
            destination.attrs["normal_fields"] = _json_attr(NORMAL_FIELDS)
            destination.attrs["target_fields"] = _json_attr(TARGET_FIELDS)
            destination.attrs["raw_training_minimum"] = (
                coordinate_transform.raw_minimum
            )
            destination.attrs["raw_training_maximum"] = (
                coordinate_transform.raw_maximum
            )
            destination.attrs["coordinate_center"] = coordinate_transform.center
            destination.attrs["coordinate_half_span"] = (
                coordinate_transform.half_span
            )
            destination.attrs["coordinate_scale_source"] = (
                coordinate_transform.source
            )
            destination.attrs["coordinate_scale_margin"] = (
                coordinate_transform.margin
            )
            destination.attrs["coordinate_min"] = coordinate_transform.minimum
            destination.attrs["coordinate_max"] = coordinate_transform.maximum
            destination.attrs["coordinates_clipped"] = False
            destination.attrs["area_normalized_moments"] = False
            destination.attrs["moments_are_projection_coefficients"] = False
            destination.attrs["moment_source"] = "full_raw"

            raw_manifest = json.loads(
                (raw_dir / "data_manifest.json").read_text(encoding="utf-8")
            )
            for split in EXPECTED_CASES:
                details = raw_manifest["splits"][split]
                cases = details["cases"]
                output_split = destination.create_group(split)
                for case_index, case in enumerate(cases, start=1):
                    key = str(case["key"])
                    print(f"[{split}] moment {case_index}/{len(cases)} {key}", flush=True)
                    filename = raw_dir / str(details.get("directory", split)) / str(case["file"])
                    record = np.load(filename, mmap_mode="r", allow_pickle=False)[0]
                    geometry, _ = tensor_legendre_surface_moments(
                        sample=_raw_sample(record),
                        mode=mode,
                        coordinate_center=coordinate_transform.center,
                        coordinate_half_span=coordinate_transform.half_span,
                        chunk_size=chunk_size,
                        clip_coordinates=False,
                        normalize_area=False,
                    )
                    group = output_split.create_group(key)
                    group.attrs["source_sample"] = key
                    group.create_dataset(
                        "geometry_moments",
                        data=geometry.astype(np.float32),
                        compression="lzf",
                        shuffle=True,
                    )
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    result: dict[str, object] = {
        "format": FORMAT_ID,
        "dataset": "NASA-CRM",
        "raw": inventory,
        "moment": {
            "file": f"moment/{output.name}",
            "bytes": output.stat().st_size,
            "sha256": _sha256(output),
            "mode": mode,
            "basis": BASIS,
            "source": "full_raw",
            "coordinate_transform": coordinate_transform.metadata(),
        },
        "splits": {name: int(inventory[name]["samples"]) for name in EXPECTED_CASES},
        "query_sampling": QUERY_SAMPLING,
    }
    return result


def validate_canonical(data_dir: Path) -> dict[str, object]:
    """Validate canonical raw/moment files and their manifest without writing."""

    data_dir = data_dir.resolve()
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT_ID:
        raise ValueError(
            "unexpected NASA manifest format: "
            f"format={manifest.get('format')!r}"
        )
    mode = int(manifest.get("moment", {}).get("mode", -1))
    if mode < 1:
        raise ValueError("NASA moment mode must be positive")
    inventory = _raw_inventory(data_dir / "raw")
    for split in EXPECTED_CASES:
        expected = manifest["raw"][split]
        for field in ("samples", "point_count_total"):
            if inventory[split][field] != expected[field]:
                raise ValueError(f"NASA raw {split} manifest mismatch for {field}")
    if inventory["manifest_sha256"] != manifest["raw"]["manifest_sha256"]:
        raise ValueError("NASA raw split manifest SHA256 mismatch")

    moment_path = data_dir / manifest["moment"]["file"]
    if not moment_path.is_file():
        raise FileNotFoundError(moment_path)
    if _sha256(moment_path) != manifest["moment"]["sha256"]:
        raise ValueError("NASA moment SHA256 differs from manifest")
    with h5py.File(moment_path, "r") as moments:
        if moments.attrs.get("format") != FORMAT_ID:
            raise ValueError("NASA moment format mismatch")
        if int(moments.attrs.get("mode_per_axis", -1)) != mode:
            raise ValueError("NASA moment mode mismatch")
        for split, expected_cases in EXPECTED_CASES.items():
            keys = sorted(moments[split].keys())
            if len(keys) != expected_cases:
                raise ValueError(f"NASA moment {split} case count mismatch")
            for key in keys:
                dataset = moments[split][key].get("geometry_moments")
                if not isinstance(dataset, h5py.Dataset) or dataset.shape != (
                    4,
                    mode,
                    mode,
                    mode,
                ):
                    raise ValueError(f"invalid NASA moment {split}/{key}")
    coordinate_source, coordinate_prior = _coordinate_policy()
    load_coordinate_transform(
        data_dir,
        source=coordinate_source,
        prior_half_span=coordinate_prior,
    )
    normalization = manifest.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("NASA manifest must contain canonical normalization")
    expected_lineage = {
        "raw_train_sha256": manifest["raw"]["manifest_sha256"],
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
    if normalization.get("weighting") != NORMALIZATION_WEIGHTING:
        raise ValueError("NASA normalization weighting mismatch")
    if normalization.get("sampling") != NORMALIZATION_SAMPLING:
        raise ValueError("NASA normalization sampling mismatch")
    if normalization.get("source") != "train/full_raw+train/moment":
        raise ValueError("NASA normalization source mismatch")
    if normalization.get("lineage") != expected_lineage:
        raise ValueError("NASA normalization lineage mismatch")
    expected_relative = "moment/normalization/canonical.npz"
    if normalization.get("file") != expected_relative:
        raise ValueError("NASA normalization path mismatch")
    path = data_dir / expected_relative
    if not path.is_file():
        raise FileNotFoundError(path)
    if _sha256(path) != normalization.get("sha256"):
        raise ValueError("NASA normalization SHA256 mismatch")
    with np.load(path) as arrays:
        actual_arrays = set(arrays.files)
        if actual_arrays != expected_arrays:
            raise ValueError("NASA normalization must contain exactly six fields")
        for name in actual_arrays:
            values = np.asarray(arrays[name])
            if not np.all(np.isfinite(values)):
                raise ValueError(f"NASA normalization/{name} contains NaN/Inf")
        for name in ("input_std", "function_std", "target_std"):
            if np.any(np.asarray(arrays[name]) <= 0.0):
                raise ValueError(f"NASA normalization/{name} is not positive")
        shapes = {name: np.asarray(arrays[name]).shape for name in expected_arrays}
        expected_shapes = {
            "input_mean": (mode**3, 10),
            "input_std": (mode**3, 10),
            "function_mean": (9,),
            "function_std": (9,),
            "target_mean": (4,),
            "target_std": (4,),
        }
        if shapes != expected_shapes:
            raise ValueError(f"NASA normalization shapes={shapes}")
    return manifest


def build_normalization(data_dir: Path) -> dict[str, object]:
    """Compute the sole six-field canonical normalization artifact."""

    from ..dataset import compute_statistical_scaling, read_keys

    root = data_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    lineage = {
        "raw_train_sha256": manifest["raw"]["manifest_sha256"],
        "moment_sha256": manifest["moment"]["sha256"],
    }
    keys = read_keys(root, "train")
    coordinate_source, coordinate_prior = _coordinate_policy()
    coordinate_transform = load_coordinate_transform(
        root,
        source=coordinate_source,
        prior_half_span=coordinate_prior,
    )
    output_dir = root / "moment" / "normalization"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[normalization] canonical", flush=True)
    scaling = compute_statistical_scaling(
        root,
        keys,
        True,
        coordinate_transform=coordinate_transform,
    )
    path = output_dir / "canonical.npz"
    temporary = path.with_name(f"{path.stem}.tmp-{os.getpid()}.npz")
    np.savez(
        temporary,
        **{
            name: np.asarray(getattr(scaling, name), np.float32)
            for name in scaling.STATISTICAL_FIELDS
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
    parser.add_argument("--raw-dir", type=Path, default=case_dir / "data" / "raw")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--mode", type=int, default=DEFAULT_MODE)
    parser.add_argument(
        "--manifest", type=Path, default=case_dir / "data" / "manifest.json"
    )
    # Keep a fixed accumulation order for deterministic float32 artifacts.
    parser.add_argument("--chunk-size", type=int, default=8_192)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--normalization-only", action="store_true")
    args = parser.parse_args(argv)
    if args.mode < 2:
        parser.error("--mode must be at least 2")
    if args.output is None:
        args.output = (
            case_dir / "data" / "moment" / f"nasa_crm_legendre_mode{args.mode}.h5"
        )
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.validate_only:
        payload = validate_canonical(args.manifest.resolve().parent)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if args.normalization_only:
        manifest = args.manifest.resolve()
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["normalization"] = build_normalization(manifest.parent)
        temporary = manifest.with_name(f"{manifest.name}.tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, manifest)
        print(f"Updated {manifest}")
        return
    payload = build_moments(
        args.raw_dir,
        args.output,
        mode=args.mode,
        overwrite=args.overwrite,
        chunk_size=args.chunk_size,
    )
    manifest = args.manifest.resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists() and not args.overwrite:
        raise FileExistsError(f"manifest exists; pass --overwrite: {manifest}")
    temporary = manifest.with_name(f"{manifest.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, manifest)
    payload["normalization"] = build_normalization(manifest.parent)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, manifest)
    print(f"Wrote {args.output.resolve()}")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
