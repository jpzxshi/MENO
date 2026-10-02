"""Split the two official NASA-CRM HDF5 archives into one NPY per case.

Each ``features_XXXXX.npy`` is a pickle-free structured NPY record.  The
record keeps the original point fields and scalar case attributes without
duplicating the six operating conditions over every surface point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.lib.format import open_memmap


FORMAT = "mfd-nasa-raw-npy-v1"
SOURCE_FILES = {
    "train": "trainingData_NASA-CRM.h5",
    "test": "testData_NASA-CRM.h5",
}
EXPECTED_CASES = {"train": 105, "test": 44}
COORDINATE_FIELDS = ("CoordinateX", "CoordinateY", "CoordinateZ")
NORMAL_FIELDS = ("NormalX", "NormalY", "NormalZ")
TARGET_FIELDS = ("PressureCoefficient", "cfx", "cfy", "cfz")
CONDITION_FIELDS = (
    "Mach",
    "AlphaMean",
    "aileronInboard",
    "aileronOutboard",
    "elevator",
    "htp",
)
COEFFICIENT_FIELDS = ("c_d", "c_l", "c_my")
POINT_FIELDS = (*COORDINATE_FIELDS, *NORMAL_FIELDS, "Surface", *TARGET_FIELDS)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _case_dtype(count: int) -> np.dtype:
    return np.dtype(
        [
            ("coordinates", np.float32, (count, 3)),
            ("normals", np.float32, (count, 3)),
            ("surface", np.float32, (count,)),
            ("targets", np.float32, (count, 4)),
            ("conditions", np.float64, (len(CONDITION_FIELDS),)),
            ("coefficients", np.float64, (len(COEFFICIENT_FIELDS),)),
        ]
    )


def _write_case(group: h5py.Group, destination: Path) -> dict[str, object]:
    count = int(group[COORDINATE_FIELDS[0]].shape[0])
    if count < 1:
        raise ValueError(f"{group.name} has no points")
    for name in POINT_FIELDS:
        values = group.get(name)
        if not isinstance(values, h5py.Dataset) or values.shape != (count,):
            raise ValueError(f"{group.name}/{name} must have shape {(count,)}")
        if values.dtype != np.dtype(np.float32):
            raise ValueError(f"{group.name}/{name} must be float32")
    for name in (*CONDITION_FIELDS, *COEFFICIENT_FIELDS):
        if name not in group.attrs:
            raise KeyError(f"{group.name} lacks attribute {name}")

    mapped = open_memmap(destination, mode="w+", dtype=_case_dtype(count), shape=(1,))
    record = mapped[0]
    for column, name in enumerate(COORDINATE_FIELDS):
        record["coordinates"][:, column] = group[name][...]
    for column, name in enumerate(NORMAL_FIELDS):
        record["normals"][:, column] = group[name][...]
    record["surface"][:] = group["Surface"][...]
    for column, name in enumerate(TARGET_FIELDS):
        record["targets"][:, column] = group[name][...]
    record["conditions"][:] = [group.attrs[name] for name in CONDITION_FIELDS]
    record["coefficients"][:] = [group.attrs[name] for name in COEFFICIENT_FIELDS]
    mapped.flush()
    del record, mapped

    check = np.load(destination, mmap_mode="r", allow_pickle=False)[0]
    for column, name in enumerate(COORDINATE_FIELDS):
        if not np.array_equal(check["coordinates"][:, column], group[name][...]):
            raise ValueError(f"split verification failed: {group.name}/{name}")
    for column, name in enumerate(NORMAL_FIELDS):
        if not np.array_equal(check["normals"][:, column], group[name][...]):
            raise ValueError(f"split verification failed: {group.name}/{name}")
    if not np.array_equal(check["surface"], group["Surface"][...]):
        raise ValueError(f"split verification failed: {group.name}/Surface")
    for column, name in enumerate(TARGET_FIELDS):
        if not np.array_equal(check["targets"][:, column], group[name][...]):
            raise ValueError(f"split verification failed: {group.name}/{name}")
    return {
        "points": count,
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
        "attributes": {name: _json_value(value) for name, value in group.attrs.items()},
    }


def split_hdf5(source_dir: Path, output_dir: Path) -> Path:
    source_root = source_dir.resolve()
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite NASA split output: {output}")
    staging = output.with_name(f".{output.name}.building-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    try:
        manifest: dict[str, Any] = {
            "format": FORMAT,
            "dataset": "NASA-CRM",
            "file_pattern": "{split}/features_{index:05d}.npy",
            "index_origin": 0,
            "fields": {
                "coordinates": list(COORDINATE_FIELDS),
                "normals": list(NORMAL_FIELDS),
                "surface": ["Surface"],
                "targets": list(TARGET_FIELDS),
                "conditions": list(CONDITION_FIELDS),
                "coefficients": list(COEFFICIENT_FIELDS),
            },
            "source": {},
            "splits": {},
        }
        for split, basename in SOURCE_FILES.items():
            source_file = source_root / basename
            if not source_file.is_file():
                raise FileNotFoundError(source_file)
            split_root = staging / split
            split_root.mkdir()
            cases: list[dict[str, object]] = []
            with h5py.File(source_file, "r") as source:
                keys = sorted(source.keys())
                if len(keys) != EXPECTED_CASES[split]:
                    raise ValueError(
                        f"NASA {split} cases={len(keys)}, expected={EXPECTED_CASES[split]}"
                    )
                for index, key in enumerate(keys):
                    filename = split_root / f"features_{index:05d}.npy"
                    details = _write_case(source[key], filename)
                    cases.append({"index": index, "key": key, "file": filename.name, **details})
                    print(f"[{split}] {index + 1}/{len(keys)} {key}", flush=True)
            manifest["source"][split] = {
                "file": basename,
                "bytes": source_file.stat().st_size,
                "sha256": _sha256(source_file),
            }
            manifest["splits"][split] = {
                "directory": split,
                "samples": len(cases),
                "point_count_min": min(int(case["points"]) for case in cases),
                "point_count_max": max(int(case["points"]) for case in cases),
                "point_count_total": sum(int(case["points"]) for case in cases),
                "cases": cases,
            }
        manifest_file = staging / "data_manifest.json"
        manifest_file.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "data_manifest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    nasa_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=nasa_root / "generate" / "raw_source")
    parser.add_argument("--output-dir", type=Path, default=nasa_root / "data" / "raw")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(split_hdf5(args.source_dir, args.output_dir), flush=True)


if __name__ == "__main__":
    main()
