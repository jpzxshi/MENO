"""Split the two Poisson NPZ families into one structured NPY per case."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap


FORMAT = "mfd-poisson-raw-npy-v1"
SOURCE_FORMAT = "mfd-poisson-raw"
FAMILIES = ("star", "annular")
FIELDS = ("points", "vertex", "line", "triangle", "k", "f", "g", "u")


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _case_dtype(values: dict[str, np.ndarray]) -> np.dtype:
    return np.dtype(
        [(name, values[name].dtype, values[name].shape) for name in FIELDS]
    )


def _write_case(
    source: np.lib.npyio.NpzFile,
    raw_index: int,
    destination: Path,
) -> dict[str, object]:
    values = {
        name: np.asarray(source[f"{name}_{raw_index}"])
        for name in FIELDS
    }
    points = values["points"]
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 1:
        raise ValueError(f"invalid Poisson points at raw index {raw_index}")
    count = int(points.shape[0])
    for name in ("k", "f", "u"):
        if values[name].shape != (count,):
            raise ValueError(f"invalid Poisson {name} at raw index {raw_index}")
    mapped = open_memmap(destination, mode="w+", dtype=_case_dtype(values), shape=(1,))
    record = mapped[0]
    for name in FIELDS:
        record[name][...] = values[name]
    mapped.flush()
    del record, mapped
    check = np.load(destination, mmap_mode="r", allow_pickle=False)[0]
    for name in FIELDS:
        if not np.array_equal(check[name], values[name]):
            raise ValueError(f"split verification failed: {raw_index}/{name}")
    return {
        "raw_index": raw_index,
        "file": destination.name,
        "points": count,
        "vertices": int(values["vertex"].shape[0]),
        "lines": int(values["line"].shape[0]),
        "triangles": int(values["triangle"].shape[0]),
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def split_npz(source_dir: Path, output_dir: Path) -> Path:
    source_root = source_dir.resolve()
    output = output_dir.resolve()
    source_manifest_file = source_root / "data_manifest.json"
    source_manifest: dict[str, Any] = json.loads(
        source_manifest_file.read_text(encoding="utf-8")
    )
    if source_manifest.get("format") != SOURCE_FORMAT:
        raise ValueError("Poisson source manifest format mismatch")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Poisson split output: {output}")
    staging = output.with_name(f".{output.name}.building-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    try:
        families: dict[str, object] = {}
        for family in FAMILIES:
            entry = source_manifest["families"][family]
            source_file = source_root / str(entry["file"])
            if _sha256(source_file) != entry["sha256"]:
                raise ValueError(f"Poisson source SHA256 mismatch: {source_file}")
            family_root = staging / family
            family_root.mkdir()
            cases = []
            with np.load(source_file, allow_pickle=False) as source:
                samples = int(source["num"])
                if samples != int(entry["samples"]):
                    raise ValueError(f"Poisson {family} sample count mismatch")
                for raw_index in range(samples):
                    filename = family_root / f"features_{raw_index:05d}.npy"
                    cases.append(_write_case(source, raw_index, filename))
                    if (raw_index + 1) % 100 == 0 or raw_index + 1 == samples:
                        print(f"[{family}] {raw_index + 1}/{samples}", flush=True)
            families[family] = {
                "directory": family,
                "samples": len(cases),
                "source_file": source_file.name,
                "source_bytes": source_file.stat().st_size,
                "source_sha256": entry["sha256"],
                "point_count_min": min(int(case["points"]) for case in cases),
                "point_count_max": max(int(case["points"]) for case in cases),
                "point_count_total": sum(int(case["points"]) for case in cases),
                "cases": cases,
            }
        manifest = {
            "format": FORMAT,
            "source_format": SOURCE_FORMAT,
            "source_manifest_sha256": _sha256(source_manifest_file),
            "generator": source_manifest["generator"],
            "coordinate_domain": source_manifest["coordinate_domain"],
            "file_pattern": "{family}/features_{raw_index:05d}.npy",
            "index_origin": 0,
            "fields": list(FIELDS),
            "families": families,
            "splits": source_manifest["splits"],
            "legacy_sampling": source_manifest["legacy_sampling"],
        }
        (staging / "data_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "data_manifest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    poisson_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=poisson_root / "generate" / "raw_source")
    parser.add_argument("--output-dir", type=Path, default=poisson_root / "data" / "raw")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(split_npz(args.source_dir, args.output_dir), flush=True)


if __name__ == "__main__":
    main()
