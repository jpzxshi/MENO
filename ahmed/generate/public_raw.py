"""Convert the public AhmedML VTP surface files to memory-mapped training cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


FORMAT = "mfd-ahmed-surface-raw-npy-v1"
TARGET_FIELDS = (
    "static(p)_coeffMean",
    "wallShearStressMean_x",
    "wallShearStressMean_y",
    "wallShearStressMean_z",
)
GEOMETRY_FIELDS = (
    "body-length",
    "body-height",
    "body-width",
    "front-arc-diameter",
    "slant-angle-length",
    "slant-angle-height",
    "slant-surface-length",
    "slant-angle-degrees",
)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _paper_splits() -> dict[str, list[int]]:
    permutation = np.random.default_rng(42).permutation(np.arange(1, 501))
    return {
        "train": sorted(int(value) for value in permutation[:400]),
        "validation": sorted(int(value) for value in permutation[400:450]),
        "test": sorted(int(value) for value in permutation[450:]),
    }


def _geometry_parameters(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected one geometry row in {path}")
    values = np.asarray([float(rows[0][name]) for name in GEOMETRY_FIELDS], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite geometry parameters in {path}")
    return values


def _targets(mesh: object) -> np.ndarray:
    cell_data = mesh.cell_data
    pressure = np.asarray(cell_data[TARGET_FIELDS[0]], dtype=np.float32).reshape(-1)
    if "wallShearStressMean" in cell_data:
        shear = np.asarray(cell_data["wallShearStressMean"], dtype=np.float32)
    else:
        shear = np.column_stack(
            [np.asarray(cell_data[name], dtype=np.float32).reshape(-1) for name in TARGET_FIELDS[1:]]
        )
    values = np.column_stack((pressure, shear)).astype(np.float32, copy=False)
    if values.shape != (mesh.n_cells, 4):
        raise ValueError(f"Unexpected AhmedML target shape: {values.shape}")
    return values


def _case_dtype(count: int) -> np.dtype:
    return np.dtype(
        [
            ("coordinates", np.float32, (count, 3)),
            ("normals", np.float32, (count, 3)),
            ("measure", np.float32, (count,)),
            ("targets", np.float32, (count, 4)),
            ("geometry_parameters", np.float32, (8,)),
        ]
    )


def _convert_run(source: Path, run: int, destination: Path) -> dict[str, object]:
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError("Install pyvista before converting public AhmedML VTP files") from exc

    run_dir = source / f"run_{run}"
    vtp = run_dir / f"boundary_{run}.vtp"
    area_file = run_dir / f"boundary_cell_area_{run}.npy"
    geometry_file = run_dir / f"geo_parameters_{run}.csv"
    for path in (vtp, area_file, geometry_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    mesh = pv.read(vtp)
    centers = np.asarray(mesh.cell_centers(vertex=False).points, dtype=np.float32)
    normal_mesh = mesh.compute_normals(
        cell_normals=True,
        point_normals=False,
        consistent_normals=False,
        auto_orient_normals=False,
        inplace=False,
    )
    normals = np.asarray(normal_mesh.cell_data["Normals"], dtype=np.float32)
    measure = np.asarray(np.load(area_file, allow_pickle=False), dtype=np.float32).reshape(-1)
    targets = _targets(mesh)
    parameters = _geometry_parameters(geometry_file)
    count = int(mesh.n_cells)
    if centers.shape != (count, 3) or normals.shape != (count, 3) or measure.shape != (count,):
        raise ValueError(f"AhmedML geometry arrays disagree for run {run}")
    if not all(np.isfinite(array).all() for array in (centers, normals, measure, targets)):
        raise ValueError(f"Non-finite AhmedML values in run {run}")

    mapped = open_memmap(destination, mode="w+", dtype=_case_dtype(count), shape=(1,))
    record = mapped[0]
    record["coordinates"][:] = centers
    record["normals"][:] = normals
    record["measure"][:] = measure
    record["targets"][:] = targets
    record["geometry_parameters"][:] = parameters
    mapped.flush()
    del record, mapped, mesh, normal_mesh
    return {
        "run": run,
        "file": destination.name,
        "points": count,
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def convert(source_dir: Path, output_dir: Path) -> Path:
    source = source_dir.resolve()
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Ahmed raw output: {output}")
    splits = _paper_splits()
    staging = output.with_name(f".{output.name}.building-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    try:
        cases = []
        for run in range(1, 501):
            destination = staging / f"features_{run:05d}.npy"
            cases.append(_convert_run(source, run, destination))
            print(f"[AhmedML] {run}/500", flush=True)
        manifest = {
            "format": FORMAT,
            "dataset": "AhmedML",
            "task": "surface",
            "file_pattern": "features_{run:05d}.npy",
            "index_origin": 1,
            "fields": {
                "coordinates": ["x", "y", "z"],
                "normals": ["normal_x", "normal_y", "normal_z"],
                "measure": ["cell_area"],
                "targets": list(TARGET_FIELDS),
                "geometry_parameters": list(GEOMETRY_FIELDS),
            },
            "source": {
                "repository": "https://huggingface.co/datasets/neashton/ahmedml",
                "association": "CellData",
            },
            "splits": splits,
            "cases": cases,
        }
        (staging / "data_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "data_manifest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(convert(args.source_dir, args.output_dir), flush=True)


if __name__ == "__main__":
    main()
