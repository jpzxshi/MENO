"""Generate affine-coordinate Poisson moments from the complete raw data.

The benchmark moments are intentionally *not* recomputed with the newer
standard-domain MFE encoder.  Historical data used the shifted orthonormal
Legendre basis on ``[0,1]^2`` and was later migrated with the exact operation
``float32(old_moment) * float32(0.5)``.  The final factor remains the 2-D basis
normalization conversion; the basis coordinates now use the one dataset-wide
affine transform recorded in the moment manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander

from ..dataset import (
    DATA_FORMAT,
    MOMENT_DATA_FORMAT,
    NORMALIZATION_PROTOCOL,
    RAW_DATA_FORMAT,
    fit_coordinate_transform,
)


DEFAULT_MODE = 12
INPUT_CHANNELS = ("domain", "diffusion", "source", "boundary")
LOCAL_FUNCTION_CHANNELS = ("f", "k", "g_reconstruction")
MOMENT_PROTOCOL = "float32(affine_legendre_moment) * float32(0.5)"


def _sha256_file(filename: Path) -> str:
    digest = hashlib.sha256()
    with open(filename, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_raw_manifest(raw_dir: Path, *, verify_hashes: bool) -> dict[str, Any]:
    filename = raw_dir / "data_manifest.json"
    if not filename.is_file():
        raise FileNotFoundError(f"missing raw manifest: {filename}")
    manifest = json.loads(filename.read_text(encoding="utf-8"))
    if manifest.get("format") != RAW_DATA_FORMAT:
        raise ValueError(
            f"raw format={manifest.get('format')!r}, expected={RAW_DATA_FORMAT!r}"
        )
    if manifest.get("coordinate_domain") != [0.0, 1.0]:
        raise ValueError("historical Poisson raw coordinates must use [0,1]^2")
    generator = manifest.get("generator")
    expected_generator = {
        "seed": 42,
        "mesh_size": 0.01,
        "radius_features": 1001,
        "field_features": 100,
        "samples_per_family": 5000,
    }
    if not isinstance(generator, dict) or any(
        generator.get(name) != value for name, value in expected_generator.items()
    ):
        raise ValueError("raw generator protocol does not match the benchmark")
    for family in ("star", "annular"):
        entry = manifest.get("families", {}).get(family)
        if not isinstance(entry, dict) or int(entry.get("samples", -1)) != 5000:
            raise ValueError(f"raw manifest has invalid {family} family entry")
        cases = entry.get("cases")
        if not isinstance(cases, list) or len(cases) != 5000:
            raise ValueError(f"raw manifest has invalid {family} case catalog")
        family_root = raw_dir / str(entry.get("directory", family))
        for raw_index, case in enumerate(cases):
            path = family_root / str(case.get("file"))
            if not path.is_file() or path.stat().st_size != int(case.get("bytes", -1)):
                raise ValueError(f"raw case file mismatch: {path}")
            if verify_hashes and _sha256_file(path) != case.get("sha256"):
                raise ValueError(f"raw case SHA256 mismatch: {path}")
    return manifest


def _affine_legendre(
    values: np.ndarray, center: float, half_span: float, mode: int
) -> np.ndarray:
    normalized = (np.asarray(values, dtype=np.float64) - center) / half_span
    return legvander(normalized, mode - 1) * np.sqrt(2.0 * np.arange(mode) + 1.0)


def legacy_shifted_moments(
    points: np.ndarray,
    line: np.ndarray,
    triangle: np.ndarray,
    k: np.ndarray,
    f: np.ndarray,
    g: np.ndarray,
    mode: int = DEFAULT_MODE,
    *,
    coordinate_center: np.ndarray,
    coordinate_half_span: np.ndarray,
) -> np.ndarray:
    """Encode four fields with the frozen affine coordinate transform."""

    center = np.asarray(coordinate_center, dtype=np.float64)
    half_span = np.asarray(coordinate_half_span, dtype=np.float64)
    if center.shape != (2,) or half_span.shape != (2,) or np.any(half_span <= 0.0):
        raise ValueError("Poisson coordinate transform must be finite positive [2]")

    vertices = points[triangle]
    vector1 = vertices[:, 1, :] - vertices[:, 0, :]
    vector2 = vertices[:, 2, :] - vertices[:, 0, :]
    areas = 0.5 * np.abs(vector1[:, 0] * vector2[:, 1] - vector1[:, 1] * vector2[:, 0])
    barycentric = np.asarray(
        ((1 / 6, 1 / 6, 2 / 3), (1 / 6, 2 / 3, 1 / 6), (2 / 3, 1 / 6, 1 / 6))
    )
    weights = np.asarray((1 / 3, 1 / 3, 1 / 3))
    quadrature = np.tensordot(barycentric, vertices, axes=([1], [1])).transpose(1, 0, 2)
    phi_x = _affine_legendre(quadrature[..., 0], center[0], half_span[0], mode)
    phi_y = _affine_legendre(quadrature[..., 1], center[1], half_span[1], mode)
    weighted_areas = areas[:, None] * weights[None, :]
    domain = np.einsum("tqi,tqj,tq->ij", phi_x, phi_y, weighted_areas).reshape(-1)
    k_quad = np.dot(k[triangle], barycentric.T)
    diffusion = np.einsum(
        "tqi,tqj,tq,tq->ij",
        phi_x,
        phi_y,
        k_quad,
        weighted_areas,
        optimize=True,
    ).reshape(-1)
    f_quad = np.dot(f[triangle], barycentric.T)
    source = np.einsum(
        "tqi,tqj,tq,tq->ij",
        phi_x,
        phi_y,
        f_quad,
        weighted_areas,
        optimize=True,
    ).reshape(-1)

    line_points = points[line]
    # Historical behavior.  In particular, do not replace this with a
    # vertex-index lookup: the canonical benchmark already encodes this rule.
    endpoint_g = np.stack((g, np.roll(g, -1)), axis=1)
    vectors = line_points[:, 1, :] - line_points[:, 0, :]
    lengths = np.sqrt(vectors[:, 0] ** 2 + vectors[:, 1] ** 2)
    gauss_x, gauss_w = leggauss(5)
    parameter = 0.5 * (gauss_x + 1.0)
    line_quadrature = (1.0 - parameter[None, :, None]) * line_points[
        :, 0, None, :
    ] + parameter[None, :, None] * line_points[:, 1, None, :]
    g_quad = (1.0 - parameter[None, :]) * endpoint_g[:, 0, None] + parameter[
        None, :
    ] * endpoint_g[:, 1, None]
    line_weights = 0.5 * gauss_w[None, :] * lengths[:, None]
    phi_x = _affine_legendre(line_quadrature[..., 0], center[0], half_span[0], mode)
    phi_y = _affine_legendre(line_quadrature[..., 1], center[1], half_span[1], mode)
    boundary = np.einsum(
        "eqi,eqj,eq,eq->ij",
        phi_x,
        phi_y,
        g_quad,
        line_weights,
        optimize=True,
    ).reshape(-1)
    old = np.stack((domain, diffusion, source, boundary), axis=-1).astype(np.float32)
    return np.ascontiguousarray(old * np.float32(0.5))


def _encode_case(
    payload: tuple[str, int, int, tuple[float, float], tuple[float, float]]
) -> tuple[int, np.ndarray]:
    filename, raw_index, mode, center, half_span = payload
    raw = np.load(filename, mmap_mode="r", allow_pickle=False)[0]
    moment = legacy_shifted_moments(
        raw["points"][:, :2],
        raw["line"],
        raw["triangle"],
        raw["k"],
        raw["f"],
        raw["g"],
        mode,
        coordinate_center=np.asarray(center, dtype=np.float64),
        coordinate_half_span=np.asarray(half_span, dtype=np.float64),
    )
    return raw_index, moment


def _exclusive_memmap(filename: Path, shape: tuple[int, ...]) -> np.memmap:
    if filename.exists():
        raise FileExistsError(f"refusing to overwrite: {filename}")
    return np.lib.format.open_memmap(filename, mode="w+", dtype=np.float32, shape=shape)


def _encode_family(
    family_root: Path,
    cases: list[dict[str, object]],
    family_offset: int,
    train: np.memmap,
    test: np.memmap,
    *,
    workers: int,
    mode: int,
    coordinate_center: np.ndarray,
    coordinate_half_span: np.ndarray,
) -> None:
    center = tuple(float(value) for value in coordinate_center)
    half_span = tuple(float(value) for value in coordinate_half_span)
    payloads = (
        (
            str(family_root / str(cases[raw_index]["file"])),
            raw_index,
            mode,
            center,
            half_span,
        )
        for raw_index in range(5000)
    )

    def publish(raw_index: int, moment: np.ndarray) -> None:
        if raw_index < 4500:
            train[family_offset * 4500 + raw_index] = moment
        else:
            test[family_offset * 500 + raw_index - 4500] = moment
        if (raw_index + 1) % 100 == 0:
            print(
                f"{family_root.name}: encoded {raw_index + 1}/5000 moments",
                flush=True,
            )

    if workers <= 1:
        for payload in payloads:
            publish(*_encode_case(payload))
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending: set[Any] = set()
        iterator = iter(payloads)
        for _ in range(max(2 * workers, 1)):
            try:
                pending.add(executor.submit(_encode_case, next(iterator)))
            except StopIteration:
                break
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                publish(*future.result())
                try:
                    pending.add(executor.submit(_encode_case, next(iterator)))
                except StopIteration:
                    pass


def _training_coordinate_policy() -> tuple[str, tuple[float, ...] | None]:
    """Return the fixed paper coordinate policy without importing PyTorch."""

    return "prior", (0.5, 0.5)


def _publish_dataset_manifest(
    raw_root: Path,
    moment_root: Path,
    coordinate_transform: dict[str, object],
    mode: int,
) -> Path | None:
    if (
        raw_root.parent != moment_root.parent
        or raw_root.name != "raw"
        or moment_root.name != "moment"
    ):
        return None
    root = raw_root.parent
    manifest_file = root / "manifest.json"
    if manifest_file.exists():
        raise FileExistsError(
            f"refusing to overwrite dataset manifest: {manifest_file}"
        )
    payload = {
        "format": DATA_FORMAT,
        "raw_manifest": "raw/data_manifest.json",
        "raw_manifest_sha256": _sha256_file(raw_root / "data_manifest.json"),
        "moment_manifest": "moment/data_manifest.json",
        "moment_manifest_sha256": _sha256_file(moment_root / "data_manifest.json"),
        "mode": mode,
        "dimension": 2,
        "input_channels": list(INPUT_CHANNELS),
        "local_function_channels": list(LOCAL_FUNCTION_CHANNELS),
        "normalization": NORMALIZATION_PROTOCOL,
        "coordinate_transform": coordinate_transform,
        "splits": {"train": {"samples": 9000}, "test": {"samples": 1000}},
    }
    temporary = manifest_file.with_name(f".{manifest_file.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest_file)
    return manifest_file


def generate_moments(
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    mode: int,
    workers: int,
    verify_raw_hashes: bool,
) -> Path:
    raw_root = Path(raw_dir).resolve()
    output = Path(output_dir).resolve()
    if mode < 2:
        raise ValueError("mode must be at least 2")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    raw_manifest = _load_raw_manifest(raw_root, verify_hashes=verify_raw_hashes)
    coordinate_source, coordinate_prior = _training_coordinate_policy()
    coordinate_transform = fit_coordinate_transform(
        raw_root,
        source=coordinate_source,
        prior_half_span=coordinate_prior,
        raw_manifest=raw_manifest,
    )
    staging = output.with_name(f".{output.name}.building-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        train = _exclusive_memmap(staging / "train_moments.npy", (9000, mode**2, 4))
        test = _exclusive_memmap(staging / "test_moments.npy", (1000, mode**2, 4))
        for family_offset, family in enumerate(("star", "annular")):
            entry = raw_manifest["families"][family]
            _encode_family(
                raw_root / str(entry.get("directory", family)),
                entry["cases"],
                family_offset,
                train,
                test,
                workers=workers,
                mode=mode,
                coordinate_center=coordinate_transform.center,
                coordinate_half_span=coordinate_transform.half_span,
            )
        train.flush()
        test.flush()
        del train, test

        raw_manifest_file = raw_root / "data_manifest.json"
        manifest = {
            "format": MOMENT_DATA_FORMAT,
            "raw_format": RAW_DATA_FORMAT,
            "raw_manifest_sha256": _sha256_file(raw_manifest_file),
            "mode": mode,
            "dimension": 2,
            "basis": "orthonormal_tensor_legendre_2d",
            "basis_domain": [-1.0, 1.0],
            "flatten_order": "C: m=i*mode+j",
            "input_channels": list(INPUT_CHANNELS),
            "moment_protocol": MOMENT_PROTOCOL,
            "normalization": NORMALIZATION_PROTOCOL,
            "coordinate_transform": coordinate_transform.metadata(),
            "splits": {
                "train": {
                    "samples": 9000,
                    "moments": "train_moments.npy",
                    "sha256": _sha256_file(staging / "train_moments.npy"),
                },
                "test": {
                    "samples": 1000,
                    "moments": "test_moments.npy",
                    "sha256": _sha256_file(staging / "test_moments.npy"),
                },
            },
        }
        manifest_file = staging / "data_manifest.json"
        manifest_file.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging.replace(output)
        _publish_dataset_manifest(
            raw_root, output, coordinate_transform.metadata(), mode
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "data_manifest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    poisson_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=poisson_root / "data" / "raw", type=Path)
    parser.add_argument(
        "--output-dir", default=poisson_root / "data" / "moment", type=Path
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, min(4, os.cpu_count() or 1))
    )
    parser.add_argument("--mode", type=int, default=DEFAULT_MODE)
    parser.add_argument("--verify-raw-hashes", action="store_true")
    args = parser.parse_args(argv)
    if args.mode < 2:
        parser.error("--mode must be at least 2")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = generate_moments(
        args.raw_dir,
        args.output_dir,
        mode=args.mode,
        workers=args.workers,
        verify_raw_hashes=args.verify_raw_hashes,
    )
    print(f"published Poisson moment manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
