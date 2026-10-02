"""Generate the sole canonical Poisson normalization from complete raw training data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ..dataset import (
    NORMALIZATION_DATA_FORMAT,
    NORMALIZATION_NAME,
    NORMALIZATION_PROTOCOL,
    SCALING_NAMES,
    compute_statistical_scaling,
    load_coordinate_transform,
    load_manifest,
)


def _sha256_file(filename: Path) -> str:
    digest = hashlib.sha256()
    with open(filename, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _training_coordinate_policy() -> tuple[str, tuple[float, ...] | None]:
    """Return the fixed paper coordinate policy without importing PyTorch."""

    return "prior", (0.5, 0.5)


def generate_normalizations(
    data_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    data_root, data_manifest = load_manifest(data_dir)
    coordinate_source, coordinate_prior = _training_coordinate_policy()
    coordinate_transform = load_coordinate_transform(
        data_root,
        source=coordinate_source,
        prior_half_span=coordinate_prior,
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    filename = output / f"{NORMALIZATION_NAME}.npz"
    temporary_file = output / f".{NORMALIZATION_NAME}.tmp-{os.getpid()}.npz"
    manifest_file = output / "manifest.json"
    temporary_manifest = output / f".manifest.tmp-{os.getpid()}.json"
    for temporary in (temporary_file, temporary_manifest):
        if temporary.exists():
            raise FileExistsError(f"normalization temporary path exists: {temporary}")
    try:
        training_cases = int(data_manifest["splits"]["train"]["samples"])
        training_indices = list(range(training_cases))
        print("computing canonical Poisson normalization", flush=True)
        scaling = compute_statistical_scaling(
            data_root,
            training_indices,
            int(data_manifest["mode"]),
            coordinate_transform=coordinate_transform,
        )
        np.savez(
            temporary_file,
            **{
                name: np.asarray(getattr(scaling, name), dtype=np.float32)
                for name in SCALING_NAMES
            },
        )
        os.replace(temporary_file, filename)

        manifest = {
            "format": NORMALIZATION_DATA_FORMAT,
            "raw_manifest_sha256": _sha256_file(
                data_root / "raw" / "data_manifest.json"
            ),
            "moment_manifest_sha256": _sha256_file(
                data_root / "moment" / "data_manifest.json"
            ),
            "training_split": "train",
            "training_cases": training_cases,
            "normalization": NORMALIZATION_PROTOCOL,
            "artifact": {
                "name": NORMALIZATION_NAME,
                "file": filename.name,
                "sha256": _sha256_file(filename),
            },
        }
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_file)
        if output == data_root / "moment" / "normalization":
            dataset_manifest_file = data_root / "manifest.json"
            dataset_payload = json.loads(
                dataset_manifest_file.read_text(encoding="utf-8")
            )
            dataset_payload["normalization_manifest"] = (
                "moment/normalization/manifest.json"
            )
            dataset_payload["normalization_manifest_sha256"] = _sha256_file(
                output / "manifest.json"
            )
            temporary = dataset_manifest_file.with_name(
                f".{dataset_manifest_file.name}.tmp-{os.getpid()}"
            )
            temporary.write_text(
                json.dumps(dataset_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(dataset_manifest_file)
    finally:
        for temporary in (temporary_file, temporary_manifest):
            temporary.unlink(missing_ok=True)
    return manifest_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    poisson_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=poisson_root / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=poisson_root / "data" / "moment" / "normalization",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = generate_normalizations(args.data_dir, args.output_dir)
    print(f"published canonical Poisson normalization: {manifest}", flush=True)


if __name__ == "__main__":
    main()
