"""Build every derived dataset artifact from files placed in data/raw_zipped."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS = ("poisson", "darcy", "nasa", "ahmed")
DEFAULT_MODES = {"poisson": 12, "darcy": 32, "nasa": 8, "ahmed": 16}


def _run_module(module: str, *arguments: object) -> None:
    command = [sys.executable, "-m", module, *(str(value) for value in arguments)]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _safe_target(root: Path, member: str) -> Path:
    target = (root / member).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Archive member escapes the extraction directory: {member}"
        ) from exc
    return target


def _is_macos_metadata(member: str) -> bool:
    """Ignore Finder/AppleDouble metadata, including its duplicate directories."""

    return any(
        part == "__MACOSX" or part == ".DS_Store" or part.startswith("._")
        for part in member.replace("\\", "/").split("/")
    )


def _extract_archives(raw_zipped: Path, destination: Path) -> None:
    archives = sorted(
        path
        for path in raw_zipped.iterdir()
        if path.is_file()
        and (path.suffix.lower() == ".zip" or tarfile.is_tarfile(path))
    )
    if not archives:
        return
    destination.mkdir(parents=True, exist_ok=False)
    for archive in archives:
        if archive.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive) as handle:
                for member in handle.infolist():
                    _safe_target(destination, member.filename)
                    if member.is_dir() or _is_macos_metadata(member.filename):
                        continue
                    target = _safe_target(destination, member.filename)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with handle.open(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
        else:
            with tarfile.open(archive) as handle:
                members = handle.getmembers()
                for member in members:
                    _safe_target(destination, member.name)
                    if member.issym() or member.islnk():
                        raise ValueError(
                            f"Archive links are not accepted: {member.name}"
                        )
                handle.extractall(
                    destination,
                    members=[m for m in members if not _is_macos_metadata(m.name)],
                )


def _source_roots(raw_zipped: Path, temporary: Path) -> list[Path]:
    roots = [raw_zipped]
    extracted = temporary / "extracted"
    _extract_archives(raw_zipped, extracted)
    if extracted.exists():
        roots.insert(0, extracted)
    return roots


def _find_file(roots: list[Path], name: str) -> Path:
    matches = [match for root in roots for match in root.rglob(name)]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {name!r}; found {len(matches)}")
    return matches[0]


def _build_raw(dataset: str, roots: list[Path], data_dir: Path) -> None:
    raw_dir = data_dir / "raw"
    if raw_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing raw data: {raw_dir}")

    if dataset == "nasa":
        training = _find_file(roots, "trainingData_NASA-CRM.h5")
        test = _find_file(roots, "testData_NASA-CRM.h5")
        if training.parent != test.parent:
            raise ValueError(
                "The NASA training and test HDF5 files must share a directory"
            )
        _run_module(
            "nasa.generate.split",
            "--source-dir",
            training.parent,
            "--output-dir",
            raw_dir,
        )
        return

    if dataset == "ahmed":
        public_root = _find_file(roots, "boundary_1.vtp").parents[1]
        _run_module(
            "ahmed.generate.public_raw",
            "--source-dir",
            public_root,
            "--output-dir",
            raw_dir,
        )
        return

    if dataset == "poisson":
        candidates: list[Path] = []
        for root in roots:
            for manifest in root.rglob("data_manifest.json"):
                try:
                    payload = json.loads(manifest.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if payload.get("format") == "mfd-poisson-raw":
                    candidates.append(manifest.parent)
        if len(candidates) != 1:
            raise FileNotFoundError(
                "Expected one Poisson source manifest with format mfd-poisson-raw"
            )
        _run_module(
            "poisson.generate.split",
            "--source-dir",
            candidates[0],
            "--output-dir",
            raw_dir,
        )
        return

    sources = []
    for name in ("smooth_small_scale", "smooth_large_scale"):
        matches = sorted({
            match.resolve()
            for root in roots
            for match in root.rglob(name)
            if match.is_dir()
            and not _is_macos_metadata(match.relative_to(root).as_posix())
        })
        if not matches:
            raise FileNotFoundError(
                f"Darcy data directory {name!r} is missing. Place "
                "deformed_domain_darcy.zip directly in darcy/data/raw_zipped; "
                "manual extraction is not required."
            )
        if len(matches) != 1:
            raise ValueError(
                f"Multiple Darcy data directories named {name!r}: "
                f"{[str(path) for path in matches]}. Keep either the archive "
                "or one extracted copy in raw_zipped, not both."
            )
        sources.append(matches[0])
    raw_dir.mkdir(parents=True)
    for source in sources:
        shutil.copytree(
            source, raw_dir / source.name,
            ignore=shutil.ignore_patterns("__MACOSX", ".DS_Store", "._*"),
        )


def _build_moments(
    dataset: str, data_dir: Path, workers: int | None, mode: int
) -> None:
    if not (data_dir / "raw").is_dir():
        raise FileNotFoundError(f"Raw data is missing: {data_dir / 'raw'}")
    if dataset == "nasa":
        _run_module("nasa.generate.canonical", "--mode", mode)
    elif dataset == "ahmed":
        if workers is None:
            raise ValueError("AhmedML moment generation requires --workers")
        _run_module("ahmed.generate.canonical", "--mode", mode, "--workers", workers)
    elif dataset == "poisson":
        if workers is None:
            raise ValueError("Poisson moment generation requires --workers")
        _run_module(
            "poisson.generate.moment",
            "--raw-dir",
            data_dir / "raw",
            "--output-dir",
            data_dir / "moment",
            "--workers",
            workers,
            "--mode",
            mode,
            "--verify-raw-hashes",
        )
        _run_module(
            "poisson.generate.normalization",
            "--data-dir",
            data_dir,
            "--output-dir",
            data_dir / "moment" / "normalization",
        )
    else:
        if workers is None:
            raise ValueError("Darcy moment generation requires --workers")
        derived = data_dir.parent / ".darcy-derived"
        _run_module(
            "darcy.generate.moment",
            "--raw-dir",
            data_dir / "raw",
            "--output-dir",
            derived,
            "--workers",
            workers,
            "--mode",
            mode,
        )
        shutil.move(str(derived / "moment"), str(data_dir / "moment"))
        shutil.move(str(derived / "manifest.json"), str(data_dir / "manifest.json"))
        derived.rmdir()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    datasets = parser.add_subparsers(dest="dataset", required=True)
    for dataset in DATASETS:
        dataset_parser = datasets.add_parser(dataset)
        dataset_parser.add_argument(
            "--stage", choices=("all", "raw", "moment"), default="all"
        )
        if dataset != "nasa":
            dataset_parser.add_argument("--workers", type=int, default=4)
        dataset_parser.add_argument(
            "--replace-derived",
            action="store_true",
            help="replace only moment/statistics and manifest; raw files are preserved",
        )
        dataset_parser.add_argument(
            "--mode",
            type=int,
            default=DEFAULT_MODES[dataset],
            help="Legendre moment order",
        )
    args = parser.parse_args(argv)
    if getattr(args, "workers", 1) < 1:
        parser.error("--workers must be positive")
    if args.mode < 2:
        parser.error("--mode must be at least 2")
    if args.replace_derived and args.stage != "moment":
        parser.error("--replace-derived requires --stage moment")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir = PROJECT_ROOT / args.dataset / "data"
    raw_zipped = data_dir / "raw_zipped"
    data_dir.mkdir(parents=True, exist_ok=True)
    if args.replace_derived:
        moment_dir = data_dir / "moment"
        manifest = data_dir / "manifest.json"
        if moment_dir.exists():
            shutil.rmtree(moment_dir)
        if manifest.exists():
            manifest.unlink()
    if args.stage in ("all", "raw"):
        if not raw_zipped.is_dir():
            raise FileNotFoundError(
                f"Create {raw_zipped} and place the downloaded files there first"
            )
        with tempfile.TemporaryDirectory(prefix=f"meno-{args.dataset}-") as temporary:
            roots = _source_roots(raw_zipped, Path(temporary))
            _build_raw(args.dataset, roots, data_dir)
    if args.stage in ("all", "moment"):
        _build_moments(
            args.dataset,
            data_dir,
            getattr(args, "workers", None),
            args.mode,
        )


if __name__ == "__main__":
    main()
