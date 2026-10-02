"""Train the manifest-driven native Darcy benchmark.

Global moments and normalization are generated from complete raw training
cases.  Local training queries are resampled directly from the same raw cases
each epoch; selection and test metrics use complete raw cases.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    project_root = str(_PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

try:
    from ..GlobalLocalMFE import GlobalLocalMFE, ModelScaling
    from ..dataset_common import CoordinateTransform
    from ..benchmark_common import (
        PhaseMemoryTracker,
        build_history_row,
        evaluate,
        format_gpu_memory_metrics,
        history_fieldnames,
        make_output_directory,
        restore_rng_state,
        save_checkpoint,
        scheduler_for,
        set_seed,
        train_one_epoch,
        validate_common_training_arguments,
    )
    from ..query_sampling import set_epoch_recursive
    from ..training_protocol import (
        OPTIMIZER,
        SCHEDULER,
        TrainingDefaults,
        add_training_arguments,
        apply_fixed_protocol,
        build_adamw,
        normalize_dropout_arguments,
        normalize_reproducibility_arguments,
        normalize_width_arguments,
    )
except ImportError:  # Support direct execution from the directory with spaces.
    from GlobalLocalMFE import GlobalLocalMFE, ModelScaling
    from dataset_common import CoordinateTransform
    from benchmark_common import (
        PhaseMemoryTracker,
        build_history_row,
        evaluate,
        format_gpu_memory_metrics,
        history_fieldnames,
        make_output_directory,
        restore_rng_state,
        save_checkpoint,
        scheduler_for,
        set_seed,
        train_one_epoch,
        validate_common_training_arguments,
    )
    from query_sampling import set_epoch_recursive
    from training_protocol import (
        OPTIMIZER,
        SCHEDULER,
        TrainingDefaults,
        add_training_arguments,
        apply_fixed_protocol,
        build_adamw,
        normalize_dropout_arguments,
        normalize_reproducibility_arguments,
        normalize_width_arguments,
    )

if __package__ in (None, ""):
    from darcy.dataset import (
        DIMENSION,
        GLOBAL_MOMENT_CHANNELS,
        GLOBAL_MOMENT_CHANNEL_INDEX,
        LOCAL_FUNCTION_CHANNELS,
        DEFAULT_MODE,
        TARGET_CHANNELS,
        load_coordinate_transform,
        load_data_manifest,
        load_scaling,
        validate_raw_inventory,
    )
    from darcy.metric import (
        FullCaseRelativeL2,
        RELATIVE_L2_LOSS,
        summarize_full_case_metrics,
    )
    from darcy.native import NativeDarcyDataset, combine_resolution_metrics
else:
    from .dataset import (
        DIMENSION,
        GLOBAL_MOMENT_CHANNELS,
        GLOBAL_MOMENT_CHANNEL_INDEX,
        LOCAL_FUNCTION_CHANNELS,
        DEFAULT_MODE,
        TARGET_CHANNELS,
        load_coordinate_transform,
        load_data_manifest,
        load_scaling,
        validate_raw_inventory,
    )
    from .metric import (
        FullCaseRelativeL2,
        RELATIVE_L2_LOSS,
        summarize_full_case_metrics,
    )
    from .native import NativeDarcyDataset, combine_resolution_metrics


METRIC_PROTOCOL = "physical_full_case_then_sample_mean_relative_l2"
LOSS_PROTOCOL = "physical_scalar_relative_l2"
MODEL_ARCHITECTURE = "GlobalLocalMFE"
DATASET_NAME = "deformed_domain_darcy_native"
COORDINATE_SCALE_SOURCE = "prior"
COORDINATE_SCALE_PRIOR: tuple[float, float] | None = (2.0, 2.0)
DEFAULTS = TrainingDefaults(
    mode=DEFAULT_MODE,
    batch_size=4,
    minimum_learning_rate=1.25e-5,
    weight_decay=0.011,
    global_width=160,
    local_width=160,
    heads=10,
    layers=4,
    feedforward=280,
    global_dropout=0.0,
    local_dropout=0.0,
    global_position_frequencies=6,
    local_fourier_frequencies=3,
    train_query_limit=2500,
    decode_chunk_size=5000,
    log_every=1,
    checkpoint_every=25,
    normalize_inputs=True,
    normalize_outputs=True,
    spectral_gradient_injection=True,
)

RESUME_COMPATIBILITY_FIELDS = (
    "protocol_version",
    "architecture",
    "dataset",
    "device",
    "no_amp",
    "precision",
    "expected_mode",
    "data_manifest_sha256",
    "raw_inventory_preflight",
    "coordinate_scale_source",
    "coordinate_scale_prior",
    "coordinate_transform",
    "query_sampling_protocol",
    "epochs",
    "batch_size",
    "learning_rate",
    "minimum_learning_rate",
    "weight_decay",
    "warmup_ratio",
    "gradient_clip",
    "global_width",
    "local_width",
    "heads",
    "layers",
    "feedforward",
    "global_dropout",
    "local_dropout",
    "global_position_frequencies",
    "local_fourier_frequencies",
    "spectral_gradient_injection",
    "spectral_gradient_projection",
    "gradient_injection_architecture",
    "deterministic",
    "normalize_inputs",
    "normalize_outputs",
    "train_query_limit",
    "decode_chunk_size",
    "validation_every",
    "log_every",
    "checkpoint_every",
    "num_workers",
    "seed",
    "skip_test",
    "loss_protocol",
    "metric_protocol",
    "input_channels",
    "function_channels",
    "output_channels",
    "dimension",
    "train_samples",
    "selection_samples",
    "test_samples",
    "scheduler_step_interval",
    "scheduler_parameters",
    "optimizer_parameter_groups",
)

RESUME_PATH_FIELDS = ("data_root", "data_manifest")


def _sha256_file(filename: Path) -> str:
    digest = hashlib.sha256()
    with filename.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(filename: Path, payload: dict[str, object]) -> None:
    temporary = filename.with_suffix(filename.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(filename)


def _read_json(filename: Path) -> dict[str, object]:
    if not filename.is_file():
        return {}
    loaded = json.loads(filename.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"JSON top-level value must be an object: {filename}")
    return loaded


def _archive_copy(filename: Path, label: str) -> Path:
    """Archive pre-resume evidence without overwriting an existing archive."""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for suffix in range(1000):
        serial = "" if suffix == 0 else f"-{suffix}"
        archive = filename.with_name(
            f"{filename.stem}.{label}-{timestamp}{serial}{filename.suffix}"
        )
        if not archive.exists():
            shutil.copy2(filename, archive)
            return archive
    raise RuntimeError(f"cannot allocate a unique archive name for {filename}")


def _load_checkpoint(filename: Path, *, label: str) -> dict[str, object]:
    loaded = torch.load(filename, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise TypeError(f"{label} top-level value must be a dictionary")
    required = {
        "epoch",
        "score",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "arguments",
    }
    missing = required.difference(loaded)
    if missing:
        raise KeyError(f"{label} is missing fields: {sorted(missing)}")
    if not isinstance(loaded["arguments"], Mapping):
        raise TypeError(f"{label}.arguments must be a dictionary")
    return loaded


def validate_resume_arguments(
    saved: Mapping[str, object], current: Mapping[str, object]
) -> None:
    """Reject checkpoints from a different dataset, model, or training protocol."""

    saved = normalize_dropout_arguments(normalize_width_arguments(saved))
    current = normalize_dropout_arguments(normalize_width_arguments(current))
    saved = normalize_reproducibility_arguments(saved)
    current = normalize_reproducibility_arguments(current)
    incompatible: dict[str, tuple[object, object]] = {}
    for name in RESUME_COMPATIBILITY_FIELDS:
        if name == "no_amp" and name not in saved:
            saved_value = True
        elif name not in saved:
            incompatible[name] = ("<missing>", current.get(name))
            continue
        else:
            saved_value = saved[name]
        if saved_value != current.get(name):
            incompatible[name] = (saved_value, current.get(name))

    for name in RESUME_PATH_FIELDS:
        if name not in saved:
            incompatible[name] = ("<missing>", current.get(name))
            continue
        saved_path = Path(str(saved[name])).resolve()
        current_path = Path(str(current.get(name))).resolve()
        if saved_path != current_path:
            incompatible[name] = (str(saved_path), str(current_path))

    if incompatible:
        details = ", ".join(
            f"{name}: checkpoint={old!r}, current={new!r}"
            for name, (old, new) in incompatible.items()
        )
        raise ValueError(
            f"resume configuration is incompatible with checkpoint: {details}"
        )


def _finite_history_peak(rows: list[dict[str, str]], field: str) -> float | None:
    values: list[float] = []
    for row in rows:
        text = row.get(field, "")
        if not text:
            continue
        value = float(text)
        if not math.isfinite(value):
            raise ValueError(
                f"history.csv field {field} contains a non-finite value: {text}"
            )
        values.append(value)
    return max(values) if values else None


def prepare_history_for_resume(
    history_path: Path,
    completed_epoch: int,
    memory_tracker: PhaseMemoryTracker,
) -> dict[str, object]:
    """Archive and trim newer history while restoring timing and memory evidence."""

    if not history_path.is_file() or history_path.stat().st_size == 0:
        warnings.warn(
            "resume checkpoint exists but history.csv is missing or empty; creating new history",
            RuntimeWarning,
            stacklevel=2,
        )
        return {
            "elapsed_offset_seconds": 0.0,
            "kept_rows": 0,
            "kept_epoch": None,
            "kept_optimizer_steps": None,
            "discarded_rows": 0,
            "archive": None,
        }

    expected_fields = list(history_fieldnames("scalar"))
    phase_memory_fields = {
        f"{phase}_peak_{kind}_mib"
        for phase in ("train", "validation", "test", "overall")
        for kind in ("allocated", "reserved")
    }
    legacy_fields = [
        field for field in expected_fields if field not in phase_memory_fields
    ]
    with history_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        actual_fields = reader.fieldnames
        if actual_fields not in (expected_fields, legacy_fields):
            raise ValueError(
                "history.csv columns do not match the current protocol: "
                f"expected={expected_fields}, actual={actual_fields}"
            )
        all_rows = list(reader)

    epochs = [int(row["epoch"]) for row in all_rows]
    if epochs != sorted(set(epochs)):
        raise ValueError("history.csv epochs must be strictly increasing and unique")
    for row in all_rows:
        elapsed = float(row.get("elapsed_seconds") or 0.0)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError(
                "history.csv elapsed_seconds must be finite and non-negative"
            )
        steps = int(row.get("optimizer_steps") or 0)
        if steps < 0:
            raise ValueError("history.csv optimizer_steps cannot be negative")

    elapsed_offset = (
        float(all_rows[-1].get("elapsed_seconds") or 0.0) if all_rows else 0.0
    )
    legacy_history = actual_fields == legacy_fields
    if legacy_history:
        memory_tracker.seed_phase(
            "train",
            peak_allocated_mib=_finite_history_peak(all_rows, "peak_allocated_mib"),
            peak_reserved_mib=_finite_history_peak(all_rows, "peak_reserved_mib"),
        )
    else:
        for phase in ("train", "validation", "test"):
            memory_tracker.seed_phase(
                phase,
                peak_allocated_mib=_finite_history_peak(
                    all_rows, f"{phase}_peak_allocated_mib"
                ),
                peak_reserved_mib=_finite_history_peak(
                    all_rows, f"{phase}_peak_reserved_mib"
                ),
            )

    kept_rows = [row for row in all_rows if int(row["epoch"]) <= completed_epoch]
    discarded_rows = len(all_rows) - len(kept_rows)
    archive: Path | None = None
    rewrite = legacy_history or discarded_rows > 0
    if rewrite:
        archive = _archive_copy(history_path, "pre-resume")
        if legacy_history:
            for row in kept_rows:
                allocated = row.get("peak_allocated_mib", "")
                reserved = row.get("peak_reserved_mib", "")
                row.update(
                    {
                        "train_peak_allocated_mib": "",
                        "train_peak_reserved_mib": "",
                        "validation_peak_allocated_mib": "",
                        "validation_peak_reserved_mib": "",
                        "test_peak_allocated_mib": "",
                        "test_peak_reserved_mib": "",
                        "overall_peak_allocated_mib": allocated,
                        "overall_peak_reserved_mib": reserved,
                    }
                )
        temporary = history_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=expected_fields)
            writer.writeheader()
            writer.writerows(kept_rows)
        temporary.replace(history_path)

    kept_epoch = int(kept_rows[-1]["epoch"]) if kept_rows else None
    kept_steps = int(kept_rows[-1].get("optimizer_steps") or 0) if kept_rows else None
    if kept_epoch is None or kept_epoch < completed_epoch:
        warnings.warn(
            "history.csv is behind the checkpoint; missing epochs will not be fabricated: "
            f"history_epoch={kept_epoch}, checkpoint_epoch={completed_epoch}",
            RuntimeWarning,
            stacklevel=2,
        )
    return {
        "elapsed_offset_seconds": elapsed_offset,
        "kept_rows": len(kept_rows),
        "kept_epoch": kept_epoch,
        "kept_optimizer_steps": kept_steps,
        "discarded_rows": discarded_rows,
        "archive": None if archive is None else str(archive),
    }


def prepare_selection_for_resume(
    selection_path: Path, completed_epoch: int
) -> tuple[list[dict[str, object]], str | None]:
    if not selection_path.is_file():
        return [], None
    payload = _read_json(selection_path)
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list) or any(
        not isinstance(record, dict) for record in evaluations
    ):
        raise TypeError("selection_metrics.json evaluations must be a list of objects")
    records = [dict(record) for record in evaluations]
    epochs = [int(record["epoch"]) for record in records]
    if epochs != sorted(set(epochs)):
        raise ValueError(
            "selection_metrics.json epochs must be strictly increasing and unique"
        )
    kept = [record for record in records if int(record["epoch"]) <= completed_epoch]
    archive: Path | None = None
    if len(kept) != len(records):
        archive = _archive_copy(selection_path, "pre-resume")
        _write_json(selection_path, {"split": "selection", "evaluations": kept})
    return kept, None if archive is None else str(archive)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the shared training protocol plus Darcy data/runtime options."""

    directory = Path(__file__).resolve().parent
    tokens = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=directory / "data")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a checkpoint in the same run directory",
    )
    add_training_arguments(parser, DEFAULTS)
    # Match the seed-42 FP32 baseline (0.6957916% combined test relative L2).
    parser.set_defaults(no_amp=True)
    parser.add_argument("--validation-every", type=int, default=25)
    args = parser.parse_args(tokens)
    apply_fixed_protocol(args, DEFAULTS, spectral_gradient_projection="ambient")
    args.command_line = [sys.executable, str(Path(__file__).resolve()), *tokens]
    args.coordinate_scale_source = COORDINATE_SCALE_SOURCE
    args.coordinate_scale_prior = (
        None if COORDINATE_SCALE_PRIOR is None else list(COORDINATE_SCALE_PRIOR)
    )
    return args


def _build_model(scaling: ModelScaling, args: argparse.Namespace) -> GlobalLocalMFE:
    return GlobalLocalMFE(
        mode=args.mode,
        input_channels=len(GLOBAL_MOMENT_CHANNELS),
        output_channels=len(TARGET_CHANNELS),
        dimension=DIMENSION,
        function_channels=len(LOCAL_FUNCTION_CHANNELS),
        scaling=scaling,
        global_width=args.global_width,
        local_width=args.local_width,
        heads=args.heads,
        layers=args.layers,
        feedforward=args.feedforward,
        global_dropout=args.global_dropout,
        local_dropout=args.local_dropout,
        global_position_frequencies=args.global_position_frequencies,
        local_fourier_frequencies=args.local_fourier_frequencies,
        spectral_gradient_injection=args.spectral_gradient_injection,
        spectral_gradient_projection=args.spectral_gradient_projection,
        normalize_inputs=args.normalize_inputs,
        normalize_outputs=args.normalize_outputs,
    )


def _loader(
    dataset: NativeDarcyDataset,
    *,
    batch_size: int,
    shuffle: bool,
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        # PyTorch 2.8 on this Windows host repeatedly faulted in the synchronous
        # pin-memory conversion path; pageable transfers preserve values/order.
        pin_memory=False,
        persistent_workers=False,
    )


def _evaluation_loaders(
    data_root: Path,
    split: str,
    args: argparse.Namespace,
    device: torch.device,
    coordinate_transform: CoordinateTransform,
    *,
    cache_cases: bool,
) -> dict[str, DataLoader]:
    return {
        resolution: _loader(
            NativeDarcyDataset(
                data_root,
                split,
                coordinate_transform=coordinate_transform,
                resolutions=(resolution,),
                cache_cases=cache_cases,
            ),
            batch_size=1,
            shuffle=False,
            args=args,
            device=device,
        )
        for resolution in ("fine", "coarse")
    }


def _evaluate_resolutions(
    model: GlobalLocalMFE,
    loaders: dict[str, DataLoader],
    device: torch.device,
    chunk_size: int,
    use_amp: bool,
) -> dict[str, dict[str, float | str]]:
    results: dict[str, dict[str, float | str]] = {}
    for resolution in ("fine", "coarse"):
        results[resolution] = evaluate(
            model,
            loaders[resolution],
            device,
            chunk_size,
            use_amp,
            LOSS_PROTOCOL,
            METRIC_PROTOCOL,
            FullCaseRelativeL2,
            summarize_full_case_metrics,
        )
    results["combined"] = combine_resolution_metrics(results["fine"], results["coarse"])
    return results


def _prepare_run_arguments(
    args: argparse.Namespace,
    output: Path,
) -> tuple[Path, dict[str, object], ModelScaling, CoordinateTransform]:
    data_root, manifest = load_data_manifest(args.data_root)
    validate_common_training_arguments(args)
    if int(manifest["mode"]) != args.mode:
        raise ValueError(
            f"data mode={manifest['mode']}, requested mode={args.mode}; "
            "regenerate derived data with the same --mode"
        )
    if args.validation_every < 1:
        raise ValueError("validation-every must be positive")
    coordinate_transform = load_coordinate_transform(
        data_root,
        source=COORDINATE_SCALE_SOURCE,
        prior_half_span=COORDINATE_SCALE_PRIOR,
    )
    scaling, _ = load_scaling(data_root, coordinate_transform=coordinate_transform)
    args.data_root = data_root
    args.data_manifest = data_root / "manifest.json"
    args.data_manifest_sha256 = _sha256_file(args.data_manifest)
    args.output_dir = output.resolve()
    args.architecture = MODEL_ARCHITECTURE
    args.dataset = DATASET_NAME
    args.input_channels = len(GLOBAL_MOMENT_CHANNELS)
    args.function_channels = len(LOCAL_FUNCTION_CHANNELS)
    args.output_channels = len(TARGET_CHANNELS)
    args.dimension = DIMENSION
    args.train_samples = int(manifest["splits"]["train"]["samples"])
    args.selection_samples = int(manifest["splits"]["selection"]["samples"])
    args.test_samples = int(manifest["splits"]["test"]["samples"])
    args.loss_protocol = LOSS_PROTOCOL
    args.metric_protocol = METRIC_PROTOCOL
    args.query_sampling_protocol = {
        "source": "raw",
        "population": "uniform_over_raw_vertex_indices",
        "seed": "SeedSequence([seed,epoch,source_index,dataset_index])",
        "overflow": "all_with_replacement",
        "resampled_each_epoch": True,
    }
    args.raw_case_cache = {
        "train": "lazy_process_local_full_case",
        "selection": "lazy_process_local_full_case",
        "test": "disabled_one_shot_evaluation",
    }
    args.coordinate_transform = coordinate_transform.metadata()
    args.python = sys.executable
    args.torch_version = torch.__version__
    args.cuda_version = torch.version.cuda
    args.cudnn_version = torch.backends.cudnn.version()
    args.scheduler_step_interval = "optimizer-update"
    args.scheduler_parameters = {
        "warmup_ratio": args.warmup_ratio,
        "minimum_learning_rate": args.minimum_learning_rate,
    }
    return data_root, manifest, scaling, coordinate_transform


def run_training(args: argparse.Namespace) -> Path:
    requested = args.output_dir
    resume_path = Path(args.resume).resolve() if args.resume is not None else None
    checkpoint: dict[str, object] | None = None
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
        requested_output = (
            Path(requested).resolve() if requested is not None else resume_path.parent
        )
        if requested_output != resume_path.parent:
            raise ValueError(
                "resuming must reuse the checkpoint directory: "
                f"checkpoint_dir={resume_path.parent}, output_dir={requested_output}"
            )
        if not requested_output.is_dir():
            raise FileNotFoundError(
                f"resume output directory does not exist: {requested_output}"
            )
        output = requested_output
        checkpoint = _load_checkpoint(resume_path, label="resume checkpoint")
        args.resume = resume_path
    else:
        if requested is not None and Path(requested).exists():
            raise FileExistsError(
                f"refusing to overwrite output directory: {requested}"
            )

    preflight_root, preflight_manifest = load_data_manifest(args.data_root)
    args.raw_inventory_preflight = validate_raw_inventory(
        preflight_root, preflight_manifest
    )
    if resume_path is None:
        output = make_output_directory(
            requested,
            experiment_name="darcy",
            protocol_name="canonical-affine",
            base_directory=Path(__file__).resolve().parent / "result",
        )

    state_file = output / "run_state.json"
    process_started_at = datetime.now().isoformat(timespec="seconds")
    if resume_path is None:
        state: dict[str, object] = {
            "status": "initializing",
            "pid": os.getpid(),
            "started_at": process_started_at,
            "process_started_at": process_started_at,
            "output": str(output),
            "resume_events": [],
        }
    else:
        state = _read_json(state_file)
        resume_events = state.get("resume_events", [])
        if not isinstance(resume_events, list):
            raise TypeError("run_state resume_events must be a list")
        resume_events = list(resume_events)
        resume_events.append(
            {
                "attempt": len(resume_events) + 1,
                "started_at": process_started_at,
                "source": str(resume_path),
                "checkpoint_epoch": int(checkpoint["epoch"]),
                "previous_status": state.get("status"),
                "previous_epoch": state.get("epoch"),
                "previous_optimizer_steps": state.get("optimizer_steps"),
                "previous_error": state.get("error"),
            }
        )
        state.update(
            {
                "status": "resuming",
                "pid": os.getpid(),
                "process_started_at": process_started_at,
                "output": str(output),
                "resume_events": resume_events,
            }
        )
    _write_json(state_file, state)

    try:
        data_root, _, scaling, coordinate_transform = _prepare_run_arguments(
            args, output
        )
        args.reproducibility = set_seed(args.seed, deterministic=args.deterministic)
        device = torch.device(args.device)
        use_amp = device.type == "cuda" and not args.no_amp
        args.precision = "amp-fp16" if use_amp else "float32"

        scaling_file = output / "scaling.npz"
        if checkpoint is not None:
            if not scaling_file.is_file():
                raise FileNotFoundError(
                    f"resuming requires the original scaling.npz: {scaling_file}"
                )
            scaling = ModelScaling.load(
                str(scaling_file),
                coordinate_scale=np.asarray(
                    coordinate_transform.half_span, dtype=np.float32
                ),
            )

        training_dataset = NativeDarcyDataset(
            data_root,
            "train",
            coordinate_transform=coordinate_transform,
            query_count=args.train_query_limit,
            random_queries=True,
            seed=args.seed,
            cache_cases=True,
        )
        training_loader = _loader(
            training_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            args=args,
            device=device,
        )
        selection_loaders = _evaluation_loaders(
            data_root,
            "selection",
            args,
            device,
            coordinate_transform,
            cache_cases=True,
        )
        model = _build_model(scaling, args).to(device)
        optimizer, parameter_groups = build_adamw(
            model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        args.optimizer_parameter_groups = parameter_groups
        total_steps = max(args.epochs * len(training_loader), 1)
        scheduler = scheduler_for(
            optimizer,
            total_steps=total_steps,
            warmup_ratio=args.warmup_ratio,
            minimum_ratio=args.minimum_learning_rate / args.learning_rate,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        resume_state_source = resume_path
        best_file = output / "best.pt"
        best_checkpoint: dict[str, object] | None = None
        if checkpoint is not None:
            config_file = output / "config.json"
            if not config_file.is_file():
                raise FileNotFoundError(
                    f"resuming requires the original config.json: {config_file}"
                )
            saved_config = _read_json(config_file)
            validate_resume_arguments(saved_config, vars(args))
            saved_arguments = checkpoint["arguments"]
            assert isinstance(saved_arguments, Mapping)
            validate_resume_arguments(saved_arguments, vars(args))
            for name in ("architecture", "dataset"):
                top_level = checkpoint.get(name)
                if top_level is not None and top_level != getattr(args, name):
                    raise ValueError(
                        f"resume checkpoint.{name}={top_level!r} is incompatible "
                        f"with current value {getattr(args, name)!r}"
                    )
            checkpoint_epoch = int(checkpoint["epoch"])
            if checkpoint_epoch < 0 or checkpoint_epoch > args.epochs:
                raise ValueError(
                    f"checkpoint epoch={checkpoint_epoch} is outside [0,{args.epochs}]"
                )

            if best_file.is_file():
                best_checkpoint = _load_checkpoint(best_file, label="best.pt")
                best_arguments = best_checkpoint["arguments"]
                assert isinstance(best_arguments, Mapping)
                validate_resume_arguments(best_arguments, vars(args))
                for name in ("architecture", "dataset"):
                    top_level = best_checkpoint.get(name)
                    if top_level is not None and top_level != getattr(args, name):
                        raise ValueError(
                            f"best.pt.{name}={top_level!r} is incompatible "
                            f"with current value {getattr(args, name)!r}"
                        )
                best_epoch = int(best_checkpoint["epoch"])
                if best_epoch < 0 or best_epoch > args.epochs:
                    raise ValueError(
                        f"best.pt epoch={best_epoch} is outside [0,{args.epochs}]"
                    )
                best_checkpoint_score = float(best_checkpoint["score"])
                if not math.isfinite(best_checkpoint_score):
                    raise ValueError("best.pt score must be finite")
                if best_epoch > checkpoint_epoch:
                    checkpoint = best_checkpoint
                    checkpoint_epoch = best_epoch
                    resume_state_source = best_file
                    print(
                        "Warning: best.pt is newer than last.pt; resuming from the "
                        f"newer complete state at epoch={best_epoch}",
                        flush=True,
                    )

        memory_tracker = PhaseMemoryTracker(device)
        start_epoch = 1
        elapsed_offset = 0.0
        best_score = float("inf")
        latest_score = float("inf")
        completed_steps = 0
        validation_records: list[dict[str, object]] = []
        history_path = output / "history.csv"
        resume_history: dict[str, object] | None = None
        selection_archive: str | None = None
        resume_rng_restored = False
        resume_scaler_restored = False
        resume_rng_note = "fresh run"
        resume_scaler_note = "fresh run"

        if checkpoint is not None:
            model.load_state_dict(checkpoint["model_state"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            completed_epoch = int(checkpoint["epoch"])
            start_epoch = completed_epoch + 1
            latest_score = float(checkpoint["score"])
            if not math.isfinite(latest_score):
                raise ValueError("resume checkpoint score must be finite")
            completed_steps = int(scheduler.last_epoch)
            if completed_steps < 0:
                raise ValueError("resume scheduler last_epoch cannot be negative")

            scaler_state = checkpoint.get("scaler_state")
            if use_amp:
                if scaler_state is None:
                    resume_scaler_note = "AMP checkpoint has no scaler_state; continuing with a new GradScaler"
                    warnings.warn(resume_scaler_note, RuntimeWarning, stacklevel=2)
                elif not isinstance(scaler_state, Mapping):
                    raise TypeError("checkpoint scaler_state must be a mapping")
                else:
                    scaler.load_state_dict(dict(scaler_state))
                    resume_scaler_restored = True
                    resume_scaler_note = "AMP GradScaler state restored"
            else:
                resume_scaler_note = "AMP disabled; no scaler state required"

            rng_state = checkpoint.get("rng_state")
            if rng_state is None:
                resume_rng_note = "legacy checkpoint has no RNG state; random streams restart from the configured seed"
                warnings.warn(resume_rng_note, RuntimeWarning, stacklevel=2)
            elif not isinstance(rng_state, Mapping):
                raise TypeError("checkpoint rng_state must be a mapping")

            previous_resume_file = output / "resume_config.json"
            previous_resume = (
                _read_json(previous_resume_file)
                if previous_resume_file.is_file()
                else {}
            )
            resume_history = prepare_history_for_resume(
                history_path, completed_epoch, memory_tracker
            )
            elapsed_offset = float(resume_history["elapsed_offset_seconds"])
            previous_elapsed = float(
                previous_resume.get("resume_elapsed_offset_seconds") or 0.0
            )
            elapsed_offset = max(elapsed_offset, previous_elapsed)
            previous_memory = previous_resume.get("resume_memory")
            if isinstance(previous_memory, Mapping):
                for phase in ("train", "validation", "test"):
                    phase_memory = previous_memory.get(phase)
                    if isinstance(phase_memory, Mapping):
                        memory_tracker.seed_phase(
                            phase,
                            peak_allocated_bytes=phase_memory.get(
                                "peak_allocated_bytes"
                            ),
                            peak_reserved_bytes=phase_memory.get("peak_reserved_bytes"),
                            peak_allocated_mib=phase_memory.get("peak_allocated_mib"),
                            peak_reserved_mib=phase_memory.get("peak_reserved_mib"),
                        )
            kept_epoch = resume_history["kept_epoch"]
            kept_steps = resume_history["kept_optimizer_steps"]
            if kept_epoch == completed_epoch and kept_steps != completed_steps:
                raise ValueError(
                    "history and checkpoint optimizer steps disagree: "
                    f"history={kept_steps}, scheduler={completed_steps}"
                )

            validation_records, selection_archive = prepare_selection_for_resume(
                output / "selection_metrics.json", completed_epoch
            )
            if best_checkpoint is not None:
                best_epoch = int(best_checkpoint["epoch"])
                if best_epoch > completed_epoch:
                    raise ValueError(
                        f"best.pt epoch={best_epoch} is newer than resume state "
                        f"epoch={completed_epoch}"
                    )
                best_score = float(best_checkpoint["score"])
                recorded_scores = [
                    float(record["combined"]["case_relative_l2"])
                    for record in validation_records
                    if isinstance(record.get("combined"), Mapping)
                ]
                if recorded_scores and not math.isclose(
                    min(recorded_scores),
                    best_score,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                ):
                    raise ValueError(
                        "best.pt score does not match selection_metrics.json: "
                        f"best={best_score}, records={min(recorded_scores)}"
                    )
            elif validation_records:
                raise FileNotFoundError(
                    "selection_metrics.json has evaluations but best.pt is missing"
                )

            if rng_state is not None:
                restore_rng_state(rng_state)
                resume_rng_restored = True
                resume_rng_note = "Python/NumPy/Torch RNG states restored"

            resume_events = state.get("resume_events", [])
            assert isinstance(resume_events, list) and resume_events
            resume_attempt = int(resume_events[-1]["attempt"])
            resume_record = {
                **vars(args),
                "resume_attempt": resume_attempt,
                "resume_state_source": str(resume_state_source),
                "resume_checkpoint_epoch": completed_epoch,
                "resume_start_epoch": start_epoch,
                "resume_optimizer_steps": completed_steps,
                "resume_best_score": best_score,
                "resume_rng_state_restored": resume_rng_restored,
                "resume_rng_note": resume_rng_note,
                "resume_scaler_state_restored": resume_scaler_restored,
                "resume_scaler_note": resume_scaler_note,
                "resume_history": resume_history,
                "resume_selection_archive": selection_archive,
                "resume_elapsed_offset_seconds": elapsed_offset,
                "resume_memory": memory_tracker.snapshot(
                    elapsed_seconds=elapsed_offset
                ),
            }
            resume_config = output / "resume_config.json"
            if resume_config.exists():
                _archive_copy(resume_config, "previous")
            _write_json(resume_config, resume_record)
            _write_json(
                output / f"resume_config.attempt-{resume_attempt}.json",
                resume_record,
            )
            print(
                f"resume={resume_state_source} completed_epoch={completed_epoch} "
                f"next_epoch={start_epoch} target_epoch={args.epochs} "
                f"optimizer_steps={completed_steps} "
                f"rng_restored={resume_rng_restored} "
                f"scaler_restored={resume_scaler_restored}",
                flush=True,
            )

        model_summary = {
            **model.parameter_groups(),
            "architecture": MODEL_ARCHITECTURE,
            "dataset": DATASET_NAME,
            "mode": args.mode,
            "dimension": DIMENSION,
            "layers": args.layers,
            "epochs": args.epochs,
            "optimizer": OPTIMIZER,
            "optimizer_steps": total_steps,
            "optimizer_parameter_groups": parameter_groups,
            "scheduler": SCHEDULER,
            "scheduler_steps": total_steps,
            "precision": args.precision,
            "global_moment_channel_index": GLOBAL_MOMENT_CHANNEL_INDEX,
            "spectral_gradient_injection": args.spectral_gradient_injection,
            "spectral_gradient_projection": args.spectral_gradient_projection,
            "gradient_injection_architecture": args.gradient_injection_architecture,
            "normalize_inputs": args.normalize_inputs,
            "normalize_outputs": args.normalize_outputs,
            "coordinate_scale": np.asarray(scaling.coordinate_scale).tolist(),
            "coordinate_scale_source": args.coordinate_scale_source,
            "coordinate_scale_prior": args.coordinate_scale_prior,
            "coordinate_transform": args.coordinate_transform,
        }
        if checkpoint is None:
            scaling.save(str(scaling_file))
            _write_json(output / "config.json", vars(args))
            _write_json(output / "model_summary.json", model_summary)
        else:
            resume_events = state.get("resume_events", [])
            assert isinstance(resume_events, list) and resume_events
            resume_attempt = int(resume_events[-1]["attempt"])
            resume_summary = output / "resume_model_summary.json"
            if resume_summary.exists():
                _archive_copy(resume_summary, "previous")
            _write_json(resume_summary, model_summary)
            _write_json(
                output / f"resume_model_summary.attempt-{resume_attempt}.json",
                model_summary,
            )
        print(
            f"device={device}, precision={args.precision}, "
            f"batch_size={args.batch_size}, train_queries={args.train_query_limit}",
            flush=True,
        )
        state.update(
            {
                "status": "running",
                "epoch": start_epoch - 1,
                "epochs": args.epochs,
                "optimizer_steps": completed_steps,
                "model_summary": model_summary,
            }
        )
        if checkpoint is not None:
            state.pop("latest", None)
            state["resume_checkpoint_epoch"] = start_epoch - 1
            state["resume_state_source"] = str(resume_state_source)
        state.pop("error", None)
        state.pop("traceback", None)
        _write_json(state_file, state)

        measured_start = time.perf_counter()
        fields = history_fieldnames("scalar")
        append_history = checkpoint is not None and history_path.is_file()
        history_mode = "a" if append_history else "w"
        with history_path.open(history_mode, newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not append_history or history_path.stat().st_size == 0:
                writer.writeheader()
            for epoch in range(start_epoch, args.epochs + 1):
                set_epoch_recursive(training_dataset, epoch)
                memory_tracker.begin("train")
                train_metrics = train_one_epoch(
                    model,
                    training_loader,
                    optimizer,
                    scheduler,
                    scaler,
                    device,
                    args.decode_chunk_size,
                    use_amp,
                    args.gradient_clip,
                    RELATIVE_L2_LOSS,
                    step_scheduler_per_batch=True,
                    record_optimizer_updates=True,
                )
                completed_steps += int(train_metrics.pop("optimizer_updates"))
                memory_tracker.end(
                    "train",
                    elapsed_seconds=(
                        elapsed_offset + time.perf_counter() - measured_start
                    ),
                )
                latest_score = float(train_metrics["loss"])
                if not math.isfinite(latest_score):
                    raise ValueError("Darcy training loss is not finite")

                selection: dict[str, float | str] | None = None
                if epoch % args.validation_every == 0 or epoch == args.epochs:
                    memory_tracker.begin("validation")
                    resolution_metrics = _evaluate_resolutions(
                        model,
                        selection_loaders,
                        device,
                        args.decode_chunk_size,
                        use_amp,
                    )
                    memory_tracker.end(
                        "validation",
                        elapsed_seconds=(
                            elapsed_offset + time.perf_counter() - measured_start
                        ),
                    )
                    selection = resolution_metrics["combined"]
                    score = float(selection["case_relative_l2"])
                    validation_records.append({"epoch": epoch, **resolution_metrics})
                    _write_json(
                        output / "selection_metrics.json",
                        {"split": "selection", "evaluations": validation_records},
                    )
                    if score < best_score:
                        best_score = score
                        save_checkpoint(
                            output / "best.pt",
                            model,
                            optimizer,
                            scheduler,
                            epoch,
                            score,
                            args,
                            scaler=scaler,
                        )

                memory = memory_tracker.snapshot(
                    elapsed_seconds=(
                        elapsed_offset + time.perf_counter() - measured_start
                    )
                )
                row = build_history_row(
                    epoch=epoch,
                    optimizer=optimizer,
                    train_metrics=train_metrics,
                    memory=memory,
                    elapsed_seconds=(
                        elapsed_offset + time.perf_counter() - measured_start
                    ),
                    metric_type="scalar",
                    validation_metrics=selection,
                    scheduler=scheduler,
                    fill_optimizer_steps=False,
                )
                row["optimizer_steps"] = completed_steps
                writer.writerow(row)
                stream.flush()
                state.update(
                    {
                        "epoch": epoch,
                        "optimizer_steps": completed_steps,
                        "latest": row,
                        "best_selection": best_score,
                    }
                )
                _write_json(state_file, state)
                if args.checkpoint_every and epoch % args.checkpoint_every == 0:
                    save_checkpoint(
                        output / "last.pt",
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        latest_score,
                        args,
                        scaler=scaler,
                    )
                if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                    selection_text = (
                        ""
                        if selection is None
                        else f" selection_rL2={100.0 * float(selection['case_relative_l2']):.3f}%"
                    )
                    print(
                        f"epoch={epoch:03d}/{args.epochs} "
                        f"loss={latest_score:.6e} "
                        f"train_rL2={100.0 * float(train_metrics['u_relative_l2']):.3f}%"
                        f"{selection_text} "
                        f"{format_gpu_memory_metrics(device, memory)}",
                        flush=True,
                    )

        save_checkpoint(
            output / "last.pt",
            model,
            optimizer,
            scheduler,
            args.epochs,
            latest_score,
            args,
            scaler=scaler,
        )
        if not args.skip_test:
            result_file = output / "test_metrics.json"
            if result_file.exists():
                if checkpoint is None or start_epoch <= args.epochs:
                    raise FileExistsError(
                        f"refusing repeated formal test: {result_file}"
                    )
                existing_test = _read_json(result_file)
                if (
                    Path(str(existing_test.get("checkpoint"))).resolve()
                    != best_file.resolve()
                ):
                    raise ValueError(
                        "existing test_metrics.json checkpoint does not match best.pt"
                    )
                test_metrics = existing_test.get("metrics")
                test_memory = existing_test.get("gpu_memory")
                if not isinstance(test_metrics, dict) or not isinstance(
                    test_memory, dict
                ):
                    raise TypeError(
                        "existing test_metrics.json must contain metrics and gpu_memory objects"
                    )
                test_phase = test_memory.get("test")
                if isinstance(test_phase, Mapping):
                    memory_tracker.seed_phase(
                        "test",
                        peak_allocated_bytes=test_phase.get("peak_allocated_bytes"),
                        peak_reserved_bytes=test_phase.get("peak_reserved_bytes"),
                        peak_allocated_mib=test_phase.get("peak_allocated_mib"),
                        peak_reserved_mib=test_phase.get("peak_reserved_mib"),
                    )
                selected_checkpoint = _load_checkpoint(best_file, label="best.pt")
                print("Reusing the existing compatible test_metrics.json", flush=True)
            else:
                selected_checkpoint = _load_checkpoint(best_file, label="best.pt")
                model.load_state_dict(selected_checkpoint["model_state"], strict=True)
                memory_tracker.begin("test")
                test_metrics = _evaluate_resolutions(
                    model,
                    _evaluation_loaders(
                        data_root,
                        "test",
                        args,
                        device,
                        coordinate_transform,
                        cache_cases=False,
                    ),
                    device,
                    args.decode_chunk_size,
                    use_amp,
                )
                test_memory = memory_tracker.end(
                    "test",
                    elapsed_seconds=(
                        elapsed_offset + time.perf_counter() - measured_start
                    ),
                )
                _write_json(
                    result_file,
                    {
                        "checkpoint": str(best_file),
                        "selected_epoch": int(selected_checkpoint["epoch"]),
                        "split": "test",
                        "gradient_injection_architecture": args.gradient_injection_architecture,
                        "metrics": test_metrics,
                        "gpu_memory": test_memory,
                    },
                )
            state["test"] = test_metrics["combined"]
            combined = test_metrics["combined"]
            print(
                "test: "
                f"selected_epoch={int(selected_checkpoint['epoch'])}, "
                f"cases={int(float(combined['samples']))}, "
                "case_L2RE fine/coarse/combined="
                f"{100.0 * float(test_metrics['fine']['case_relative_l2']):.3f}%/"
                f"{100.0 * float(test_metrics['coarse']['case_relative_l2']):.3f}%/"
                f"{100.0 * float(combined['case_relative_l2']):.3f}%, "
                f"test_elapsed={float(combined['elapsed_seconds']):.2f}s, "
                f"total_elapsed={float(test_memory['elapsed_seconds']):.2f}s, "
                f"{format_gpu_memory_metrics(device, test_memory)}",
                flush=True,
            )

        memory = memory_tracker.snapshot(
            elapsed_seconds=(elapsed_offset + time.perf_counter() - measured_start)
        )
        memory_file = output / "gpu_memory.json"
        if checkpoint is not None and memory_file.exists():
            _archive_copy(memory_file, "pre-resume")
        _write_json(memory_file, memory)
        state.update(
            {
                "status": "completed",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "gpu_memory": memory,
            }
        )
        _write_json(state_file, state)
        print(f"output={output}", flush=True)
        return output
    except BaseException as error:
        state.update(
            {
                "status": (
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                ),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(state_file, state)
        raise


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run one clean Darcy training job."""

    run_training(parse_args(argv))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULTS",
    "RESUME_COMPATIBILITY_FIELDS",
    "parse_args",
    "prepare_history_for_resume",
    "prepare_selection_for_resume",
    "run_training",
    "validate_resume_arguments",
]
