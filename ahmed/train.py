"""Train, validate, and test MENO on the AhmedML surface benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, "") and str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from ..GlobalLocalMFE import GlobalLocalMFE, ModelScaling
    from ..training_protocol import (
        TrainingDefaults,
        add_training_arguments,
        apply_fixed_protocol,
        build_adamw,
        normalize_dropout_arguments,
        normalize_reproducibility_arguments,
        normalize_width_arguments,
    )
    from ..benchmark_common import (
        PhaseMemoryTracker,
        evaluate as evaluate_common,
        make_output_directory,
        build_history_row,
        format_gpu_memory_metrics,
        history_fieldnames,
        restore_rng_state,
        save_checkpoint,
        scheduler_for,
        set_seed,
        train_one_epoch as train_one_epoch_common,
        validate_common_training_arguments,
    )
    from ..query_sampling import set_epoch_recursive
except ImportError:
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from GlobalLocalMFE import GlobalLocalMFE, ModelScaling
    from training_protocol import (
        TrainingDefaults,
        add_training_arguments,
        apply_fixed_protocol,
        build_adamw,
        normalize_dropout_arguments,
        normalize_reproducibility_arguments,
        normalize_width_arguments,
    )
    from benchmark_common import (
        PhaseMemoryTracker,
        evaluate as evaluate_common,
        make_output_directory,
        build_history_row,
        format_gpu_memory_metrics,
        history_fieldnames,
        restore_rng_state,
        save_checkpoint,
        scheduler_for,
        set_seed,
        train_one_epoch as train_one_epoch_common,
        validate_common_training_arguments,
    )
    from query_sampling import set_epoch_recursive

if __package__ in (None, ""):
    from ahmed.dataset import (
        AhmedMLDataset,
        GLOBAL_MOMENT_CHANNELS,
        GLOBAL_MOMENT_CHANNEL_INDEX,
        TASKS,
        load_coordinate_transform,
        read_splits,
        validate_ahmed_schema,
    )
    from ahmed.native import (
        load_data_paths,
        normalization_file,
    )
    from ahmed.metric import (
        FullCaseRelativeL2,
        RELATIVE_MSE_LOSS,
        summarize_full_case_metrics,
    )
else:
    from .dataset import (
        AhmedMLDataset,
        GLOBAL_MOMENT_CHANNELS,
        GLOBAL_MOMENT_CHANNEL_INDEX,
        TASKS,
        load_coordinate_transform,
        read_splits,
        validate_ahmed_schema,
    )
    from .native import (
        load_data_paths,
        normalization_file,
    )
    from .metric import (
        FullCaseRelativeL2,
        RELATIVE_MSE_LOSS,
        summarize_full_case_metrics,
    )


LOSS_PROTOCOL = "physical_p_vector_mean_relative_mse"
METRIC_PROTOCOL = "physical_full_case_then_sample_mean_p_vector_magnitude_relative_l2"
MODEL_ARCHITECTURE = "GlobalLocalMFE"
COORDINATE_SCALE_SOURCE = "train_bounds"
COORDINATE_SCALE_PRIOR = None
DEFAULT_MODE = 16
DEFAULTS = TrainingDefaults(
    mode=DEFAULT_MODE,
    batch_size=1,
    minimum_learning_rate=3.0e-5,
    weight_decay=1.0e-2,
    global_width=512,
    local_width=512,
    heads=8,
    layers=6,
    feedforward=256,
    global_dropout=0.0,
    local_dropout=0.0,
    global_position_frequencies=4,
    local_fourier_frequencies=6,
    train_query_limit=16384,
    decode_chunk_size=16384,
    log_every=1,
    checkpoint_every=0,
    spectral_gradient_injection=True,
)


def train_one_epoch(
    model: GlobalLocalMFE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    chunk_size: int,
    use_amp: bool,
    gradient_clip: float,
) -> dict[str, float]:
    """Train one epoch with the AhmedML grouped physical-space loss."""

    return train_one_epoch_common(
        model,
        loader,
        optimizer,
        scheduler,
        scaler,
        device,
        chunk_size,
        use_amp,
        gradient_clip,
        RELATIVE_MSE_LOSS,
    )


def evaluate(
    model: GlobalLocalMFE,
    loader: DataLoader,
    device: torch.device,
    chunk_size: int,
    use_amp: bool,
) -> dict[str, float | str]:
    """Evaluate full AhmedML cases with chunked local decoding."""

    return evaluate_common(
        model,
        loader,
        device,
        chunk_size,
        use_amp,
        LOSS_PROTOCOL,
        METRIC_PROTOCOL,
        FullCaseRelativeL2,
        summarize_full_case_metrics,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the AhmedML surface benchmark and explicit model settings."""

    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--resume", default=None, help="Not supported for paper-aligned best-only runs"
    )
    parser.add_argument("--data-root", dest="data", default=str(project / "data"))
    add_training_arguments(parser, DEFAULTS)
    # Dataset-local defaults: do not change Darcy/Poisson precision.
    parser.set_defaults(no_amp=True)
    parser.add_argument(
        "--cache-raw-cases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache per-case NPY arrays in memory; disable to mmap them directly",
    )
    parser.add_argument("--eval-every", type=int, default=1)
    args = parser.parse_args(argv)
    if args.resume is not None:
        parser.error("Paper-aligned best-only runs must start fresh; --resume is unsupported.")
    if args.eval_every != 1 or args.log_every != 1 or args.checkpoint_every != 0:
        parser.error("Paper-aligned Ahmed requires eval/log every epoch and checkpoint-every=0.")
    apply_fixed_protocol(args, DEFAULTS, spectral_gradient_projection="ambient")
    args.coordinate_scale_source = COORDINATE_SCALE_SOURCE
    args.coordinate_scale_prior = (
        None if COORDINATE_SCALE_PRIOR is None else list(COORDINATE_SCALE_PRIOR)
    )
    args.task = "surface"
    args.train_runs = []
    args.validation_runs = []
    args.test_runs = []
    args.validation_query_limit = 16384
    args.test_query_limit = -1
    args.query_sampling = "dynamic_uniform_native"
    args.selection_split = "validation"
    args.shuffle_protocol = "independent_torch_generator_seed42_or_cli_seed"

    return args


def main(argv: list[str] | None = None) -> None:
    """Train, select the best validation checkpoint, and run the final test."""

    args = parse_args(argv)
    args.architecture = MODEL_ARCHITECTURE
    args.dataset = "ahmed"
    args.loss_protocol = LOSS_PROTOCOL
    args.metric_protocol = METRIC_PROTOCOL
    validate_common_training_arguments(args)

    if args.no_amp or args.device == "cpu":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    args.reproducibility = set_seed(args.seed, deterministic=args.deterministic)
    args.reproducibility.update(
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
    )
    data = Path(args.data).resolve()
    schema = validate_ahmed_schema(
        data, expected_mode=args.expected_mode, verify_payload_hashes=True
    )
    data_paths = load_data_paths(data)
    normalization_source = normalization_file(data_paths)
    normalization_details = data_paths.manifest["normalization"]
    args.normalization_source = str(normalization_source)
    args.normalization_sha256 = normalization_details["sha256"]
    args.normalization_lineage = normalization_details["lineage"]
    split = read_splits(data)
    args.train_runs = split["train"]
    args.validation_runs = split["validation"]
    args.test_runs = split["test"]
    coordinate_transform = load_coordinate_transform(
        data,
        source=COORDINATE_SCALE_SOURCE,
        prior_half_span=COORDINATE_SCALE_PRIOR,
    )
    args.coordinate_transform = coordinate_transform.metadata()
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    args.precision = "amp-fp16" if use_amp else "float32"
    mode = args.expected_mode
    spec = TASKS["surface"]

    args.data = str(data)
    args.train_point_pool_min = schema["train_point_pool_min"]
    args.train_point_pool_max = schema["train_point_pool_max"]
    args.raw_case_cache = {
        split_name: (
            "lazy_process_local_full_case"
            if args.cache_raw_cases
            else "disabled_direct_per_case_npy"
        )
        for split_name in ("train", "validation", "test")
    }
    resume_path = Path(args.resume).resolve() if args.resume else None
    resume_state_source = resume_path
    checkpoint: dict[str, object] | None = None
    resume_rng_note: str | None = None
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
        requested_output = (
            Path(args.output_dir).resolve()
            if args.output_dir is not None
            else resume_path.parent
        )
        if requested_output != resume_path.parent:
            raise ValueError(
                "resuming must reuse the checkpoint directory so history.csv can be appended safely: "
                f"checkpoint_dir={resume_path.parent}, output_dir={requested_output}"
            )
        output = requested_output
        if not output.is_dir():
            raise NotADirectoryError(
                f"resume output directory does not exist: {output}"
            )
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise TypeError("resume checkpoint must contain a dictionary")
        required_checkpoint_fields = {
            "epoch",
            "score",
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "arguments",
        }
        missing_checkpoint_fields = required_checkpoint_fields.difference(checkpoint)
        if missing_checkpoint_fields:
            raise KeyError(
                "resume checkpoint is missing fields: "
                f"{sorted(missing_checkpoint_fields)}"
            )
        saved_arguments = checkpoint["arguments"]
        if not isinstance(saved_arguments, dict):
            raise TypeError("resume checkpoint arguments must be a dictionary")
        saved_arguments = normalize_dropout_arguments(
            normalize_width_arguments(saved_arguments)
        )
        saved_arguments = normalize_reproducibility_arguments(
            saved_arguments,
            expected=args.deterministic,
        )

        saved_arguments.setdefault("no_amp", True)
        config_file = output / "config.json"
        if not config_file.is_file():
            raise FileNotFoundError(
                f"resuming requires the original config.json: {config_file}"
            )
        with open(config_file, "r", encoding="utf-8") as file:
            original_config = json.load(file)
        if not isinstance(original_config, dict):
            raise TypeError("the original config.json must contain a JSON object")
        original_config = normalize_dropout_arguments(
            normalize_width_arguments(original_config)
        )
        original_config = normalize_reproducibility_arguments(
            original_config,
            expected=args.deterministic,
        )
        original_config.setdefault("no_amp", True)

        compatibility_fields = (
            "architecture",
            "dataset",
            "task",
            "device",
            "no_amp",
            "expected_mode",
            "normalization_source",
            "normalization_sha256",
            "normalization_lineage",
            "coordinate_scale_source",
            "coordinate_scale_prior",
            "coordinate_transform",
            "query_sampling",
            "cache_raw_cases",
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
            "train_runs",
            "validation_runs",
            "test_runs",
            "train_query_limit",
            "validation_query_limit",
            "test_query_limit",
            "decode_chunk_size",
            "eval_every",
            "log_every",
            "checkpoint_every",
            "num_workers",
            "seed",
            "loss_protocol",
            "metric_protocol",
        )
        current_arguments = vars(args)
        incompatible = {
            name: (saved_arguments.get(name), current_arguments.get(name))
            for name in compatibility_fields
            if saved_arguments.get(name) != current_arguments.get(name)
        }
        incompatible.update(
            {
                f"config.{name}": (
                    original_config.get(name),
                    current_arguments.get(name),
                )
                for name in compatibility_fields
                if original_config.get(name) != current_arguments.get(name)
            }
        )
        for name in ("data", "normalization_source"):
            saved_value = saved_arguments.get(name)
            current_value = current_arguments.get(name)
            if saved_value is None or current_value is None:
                if saved_value != current_value:
                    incompatible[name] = (saved_value, current_value)
            elif Path(saved_value).resolve() != Path(current_value).resolve():
                incompatible[name] = (saved_value, current_value)
            config_value = original_config.get(name)
            if config_value is None or current_value is None:
                if config_value != current_value:
                    incompatible[f"config.{name}"] = (
                        config_value,
                        current_value,
                    )
            elif Path(config_value).resolve() != Path(current_value).resolve():
                incompatible[f"config.{name}"] = (config_value, current_value)
        for name in ("architecture", "dataset"):
            top_level = checkpoint.get(name)
            if top_level is not None and top_level != current_arguments[name]:
                incompatible[f"checkpoint.{name}"] = (
                    top_level,
                    current_arguments[name],
                )
        if incompatible:
            details = ", ".join(
                f"{name}: checkpoint={old!r}, current={new!r}"
                for name, (old, new) in incompatible.items()
            )
            raise ValueError(
                f"resume configuration is incompatible with checkpoint: {details}"
            )
        args.resume = str(resume_path)
        if "rng_state" not in checkpoint:
            resume_rng_note = (
                "legacy checkpoint has no RNG state; model, optimizer, and scheduler "
                "will resume, but sampling, shuffle, and dropout restart from the configured seed"
            )
            print(f"Warning: {resume_rng_note}", flush=True)
    else:
        output = make_output_directory(
            args.output_dir,
            f"ahmed-{args.task}",
            base_directory=Path(__file__).resolve().parent / "result",
        )
        with open(output / "config.json", "w", encoding="utf-8") as file:
            json.dump(vars(args), file, ensure_ascii=False, indent=2)

    input_channels = len(GLOBAL_MOMENT_CHANNELS)
    if input_channels != spec.moment_channels + spec.condition_channels:
        raise RuntimeError(
            "Ahmed global moment channels do not match the selected TaskSpec"
        )
    scaling_file = output / "scaling.npz"
    if resume_path is not None:
        if not scaling_file.is_file():
            raise FileNotFoundError(
                f"resuming requires the original scaling.npz: {scaling_file}"
            )
        scaling = ModelScaling.load(
            str(scaling_file),
            coordinate_scale=coordinate_transform.half_span,
        )
    else:
        print(
            "Loading canonical data derived from the raw training split.",
            flush=True,
        )
        scaling = ModelScaling.load(
            str(normalization_source),
            coordinate_scale=coordinate_transform.half_span,
        )
        scaling.save(str(scaling_file))

    datasets = {
        "train": AhmedMLDataset(
            data,
            args.task,
            args.train_runs,
            mode,
            args.train_query_limit,
            random_queries=True,
            coordinate_transform=coordinate_transform,
            cache_points=args.cache_raw_cases,
            query_seed=args.seed,
            use_condition_moments=True,
        ),
        "validation": AhmedMLDataset(
            data,
            args.task,
            args.validation_runs,
            mode,
            args.validation_query_limit,
            random_queries=True,
            coordinate_transform=coordinate_transform,
            cache_points=args.cache_raw_cases,
            query_seed=args.seed,
            use_condition_moments=True,
        ),
        "test": AhmedMLDataset(
            data,
            args.task,
            args.test_runs,
            mode,
            args.test_query_limit,
            random_queries=True,
            coordinate_transform=coordinate_transform,
            cache_points=args.cache_raw_cases,
            query_seed=args.seed,
            use_condition_moments=True,
        ),
    }
    loader_common = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
    }
    train_order_generator = torch.Generator()
    train_order_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_order_generator,
        **loader_common,
    )
    validation_loader = (
        DataLoader(datasets["validation"], batch_size=1, **loader_common)
        if args.validation_runs
        else None
    )
    test_loader = (
        DataLoader(datasets["test"], batch_size=1, **loader_common)
        if args.test_runs
        else None
    )

    model = GlobalLocalMFE(
        mode=mode,
        input_channels=input_channels,
        output_channels=4,
        dimension=3,
        function_channels=(spec.normal_channels + spec.condition_channels),
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
    ).to(device)
    optimizer, _ = build_adamw(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = scheduler_for(
        optimizer,
        total_steps=max(args.epochs * len(train_loader), 1),
        warmup_ratio=args.warmup_ratio,
        minimum_ratio=args.minimum_learning_rate / args.learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 1
    elapsed_offset = 0.0
    best_score = float("inf")
    best_file = output / "best.pt"
    history_path = output / "history.csv"
    memory_tracker = PhaseMemoryTracker(device)
    history_rows: list[dict[str, str]] = []
    history_fields = history_fieldnames("p_vector")

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])

        completed_epoch = int(checkpoint["epoch"])
        if completed_epoch < 0 or completed_epoch > args.epochs:
            raise ValueError(
                f"checkpoint epoch={completed_epoch} is outside [0,{args.epochs}]"
            )
        start_epoch = completed_epoch + 1
        checkpoint_best_score = float(checkpoint["score"])
        if math.isnan(checkpoint_best_score):
            raise ValueError("resume checkpoint best score cannot be NaN")

        if best_file.is_file():
            best_checkpoint = torch.load(
                best_file, map_location="cpu", weights_only=False
            )
            if not isinstance(best_checkpoint, dict):
                raise TypeError("best.pt must contain a dictionary")
            missing_best_fields = {
                "epoch",
                "score",
                "model_state",
                "optimizer_state",
                "scheduler_state",
                "arguments",
            }.difference(best_checkpoint)
            if missing_best_fields:
                raise KeyError(
                    f"best.pt is missing fields: {sorted(missing_best_fields)}"
                )
            best_epoch = int(best_checkpoint["epoch"])
            if best_epoch < 0 or best_epoch > args.epochs:
                raise ValueError(
                    f"best.pt epoch={best_epoch} is outside [0,{args.epochs}]"
                )
            best_arguments = best_checkpoint["arguments"]
            if not isinstance(best_arguments, dict):
                raise TypeError("best.pt arguments must be a dictionary")
            best_arguments = normalize_dropout_arguments(
                normalize_width_arguments(best_arguments)
            )
            best_arguments = normalize_reproducibility_arguments(
                best_arguments,
                expected=args.deterministic,
            )
            best_arguments.setdefault("no_amp", True)
            best_incompatible = {
                name: (best_arguments.get(name), current_arguments.get(name))
                for name in compatibility_fields
                if best_arguments.get(name) != current_arguments.get(name)
            }
            for name in ("data", "normalization_source"):
                best_value = best_arguments.get(name)
                current_value = current_arguments.get(name)
                if best_value is None or current_value is None:
                    if best_value != current_value:
                        best_incompatible[name] = (best_value, current_value)
                elif Path(best_value).resolve() != Path(current_value).resolve():
                    best_incompatible[name] = (best_value, current_value)
            for name in ("architecture", "dataset"):
                best_top_level = best_checkpoint.get(name)
                if (
                    best_top_level is not None
                    and best_top_level != current_arguments[name]
                ):
                    best_incompatible[f"checkpoint.{name}"] = (
                        best_top_level,
                        current_arguments[name],
                    )
            if best_incompatible:
                details = ", ".join(
                    f"{name}: best={old!r}, current={new!r}"
                    for name, (old, new) in best_incompatible.items()
                )
                raise ValueError(f"best.pt configuration is incompatible: {details}")

            if best_epoch > completed_epoch:
                model.load_state_dict(best_checkpoint["model_state"], strict=True)
                optimizer.load_state_dict(best_checkpoint["optimizer_state"])
                scheduler.load_state_dict(best_checkpoint["scheduler_state"])
                checkpoint = best_checkpoint
                completed_epoch = best_epoch
                start_epoch = completed_epoch + 1
                checkpoint_best_score = float(best_checkpoint["score"])
                resume_state_source = best_file
                print(
                    "Warning: best.pt is newer than last.pt; resuming from the newer "
                    f"complete state at epoch={best_epoch}",
                    flush=True,
                )
            best_score = float(best_checkpoint["score"])
            if not math.isfinite(best_score):
                raise ValueError(f"best.pt score must befinite,got {best_score}")
            if math.isfinite(checkpoint_best_score) and not math.isclose(
                best_score,
                checkpoint_best_score,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ):
                raise ValueError(
                    "best score in last.pt does not match best.pt: "
                    f"last={checkpoint_best_score}, best={best_score}"
                )
            if checkpoint is not best_checkpoint:
                del best_checkpoint
        elif math.isfinite(checkpoint_best_score):
            raise FileNotFoundError(
                "resume checkpoint records a finite best score, but best.pt is missing"
            )
        else:
            best_score = checkpoint_best_score

        resume_scaler_restored = False
        if use_amp:
            scaler_state = checkpoint.get("scaler_state")
            if scaler_state is None:
                resume_scaler_note = "AMP checkpoint has no scaler_state; continuing with a new GradScaler"
                warnings.warn(resume_scaler_note, RuntimeWarning, stacklevel=2)
            elif not isinstance(scaler_state, dict):
                raise TypeError("checkpoint scaler_state must be a dictionary")
            else:
                scaler.load_state_dict(scaler_state)
                resume_scaler_restored = True
                resume_scaler_note = "AMP GradScaler state restored"
        else:
            resume_scaler_note = "AMP disabled; no scaler state required"

        if history_path.is_file() and history_path.stat().st_size:
            with open(history_path, newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                expected_fields = list(history_fields)
                phase_memory_fields = {
                    f"{phase}_peak_{kind}_mib"
                    for phase in ("train", "validation", "test", "overall")
                    for kind in ("allocated", "reserved")
                }
                legacy_fields = [
                    field
                    for field in expected_fields
                    if field not in phase_memory_fields
                ]
                actual_fields = reader.fieldnames
                if actual_fields not in (expected_fields, legacy_fields):
                    raise ValueError(
                        "history.csv columns do not match the current protocol: "
                        f"expected={expected_fields}, actual={actual_fields}"
                    )
                history_rows = list(reader)
            legacy_history = actual_fields == legacy_fields
            if history_rows:
                history_epoch = int(history_rows[-1]["epoch"])
                if history_epoch > completed_epoch:
                    raise ValueError(
                        "history.csv is newer than the checkpoint: "
                        f"history_epoch={history_epoch}, "
                        f"checkpoint_epoch={completed_epoch}"
                    )
                if history_epoch < completed_epoch:
                    print(
                        "Warning: history.csv is behind the checkpoint; missing epochs will not be fabricated: "
                        f"history_epoch={history_epoch}, "
                        f"checkpoint_epoch={completed_epoch}",
                        flush=True,
                    )
                elapsed_offset = float(history_rows[-1].get("elapsed_seconds") or 0.0)

                if legacy_history:
                    allocated_values = [
                        float(row["peak_allocated_mib"])
                        for row in history_rows
                        if row.get("peak_allocated_mib")
                    ]
                    reserved_values = [
                        float(row["peak_reserved_mib"])
                        for row in history_rows
                        if row.get("peak_reserved_mib")
                    ]

                    memory_tracker.seed_phase(
                        "train",
                        peak_allocated_mib=(
                            max(allocated_values) if allocated_values else None
                        ),
                        peak_reserved_mib=(
                            max(reserved_values) if reserved_values else None
                        ),
                    )
                else:
                    for phase in ("train", "validation", "test"):
                        allocated_field = f"{phase}_peak_allocated_mib"
                        reserved_field = f"{phase}_peak_reserved_mib"
                        allocated_values = [
                            float(row[allocated_field])
                            for row in history_rows
                            if row.get(allocated_field)
                        ]
                        reserved_values = [
                            float(row[reserved_field])
                            for row in history_rows
                            if row.get(reserved_field)
                        ]
                        memory_tracker.seed_phase(
                            phase,
                            peak_allocated_mib=(
                                max(allocated_values) if allocated_values else None
                            ),
                            peak_reserved_mib=(
                                max(reserved_values) if reserved_values else None
                            ),
                        )

            if legacy_history:
                for row in history_rows:
                    allocated = row.get("peak_allocated_mib", "")
                    reserved = row.get("peak_reserved_mib", "")
                    row.update(
                        {
                            "train_peak_allocated_mib": allocated,
                            "train_peak_reserved_mib": reserved,
                            "validation_peak_allocated_mib": "",
                            "validation_peak_reserved_mib": "",
                            "test_peak_allocated_mib": "",
                            "test_peak_reserved_mib": "",
                            "overall_peak_allocated_mib": allocated,
                            "overall_peak_reserved_mib": reserved,
                        }
                    )
                temporary_history = history_path.with_suffix(".csv.tmp")
                with open(temporary_history, "w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=history_fields)
                    writer.writeheader()
                    writer.writerows(history_rows)
                temporary_history.replace(history_path)

        rng_restored = False
        if "rng_state" in checkpoint:
            rng_state = checkpoint["rng_state"]
            if not isinstance(rng_state, dict):
                raise TypeError("checkpoint rng_state must be a dictionary")
            restore_rng_state(rng_state)
            rng_restored = True
        elif resume_rng_note is None:
            resume_rng_note = (
                "selected resume checkpoint has no RNG state; sampling, shuffle, and "
                "dropout restart from the configured seed"
            )
            print(f"Warning: {resume_rng_note}", flush=True)
        resume_record = {
            **vars(args),
            "resume_checkpoint_epoch": completed_epoch,
            "resume_state_source": str(resume_state_source),
            "resume_next_epoch": start_epoch,
            "resume_best_score": best_score,
            "resume_rng_state_restored": rng_restored,
            "resume_rng_note": resume_rng_note,
            "resume_scaler_state_restored": resume_scaler_restored,
            "resume_scaler_note": resume_scaler_note,
            "resume_history_rows": len(history_rows),
            "resume_elapsed_offset_seconds": elapsed_offset,
        }
        with open(output / "resume_config.json", "w", encoding="utf-8") as file:
            json.dump(resume_record, file, ensure_ascii=False, indent=2)
        print(
            f"resume={resume_state_source} completed_epoch={completed_epoch} "
            f"next_epoch={start_epoch} target_epoch={args.epochs} "
            f"best_score={best_score:.6e} rng_restored={rng_restored} "
            f"scaler_restored={resume_scaler_restored}",
            flush=True,
        )

        for state_name in (
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "scaler_state",
            "rng_state",
        ):
            checkpoint.pop(state_name, None)

    parameter_counts = model.parameter_groups()
    model_summary_file = (
        output / "resume_model_summary.json"
        if checkpoint is not None
        else output / "model_summary.json"
    )
    with open(model_summary_file, "w", encoding="utf-8") as file:
        json.dump(
            {
                **parameter_counts,
                "task": args.task,
                "mode": mode,
                "moment_channels": spec.moment_channels,
                "condition_moment_channels": spec.condition_channels,
                "input_channels": input_channels,
                "normal_channels": spec.normal_channels,
                "condition_channels": spec.condition_channels,
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
                "dataset": args.dataset,
                "global_position_frequencies": args.global_position_frequencies,
                "architecture": args.architecture,
                "precision": args.precision,
            },
            file,
            indent=2,
        )
    print(
        f"task={args.task}, architecture={args.architecture}, device={device}, "
        f"precision={args.precision}, mode={mode}, "
        f"input_channels={input_channels}, attention=full, parameters={parameter_counts}",
        flush=True,
    )
    print(
        f"runs train/val/test={args.train_runs}/{args.validation_runs}/{args.test_runs}, "
        f"queries train/val/test={args.train_query_limit}/"
        f"{args.validation_query_limit}/{args.test_query_limit}, output={output}",
        flush=True,
    )

    measured_start = time.perf_counter()

    append_history = checkpoint is not None and history_path.is_file()
    history_mode = "a" if append_history else "w"
    with open(history_path, history_mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history_fields)
        if not append_history or history_path.stat().st_size == 0:
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs + 1):
            set_epoch_recursive(datasets["train"], epoch)

            memory_tracker.begin("train")
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                device,
                args.decode_chunk_size,
                use_amp,
                args.gradient_clip,
            )
            memory_tracker.end(
                "train",
                elapsed_seconds=(elapsed_offset + time.perf_counter() - measured_start),
            )

            validation_metrics = None
            if validation_loader is not None and (
                epoch == args.epochs or epoch % args.eval_every == 0
            ):
                memory_tracker.begin("validation")
                validation_metrics = evaluate(
                    model,
                    validation_loader,
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

                score = float(
                    validation_metrics["case_p_relative_l2"]
                    + validation_metrics["case_vector_relative_l2"]
                )
                if not math.isfinite(score):
                    raise ValueError("validation score is not finite")
                if score < best_score:
                    best_score = score
                    save_checkpoint(
                        best_file,
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        score,
                        args,
                        scaler=scaler,
                    )

            if epoch % args.log_every != 0 and epoch != args.epochs:
                continue
            memory_metrics = memory_tracker.snapshot(
                elapsed_seconds=(elapsed_offset + time.perf_counter() - measured_start)
            )
            row = build_history_row(
                epoch=epoch,
                optimizer=optimizer,
                train_metrics=train_metrics,
                memory=memory_metrics,
                elapsed_seconds=(elapsed_offset + time.perf_counter() - measured_start),
                metric_type="p_vector",
                validation_metrics=validation_metrics,
                scheduler=scheduler,
            )
            writer.writerow(row)
            file.flush()
            vector_label = "F" if args.task == "surface" else "U"
            train_relative_l2 = tuple(
                100.0 * train_metrics[field]
                for field in (
                    "p_relative_l2",
                    "vector_relative_l2",
                    "magnitude_relative_l2",
                )
            )
            message = (
                f"epoch {epoch:03d}/{args.epochs}, "
                f"train_loss={train_metrics['loss']:.6e}, "
                f"train_relative_MSE P/{vector_label}="
                f"{train_metrics['p_relative_mse']:.3e}/"
                f"{train_metrics['vector_relative_mse']:.3e}, "
                f"train_relative_L2 P/{vector_label}/|{vector_label}|="
                f"{train_relative_l2[0]:.2f}%/"
                f"{train_relative_l2[1]:.2f}%/"
                f"{train_relative_l2[2]:.2f}%, "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )
            if validation_metrics is not None:
                message += (
                    f", val_case_L2RE P/{vector_label}/|{vector_label}|="
                    f"{100*validation_metrics['case_p_relative_l2']:.3f}%/"
                    f"{100*validation_metrics['case_vector_relative_l2']:.3f}%/"
                    f"{100*validation_metrics['case_magnitude_relative_l2']:.3f}%"
                )
            message += f", {format_gpu_memory_metrics(device, memory_metrics)}"
            print(message, flush=True)

    if not best_file.exists():
        raise FileNotFoundError("Training produced no valid validation-selected best.pt")
    test_checkpoint = torch.load(best_file, map_location="cpu", weights_only=False)
    model.load_state_dict(test_checkpoint["model_state"], strict=True)
    selected_epoch = int(test_checkpoint["epoch"])
    best_score = float(test_checkpoint["score"])
    del test_checkpoint
    selection = {
        "split": "validation",
        "score_name": "case_p_relative_l2+case_vector_relative_l2",
        "score": best_score,
        "selected_epoch": selected_epoch,
        "checkpoint": str(best_file),
    }
    with open(output / "selection.json", "w", encoding="utf-8") as file:
        json.dump(selection, file, ensure_ascii=False, indent=2)
    if test_loader is not None and not args.skip_test:
        memory_tracker.begin("test")
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            args.decode_chunk_size,
            use_amp,
        )
        test_memory = memory_tracker.end(
            "test",
            elapsed_seconds=(elapsed_offset + time.perf_counter() - measured_start),
        )
        report = {
            "checkpoint": str(best_file),
            "selection": selection,
            "gradient_injection_architecture": args.gradient_injection_architecture,
            "selected_epoch": selected_epoch,
            "task": args.task,
            "dataset": args.dataset,
            "coordinate_scale_source": args.coordinate_scale_source,
            "coordinate_scale_prior": args.coordinate_scale_prior,
            "coordinate_transform": args.coordinate_transform,
            "normalize_inputs": args.normalize_inputs,
            "normalize_outputs": args.normalize_outputs,
            "precision": args.precision,
            "loss_protocol": args.loss_protocol,
            "metric_protocol": args.metric_protocol,
            "parameters": parameter_counts,
            "metrics": test_metrics,
            "gpu_memory": test_memory,
        }
        with open(output / "test_metrics.json", "w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
        print(
            "test: "
            f"selected_epoch={selected_epoch}, "
            f"cases={int(float(test_metrics['samples']))}, "
            f"case_L2RE P/{'F/|F|' if args.task == 'surface' else 'U/|U|'}="
            f"{100*test_metrics['case_p_relative_l2']:.3f}%/"
            f"{100*test_metrics['case_vector_relative_l2']:.3f}%/"
            f"{100*test_metrics['case_magnitude_relative_l2']:.3f}%, "
            f"test_elapsed={float(test_metrics['elapsed_seconds']):.2f}s, "
            f"total_elapsed={float(test_memory['elapsed_seconds']):.2f}s, "
            f"{format_gpu_memory_metrics(device, test_memory)}",
            flush=True,
        )

    memory = memory_tracker.snapshot(
        elapsed_seconds=elapsed_offset + time.perf_counter() - measured_start
    )
    with open(output / "gpu_memory.json", "w", encoding="utf-8") as file:
        json.dump(memory, file, ensure_ascii=False, indent=2)
    print(format_gpu_memory_metrics(device, memory), flush=True)


if __name__ == "__main__":
    main()
