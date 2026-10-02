"""Train and test MENO on the NASA CRM surface benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import warnings
from functools import partial
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
    from nasa.dataset import (
        GLOBAL_MOMENT_CHANNELS,
        GLOBAL_MOMENT_CHANNEL_INDEX,
        NASACRMSurfaceDataset,
        load_coordinate_transform,
        read_keys,
        validate_nasa_schema,
    )
    from nasa.native import (
        load_data_paths,
        normalization_file,
    )
    from nasa.metric import (
        FullCaseRelativeL2,
        RELATIVE_MSE_LOSS,
        summarize_full_case_metrics,
    )
else:
    from .dataset import (
        GLOBAL_MOMENT_CHANNELS,
        GLOBAL_MOMENT_CHANNEL_INDEX,
        NASACRMSurfaceDataset,
        load_coordinate_transform,
        read_keys,
        validate_nasa_schema,
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
WEIGHTED_LOSS_PROTOCOL = "physical_p_vector_weighted_relative_mse"
DEFAULT_F_LOSS_WEIGHT = 1.0
METRIC_PROTOCOL = "physical_full_case_then_sample_mean_p_vector_magnitude_relative_l2"
MODEL_ARCHITECTURE = "NormalizationAblationA"
DEFAULT_MODE = 8
COORDINATE_SCALE_SOURCE = "train_bounds"
COORDINATE_SCALE_PRIOR = None
DEFAULTS = TrainingDefaults(
    mode=DEFAULT_MODE,
    batch_size=1,
    minimum_learning_rate=2.0e-5,
    weight_decay=1.0e-3,
    global_width=512,
    local_width=512,
    heads=8,
    layers=6,
    feedforward=512,
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


def validate_f_loss_weight(value: object) -> float:
    """Validate a saved/configured scalar without coercing booleans or strings."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("f_loss_weight must be a finite number greater than zero")
    weight = float(value)
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("f_loss_weight must be a finite number greater than zero")
    return weight


def loss_protocol_for_weight(f_loss_weight: float) -> str:
    weight = validate_f_loss_weight(f_loss_weight)
    return LOSS_PROTOCOL if weight == 1.0 else WEIGHTED_LOSS_PROTOCOL


def normalize_loss_arguments(
    arguments: dict[str, object],
    *,
    expected_f_loss_weight: float | None = None,
) -> dict[str, object]:
    """Only legacy equal-weight checkpoints may omit f_loss_weight.

    The optional expected value enforces resume compatibility before any run
    artifact is written; changing a loss weight requires a separate run.
    """

    result = dict(arguments)
    protocol = result.get("loss_protocol")
    if "f_loss_weight" not in result:
        if protocol != LOSS_PROTOCOL:
            raise ValueError(
                "only legacy equal-weight checkpoints may omit f_loss_weight"
            )
        result["f_loss_weight"] = 1.0
    weight = validate_f_loss_weight(result["f_loss_weight"])
    expected_protocol = loss_protocol_for_weight(weight)
    if protocol != expected_protocol:
        raise ValueError(
            "loss_protocol and f_loss_weight are inconsistent: "
            f"protocol={protocol!r}, weight={weight}, expected={expected_protocol!r}"
        )
    if expected_f_loss_weight is not None:
        requested = validate_f_loss_weight(expected_f_loss_weight)
        if weight != requested:
            raise ValueError(
                "f_loss_weight cannot change when resuming: "
                f"checkpoint={weight}, current={requested}"
            )
    result["f_loss_weight"] = weight
    return result


def relative_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    f_loss_weight: float = DEFAULT_F_LOSS_WEIGHT,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weight the differentiable group MSEs, retaining all original metrics."""

    weight = validate_f_loss_weight(f_loss_weight)
    original_loss, metrics = RELATIVE_MSE_LOSS(prediction, target)
    if weight == 1.0:
        # Keep the original operation graph, rounding and gradients exactly.
        return original_loss, metrics
    loss = (metrics["p_relative_mse"] + weight * metrics["vector_relative_mse"]) / (
        1.0 + weight
    )
    return loss, metrics


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
    f_loss_weight: float = DEFAULT_F_LOSS_WEIGHT,
) -> dict[str, float]:
    """Train one epoch with the NASA grouped physical-space loss."""

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
        partial(relative_mse_loss, f_loss_weight=validate_f_loss_weight(f_loss_weight)),
    )


def evaluate(
    model: GlobalLocalMFE,
    loader: DataLoader,
    device: torch.device,
    chunk_size: int,
    use_amp: bool,
    f_loss_weight: float = DEFAULT_F_LOSS_WEIGHT,
) -> dict[str, float | str]:
    """Evaluate full NASA cases with chunked local decoding."""

    result = evaluate_common(
        model,
        loader,
        device,
        chunk_size,
        use_amp,
        loss_protocol_for_weight(f_loss_weight),
        METRIC_PROTOCOL,
        FullCaseRelativeL2,
        summarize_full_case_metrics,
    )
    result["f_loss_weight"] = validate_f_loss_weight(f_loss_weight)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the NASA benchmark training and runtime options."""

    case_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--resume", default=None, help="Not supported for paper-aligned best-only runs"
    )
    parser.add_argument(
        "--data-root", dest="data", default=str(case_directory / "data")
    )
    parser.add_argument(
        "--f-loss-weight",
        type=float,
        default=DEFAULT_F_LOSS_WEIGHT,
        help="Physical Relative-MSE P:F weighting is 1:w; w must be finite and positive",
    )
    add_training_arguments(parser, DEFAULTS)
    # Dataset-local defaults: do not change Darcy/Poisson precision.
    parser.set_defaults(no_amp=True)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--cache-raw-cases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache per-case NPY arrays in memory; disable to mmap them directly",
    )
    args = parser.parse_args(argv)
    if args.resume is not None:
        parser.error("Paper-aligned best-only runs must start fresh; --resume is unsupported.")
    if args.eval_every != 1 or args.log_every != 1 or args.checkpoint_every != 0:
        parser.error("Paper-aligned NASA requires eval/log every epoch and checkpoint-every=0.")
    if args.skip_test:
        parser.error("NASA test-oracle selection requires the test set; --skip-test is unsupported.")
    if not args.normalize_inputs or not args.normalize_outputs:
        parser.error("NASA variant A requires Local input and output normalization.")
    try:
        args.f_loss_weight = validate_f_loss_weight(args.f_loss_weight)
    except ValueError as error:
        parser.error(str(error))
    apply_fixed_protocol(args, DEFAULTS, spectral_gradient_projection="ambient")
    args.coordinate_scale_source = COORDINATE_SCALE_SOURCE
    args.coordinate_scale_prior = (
        None if COORDINATE_SCALE_PRIOR is None else list(COORDINATE_SCALE_PRIOR)
    )
    args.query_sampling = "dynamic_uniform_native"
    args.selection_split = "test_oracle"
    args.shuffle_protocol = "independent_torch_generator_seed42_or_cli_seed"
    args.normalization_ablation = "A"
    args.normalize_global_moments = False
    args.normalize_local_functions = True
    args.legendre_basis = "orthonormal"
    args.learned_fusion = False
    return args


def main(argv: list[str] | None = None) -> None:
    """Train NASA variant A with per-epoch test-oracle checkpoint selection."""

    args = parse_args(argv)
    args.architecture = MODEL_ARCHITECTURE
    args.dataset = "nasa"

    args.loss_protocol = loss_protocol_for_weight(args.f_loss_weight)
    args.metric_protocol = METRIC_PROTOCOL
    validate_common_training_arguments(args)

    data = Path(args.data).resolve()
    data_paths = load_data_paths(data)
    coordinate_transform = load_coordinate_transform(
        data,
        source=COORDINATE_SCALE_SOURCE,
        prior_half_span=COORDINATE_SCALE_PRIOR,
    )
    schema = validate_nasa_schema(
        data, expected_mode=args.expected_mode, verify_payload_hashes=True
    )
    normalization_source = normalization_file(data_paths)
    normalization_details = data_paths.manifest["normalization"]
    args.normalization_source = str(normalization_source)
    args.normalization_sha256 = normalization_details["sha256"]
    args.normalization_lineage = normalization_details["lineage"]
    args.coordinate_transform = coordinate_transform.metadata()
    mode = schema["mode"]
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    precision = "amp-fp16" if use_amp else "float32"
    args.precision = precision
    if not use_amp:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    print(
        f"device={device}, precision={precision}, "
        f"loss_protocol={args.loss_protocol}, P:F_loss_weight=1:{args.f_loss_weight:g}",
        flush=True,
    )
    args.reproducibility = set_seed(args.seed, deterministic=args.deterministic)
    args.reproducibility.update(
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
    )

    training_keys = read_keys(data, "train")
    test_keys = read_keys(data, "test")
    args.data = str(data)
    args.train_cases = len(training_keys)
    args.test_cases = len(test_keys)
    args.train_point_pool_min = schema["train_point_pool_min"]
    args.train_point_pool_max = schema["train_point_pool_max"]
    args.dynamic_full_surface_queries = True
    args.raw_case_cache = {
        "train": (
            "lazy_process_local_full_case"
            if args.cache_raw_cases
            else "disabled_direct_per_case_npy"
        ),
        "test": "disabled_full_case_per_epoch_evaluation",
    }

    resume_path = Path(args.resume).resolve() if args.resume else None
    checkpoint: dict[str, object] | None = None
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
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
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
        saved_arguments = normalize_loss_arguments(
            normalize_dropout_arguments(normalize_width_arguments(saved_arguments)),
            expected_f_loss_weight=args.f_loss_weight,
        )
        saved_arguments = normalize_reproducibility_arguments(
            saved_arguments,
            expected=args.deterministic,
        )

        saved_arguments.setdefault("no_amp", True)
        compatibility_fields = (
            "architecture",
            "dataset",
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
            "seed",
            "loss_protocol",
            "f_loss_weight",
            "metric_protocol",
            "train_cases",
            "test_cases",
        )
        current_arguments = vars(args)
        incompatible = {
            name: (saved_arguments.get(name), current_arguments.get(name))
            for name in compatibility_fields
            if saved_arguments.get(name) != current_arguments.get(name)
        }
        for name in ("data", "normalization_source"):
            saved_value = saved_arguments.get(name)
            current_value = current_arguments.get(name)
            if saved_value is None or current_value is None:
                if saved_value != current_value:
                    incompatible[name] = (saved_value, current_value)
            elif Path(saved_value).resolve() != Path(current_value).resolve():
                incompatible[name] = (saved_value, current_value)
        if incompatible:
            details = ", ".join(
                f"{name}: checkpoint={old!r}, current={new!r}"
                for name, (old, new) in incompatible.items()
            )
            raise ValueError(
                f"resume configuration is incompatible with checkpoint: {details}"
            )
        args.resume = str(resume_path)
        rng_state_available = isinstance(checkpoint.get("rng_state"), dict)
        resume_record = {
            **vars(args),
            "resume_checkpoint_epoch": int(checkpoint["epoch"]),
            "resume_rng_state_restored": rng_state_available,
            "resume_rng_note": (
                "Python, NumPy, and Torch RNG states will be restored after model loading"
                if rng_state_available
                else "legacy checkpoint has no RNG state; sampling and shuffle restart from the configured seed"
            ),
        }
        with open(output / "resume_config.json", "w", encoding="utf-8") as file:
            json.dump(resume_record, file, ensure_ascii=False, indent=2)
    else:
        output = make_output_directory(
            args.output_dir,
            "nasa",
            base_directory=Path(__file__).resolve().parent / "result",
        )
        with open(output / "config.json", "w", encoding="utf-8") as file:
            json.dump(vars(args), file, ensure_ascii=False, indent=2)

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
        scaling = ModelScaling.load(
            str(normalization_source),
            coordinate_scale=coordinate_transform.half_span,
        )
    if resume_path is None:
        scaling.save(str(scaling_file))

    datasets = {
        "train": NASACRMSurfaceDataset(
            data,
            "train",
            training_keys,
            args.train_query_limit,
            random_queries=True,
            coordinate_transform=coordinate_transform,
            cache_points=args.cache_raw_cases,
            use_condition_moments=True,
            query_seed=args.seed,
        ),
        "test": NASACRMSurfaceDataset(
            data,
            "test",
            test_keys,
            None,
            random_queries=False,
            coordinate_transform=coordinate_transform,
            cache_points=False,
            use_condition_moments=True,
            query_seed=args.seed,
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
    test_loader = DataLoader(datasets["test"], batch_size=1, **loader_common)

    model = GlobalLocalMFE(
        mode=mode,
        input_channels=len(GLOBAL_MOMENT_CHANNELS),
        output_channels=4,
        dimension=3,
        function_channels=9,
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
        normalize_global_moments=args.normalize_global_moments,
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
    memory_tracker = PhaseMemoryTracker(device)
    start_epoch = 1
    best_score = float("inf")
    best_epoch = 0
    best_file = output / "best.pt"
    elapsed_offset = 0.0
    history_path = output / "history.csv"
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler_state = checkpoint.get("scaler_state")
        if use_amp:
            if scaler_state is None:
                warnings.warn(
                    "AMP checkpoint has no scaler_state; continuing with a new GradScaler",
                    RuntimeWarning,
                    stacklevel=2,
                )
            elif not isinstance(scaler_state, dict):
                raise TypeError("checkpoint scaler_state must be a dictionary")
            else:
                scaler.load_state_dict(scaler_state)
        if isinstance(checkpoint.get("rng_state"), dict):
            restore_rng_state(checkpoint["rng_state"])
        completed_epoch = int(checkpoint["epoch"])
        if completed_epoch < 0 or completed_epoch > args.epochs:
            raise ValueError(
                f"checkpoint epoch={completed_epoch} is outside [0,{args.epochs}]"
            )
        start_epoch = completed_epoch + 1
        best_score = float(checkpoint["score"])
        best_epoch = completed_epoch
        if history_path.is_file() and history_path.stat().st_size:
            with open(history_path, newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                expected_fields = list(history_fieldnames("p_vector"))
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
                        f"history_epoch={history_epoch}, checkpoint_epoch={completed_epoch}"
                    )
                elapsed_offset = float(history_rows[-1]["elapsed_seconds"] or 0.0)
                allocated_field = (
                    "peak_allocated_mib"
                    if legacy_history
                    else "train_peak_allocated_mib"
                )
                reserved_field = (
                    "peak_reserved_mib" if legacy_history else "train_peak_reserved_mib"
                )
                allocated_peaks = [
                    float(row[allocated_field])
                    for row in history_rows
                    if row.get(allocated_field)
                ]
                reserved_peaks = [
                    float(row[reserved_field])
                    for row in history_rows
                    if row.get(reserved_field)
                ]
                memory_tracker.seed_phase(
                    "train",
                    peak_allocated_mib=(
                        max(allocated_peaks) if allocated_peaks else None
                    ),
                    peak_reserved_mib=(max(reserved_peaks) if reserved_peaks else None),
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
                    writer = csv.DictWriter(file, fieldnames=expected_fields)
                    writer.writeheader()
                    writer.writerows(history_rows)
                temporary_history.replace(history_path)
        print(
            f"resume={resume_path} completed_epoch={completed_epoch} "
            f"next_epoch={start_epoch} target_epoch={args.epochs} "
            + (
                "rng_state=restored"
                if isinstance(checkpoint.get("rng_state"), dict)
                else "rng_state=unavailable(seed stream restarted)"
            ),
            flush=True,
        )
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
                "architecture": args.architecture,
                "dataset": args.dataset,
                "loss_protocol": args.loss_protocol,
                "f_loss_weight": args.f_loss_weight,
                "input_channels": len(GLOBAL_MOMENT_CHANNELS),
                "global_moment_channel_index": GLOBAL_MOMENT_CHANNEL_INDEX,
                "output_channels": 4,
                "function_channels": 9,
                "spectral_gradient_injection": args.spectral_gradient_injection,
                "spectral_gradient_projection": args.spectral_gradient_projection,
                "gradient_injection_architecture": args.gradient_injection_architecture,
                "normalize_inputs": args.normalize_inputs,
                "normalize_outputs": args.normalize_outputs,
                "normalization_ablation": args.normalization_ablation,
                "normalize_global_moments": args.normalize_global_moments,
                "normalize_local_functions": args.normalize_local_functions,
                "coordinate_scale_source": args.coordinate_scale_source,
                "coordinate_scale_prior": args.coordinate_scale_prior,
                "coordinate_transform": args.coordinate_transform,
                "coordinate_scale": np.asarray(scaling.coordinate_scale).tolist(),
                "precision": precision,
            },
            file,
            indent=2,
        )
    measured_start = time.perf_counter()

    history_fields = history_fieldnames("p_vector")
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
                args.f_loss_weight,
            )
            memory_tracker.end(
                "train",
                elapsed_seconds=(elapsed_offset + time.perf_counter() - measured_start),
            )
            train_score = float(train_metrics["loss"])
            if not math.isfinite(train_score):
                raise ValueError(
                    "training loss is not finite; refusing to save a checkpoint"
                )

            memory_tracker.begin("test")
            oracle_metrics = evaluate(
                model,
                test_loader,
                device,
                args.decode_chunk_size,
                use_amp,
                args.f_loss_weight,
            )
            memory_tracker.end(
                "test",
                elapsed_seconds=(elapsed_offset + time.perf_counter() - measured_start),
            )
            score = float(
                oracle_metrics["case_p_relative_l2"]
                + oracle_metrics["case_vector_relative_l2"]
            )
            if not math.isfinite(score):
                raise ValueError("NASA test-oracle selection score is not finite")
            if score < best_score:
                best_score = score
                best_epoch = epoch
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

            run_elapsed = time.perf_counter() - measured_start
            memory_metrics = memory_tracker.snapshot(
                elapsed_seconds=elapsed_offset + run_elapsed
            )
            row = build_history_row(
                epoch=epoch,
                optimizer=optimizer,
                train_metrics=train_metrics,
                memory=memory_metrics,
                elapsed_seconds=elapsed_offset + run_elapsed,
                metric_type="p_vector",
                # The shared CSV uses val_* names; selection_split identifies
                # these columns as test-oracle metrics, not validation metrics.
                validation_metrics=oracle_metrics,
                scheduler=scheduler,
            )
            writer.writerow(row)
            file.flush()

            train_relative_l2 = tuple(
                100.0 * train_metrics[field]
                for field in (
                    "p_relative_l2",
                    "vector_relative_l2",
                    "magnitude_relative_l2",
                )
            )
            print(
                f"epoch={epoch:03d}/{args.epochs} "
                f"train_loss={train_metrics['loss']:.6e} "
                "train_relative_mse P/F="
                f"{train_metrics['p_relative_mse']:.3e}/"
                f"{train_metrics['vector_relative_mse']:.3e} "
                "train_relative_l2 P/F/|F|="
                f"{train_relative_l2[0]:.2f}%/"
                f"{train_relative_l2[1]:.2f}%/"
                f"{train_relative_l2[2]:.2f}% "
                f"test_oracle_score={score:.8f} best_epoch={best_epoch} "
                f"{format_gpu_memory_metrics(device, memory_metrics)}",
                flush=True,
            )

    if not best_file.is_file():
        raise FileNotFoundError("Training produced no valid test-selected best.pt")
    best_checkpoint = torch.load(best_file, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"], strict=True)
    best_epoch = int(best_checkpoint["epoch"])
    best_score = float(best_checkpoint["score"])
    del best_checkpoint
    selection = {
        "split": "test_oracle",
        "score_name": "case_p_relative_l2+case_vector_relative_l2",
        "score": best_score,
        "selected_epoch": best_epoch,
        "checkpoint": str(best_file),
    }
    with open(output / "selection.json", "w", encoding="utf-8") as file:
        json.dump(selection, file, ensure_ascii=False, indent=2)

    if not args.skip_test:
        memory_tracker.begin("test")
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            args.decode_chunk_size,
            use_amp,
            args.f_loss_weight,
        )
        test_memory = memory_tracker.end(
            "test",
            elapsed_seconds=(elapsed_offset + time.perf_counter() - measured_start),
        )
        report = {
            "checkpoint": str(best_file),
            "selected_epoch": best_epoch,
            "selection": selection,
            "architecture": args.architecture,
            "dataset": args.dataset,
            "gradient_injection_architecture": args.gradient_injection_architecture,
            "loss_protocol": args.loss_protocol,
            "f_loss_weight": args.f_loss_weight,
            "metric_protocol": args.metric_protocol,
            "normalize_inputs": args.normalize_inputs,
            "normalize_outputs": args.normalize_outputs,
            "normalization_ablation": args.normalization_ablation,
            "normalize_global_moments": args.normalize_global_moments,
            "normalize_local_functions": args.normalize_local_functions,
            "precision": precision,
            "parameters": parameter_counts,
            "metrics": test_metrics,
            "gpu_memory": test_memory,
        }
        with open(output / "test_metrics.json", "w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
        print(
            "test: "
            f"selected_epoch={best_epoch}, "
            f"cases={int(float(test_metrics['samples']))}, "
            "case_L2RE P/F/|F|="
            f"{100.0 * test_metrics['case_p_relative_l2']:.3f}%/"
            f"{100.0 * test_metrics['case_vector_relative_l2']:.3f}%/"
            f"{100.0 * test_metrics['case_magnitude_relative_l2']:.3f}%, "
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
