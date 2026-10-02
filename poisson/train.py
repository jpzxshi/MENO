"""Train the canonical affine-coordinate Poisson raw-native benchmark.

Global moments are derived exactly from the complete historical FEM raw data;
Local queries are sampled directly from that same raw source.  Statistical
normalization is uniformly weighted per case over every raw training node.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path

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
    from poisson.dataset import (
        GLOBAL_MOMENT_CHANNELS,
        GLOBAL_MOMENT_CHANNEL_INDEX,
        NORMALIZATION_PROTOCOL,
        load_coordinate_transform,
        load_precomputed_scaling,
        local_function_channels,
        verify_raw_payload_hashes,
    )
    from poisson.metric import (
        FullCaseRelativeL2,
        RELATIVE_MSE_LOSS,
        summarize_full_case_metrics,
    )
    from poisson.native import (
        NATIVE_QUERY_SAMPLING_PROTOCOL,
        NativePoissonQueryDataset,
    )
else:
    from .dataset import (
        GLOBAL_MOMENT_CHANNELS,
        GLOBAL_MOMENT_CHANNEL_INDEX,
        NORMALIZATION_PROTOCOL,
        load_coordinate_transform,
        load_precomputed_scaling,
        local_function_channels,
        verify_raw_payload_hashes,
    )
    from .metric import (
        FullCaseRelativeL2,
        RELATIVE_MSE_LOSS,
        summarize_full_case_metrics,
    )
    from .native import NATIVE_QUERY_SAMPLING_PROTOCOL, NativePoissonQueryDataset

LOSS_PROTOCOL = "physical_scalar_relative_mse"
METRIC_PROTOCOL = "physical_full_case_then_sample_mean_relative_l2"
MODEL_ARCHITECTURE = "GlobalLocalMFE"
DEFAULT_MODE = 12
COORDINATE_SCALE_SOURCE = "prior"
COORDINATE_SCALE_PRIOR: tuple[float, float] | None = (0.5, 0.5)
DEFAULTS = TrainingDefaults(
    mode=DEFAULT_MODE,
    batch_size=10,
    minimum_learning_rate=1.0e-5,
    weight_decay=1.0e-2,
    global_width=128,
    heads=8,
    layers=4,
    feedforward=256,
    global_dropout=0.0,
    local_dropout=0.0,
    global_position_frequencies=4,
    local_fourier_frequencies=6,
    train_query_limit=10000,
    decode_chunk_size=16384,
    log_every=1,
    checkpoint_every=1,
    spectral_gradient_injection=True,
)

RESUME_COMPATIBILITY_FIELDS = (
    "architecture",
    "dataset",
    "device",
    "no_amp",
    "data_format",
    "raw_data_format",
    "moment_data_format",
    "raw_manifest_sha256",
    "moment_manifest_sha256",
    "normalization_manifest",
    "normalization_manifest_sha256",
    "expected_mode",
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
    "log_every",
    "checkpoint_every",
    "num_workers",
    "seed",
    "skip_test",
    "loss_protocol",
    "metric_protocol",
    "local_function_channels",
    "train_cases",
    "validation_cases",
    "test_cases",
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
    """Train one epoch with the scalar Poisson Relative-MSE loss."""

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
    """Accumulate errors over complete PDE cases and average across cases."""

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
    """Resolve the sole canonical affine-coordinate benchmark."""

    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--resume", default=None, help="Resume from last.pt in the same run directory"
    )
    add_training_arguments(parser, DEFAULTS)
    parser.set_defaults(no_amp=True)
    parser.add_argument(
        "--cache-raw-cases",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cache per-case NPY arrays in memory; default behavior uses mmap directly",
    )
    args = parser.parse_args(argv)
    apply_fixed_protocol(args, DEFAULTS, spectral_gradient_projection="ambient")
    args.data = str(directory / "data")
    args.coordinate_scale_source = COORDINATE_SCALE_SOURCE
    args.coordinate_scale_prior = (
        None if COORDINATE_SCALE_PRIOR is None else list(COORDINATE_SCALE_PRIOR)
    )
    args.query_sampling_protocol = NATIVE_QUERY_SAMPLING_PROTOCOL
    args.raw_access_mode = "npy"
    return args


def validate_resume_arguments(
    saved: Mapping[str, object], current: Mapping[str, object]
) -> None:
    """Reject checkpoints from a different model, split, or training protocol."""

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

    if "data" not in saved:
        incompatible["data"] = ("<missing>", current.get("data"))
    else:
        saved_data = Path(str(saved["data"])).resolve()
        current_data = Path(str(current["data"])).resolve()
        if saved_data != current_data:
            incompatible["data"] = (str(saved_data), str(current_data))

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
        if math.isfinite(value):
            values.append(value)
    return max(values) if values else None


def prepare_history_for_resume(
    history_path: Path,
    completed_epoch: int,
    memory_tracker: PhaseMemoryTracker,
) -> float:
    """Validate resume history and restore elapsed-time and memory evidence."""

    if not history_path.is_file() or history_path.stat().st_size == 0:
        warnings.warn(
            "resume checkpoint exists but history.csv is missing or empty; prior "
            "epoch logs, elapsed time, and memory peaks cannot be restored",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.0

    expected_fields = list(history_fieldnames("scalar"))
    phase_memory_fields = {
        f"{phase}_peak_{kind}_mib"
        for phase in ("train", "validation", "test", "overall")
        for kind in ("allocated", "reserved")
    }
    legacy_fields = [
        field for field in expected_fields if field not in phase_memory_fields
    ]
    with open(history_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        actual_fields = reader.fieldnames
        if actual_fields not in (expected_fields, legacy_fields):
            raise ValueError(
                "history.csv columns do not match the current protocol: "
                f"expected={expected_fields}, actual={actual_fields}"
            )
        rows = list(reader)

    epochs = [int(row["epoch"]) for row in rows]
    if epochs != sorted(set(epochs)):
        raise ValueError("history.csv epochs must be strictly increasing and unique")
    if epochs and epochs[-1] > completed_epoch:
        raise ValueError(
            "history.csv is newer than the checkpoint: "
            f"history_epoch={epochs[-1]}, checkpoint_epoch={completed_epoch}"
        )
    if not epochs or epochs[-1] < completed_epoch:
        warnings.warn(
            "history.csv is behind the checkpoint; missing epochs will not be fabricated: "
            f"history_epoch={epochs[-1] if epochs else None}, "
            f"checkpoint_epoch={completed_epoch}",
            RuntimeWarning,
            stacklevel=2,
        )

    legacy_history = actual_fields == legacy_fields
    if legacy_history:

        memory_tracker.seed_phase(
            "train",
            peak_allocated_mib=_finite_history_peak(rows, "peak_allocated_mib"),
            peak_reserved_mib=_finite_history_peak(rows, "peak_reserved_mib"),
        )
        for row in rows:
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
        with open(temporary, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=expected_fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(history_path)
    else:

        for phase in ("train", "test"):
            memory_tracker.seed_phase(
                phase,
                peak_allocated_mib=_finite_history_peak(
                    rows, f"{phase}_peak_allocated_mib"
                ),
                peak_reserved_mib=_finite_history_peak(
                    rows, f"{phase}_peak_reserved_mib"
                ),
            )

    return float(rows[-1].get("elapsed_seconds") or 0.0) if rows else 0.0


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.architecture = MODEL_ARCHITECTURE
    args.dataset = "poisson"
    args.loss_protocol = LOSS_PROTOCOL
    args.metric_protocol = METRIC_PROTOCOL
    validate_common_training_arguments(args)
    data = Path(args.data).resolve()
    _, manifest = verify_raw_payload_hashes(data)
    coordinate_transform = load_coordinate_transform(
        data,
        source=COORDINATE_SCALE_SOURCE,
        prior_half_span=COORDINATE_SCALE_PRIOR,
    )
    print("Poisson full-raw SHA256 preflight passed", flush=True)
    args.reproducibility = set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    precision = "amp-fp16" if use_amp else "float32"
    print(f"device={device}, precision={precision}", flush=True)
    mode = int(manifest["mode"])
    local_channels = local_function_channels(manifest)
    function_channels = len(local_channels)
    if mode != args.expected_mode:
        raise ValueError(f"data mode={mode}, expected-mode={args.expected_mode}")
    if int(manifest["dimension"]) != 2:
        raise ValueError("Poisson data dimension must be 2")
    train_count = int(manifest["splits"]["train"]["samples"])
    test_count = int(manifest["splits"]["test"]["samples"])
    training_indices = list(range(train_count))
    test_indices = list(range(test_count))
    if not training_indices:
        raise ValueError("the Poisson training split is empty")

    args.data = str(data)
    args.data_format = str(manifest["format"])
    args.raw_data_format = str(manifest["raw"]["format"])
    args.moment_data_format = str(manifest["moment"]["format"])
    args.raw_manifest_sha256 = str(manifest["raw_manifest_sha256"])
    args.moment_manifest_sha256 = str(manifest["moment_manifest_sha256"])
    args.normalization_manifest = str(manifest["normalization_manifest"])
    args.normalization_manifest_sha256 = str(manifest["normalization_manifest_sha256"])
    args.local_function_channels = list(local_channels)
    args.train_cases = len(training_indices)
    args.validation_cases = 0
    args.test_cases = len(test_indices)
    args.coordinate_transform = coordinate_transform.metadata()
    args.raw_case_cache = {
        "train": (
            "lazy_process_local_full_case"
            if args.cache_raw_cases
            else "disabled_direct_per_case_npy"
        ),
        "test": "disabled_one_shot_evaluation",
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
        loaded = torch.load(resume_path, map_location="cpu", weights_only=False)
        if not isinstance(loaded, dict):
            raise TypeError("resume checkpoint must contain a dictionary")
        checkpoint = loaded
        required_fields = {
            "epoch",
            "score",
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "arguments",
        }
        missing_fields = required_fields.difference(checkpoint)
        if missing_fields:
            raise KeyError(
                f"resume checkpoint is missing fields: {sorted(missing_fields)}"
            )
        saved_arguments = checkpoint["arguments"]
        if not isinstance(saved_arguments, Mapping):
            raise TypeError("resume checkpoint arguments must be a mapping")
        validate_resume_arguments(saved_arguments, vars(args))
        args.resume = str(resume_path)
    else:
        output = make_output_directory(
            args.output_dir,
            "poisson",
            "sg-raw-native",
            base_directory=Path(__file__).resolve().parent / "result",
        )
        with open(output / "config.json", "w", encoding="utf-8") as file:
            json.dump(vars(args), file, ensure_ascii=False, indent=2, default=str)

    scaling_file = output / "scaling.npz"
    if resume_path is not None:
        if not scaling_file.is_file():
            raise FileNotFoundError(
                f"resuming requires the original scaling.npz: {scaling_file}"
            )
        scaling = ModelScaling.load(
            str(scaling_file),
            coordinate_scale=np.asarray(coordinate_transform.half_span, np.float32),
        )
    else:
        print("Loading canonical data derived from the raw training split.", flush=True)
        scaling = load_precomputed_scaling(
            data, coordinate_transform=coordinate_transform
        )
        scaling.save(str(scaling_file))

    datasets = {
        "train": NativePoissonQueryDataset(
            data,
            "train",
            training_indices,
            args.train_query_limit,
            random_queries=True,
            query_seed=args.seed,
            coordinate_transform=coordinate_transform,
            cache_cases=args.cache_raw_cases,
            raw_access_mode=args.raw_access_mode,
        ),
    }
    loader_common = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
    }
    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        **loader_common,
    )
    model = GlobalLocalMFE(
        mode=mode,
        input_channels=len(GLOBAL_MOMENT_CHANNELS),
        output_channels=1,
        dimension=2,
        function_channels=function_channels,
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
        spectral_gradient_projection="ambient",
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
    memory_tracker = PhaseMemoryTracker(device)
    start_epoch = 1
    latest_train_loss = float("nan")
    elapsed_offset = 0.0
    history_path = output / "history.csv"

    resume_rng_restored = False
    resume_rng_note = "fresh run"
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
            elif not isinstance(scaler_state, Mapping):
                raise TypeError("checkpoint scaler_state must be a mapping")
            else:
                scaler.load_state_dict(dict(scaler_state))

        completed_epoch = int(checkpoint["epoch"])
        if completed_epoch < 0 or completed_epoch > args.epochs:
            raise ValueError(
                f"checkpoint epoch={completed_epoch} is outside [0,{args.epochs}]"
            )
        start_epoch = completed_epoch + 1
        latest_train_loss = float(checkpoint["score"])
        elapsed_offset = prepare_history_for_resume(
            history_path, completed_epoch, memory_tracker
        )

        rng_state = checkpoint.get("rng_state")
        if rng_state is None:
            resume_rng_note = (
                "legacy checkpoint has no rng_state; Python/NumPy/Torch RNG "
                "streams restarted from configured seed"
            )
            warnings.warn(resume_rng_note, RuntimeWarning, stacklevel=2)
        elif not isinstance(rng_state, Mapping):
            raise TypeError("checkpoint rng_state must be a mapping")
        else:

            restore_rng_state(rng_state)
            resume_rng_restored = True
            resume_rng_note = "Python/NumPy/Torch RNG states restored"

        with open(output / "resume_config.json", "w", encoding="utf-8") as file:
            json.dump(
                {
                    **vars(args),
                    "resume_checkpoint_epoch": completed_epoch,
                    "resume_start_epoch": start_epoch,
                    "resume_rng_state_restored": resume_rng_restored,
                    "resume_rng_note": resume_rng_note,
                },
                file,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        print(
            f"resume={resume_path} completed_epoch={completed_epoch} "
            f"next_epoch={start_epoch} target_epoch={args.epochs} "
            f"rng_state={'restored' if resume_rng_restored else 'legacy-missing'}",
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
                "mode": mode,
                "tokens": mode**2,
                "dimension": 2,
                "input_channels": len(GLOBAL_MOMENT_CHANNELS),
                "global_moment_channel_index": GLOBAL_MOMENT_CHANNEL_INDEX,
                "function_channels": function_channels,
                "local_function_channels": list(local_channels),
                "data_format": manifest["format"],
                "raw_data_format": manifest["raw"]["format"],
                "moment_data_format": manifest["moment"]["format"],
                "raw_manifest_sha256": manifest["raw_manifest_sha256"],
                "moment_manifest_sha256": manifest["moment_manifest_sha256"],
                "normalization_manifest": args.normalization_manifest,
                "normalization_manifest_sha256": args.normalization_manifest_sha256,
                "output_channels": 1,
                "dataset": args.dataset,
                "architecture": args.architecture,
                "basis": "orthonormal_tensor_legendre_2d",
                "basis_domain": [-1.0, 1.0],
                "spectral_gradient_injection": args.spectral_gradient_injection,
                "spectral_gradient_projection": args.spectral_gradient_projection,
                "gradient_injection_architecture": args.gradient_injection_architecture,
                "normalize_inputs": args.normalize_inputs,
                "normalize_outputs": args.normalize_outputs,
                "coordinate_scale": np.asarray(scaling.coordinate_scale).tolist(),
                "coordinate_scale_source": args.coordinate_scale_source,
                "coordinate_scale_prior": args.coordinate_scale_prior,
                "coordinate_transform": args.coordinate_transform,
                "normalization": NORMALIZATION_PROTOCOL,
                "query_sampling_protocol": args.query_sampling_protocol,
                "precision": precision,
            },
            file,
            indent=2,
        )
    measured_start = time.perf_counter()

    fields = history_fieldnames("scalar")
    append_history = checkpoint is not None and history_path.is_file()
    history_mode = "a" if append_history else "w"
    with open(history_path, history_mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
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
            latest_train_loss = float(train_metrics["loss"])
            if not math.isfinite(latest_train_loss):
                raise ValueError(
                    "training loss is not finite; refusing to save a checkpoint"
                )
            if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
                save_checkpoint(
                    output / "last.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    latest_train_loss,
                    args,
                    scaler=scaler,
                )
            if epoch % args.log_every != 0 and epoch != args.epochs:
                continue
            memory = memory_tracker.snapshot(
                elapsed_seconds=(elapsed_offset + time.perf_counter() - measured_start)
            )
            row = build_history_row(
                epoch=epoch,
                optimizer=optimizer,
                train_metrics=train_metrics,
                memory=memory,
                elapsed_seconds=(elapsed_offset + time.perf_counter() - measured_start),
                metric_type="scalar",
                scheduler=scheduler,
            )
            writer.writerow(row)
            file.flush()
            vram_text = format_gpu_memory_metrics(device, memory)
            print(
                f"epoch={epoch:03d}/{args.epochs} "
                f"loss={train_metrics['loss']:.6e} "
                f"train_rMSE={train_metrics['u_relative_mse']:.6e} "
                f"train_rL2={100.0*train_metrics['u_relative_l2']:.3f}%"
                f" {vram_text}",
                flush=True,
            )

    last_file = output / "last.pt"
    if not math.isfinite(latest_train_loss):
        raise ValueError(
            "cannot save last.pt because the final training loss is not finite"
        )
    save_checkpoint(
        last_file,
        model,
        optimizer,
        scheduler,
        args.epochs,
        latest_train_loss,
        args,
        scaler=scaler,
    )
    if not args.skip_test:
        # Keep optimizer/RNG tensors from the full training checkpoint on
        # CPU and release them before starting the test-phase VRAM measurement.
        # Otherwise map_location=device would count a second model plus optimizer
        # state as "test" memory even though evaluation does not use either.
        checkpoint = torch.load(last_file, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        selected_epoch = int(checkpoint["epoch"])
        del checkpoint

        test_dataset = NativePoissonQueryDataset(
            data,
            "test",
            test_indices,
            None,
            random_queries=False,
            query_seed=args.seed,
            coordinate_transform=coordinate_transform,
            cache_cases=False,
            raw_access_mode=args.raw_access_mode,
        )
        test_loader = DataLoader(test_dataset, batch_size=1, **loader_common)
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
            "checkpoint": str(last_file),
            "gradient_injection_architecture": args.gradient_injection_architecture,
            "selected_epoch": selected_epoch,
            "architecture": args.architecture,
            "dataset": args.dataset,
            "loss_protocol": args.loss_protocol,
            "metric_protocol": args.metric_protocol,
            "normalize_inputs": args.normalize_inputs,
            "normalize_outputs": args.normalize_outputs,
            "query_sampling_protocol": args.query_sampling_protocol,
            "precision": precision,
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
            f"case_L2RE u={100.0 * float(test_metrics['case_relative_l2']):.3f}%, "
            f"test_elapsed={float(test_metrics['elapsed_seconds']):.2f}s, "
            f"total_elapsed={float(test_memory['elapsed_seconds']):.2f}s, "
            f"{format_gpu_memory_metrics(device, test_memory)}",
            flush=True,
        )
    final_memory = memory_tracker.snapshot(
        elapsed_seconds=elapsed_offset + time.perf_counter() - measured_start
    )
    with open(output / "gpu_memory.json", "w", encoding="utf-8") as file:
        json.dump(final_memory, file, ensure_ascii=False, indent=2)
    print(format_gpu_memory_metrics(device, final_memory), flush=True)
    print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
