"""Shared runtime for channel-independent training and chunked evaluation."""

from __future__ import annotations

import math
import os
import random
import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .GlobalLocalMFE import GlobalLocalMFE
    from .dataset_common import compose_function_scaling, derive_condition_moments
except ImportError:  # Support direct execution from the project directory.
    from GlobalLocalMFE import GlobalLocalMFE
    from dataset_common import compose_function_scaling, derive_condition_moments


LossAndMetrics = Callable[
    [torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, Mapping[str, torch.Tensor]],
]


class FullCaseMetricAccumulator(Protocol):
    """Minimal interface implemented by benchmark metric accumulators."""

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None: ...

    def finalize(self) -> Mapping[str, torch.Tensor]: ...


FullCaseMetricFactory = Callable[[int, torch.device], FullCaseMetricAccumulator]
FullCaseMetricSummary = Callable[[Mapping[str, Sequence[float]]], Mapping[str, float]]


HISTORY_FIELDS_MEMORY = (
    # Legacy fields remain in their original positions; peak_* equals overall.
    "peak_allocated_mib",
    "peak_reserved_mib",
    "train_peak_allocated_mib",
    "train_peak_reserved_mib",
    "validation_peak_allocated_mib",
    "validation_peak_reserved_mib",
    "test_peak_allocated_mib",
    "test_peak_reserved_mib",
    "overall_peak_allocated_mib",
    "overall_peak_reserved_mib",
)

HISTORY_FIELDS_SCALAR = (
    "epoch",
    "optimizer_steps",
    "elapsed_seconds",
    "learning_rate",
    "train_loss",
    "train_relative_mse_u",
    "train_relative_l2_u",
    "val_relative_mse_u",
    "val_relative_l2_u",
    *HISTORY_FIELDS_MEMORY,
)

HISTORY_FIELDS_P_VECTOR = (
    "epoch",
    "optimizer_steps",
    "elapsed_seconds",
    "learning_rate",
    "train_loss",
    "train_relative_mse_p",
    "train_relative_mse_f_or_u",
    "train_relative_l2_p",
    "train_relative_l2_f_or_u",
    "train_relative_l2_magnitude",
    "val_relative_mse_p",
    "val_relative_mse_f_or_u",
    "val_relative_l2_p",
    "val_relative_l2_f_or_u",
    "val_relative_l2_magnitude",
    *HISTORY_FIELDS_MEMORY,
)


def history_fieldnames(metric_type: str = "scalar") -> tuple[str, ...]:
    """Return the unified history schema for scalar or P-vector metrics."""

    if metric_type in {"scalar", "darcy", "poisson"}:
        return HISTORY_FIELDS_SCALAR
    if metric_type in {"p_vector", "vector", "nasa", "ahmed"}:
        return HISTORY_FIELDS_P_VECTOR
    raise ValueError(
        f"unknown metric_type={metric_type!r}; expected 'scalar' or 'p_vector'"
    )


def normalize_history_row(
    row: Mapping[str, object], fieldnames: Sequence[str]
) -> dict[str, object]:
    """Fill absent history fields with empty strings for a stable CSV schema."""

    return {name: row.get(name, "") for name in fieldnames}


def format_gpu_memory_metrics(
    device: torch.device,
    memory: Mapping[str, object],
) -> str:
    """Format unified peak CUDA-memory statistics for console output."""

    if isinstance(memory.get("overall"), Mapping):
        phase_parts: list[str] = []
        for phase, label in (
            ("train", "train"),
            ("validation", "val"),
            ("test", "test"),
            ("overall", "overall"),
        ):
            phase_memory = memory[phase]
            assert isinstance(phase_memory, Mapping)
            allocated = phase_memory.get("peak_allocated_mib")
            reserved = phase_memory.get("peak_reserved_mib")
            value = (
                "N/A"
                if allocated is None or reserved is None
                else f"{allocated:.1f}/{reserved:.1f}"
            )
            phase_parts.append(f"{label}={value}")
        return (
            "VRAM_peak_MiB[allocated/reserved] " + ", ".join(phase_parts)
        )
    if memory["peak_allocated_mib"] is None:
        return f"VRAM_peak=N/A ({device})"
    return (
        f"VRAM_peak={memory['peak_allocated_mib']:.1f}/"
        f"{memory['peak_reserved_mib']:.1f} MiB (allocated/reserved)"
    )


def gpu_memory_memory_row(
    memory: Mapping[str, object],
) -> dict[str, object]:
    """Convert phased or legacy memory statistics to history columns."""

    row: dict[str, object] = {}
    for phase in ("train", "validation", "test", "overall"):
        phase_memory = memory.get(phase)
        for kind in ("allocated", "reserved"):
            key = f"peak_{kind}_mib"
            value = (
                phase_memory.get(key)
                if isinstance(phase_memory, Mapping)
                else None
            )
            row[f"{phase}_peak_{kind}_mib"] = "" if value is None else value

    allocated = memory.get("peak_allocated_mib")
    reserved = memory.get("peak_reserved_mib")
    row["peak_allocated_mib"] = "" if allocated is None else allocated
    row["peak_reserved_mib"] = "" if reserved is None else reserved
    if not isinstance(memory.get("overall"), Mapping):
        row["overall_peak_allocated_mib"] = row["peak_allocated_mib"]
        row["overall_peak_reserved_mib"] = row["peak_reserved_mib"]
    return row


def build_history_row(
    *,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    train_metrics: Mapping[str, object],
    memory: Mapping[str, object],
    elapsed_seconds: float,
    metric_type: str,
    validation_metrics: Mapping[str, object] | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    fill_optimizer_steps: bool = True,
) -> dict[str, object]:
    """Assemble one normalized training-history row."""

    if fill_optimizer_steps and scheduler is not None:
        optimizer_steps = int(scheduler.last_epoch)
    elif fill_optimizer_steps:
        optimizer_steps = 0
    else:
        optimizer_steps = ""
    train_loss = float(train_metrics["loss"])
    row: dict[str, object] = {
        "epoch": int(epoch),
        "optimizer_steps": optimizer_steps,
        "elapsed_seconds": float(elapsed_seconds),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "train_loss": train_loss,
    }
    if metric_type in {"p_vector", "vector", "nasa", "ahmed"}:
        row.update(
            {
                "train_relative_mse_p": float(train_metrics["p_relative_mse"]),
                "train_relative_mse_f_or_u": float(
                    train_metrics["vector_relative_mse"]
                ),
                "train_relative_l2_p": float(train_metrics["p_relative_l2"]),
                "train_relative_l2_f_or_u": float(train_metrics["vector_relative_l2"]),
                "train_relative_l2_magnitude": float(
                    train_metrics["magnitude_relative_l2"]
                ),
            }
        )
    else:
        row.update(
            {
                "train_relative_mse_u": float(train_metrics["u_relative_mse"]),
                "train_relative_l2_u": float(train_metrics["u_relative_l2"]),
            }
        )

    if validation_metrics is not None:
        if metric_type in {"p_vector", "vector", "nasa", "ahmed"}:
            row.update(
                {
                    "val_relative_mse_p": "",
                    "val_relative_mse_f_or_u": "",
                    "val_relative_l2_p": float(
                        validation_metrics["case_p_relative_l2"]
                    ),
                    "val_relative_l2_f_or_u": float(
                        validation_metrics["case_vector_relative_l2"]
                    ),
                    "val_relative_l2_magnitude": float(
                        validation_metrics["case_magnitude_relative_l2"]
                    ),
                }
            )
        else:
            row.update(
                {
                    "val_relative_mse_u": float(
                        validation_metrics["case_relative_mse"]
                    ),
                    "val_relative_l2_u": float(validation_metrics["case_relative_l2"]),
                }
            )
    else:
        # Preserve all schema keys even when validation is not run.
        if metric_type in {"p_vector", "vector", "nasa", "ahmed"}:
            row.update(
                {
                    "val_relative_mse_p": "",
                    "val_relative_mse_f_or_u": "",
                    "val_relative_l2_p": "",
                    "val_relative_l2_f_or_u": "",
                    "val_relative_l2_magnitude": "",
                }
            )
        else:
            row.update(
                {
                    "val_relative_mse_u": "",
                    "val_relative_l2_u": "",
                }
            )

    row.update(gpu_memory_memory_row(memory))
    return normalize_history_row(row, history_fieldnames(metric_type))


def scheduler_for(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
    minimum_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a per-step linear-warmup and cosine-decay scheduler."""

    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def set_seed(seed: int, *, deterministic: bool = True) -> dict[str, object]:
    """Seed all RNGs and configure deterministic CUDA execution.

    Call this before creating CUDA tensors. In deterministic mode, operations
    without a deterministic implementation raise instead of merely warning.
    """

    if deterministic:
        workspace = ":4096:8"
        if (
            torch.cuda.is_initialized()
            and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != workspace
        ):
            raise RuntimeError(
                "deterministic settings must be applied before CUDA initialization; "
                "start a new process or pre-set CUBLAS_WORKSPACE_CONFIG=:4096:8"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def validate_common_training_arguments(args: Namespace) -> None:
    """Validate optimizer, model, and runtime arguments shared by benchmarks."""

    positive_integers = (
        "mode",
        "epochs",
        "batch_size",
        "global_width",
        "local_width",
        "heads",
        "layers",
        "feedforward",
        "decode_chunk_size",
        "log_every",
    )
    required = (
        *positive_integers,
        "learning_rate",
        "minimum_learning_rate",
        "weight_decay",
        "warmup_ratio",
        "gradient_clip",
        "global_dropout",
        "local_dropout",
        "global_position_frequencies",
        "local_fourier_frequencies",
        "checkpoint_every",
        "num_workers",
    )
    missing = [name for name in required if not hasattr(args, name)]
    if missing:
        raise AttributeError(f"training entry point is missing common fields: {missing}")

    boolean_fields = (
        "no_amp",
        "normalize_inputs",
        "normalize_outputs",
    )
    missing_booleans = [name for name in boolean_fields if not hasattr(args, name)]
    if missing_booleans:
        raise AttributeError(
            f"training entry point is missing boolean fields: {missing_booleans}"
        )
    invalid_booleans = [
        name for name in boolean_fields if not isinstance(getattr(args, name), bool)
    ]
    if invalid_booleans:
        raise TypeError(f"common model switches must be bool: {invalid_booleans}")

    for name in positive_integers:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be a positive integer")
    if args.mode < 2:
        raise ValueError("mode must be at least 2 for spectral-gradient injection")
    for name in ("train_query_limit", "eval_every", "validation_every"):
        if hasattr(args, name) and int(getattr(args, name)) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be a positive integer")
    if args.global_width % args.heads:
        raise ValueError("global-width must be divisible by heads")
    finite_fields = (
        "learning_rate",
        "minimum_learning_rate",
        "weight_decay",
        "warmup_ratio",
        "gradient_clip",
        "global_dropout",
        "local_dropout",
    )
    invalid_finite = [
        name for name in finite_fields if not math.isfinite(float(getattr(args, name)))
    ]
    if invalid_finite:
        raise ValueError(f"training float arguments must be finite: {invalid_finite}")
    if args.learning_rate <= 0.0 or args.minimum_learning_rate <= 0.0:
        raise ValueError("learning-rate and minimum-learning-rate must be positive")
    if args.minimum_learning_rate > args.learning_rate:
        raise ValueError("minimum-learning-rate cannot exceed learning-rate")
    if args.weight_decay < 0.0:
        raise ValueError("weight-decay must be non-negative")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup-ratio must be in [0, 1)")
    if args.gradient_clip < 0.0:
        raise ValueError("gradient-clip must be non-negative; zero disables clipping")
    for name in ("global_dropout", "local_dropout"):
        if not 0.0 <= getattr(args, name) < 1.0:
            raise ValueError(f"{name} must be in [0,1)")
    if args.global_position_frequencies < 0 or args.local_fourier_frequencies < 0:
        raise ValueError("Global/Local encoding frequency counts must be non-negative")
    if args.checkpoint_every < 0 or args.num_workers < 0:
        raise ValueError("checkpoint-every and num-workers must be non-negative")
    if getattr(args, "device", None) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch runtime")


def make_output_directory(
    requested: str | Path | None,
    experiment_name: str,
    protocol_name: str = "p-vector-relative-mse",
    base_directory: str | Path | None = None,
) -> Path:
    """Return an explicit output directory or create a timestamped result path."""

    if requested:
        output = Path(requested).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_root = (
            Path(base_directory).resolve()
            if base_directory is not None
            else Path(__file__).resolve().parent / "result"
        )
        output = result_root / f"{experiment_name}-{protocol_name}-{timestamp}"
    output.mkdir(parents=True, exist_ok=True)
    return output


def move_batch(batch: dict, device: torch.device) -> dict:
    """Move model input tensors to a device while retaining CPU metadata."""

    model_keys = {
        "moments",
        "coordinates",
        "functions",
        "targets",
        "normals",
    }
    return {
        key: (
            value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor) and key in model_keys
            else value
        )
        for key, value in batch.items()
    }


def autocast_context(
    device: torch.device, enabled: bool
) -> torch.autocast:
    """Return the CUDA autocast context used by every benchmark runtime."""

    if enabled and device.type != "cuda":
        raise ValueError("AMP is supported only on CUDA; disable it for CPU runs")
    return torch.autocast(device_type=device.type, enabled=enabled)


def validate_batch_contract(batch: Mapping[str, object], model: GlobalLocalMFE) -> None:
    """Validate the shared tensor shapes and dtypes for every benchmark batch."""

    required = ("moments", "coordinates", "functions", "targets")
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError(f"batch is missing required fields: {missing}")
    tensors: dict[str, torch.Tensor] = {}
    for name in required:
        value = batch[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch[{name!r}] must be a torch.Tensor")
        if value.dtype != torch.float32:
            raise TypeError(f"batch[{name!r}] must be float32, got {value.dtype}")
        if value.ndim != 3:
            raise ValueError(
                f"batch[{name!r}] must be rank 3 and include a batch dimension; "
                f"got {tuple(value.shape)}"
            )
        tensors[name] = value

    moments = tensors["moments"]
    coordinates = tensors["coordinates"]
    functions = tensors["functions"]
    targets = tensors["targets"]
    batch_size, query_count = coordinates.shape[:2]
    expected = {
        "moments": (batch_size, model.tokens, model.input_channels),
        "coordinates": (batch_size, query_count, model.dimension),
        "functions": (batch_size, query_count, model.function_channels),
        "targets": (batch_size, query_count, model.output_channels),
    }
    actual = {
        "moments": tuple(moments.shape),
        "coordinates": tuple(coordinates.shape),
        "functions": tuple(functions.shape),
        "targets": tuple(targets.shape),
    }
    errors = [
        f"{name}: expected={expected[name]}, actual={actual[name]}"
        for name in required
        if actual[name] != expected[name]
    ]
    if errors:
        raise ValueError("batch violates the shared data contract: " + "; ".join(errors))

    if model.requires_normals and "normals" not in batch:
        raise KeyError(
            "tangent spectral gradients require explicit batch['normals']; "
            "normals are not inferred from function channels"
        )
    if "normals" in batch:
        normals = batch["normals"]
        if not isinstance(normals, torch.Tensor):
            raise TypeError("batch['normals'] must be a torch.Tensor")
        if normals.dtype != torch.float32:
            raise TypeError(
                f"batch['normals'] must be float32, got {normals.dtype}"
            )
        if tuple(normals.shape) != tuple(coordinates.shape):
            raise ValueError(
                "batch['normals'] must match coordinates shape; "
                f"expected={tuple(coordinates.shape)}, actual={tuple(normals.shape)}"
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
    loss_and_metrics: LossAndMetrics,
    step_scheduler_per_batch: bool = True,
    record_optimizer_updates: bool = False,
) -> dict[str, float | int]:
    """Run one training epoch under the shared batch and metric protocols."""

    if use_amp and not scaler.is_enabled():
        raise ValueError("use_amp=True requires an enabled GradScaler")
    model.train()
    loss_sum = 0.0
    metric_sums: dict[str, float] = {}
    metric_names: tuple[str, ...] | None = None
    samples = 0
    optimizer_updates = 0
    optimizer_step_completed = False

    def mark_optimizer_step(
        _optimizer: torch.optim.Optimizer,
        _args: tuple[object, ...],
        _kwargs: dict[str, object],
    ) -> None:
        nonlocal optimizer_step_completed
        optimizer_step_completed = True

    step_hook = optimizer.register_step_post_hook(mark_optimizer_step)
    try:
        for raw_batch in loader:
            validate_batch_contract(raw_batch, model)
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, use_amp):
                normalized, _, _ = model(
                    batch["moments"],
                    batch["coordinates"],
                    batch["functions"],
                    chunk_size,
                    normals=(
                        batch.get("normals") if model.requires_normals else None
                    ),
                )
                prediction = model.denormalize_targets(normalized)
                loss, batch_metrics = loss_and_metrics(
                    prediction, batch["targets"]
                )
            if loss.ndim != 0:
                raise ValueError("the benchmark loss callback must return a scalar loss")
            # The loop already needs this host scalar for history aggregation.
            # Materialize it once before backward so fail-fast adds no extra
            # device synchronization to every successful batch.
            loss_value = float(loss.detach().double().cpu())
            if not math.isfinite(loss_value):
                raise FloatingPointError(
                    "training loss is NaN or Inf; stopped before backward and "
                    "scheduler.step"
                )
            current_names = tuple(batch_metrics)
            if not current_names:
                raise ValueError("the benchmark loss callback must return at least one metric")
            if metric_names is None:
                metric_names = current_names
                metric_sums = {name: 0.0 for name in metric_names}
            elif current_names != metric_names:
                raise ValueError(
                    "loss callback returned inconsistent metrics across batches: "
                    f"expected={metric_names}, actual={current_names}"
                )

            optimizer_step_completed = False
            if use_amp:
                scale = float(scaler.get_scale())
                if not math.isfinite(scale) or scale <= 0.0:
                    raise FloatingPointError(f"GradScaler scale invalid: {scale}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if gradient_clip > 0.0:
                    # GradScaler owns non-finite-gradient recovery in AMP mode:
                    # it skips this optimizer step and lowers its scale.
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        gradient_clip,
                        error_if_nonfinite=False,
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if gradient_clip > 0.0:
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            gradient_clip,
                            error_if_nonfinite=True,
                        )
                    except RuntimeError as error:
                        raise FloatingPointError(
                            "float32 parameter gradients contain NaN or Inf; "
                            "stopped before optimizer.step"
                        ) from error
                optimizer.step()

            if optimizer_step_completed:
                optimizer_updates += 1
                if step_scheduler_per_batch:
                    scheduler.step()

            batch_size = int(batch["targets"].shape[0])
            loss_sum += loss_value * batch_size
            for name, value in batch_metrics.items():
                if not isinstance(value, torch.Tensor) or value.numel() != 1:
                    raise ValueError(
                        f"training metric {name!r} must be a one-element Tensor"
                    )
                metric_sums[name] += (
                    float(value.detach().double().cpu()) * batch_size
                )
            samples += batch_size
    finally:
        step_hook.remove()

    if samples and optimizer_updates == 0:
        raise FloatingPointError(
            "the epoch completed no optimizer steps; AMP overflowed on every "
            "batch, so check numerical scales or disable AMP"
        )

    denominator = max(samples, 1)
    result: dict[str, float | int] = {"loss": loss_sum / denominator}
    if record_optimizer_updates:
        # Epoch-stepped schedulers do not expose the number of minibatch
        # optimizer updates, so report it only when explicitly requested.
        result["optimizer_updates"] = optimizer_updates
    result.update({name: total / denominator for name, total in metric_sums.items()})
    return result


@torch.no_grad()
def evaluate(
    model: GlobalLocalMFE,
    loader: DataLoader,
    device: torch.device,
    chunk_size: int,
    use_amp: bool,
    loss_protocol: str,
    metric_protocol: str,
    metric_factory: FullCaseMetricFactory,
    summarize_metrics: FullCaseMetricSummary,
) -> dict[str, float | str]:
    """Decode complete cases in chunks and accumulate benchmark metrics."""

    model.eval()
    metric_values: dict[str, list[float]] = {}
    metric_names: tuple[str, ...] | None = None
    sample_count = 0
    diagnostic_count = 0
    metric_state: FullCaseMetricAccumulator | None = None
    global_square = 0.0
    local_square = 0.0
    elapsed_start = time.perf_counter()

    for raw_batch in loader:
        validate_batch_contract(raw_batch, model)
        batch = move_batch(raw_batch, device)
        with autocast_context(device, use_amp):
            state = model.encode_global(batch["moments"])
        physical_scales = (
            model._resolve_gradient_physical_scales(
                batch["coordinates"].shape[0],
                batch["coordinates"].device,
            )
            if model.uses_mfe_gradients
            else None
        )
        unit_normals = (
            model._unit_surface_normals(
                batch.get("normals"), batch["coordinates"]
            )
            if model.requires_normals
            else None
        )
        batch_size = int(batch["moments"].shape[0])
        sample_count += batch_size
        metric_state = metric_factory(batch_size, batch["moments"].device)

        for start in range(0, batch["coordinates"].shape[1], chunk_size):
            stop = min(start + chunk_size, batch["coordinates"].shape[1])
            with autocast_context(device, use_amp):
                normalized, global_part, local_part = (
                    model._decode_chunk_prevalidated(
                        state,
                        batch["coordinates"][:, start:stop],
                        batch["functions"][:, start:stop],
                        unit_normals=(
                            unit_normals[:, start:stop]
                            if unit_normals is not None
                            else None
                        ),
                        physical_scales=physical_scales,
                    )
                )
                prediction = model.denormalize_targets(normalized)
            target = batch["targets"][:, start:stop]
            metric_state.update(prediction, target)
            diagnostic_count += target.numel()
            global_square += float(global_part.float().square().sum())
            local_square += float(local_part.float().square().sum())

        batch_metrics = metric_state.finalize()
        current_names = tuple(batch_metrics)
        if not current_names:
            raise ValueError("the full-case metric accumulator returned no metrics")
        if metric_names is None:
            metric_names = current_names
            metric_values = {name: [] for name in metric_names}
        elif current_names != metric_names:
            raise ValueError(
                "full-case accumulator returned inconsistent metrics across batches: "
                f"expected={metric_names}, actual={current_names}"
            )
        for name, value in batch_metrics.items():
            if not isinstance(value, torch.Tensor) or value.shape != (batch_size,):
                raise ValueError(
                    f"full-case metric {name!r} must be a Tensor with shape "
                    f"[{batch_size}]"
                )
            metric_values[name].extend(value.detach().double().cpu().tolist())

    result: dict[str, float | str] = {
        "loss_protocol": loss_protocol,
        "metric_protocol": metric_protocol,
        "samples": float(sample_count),
        "global_output_rms": math.sqrt(global_square / max(diagnostic_count, 1)),
        "local_output_rms": math.sqrt(local_square / max(diagnostic_count, 1)),
        "elapsed_seconds": time.perf_counter() - elapsed_start,
    }
    result.update(
        {name: float(value) for name, value in summarize_metrics(metric_values).items()}
    )
    return result


def gpu_memory_metrics(
    device: torch.device,
    elapsed_seconds: float | None = None,
) -> dict[str, float | str | int | None]:
    """Return peak CUDA allocator statistics and optional elapsed time."""

    if device.type != "cuda":
        return {
            "device": str(device),
            "elapsed_seconds": elapsed_seconds,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "peak_allocated_mib": None,
            "peak_reserved_mib": None,
        }

    torch.cuda.synchronize(device)
    mib = 1024.0**2
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        "device": torch.cuda.get_device_name(device),
        "elapsed_seconds": elapsed_seconds,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_mib": peak_allocated / mib,
        "peak_reserved_mib": peak_reserved / mib,
    }


class PhaseMemoryTracker:
    """Track peak CUDA allocator use separately by execution phase.

    ``begin`` resets PyTorch peak statistics, ``end`` records the phase peak,
    and ``snapshot`` reports train, validation, test, and overall peaks. These
    values cover only the current PyTorch process, not driver-level usage.
    """

    PHASES = ("train", "validation", "test")
    _PEAK_FIELDS = (
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "peak_allocated_mib",
        "peak_reserved_mib",
    )

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._active_phase: str | None = None
        self._peaks: dict[str, dict[str, float | int | None]] = {
            phase: {field: None for field in self._PEAK_FIELDS}
            for phase in self.PHASES
        }
        self._measurements = {phase: 0 for phase in self.PHASES}

    def _validate_phase(self, phase: str) -> None:
        if phase not in self.PHASES:
            raise ValueError(
                f"unknown memory phase {phase!r}; expected one of {', '.join(self.PHASES)}"
            )

    def begin(self, phase: str) -> None:
        """Start a phase measurement and reset CUDA peak statistics."""

        self._validate_phase(phase)
        if self._active_phase is not None:
            raise RuntimeError(
                f"memory phase {self._active_phase!r} has not ended; "
                f"cannot start {phase!r}"
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._active_phase = phase

    def end(
        self,
        phase: str,
        *,
        elapsed_seconds: float | None = None,
    ) -> dict[str, object]:
        """End a phase, accumulate its peak, and return a complete snapshot."""

        self._validate_phase(phase)
        if self._active_phase != phase:
            raise RuntimeError(
                f"active memory phase is {self._active_phase!r}; cannot end {phase!r}"
            )
        measured = gpu_memory_metrics(self.device, elapsed_seconds)
        self._active_phase = None
        phase_peak = self._peaks[phase]
        for field in self._PEAK_FIELDS:
            value = measured[field]
            if value is None:
                continue
            old = phase_peak[field]
            phase_peak[field] = value if old is None else max(old, value)
        self._measurements[phase] += 1
        return self.snapshot(elapsed_seconds=elapsed_seconds)

    def seed_phase(
        self,
        phase: str,
        *,
        peak_allocated_bytes: int | None = None,
        peak_reserved_bytes: int | None = None,
        peak_allocated_mib: float | None = None,
        peak_reserved_mib: float | None = None,
    ) -> None:
        """Merge prior process peaks when resuming from a checkpoint."""

        self._validate_phase(phase)
        if self._active_phase is not None:
            raise RuntimeError("cannot seed memory peaks while a phase is active")
        mib = 1024.0**2
        values: dict[str, float | int | None] = {
            "peak_allocated_bytes": peak_allocated_bytes,
            "peak_reserved_bytes": peak_reserved_bytes,
            "peak_allocated_mib": peak_allocated_mib,
            "peak_reserved_mib": peak_reserved_mib,
        }
        if values["peak_allocated_bytes"] is None and peak_allocated_mib is not None:
            values["peak_allocated_bytes"] = int(round(peak_allocated_mib * mib))
        if values["peak_reserved_bytes"] is None and peak_reserved_mib is not None:
            values["peak_reserved_bytes"] = int(round(peak_reserved_mib * mib))
        if values["peak_allocated_mib"] is None and peak_allocated_bytes is not None:
            values["peak_allocated_mib"] = peak_allocated_bytes / mib
        if values["peak_reserved_mib"] is None and peak_reserved_bytes is not None:
            values["peak_reserved_mib"] = peak_reserved_bytes / mib
        if all(value is None for value in values.values()):
            return
        phase_peak = self._peaks[phase]
        for field, value in values.items():
            if value is None:
                continue
            old = phase_peak[field]
            phase_peak[field] = value if old is None else max(old, value)
        self._measurements[phase] = max(self._measurements[phase], 1)

    def snapshot(self, *, elapsed_seconds: float | None = None) -> dict[str, object]:
        """Return phased and overall peaks after the active phase has ended."""

        if self._active_phase is not None:
            raise RuntimeError(
                f"memory phase {self._active_phase!r} has not ended; "
                "cannot create a snapshot"
            )
        device_name = (
            torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else str(self.device)
        )
        phase_results: dict[str, dict[str, object]] = {}
        for phase in self.PHASES:
            phase_results[phase] = {
                "measurements": self._measurements[phase],
                **self._peaks[phase],
            }

        overall: dict[str, object] = {
            "measurements": sum(self._measurements.values())
        }
        for field in self._PEAK_FIELDS:
            values = [
                self._peaks[phase][field]
                for phase in self.PHASES
                if self._peaks[phase][field] is not None
            ]
            overall[field] = max(values) if values else None

        return {
            "device": device_name,
            "elapsed_seconds": elapsed_seconds,
            **{field: overall[field] for field in self._PEAK_FIELDS},
            **phase_results,
            "overall": overall,
        }


def save_checkpoint(
    filename: Path,
    model: GlobalLocalMFE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    score: float,
    args: Namespace,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    """Atomically save all state required to resume training."""

    temporary = filename.with_suffix(filename.suffix + ".tmp")
    payload: dict[str, object] = {
        "epoch": epoch,
        "score": score,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "arguments": vars(args),
        "rng_state": capture_rng_state(),
    }
    if hasattr(args, "architecture"):
        payload["architecture"] = getattr(args, "architecture")
    if hasattr(args, "dataset"):
        payload["dataset"] = getattr(args, "dataset")
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    torch.save(payload, temporary)
    temporary.replace(filename)


def capture_rng_state() -> dict[str, object]:
    """Capture Python, NumPy, and Torch RNG state for exact resumption."""

    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, object]) -> None:
    """Restore state produced by :func:`capture_rng_state`."""

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = sorted(required.difference(state))
    if missing:
        raise KeyError(f"checkpoint rng_state is missing fields: {missing}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state, but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_state)
