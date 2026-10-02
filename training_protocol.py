"""Shared training protocol for the four published benchmarks."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass

import torch


WARMUP_RATIO = 0.05
GRADIENT_CLIP = 1.0
OPTIMIZER = "adamw"
SCHEDULER = "warmup-cosine"
PROTOCOL_VERSION = "mfd-four-benchmark-v6"
GRADIENT_INJECTION_ARCHITECTURE = "h_plus_shared_w_grad_h_no_alpha"


@dataclass(frozen=True)
class TrainingDefaults:
    """Numerical and runtime defaults shared by benchmark entry points."""

    mode: int
    batch_size: int
    minimum_learning_rate: float
    weight_decay: float
    global_width: int
    heads: int
    layers: int
    feedforward: int
    global_dropout: float
    global_position_frequencies: int
    local_fourier_frequencies: int
    train_query_limit: int
    decode_chunk_size: int
    log_every: int
    checkpoint_every: int
    local_width: int | None = None
    local_dropout: float = 0.0
    normalize_inputs: bool = True
    normalize_outputs: bool = True
    spectral_gradient_injection: bool = True


def add_width_arguments(
    parser: argparse.ArgumentParser,
    *,
    global_width: int,
    local_width: int | None = None,
) -> None:
    """Add independent Global and Local hidden-width arguments."""

    parser.add_argument(
        "--global-width",
        type=int,
        default=global_width,
        help="Global Transformer hidden width; it must be divisible by --heads.",
    )
    parser.add_argument(
        "--local-width",
        type=int,
        default=local_width,
        help="Local MLP hidden width; defaults to the Global width when omitted.",
    )


def resolve_width_arguments(args: argparse.Namespace) -> None:
    """Resolve an omitted Local width to the selected Global width."""

    if args.local_width is None:
        args.local_width = args.global_width


def normalize_width_arguments(arguments: Mapping[str, object]) -> dict[str, object]:
    """Normalize width keys from current and legacy checkpoints."""

    resolved = dict(arguments)
    if "global_width" not in resolved and "width" in resolved:
        resolved["global_width"] = resolved["width"]
    if resolved.get("local_width") is None and "global_width" in resolved:
        resolved["local_width"] = resolved["global_width"]
    return resolved


def add_dropout_arguments(
    parser: argparse.ArgumentParser,
    *,
    global_dropout: float,
    local_dropout: float = 0.0,
) -> None:
    """Add independent Global and Local dropout arguments."""

    parser.add_argument(
        "--global-dropout",
        type=float,
        default=global_dropout,
        help="Dropout probability used by the Global Transformer.",
    )
    parser.add_argument(
        "--local-dropout",
        type=float,
        default=local_dropout,
        help="Dropout probability after each Local GELU activation.",
    )


def normalize_dropout_arguments(arguments: Mapping[str, object]) -> dict[str, object]:
    """Normalize dropout keys from current and legacy checkpoints."""

    resolved = dict(arguments)
    if "global_dropout" not in resolved and "dropout" in resolved:
        resolved["global_dropout"] = resolved["dropout"]
    resolved.setdefault("local_dropout", 0.0)
    return resolved


def add_training_arguments(
    parser: argparse.ArgumentParser,
    defaults: TrainingDefaults,
) -> None:
    """Add the common command-line options used by every benchmark."""

    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--mode", type=int, default=defaults.mode)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument(
        "--minimum-learning-rate",
        type=float,
        default=defaults.minimum_learning_rate,
    )
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    add_width_arguments(
        parser, global_width=defaults.global_width, local_width=defaults.local_width
    )
    parser.add_argument("--heads", type=int, default=defaults.heads)
    parser.add_argument("--layers", type=int, default=defaults.layers)
    parser.add_argument("--feedforward", type=int, default=defaults.feedforward)
    add_dropout_arguments(
        parser,
        global_dropout=defaults.global_dropout,
        local_dropout=defaults.local_dropout,
    )
    parser.add_argument(
        "--global-position-frequencies",
        type=int,
        default=defaults.global_position_frequencies,
    )
    parser.add_argument(
        "--local-fourier-frequencies",
        type=int,
        default=defaults.local_fourier_frequencies,
    )
    parser.add_argument(
        "--train-query-limit", type=int, default=defaults.train_query_limit
    )
    parser.add_argument(
        "--decode-chunk-size", type=int, default=defaults.decode_chunk_size
    )
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    parser.add_argument(
        "--checkpoint-every", type=int, default=defaults.checkpoint_every
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic algorithms and a fixed cuBLAS workspace.",
    )
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument(
        "--amp",
        dest="no_amp",
        action="store_false",
        help="Enable CUDA float16 autocast and gradient scaling.",
    )
    precision.add_argument(
        "--no-amp",
        dest="no_amp",
        action="store_true",
        help="Disable mixed precision and use float32 throughout.",
    )
    parser.set_defaults(no_amp=False)
    parser.add_argument(
        "--normalize-inputs",
        action=argparse.BooleanOptionalAction,
        default=defaults.normalize_inputs,
        help="Normalize Global moments and Local function channels.",
    )
    parser.add_argument(
        "--normalize-outputs",
        action=argparse.BooleanOptionalAction,
        default=defaults.normalize_outputs,
        help="Train in normalized output space and evaluate physical values.",
    )
    parser.add_argument(
        "--skip-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip the final test evaluation when explicitly enabled.",
    )


def normalize_reproducibility_arguments(
    arguments: Mapping[str, object],
    *,
    expected: bool | None = None,
) -> dict[str, object]:
    """Normalize and validate deterministic-training settings."""

    resolved = dict(arguments)
    enabled = resolved.setdefault("deterministic", False)
    if not isinstance(enabled, bool):
        raise ValueError("deterministic must be a boolean")
    if expected is not None and enabled != expected:
        raise ValueError(
            "deterministic cannot change when resuming: "
            f"checkpoint={enabled}, current={expected}"
        )
    return resolved


def apply_fixed_protocol(
    args: argparse.Namespace,
    defaults: TrainingDefaults,
    *,
    spectral_gradient_projection: str,
) -> None:
    """Attach fixed protocol metadata after resolving adjustable arguments."""

    resolve_width_arguments(args)
    args.protocol_version = PROTOCOL_VERSION
    args.expected_mode = int(args.mode)
    args.warmup_ratio = WARMUP_RATIO
    args.gradient_clip = GRADIENT_CLIP
    args.spectral_gradient_injection = defaults.spectral_gradient_injection
    args.spectral_gradient_projection = spectral_gradient_projection
    args.gradient_injection_architecture = GRADIENT_INJECTION_ARCHITECTURE


def build_adamw(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> tuple[torch.optim.AdamW, list[dict[str, int | float | str]]]:
    """Create the AdamW optimizer and its serializable parameter summary."""

    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    summary: list[dict[str, int | float | str]] = [
        {
            "name": "all",
            "weight_decay": float(weight_decay),
            "tensors": len(parameters),
            "parameters": sum(parameter.numel() for parameter in parameters),
        }
    ]
    return optimizer, summary


__all__ = [
    "GRADIENT_CLIP",
    "GRADIENT_INJECTION_ARCHITECTURE",
    "OPTIMIZER",
    "PROTOCOL_VERSION",
    "SCHEDULER",
    "TrainingDefaults",
    "WARMUP_RATIO",
    "add_dropout_arguments",
    "add_training_arguments",
    "apply_fixed_protocol",
    "build_adamw",
    "normalize_dropout_arguments",
    "normalize_reproducibility_arguments",
]
