"""Global-to-Local MENO model with orthonormal Legendre bases."""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator

import numpy as np
import torch

try:
    from .dataset_common import ModelScaling
except ImportError:  # Imported as a top-level module.
    from dataset_common import ModelScaling


def normalized_legendre_torch(x: torch.Tensor, mode: int) -> torch.Tensor:
    r"""Evaluate the orthonormal Legendre basis on ``[-1, 1]``.

    Channel ``n`` is ``sqrt((2n+1)/2) P_n(x)``.
    """

    if mode < 1:
        raise ValueError("mode must be a positive integer")
    values = [torch.ones_like(x)]
    if mode > 1:
        values.append(x)
    for degree in range(2, mode):
        values.append(
            ((2 * degree - 1) * x * values[-1] - (degree - 1) * values[-2]) / degree
        )
    vandermonde = torch.stack(values, dim=-1)
    degree = torch.arange(mode, device=x.device, dtype=x.dtype)
    return vandermonde * torch.sqrt((2.0 * degree + 1.0) / 2.0)


def normalized_legendre_derivative_matrix(mode: int) -> torch.Tensor:
    r"""Return the exact derivative operator for the normalized Legendre basis.

    Rows are derivative/output modes and columns are source modes.  For
    ``l_n=sqrt((2n+1)/2) P_n`` the nonzero entries are

    ``D[k,n] = sqrt((2k+1)(2n+1))`` for ``k<n`` and odd ``n-k``.

    This coefficient-space form is finite at ``x=+-1``; unlike formulas with
    a ``1-x**2`` denominator it is safe on Darcy boundary nodes.
    """

    if mode < 1:
        raise ValueError("mode must be a positive integer")
    indices = torch.arange(mode, dtype=torch.int64)
    output_degree = indices[:, None]
    source_degree = indices[None, :]
    nonzero = (source_degree > output_degree) & (
        (source_degree - output_degree).remainder(2) == 1
    )
    values = torch.sqrt(
        (2.0 * output_degree.float() + 1.0)
        * (2.0 * source_degree.float() + 1.0)
    )
    return torch.where(nonzero, values, torch.zeros_like(values))


def tensor_legendre_basis(
    coordinates: torch.Tensor, mode: int, dimension: int
) -> torch.Tensor:
    """Return a ``[B,Q,mode**dimension]`` tensor-product basis."""

    if coordinates.ndim != 3 or coordinates.shape[-1] != dimension:
        raise ValueError("coordinates must have shape [B,Q,dimension]")
    coordinates = coordinates.float()
    if dimension == 2:
        lx = normalized_legendre_torch(coordinates[..., 0], mode)
        ly = normalized_legendre_torch(coordinates[..., 1], mode)
        return torch.einsum("bqi,bqj->bqij", lx, ly).flatten(2)
    elif dimension == 3:
        lx = normalized_legendre_torch(coordinates[..., 0], mode)
        ly = normalized_legendre_torch(coordinates[..., 1], mode)
        lz = normalized_legendre_torch(coordinates[..., 2], mode)
        # Compute a pointwise outer product without summing over modal axes.
        return torch.einsum("bqi,bqj,bqk->bqijk", lx, ly, lz).flatten(2)
    else:
        raise ValueError("dimension must be 2 or 3")


def basis_multi_indices(mode: int, dimension: int) -> np.ndarray:
    if dimension == 2:
        return np.indices((mode, mode), dtype=np.int64).reshape(2, -1).T
    elif dimension == 3:
        return np.indices((mode, mode, mode), dtype=np.int64).reshape(3, -1).T
    else:
        raise ValueError("dimension must be 2 or 3")


class SpectralPositionEncoding(torch.nn.Module):
    """Encode each Global token from its Legendre multi-index."""

    def __init__(self, mode: int, width: int, frequencies: int, dimension: int) -> None:
        super().__init__()
        if frequencies < 0:
            raise ValueError("frequencies must be non-negative")
        indices = basis_multi_indices(mode, dimension).astype(np.float32)
        normalized = indices / max(mode - 1, 1)
        # features = [normalized, np.square(normalized)]
        features = [normalized]
        for level in range(frequencies):
            omega = math.pi * (2.0**level)
            features.extend([np.sin(omega * normalized), np.cos(omega * normalized)])
        fixed = np.concatenate(features, axis=1).astype(np.float32)
        self.register_buffer("fixed_features", torch.from_numpy(fixed))
        self.project = torch.nn.Sequential(
            torch.nn.Linear(fixed.shape[1], width),
            torch.nn.GELU(),
            torch.nn.Linear(width, width),
        )

    def forward(self) -> torch.Tensor:
        return self.project(self.fixed_features)


@dataclass
class GlobalState:
    injection_coefficients: tuple[torch.Tensor, ...]
    output_coefficients: torch.Tensor


class GlobalLocalMFE(torch.nn.Module):
    r"""One-way Global-to-Local MENO model.

    Each Global Transformer layer maps its normalized hidden state to MFE
    coefficients, which are evaluated at query points and injected into the
    matching Local layer. Global and Local widths may differ. The final output
    is the sum of the Global MFE reconstruction and the Local prediction.

    The optional first-order spectral branch injects ``W grad(h)`` in ambient
    space or ``W (I-nn^T) grad(h)`` on a two-dimensional surface embedded in
    ``R^3``. Physical derivatives use the affine half-spans stored in
    ``ModelScaling.coordinate_scale``.
    """

    def __init__(
        self,
        mode: int,
        input_channels: int,
        output_channels: int,
        dimension: int,
        function_channels: int,
        scaling: ModelScaling,
        global_width: int = 128,
        local_width: int | None = None,
        heads: int = 8,
        layers: int = 4,
        feedforward: int = 256,
        global_dropout: float = 0.0,
        local_dropout: float = 0.0,
        global_position_frequencies: int = 4,
        local_fourier_frequencies: int = 4,
        spectral_gradient_injection: bool = True,
        spectral_gradient_projection: str = "ambient",
        normalize_inputs: bool = True,
        normalize_outputs: bool = True,
        normalize_global_moments: bool | None = None,
    ) -> None:
        super().__init__()
        if input_channels < 1 or output_channels < 1 or function_channels < 0:
            raise ValueError(
                "input/output channels must be positive and function_channels "
                "must be non-negative"
            )
        if dimension not in (2, 3):
            raise ValueError("dimension must be 2 or 3")
        if mode < 1 or layers < 1:
            raise ValueError("mode and layers must be positive integers")
        if global_position_frequencies < 0 or local_fourier_frequencies < 0:
            raise ValueError("Fourier frequency counts must be non-negative")
        if local_width is None:
            local_width = global_width
        dimensions = {}
        for name, value in (
            ("global_width", global_width),
            ("local_width", local_width),
            ("heads", heads),
        ):
            try:
                integer_value = operator.index(value)
            except TypeError as error:
                raise ValueError(f"{name} must be a positive integer") from error
            if isinstance(value, (bool, np.bool_)) or integer_value < 1:
                raise ValueError(f"{name} must be a positive integer")
            dimensions[name] = integer_value
        global_width = dimensions["global_width"]
        local_width = dimensions["local_width"]
        heads = dimensions["heads"]
        if global_width % heads:
            raise ValueError("global_width must be divisible by heads")
        for name, probability in (
            ("global_dropout", global_dropout),
            ("local_dropout", local_dropout),
        ):
            if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1)")
        if spectral_gradient_projection not in ("ambient", "tangent"):
            raise ValueError(
                "spectral_gradient_projection must be 'ambient' or 'tangent'"
            )
        if spectral_gradient_projection == "tangent" and dimension != 3:
            raise ValueError("tangent projection requires a surface embedded in R^3")
        if spectral_gradient_injection and mode < 2:
            raise ValueError("spectral gradient injection requires mode >= 2")
        self.mode = int(mode)
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.function_channels = int(function_channels)
        self.tokens = self.mode**dimension
        self.global_width = int(global_width)
        self.local_width = int(local_width)
        self.global_dropout = float(global_dropout)
        self.layers = int(layers)
        self.local_fourier_frequencies = int(local_fourier_frequencies)
        self.coefficient_scale = 1.0 / math.sqrt(self.tokens)
        self.dimension = int(dimension)
        self.spectral_gradient_injection = bool(spectral_gradient_injection)
        self.normalize_inputs = bool(normalize_inputs)
        self.normalize_outputs = bool(normalize_outputs)
        # None preserves the original behavior for every other benchmark.
        self.normalize_global_moments = (
            self.normalize_inputs
            if normalize_global_moments is None
            else bool(normalize_global_moments)
        )
        self.spectral_gradient_projection = spectral_gradient_projection
        self.uses_mfe_gradients = self.spectral_gradient_injection
        self.requires_normals = bool(
            self.uses_mfe_gradients and self.spectral_gradient_projection == "tangent"
        )
        derivative = (
            normalized_legendre_derivative_matrix(self.mode)
            if self.uses_mfe_gradients
            else torch.empty((0, 0), dtype=torch.float32)
        )
        # This matrix is exactly derived from ``mode`` and is not checkpoint state.
        self.register_buffer(
            "_normalized_legendre_derivative",
            derivative,
            persistent=False,
        )

        coordinate_scale = (
            None
            if scaling.coordinate_scale is None
            else np.asarray(scaling.coordinate_scale, np.float32)
        )
        expected_coordinate_shape = (self.dimension,)
        if coordinate_scale is None and self.uses_mfe_gradients:
            raise ValueError(
                "scaling.coordinate_scale is required when spectral gradient "
                "injection is enabled"
            )
        if coordinate_scale is not None:
            if coordinate_scale.shape != expected_coordinate_shape:
                raise ValueError(
                    "scaling.coordinate_scale must have shape "
                    f"{expected_coordinate_shape}, got {coordinate_scale.shape}"
                )
            if not np.all(np.isfinite(coordinate_scale)) or np.any(
                coordinate_scale <= 0.0
            ):
                raise ValueError("scaling.coordinate_scale must be finite and positive")
        if coordinate_scale is None:
            coordinate_scale = np.ones(self.dimension, np.float32)
        # Geometry comes from the scaling artifact rather than checkpoint state.
        self.register_buffer(
            "_coordinate_scale",
            torch.from_numpy(np.ascontiguousarray(coordinate_scale)),
            persistent=False,
        )

        scaling_shapes = {
            "input_mean": (self.tokens, self.input_channels),
            "input_std": (self.tokens, self.input_channels),
            "function_mean": (self.function_channels,),
            "function_std": (self.function_channels,),
            "target_mean": (self.output_channels,),
            "target_std": (self.output_channels,),
        }
        buffer_shapes = {
            "input_mean": (1, self.tokens, self.input_channels),
            "input_std": (1, self.tokens, self.input_channels),
            "function_mean": (1, 1, self.function_channels),
            "function_std": (1, 1, self.function_channels),
            "target_mean": (1, 1, self.output_channels),
            "target_std": (1, 1, self.output_channels),
        }
        for name, expected_shape in scaling_shapes.items():
            array = np.asarray(getattr(scaling, name), np.float32)
            if array.shape != expected_shape:
                raise ValueError(
                    f"scaling.{name} must have shape {expected_shape}, "
                    f"got {array.shape}"
                )
            if name.endswith("_std") and (
                not np.all(np.isfinite(array)) or np.any(array <= 0.0)
            ):
                raise ValueError(f"scaling.{name} must be finite and positive")
            value = torch.from_numpy(array).reshape(buffer_shapes[name])
            self.register_buffer(name, value)

        self.global_input = torch.nn.Linear(self.input_channels, global_width)
        self.global_position = SpectralPositionEncoding(
            mode, global_width, global_position_frequencies, dimension
        )
        self.global_blocks = torch.nn.ModuleList(
            [
                torch.nn.TransformerEncoderLayer(
                    d_model=global_width,
                    nhead=heads,
                    dim_feedforward=feedforward,
                    dropout=self.global_dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.injection_norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(global_width) for _ in range(layers)]
        )
        self.injection_heads = torch.nn.ModuleList(
            [torch.nn.Linear(global_width, local_width) for _ in range(layers)]
        )
        self.global_output_norm = torch.nn.LayerNorm(global_width)
        self.global_output_head = torch.nn.Linear(global_width, self.output_channels)

        local_channels = (
            dimension
            + function_channels
            + 2 * dimension * local_fourier_frequencies
        )
        self.local_input = torch.nn.Linear(local_channels, local_width)
        self.local_layers = torch.nn.ModuleList(
            [torch.nn.Linear(local_width, local_width) for _ in range(layers)]
        )
        self.local_norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(local_width) for _ in range(layers)]
        )
        self.local_output_norm = torch.nn.LayerNorm(local_width)
        self.local_output_head = torch.nn.Linear(local_width, self.output_channels)
        self.activation = torch.nn.GELU()
        self.local_dropout = torch.nn.Dropout(p=local_dropout)
        # Injection heads already project coefficients to Local width.  This
        # single mixer is reused by every hidden layer and starts at zero correction.
        self.spectral_gradient_mixer = (
            torch.nn.Linear(dimension * local_width, local_width, bias=False)
            if self.spectral_gradient_injection
            else None
        )
        if self.spectral_gradient_mixer is not None:
            torch.nn.init.zeros_(self.spectral_gradient_mixer.weight)

        # Start from the training-set target mean rather than a random output.
        torch.nn.init.zeros_(self.global_output_head.weight)
        torch.nn.init.zeros_(self.global_output_head.bias)
        torch.nn.init.zeros_(self.local_output_head.weight)
        torch.nn.init.zeros_(self.local_output_head.bias)

    def parameter_groups(self) -> dict[str, int]:
        global_modules = (
            self.global_input,
            self.global_position,
            self.global_blocks,
            self.injection_norms,
            self.injection_heads,
            self.global_output_norm,
            self.global_output_head,
        )
        local_modules = (
            self.local_input,
            self.local_layers,
            self.local_norms,
            self.local_output_norm,
            self.local_output_head,
        )
        global_count = sum(
            p.numel() for module in global_modules for p in module.parameters()
        )
        local_count = sum(
            p.numel() for module in local_modules for p in module.parameters()
        )
        spectral_gradient_count = (
            sum(p.numel() for p in self.spectral_gradient_mixer.parameters())
            if self.spectral_gradient_mixer is not None
            else 0
        )
        return {
            "global": global_count,
            "local": local_count,
            "spectral_gradient": spectral_gradient_count,
            "total": global_count + local_count + spectral_gradient_count,
        }

    def encode_global(self, moments: torch.Tensor) -> GlobalState:
        if moments.ndim != 3 or moments.shape[1:] != (self.tokens, self.input_channels):
            raise ValueError(
                f"moments must have shape [B,{self.tokens},{self.input_channels}], "
                f"got {tuple(moments.shape)}"
            )
        raw_moments = moments.float()
        tokens = (
            (raw_moments - self.input_mean) / self.input_std
            if self.normalize_global_moments
            else raw_moments
        )
        hidden = self.global_input(tokens) + self.global_position()[None]
        injections = []
        for block, norm, head in zip(
            self.global_blocks, self.injection_norms, self.injection_heads
        ):
            hidden = block(hidden)
            injections.append(self.coefficient_scale * head(norm(hidden)))
        output = self.coefficient_scale * self.global_output_head(
            self.global_output_norm(hidden)
        )
        return GlobalState(
            tuple(injections),
            output,
        )

    def _coordinate_fourier(self, coordinates: torch.Tensor) -> torch.Tensor:
        features = []
        for level in range(self.local_fourier_frequencies):
            omega = math.pi * (2.0**level)
            features.extend(
                [torch.sin(omega * coordinates), torch.cos(omega * coordinates)]
            )
        return torch.cat(features, dim=-1) if features else coordinates[..., :0]

    def _resolve_gradient_physical_scales(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Expand the scaling artifact's affine half-span to ``[B,dimension]``."""

        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        return self._coordinate_scale.to(device=device, dtype=torch.float32)[
            None
        ].expand(batch_size, -1)

    def _differentiate_coefficients(
        self, coefficients: torch.Tensor, physical_scales: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Differentiate a 2-D/3-D tensor-product coefficient grid physically."""

        if coefficients.ndim != 3 or coefficients.shape[1] != self.tokens:
            raise ValueError(
                "coefficients must have shape [B,tokens,channels], "
                f"got {tuple(coefficients.shape)}"
            )
        channels = coefficients.shape[-1]
        expected = (coefficients.shape[0], self.tokens, channels)
        if physical_scales.shape != (coefficients.shape[0], self.dimension):
            raise ValueError(
                "physical_scales must have shape "
                f"[{coefficients.shape[0]},{self.dimension}]"
            )
        grid = coefficients.float().reshape(
            coefficients.shape[0],
            *((self.mode,) * self.dimension),
            channels,
        )
        derivative = self._normalized_legendre_derivative.float()
        gradients: list[torch.Tensor] = []
        for axis in range(self.dimension):
            tensor_axis = axis + 1  # Spatial modal axes follow the batch axis.
            moved = grid.movedim(tensor_axis, -2)
            differentiated = torch.einsum("ks,...sw->...kw", derivative, moved)
            differentiated = differentiated.movedim(-2, tensor_axis)
            gradients.append(
                differentiated.reshape(expected)
                / physical_scales[:, axis, None, None]
            )
        return tuple(gradients)

    def _unit_surface_normals(
        self, normals: torch.Tensor | None, coordinates: torch.Tensor
    ) -> torch.Tensor:
        """Validate and normalize explicit normals for a surface in ``R^3``."""

        if normals is None:
            raise ValueError(
                "tangent spectral gradients require explicit normals with "
                "shape [B,Q,3]"
            )
        if not isinstance(normals, torch.Tensor):
            raise TypeError("normals must be a torch.Tensor")
        if normals.device != coordinates.device:
            raise ValueError("normals and coordinates must be on the same device")
        if tuple(normals.shape) != tuple(coordinates.shape):
            raise ValueError(
                f"normals must match coordinates shape {tuple(coordinates.shape)}, "
                f"got {tuple(normals.shape)}"
            )
        if not normals.is_floating_point():
            raise TypeError("normals must be a floating-point tensor")
        normals = normals.float()
        if not bool(torch.isfinite(normals).all()):
            raise ValueError("normals contain NaN or Inf")
        lengths = torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
        if not bool((lengths > 1.0e-12).all()):
            raise ValueError("normals contain zero-length or near-zero vectors")
        return normals / lengths

    @staticmethod
    def _project_tangent_gradient(
        gradient: torch.Tensor, unit_normals: torch.Tensor
    ) -> torch.Tensor:
        """Compute ``(I - n n^T) gradient`` at every query point."""

        normal_component = torch.einsum(
            "bqd,bqdw->bqw", unit_normals, gradient
        )
        return gradient - unit_normals[..., None] * normal_component[:, :, None, :]

    def _map_spectral_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        """Map direction-major ``[..., dimension * width]`` gradients to width."""

        if self.spectral_gradient_mixer is None:
            raise RuntimeError("spectral gradient mixer is disabled")
        return self.spectral_gradient_mixer(gradient)

    def _ambient_gradient_correction(
        self,
        coefficients: torch.Tensor,
        physical_scales: torch.Tensor,
    ) -> torch.Tensor:
        gradient = self._differentiate_coefficients(coefficients, physical_scales)
        return self._map_spectral_gradient(torch.cat(gradient, dim=-1))

    def _tangent_injection_fused(
        self,
        coefficients: torch.Tensor,
        basis: torch.Tensor,
        unit_normals: torch.Tensor,
        physical_scales: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct ``h`` and its ambient gradient in one wide BMM.

        Concatenating the coefficient blocks is algebraically identical to four
        independent ``basis @ coefficients`` products.  It reduces repeated
        basis reads and CUDA launch overhead without changing the tangent
        projector, trainable mixer, parameters, or query set.
        """

        gradient_coefficients = self._differentiate_coefficients(
            coefficients, physical_scales
        )
        reconstructed = torch.bmm(
            basis,
            torch.cat(
                [coefficients.float(), *gradient_coefficients], dim=-1
            ),
        )
        pieces = reconstructed.split(self.local_width, dim=-1)
        injection = pieces[0]
        ambient_gradient = torch.stack(pieces[1:], dim=2)
        tangent_gradient = self._project_tangent_gradient(
            ambient_gradient, unit_normals
        )
        return injection + self._map_spectral_gradient(tangent_gradient.flatten(2))

    def _decode_chunk_prevalidated(
        self,
        state: GlobalState,
        coordinates: torch.Tensor,
        functions: torch.Tensor,
        *,
        physical_scales: torch.Tensor | None,
        unit_normals: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode inputs whose shapes and optional geometry are already valid."""

        basis = tensor_legendre_basis(coordinates, self.mode, self.dimension)
        normalized_functions = (
            (functions.float() - self.function_mean) / self.function_std
            if self.normalize_inputs
            else functions.float()
        )
        local_features = torch.cat(
            [
                coordinates.float(),
                normalized_functions,
                self._coordinate_fourier(coordinates.float()),
            ],
            dim=-1,
        )
        local = self.local_input(local_features.float())
        for linear, norm, coefficients in zip(
            self.local_layers,
            self.local_norms,
            state.injection_coefficients,
        ):
            if self.spectral_gradient_injection and self.requires_normals:
                assert (
                    physical_scales is not None
                    and unit_normals is not None
                )
                injection = self._tangent_injection_fused(
                    coefficients,
                    basis,
                    unit_normals,
                    physical_scales,
                )
            else:
                combined_coefficients = coefficients.float()
                if (
                    self.spectral_gradient_injection
                    and self.spectral_gradient_projection == "ambient"
                ):
                    assert physical_scales is not None
                    combined_coefficients = (
                        combined_coefficients
                        + self._ambient_gradient_correction(
                            coefficients,
                            physical_scales,
                        )
                    )
                injection = torch.bmm(basis, combined_coefficients.float())
            # Inject pointwise MFE features before GELU and optional dropout.
            local = self.local_dropout(
                self.activation(linear(norm(local)) + injection)
            )
        global_output = torch.bmm(basis, state.output_coefficients.float())
        local_output = self.local_output_head(self.local_output_norm(local))
        return (
            global_output + local_output,
            global_output,
            local_output,
        )

    def forward(
        self,
        moments: torch.Tensor,
        coordinates: torch.Tensor,
        functions: torch.Tensor,
        chunk_size: int,
        *,
        normals: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if (
            coordinates.ndim != 3
            or coordinates.shape[-1] != self.dimension
            or coordinates.shape[1] < 1
        ):
            raise ValueError(
                f"coordinates must have non-empty shape [B,Q,{self.dimension}], "
                f"got {tuple(coordinates.shape)}"
            )
        expected_functions = (
            coordinates.shape[0],
            coordinates.shape[1],
            self.function_channels,
        )
        if functions.shape != expected_functions:
            raise ValueError(
                f"functions must have shape {expected_functions}, "
                f"got {tuple(functions.shape)}"
            )
        if moments.ndim != 3 or moments.shape[0] != coordinates.shape[0]:
            raise ValueError("moments must be rank 3 and match the coordinates batch size")
        unit_normals = (
            self._unit_surface_normals(normals, coordinates)
            if self.requires_normals
            else None
        )
        physical_scales = (
            self._resolve_gradient_physical_scales(
                coordinates.shape[0], coordinates.device
            )
            if self.uses_mfe_gradients
            else None
        )
        state = self.encode_global(moments)
        outputs = [[], [], []]
        for start in range(0, coordinates.shape[1], chunk_size):
            stop = min(start + chunk_size, coordinates.shape[1])
            result = self._decode_chunk_prevalidated(
                state,
                coordinates[:, start:stop],
                functions[:, start:stop],
                unit_normals=(
                    unit_normals[:, start:stop] if unit_normals is not None else None
                ),
                physical_scales=physical_scales,
            )
            for destination, value in zip(outputs, result):
                destination.append(value)
        return (
            torch.cat(outputs[0], dim=1),
            torch.cat(outputs[1], dim=1),
            torch.cat(outputs[2], dim=1),
        )

    def denormalize_targets(self, target: torch.Tensor) -> torch.Tensor:
        if not self.normalize_outputs:
            return target
        return target * self.target_std + self.target_mean
