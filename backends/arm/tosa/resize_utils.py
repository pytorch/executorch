# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence

import torch

from executorch.backends.arm.tosa.specification import TosaSpecification

_MAX_RESIZE_DIMENSION = 16384
_MAX_RESIZE_SCALE_NUMERATOR = 1 << 11
_MAX_SCALE = 2048
_MAX_SCALE_LEVEL_8K = 256
_INT16_MIN = -(2**15)
_INT16_MAX = 2**15 - 1


def _as_concrete_ints(values: Sequence[int | torch.SymInt]) -> list[int] | None:
    if all(isinstance(value, int) for value in values):
        return [int(value) for value in values]
    return None


def _concrete_int_values(values: Sequence[int | torch.SymInt]) -> list[int]:
    return [int(value) for value in values if isinstance(value, int)]


def _first_outside_range(
    values: Sequence[int], min_value: int, max_value: int
) -> int | None:
    return next(
        (value for value in values if value < min_value or value > max_value), None
    )


def _max_scale(tosa_spec: TosaSpecification) -> int:
    return _MAX_SCALE_LEVEL_8K if getattr(tosa_spec, "level_8k", False) else _MAX_SCALE


def _validate_dimensions(
    input_hw: Sequence[int | torch.SymInt],
    output_hw: Sequence[int | torch.SymInt] | None,
) -> str | None:
    concrete_dimensions: list[int] = []
    input_hw_ints = _as_concrete_ints(input_hw)
    output_hw_ints = _as_concrete_ints(output_hw) if output_hw is not None else None
    if input_hw_ints is not None:
        concrete_dimensions.extend(input_hw_ints)
    if output_hw_ints is not None:
        concrete_dimensions.extend(output_hw_ints)

    invalid_dimension = next(
        (
            dimension
            for dimension in concrete_dimensions
            if dimension >= _MAX_RESIZE_DIMENSION
        ),
        None,
    )
    if invalid_dimension is not None:
        return (
            "RESIZE dimensions must be less than "
            f"{_MAX_RESIZE_DIMENSION}; got {invalid_dimension}"
        )
    return None


def get_tosa_resize_output_hw_validation_error(
    output_hw: Sequence[int | torch.SymInt] | None,
) -> str | None:
    if output_hw is None:
        return None

    output_hw_ints = _as_concrete_ints(output_hw)
    if output_hw_ints is None:
        return None

    invalid_dimension = next(
        (dimension for dimension in output_hw_ints if dimension <= 0), None
    )
    if invalid_dimension is not None:
        return f"RESIZE output dimensions must be positive; got {invalid_dimension}"

    return _validate_dimensions((), output_hw)


def _validate_scale(
    scale: Sequence[int | torch.SymInt],
    tosa_spec: TosaSpecification,
) -> str | None:
    invalid_scale = _first_outside_range(
        _concrete_int_values(scale), _INT16_MIN, _INT16_MAX
    )
    if invalid_scale is not None:
        return (
            "RESIZE scale must be in int16 range "
            f"[{_INT16_MIN}, {_INT16_MAX}]; got {invalid_scale}"
        )

    scale_ints = _as_concrete_ints(scale)
    if scale_ints is None:
        return None

    scale_y_n, scale_y_d, scale_x_n, scale_x_d = scale_ints
    if min(scale_y_n, scale_y_d, scale_x_n, scale_x_d) <= 0:
        return f"RESIZE scale values must be positive; got {scale_ints}"

    max_scale = _max_scale(tosa_spec)
    if scale_y_n > max_scale * scale_y_d or scale_x_n > max_scale * scale_x_d:
        return (
            f"RESIZE scale ratio must be <= MAX_SCALE ({max_scale}); "
            f"got y={scale_y_n}/{scale_y_d}, x={scale_x_n}/{scale_x_d}"
        )

    if (
        scale_y_n > _MAX_RESIZE_SCALE_NUMERATOR
        or scale_x_n > _MAX_RESIZE_SCALE_NUMERATOR
    ):
        return (
            "RESIZE scale numerator must be <= "
            f"{_MAX_RESIZE_SCALE_NUMERATOR}; got y={scale_y_n}, x={scale_x_n}"
        )

    # The scale values are already in the doubled rational representation that
    # TOSA RESIZE lowering emits, so the lower-bound downscale rule can be
    # checked directly against them.
    if scale_y_d >= 16 * scale_y_n or scale_x_d >= 16 * scale_x_n:
        return (
            "RESIZE downscale must be strictly greater than 1/16; "
            f"got y={scale_y_n}/{scale_y_d}, x={scale_x_n}/{scale_x_d}"
        )
    return None


def _validate_offset(
    offset: Sequence[int | torch.SymInt],
    scale_ints: list[int],
) -> str | None:
    offset_ints = _as_concrete_ints(offset)
    if offset_ints is None:
        return None

    scale_y_n, _, scale_x_n, _ = scale_ints
    offset_y, offset_x = offset_ints
    if offset_y < -scale_y_n or offset_y >= 16 * scale_y_n:
        return (
            f"RESIZE offset_y must be in [{-scale_y_n}, {16 * scale_y_n}); "
            f"got {offset_y}"
        )
    if offset_x < -scale_x_n or offset_x >= 16 * scale_x_n:
        return (
            f"RESIZE offset_x must be in [{-scale_x_n}, {16 * scale_x_n}); "
            f"got {offset_x}"
        )
    return None


def _validate_border(
    border: Sequence[int | torch.SymInt],
    scale_ints: list[int],
) -> str | None:
    invalid_border = _first_outside_range(
        _concrete_int_values(border), _INT16_MIN, _INT16_MAX
    )
    if invalid_border is not None:
        return (
            "RESIZE border must be in int16 range "
            f"[{_INT16_MIN}, {_INT16_MAX}]; got {invalid_border}"
        )

    border_ints = _as_concrete_ints(border)
    if border_ints is None:
        return None

    scale_y_n, _, scale_x_n, _ = scale_ints
    border_y, border_x = border_ints
    if border_y < -16 * scale_y_n or border_y >= scale_y_n:
        return (
            f"RESIZE border_y must be in [{-16 * scale_y_n}, {scale_y_n}); "
            f"got {border_y}"
        )
    if border_x < -16 * scale_x_n or border_x >= scale_x_n:
        return (
            f"RESIZE border_x must be in [{-16 * scale_x_n}, {scale_x_n}); "
            f"got {border_x}"
        )
    return None


def _validate_output_shape(
    input_hw: Sequence[int | torch.SymInt],
    output_hw: Sequence[int | torch.SymInt] | None,
    scale: Sequence[int | torch.SymInt],
    offset: Sequence[int | torch.SymInt],
    border: Sequence[int | torch.SymInt],
) -> str | None:
    if output_hw is None:
        return None

    output_hw_ints = _as_concrete_ints(output_hw)
    expected_output_hw = calculate_tosa_resize_output_hw(
        input_hw, scale, offset, border
    )
    if (
        output_hw_ints is not None
        and expected_output_hw is not None
        and tuple(output_hw_ints) != expected_output_hw
    ):
        return (
            "RESIZE output shape is inconsistent with input and parameters; "
            f"expected {expected_output_hw}, got {tuple(output_hw_ints)}"
        )
    return None


def calculate_tosa_resize_output_hw(
    input_hw: Sequence[int | torch.SymInt],
    scale: Sequence[int | torch.SymInt],
    offset: Sequence[int | torch.SymInt],
    border: Sequence[int | torch.SymInt],
) -> tuple[int, int] | None:
    input_hw_ints = _as_concrete_ints(input_hw)
    scale_ints = _as_concrete_ints(scale)
    offset_ints = _as_concrete_ints(offset)
    border_ints = _as_concrete_ints(border)
    if (
        input_hw_ints is None
        or scale_ints is None
        or offset_ints is None
        or border_ints is None
    ):
        return None

    input_h, input_w = input_hw_ints
    scale_y_n, scale_y_d, scale_x_n, scale_x_d = scale_ints
    offset_y, offset_x = offset_ints
    border_y, border_x = border_ints

    # RESIZE first upscales the input by an integer value to "upscale space".
    # Offset and border are encoded in that space, then RESIZE completes by
    # downscaling with another integer value, approximating multiplication by a
    # fraction.
    return (
        ((input_h - 1) * scale_y_n - offset_y + border_y) // scale_y_d + 1,
        ((input_w - 1) * scale_x_n - offset_x + border_x) // scale_x_d + 1,
    )


def get_tosa_resize_validation_error(
    *,
    input_hw: Sequence[int | torch.SymInt],
    output_hw: Sequence[int | torch.SymInt] | None,
    scale: Sequence[int | torch.SymInt],
    offset: Sequence[int | torch.SymInt],
    border: Sequence[int | torch.SymInt],
    tosa_spec: TosaSpecification,
) -> str | None:
    scale_ints = _as_concrete_ints(scale)

    validation_error = _validate_dimensions(input_hw, output_hw)
    if validation_error is not None:
        return validation_error
    validation_error = _validate_scale(scale, tosa_spec)
    if validation_error is not None:
        return validation_error
    if scale_ints is None:
        return None

    for validation_error in (
        _validate_offset(offset, scale_ints),
        _validate_border(border, scale_ints),
        _validate_output_shape(input_hw, output_hw, scale, offset, border),
    ):
        if validation_error is not None:
            return validation_error
    return None
