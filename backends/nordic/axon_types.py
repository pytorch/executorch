# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""AXON NPU type definitions and enums.

These mirror the C types in ``nrf_axon_nn_compiler_types.h`` from
Nordic's sdk-edge-ai. Used by both the compiler bridge
(``axon_compiler.py``) and the binary builder (``axon_binary.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class AxonOp(IntEnum):
    """AXON operation codes (from nrf_axon_nn_op_e)."""
    FULLY_CONNECTED = 0
    CONV2D = 1
    DEPTHWISE_CONV2D = 2
    POINTWISE_CONV2D = 3
    AVERAGE_POOLING = 4
    MAX_POOLING = 5
    ADD2 = 6
    CHANNEL_PADDING = 7
    PERSISTENT_VAR = 8
    CONCATENATE = 9
    STRIDED_SLICE = 10
    MULTIPLY = 11
    MEAN = 12
    # Op extensions — CPU+AXON hybrid ops
    SOFTMAX = 100
    SIGMOID = 101
    TANH = 102


class AxonActivation(IntEnum):
    """AXON activation function codes (from nrf_axon_nn_activation_function_e)."""
    DISABLED = 0
    RELU = 1
    PREPARE_SOFTMAX = 2  # Preceding layer outputs q11.12 INT32 for softmax
    LEAKY_RELU = 3


class AxonByteWidth(IntEnum):
    """AXON byte width codes (from nrf_axon_nn_byte_width_e)."""
    INT8 = 1
    INT16 = 2
    INT32 = 4


@dataclass
class AxonDimensions:
    """Mirrors nrf_axon_nn_compiler_model_layer_dimensions_s."""
    height: int = 1
    width: int = 1
    channel_cnt: int = 1
    byte_width: int = AxonByteWidth.INT8


@dataclass
class ActivationQuantInfo:
    """Quantization parameters for an activation op (sigmoid/tanh/softmax).

    Extracted from the PyTorch FX graph BEFORE TOSA lowering, because
    TOSA bakes scales into TABLE ops and the original values are lost.
    """
    op_type: str  # "sigmoid", "tanh", or "softmax"
    input_scale: float
    input_zp: int
    output_scale: float
    output_zp: int


@dataclass
class AxonLayer:
    """An AXON layer descriptor, mirroring nrf_axon_nn_model_layer_desc_s.

    Each layer represents one operation in the AXON command buffer.
    The ``filter_data``, ``bias_data``, ``multiplier_data``, and
    ``shift_data`` fields contain raw bytes that are packed into the
    constants section of the AXON intermediate binary.
    """
    # Input layer indices (-1 = graph input, 0+ = preceding layer output)
    input_ids: list[int] = field(default_factory=lambda: [-1])
    operation: int = AxonOp.FULLY_CONNECTED
    concatenate_axis: int = 0
    input_dimensions: list[AxonDimensions] = field(default_factory=list)
    filter_dimensions: AxonDimensions = field(default_factory=AxonDimensions)
    output_dimensions: AxonDimensions = field(default_factory=AxonDimensions)
    stride_x: int = 1
    stride_y: int = 1
    dilation_x: int = 1
    dilation_y: int = 1
    input_zero_point: int = 0
    output_zero_point: int = 0
    activation: int = AxonActivation.DISABLED
    pad_left: int = 0
    pad_right: int = 0
    pad_top: int = 0
    pad_bottom: int = 0
    # Constant data (raw bytes, packed into the binary's constants section)
    filter_data: bytes = b""       # INT8 weight data
    bias_data: bytes = b""         # INT32 bias_prime values
    multiplier_data: bytes = b""   # INT32 output rescale multipliers
    shift_data: bytes = b""        # INT8 rescale shifts
    scale_shift_cnt: int = 0       # Number of shifts (1 = shared, N = per-channel)
    cpu_op_attributes: bytes = b""  # Op extension parameters (softmax/sigmoid/tanh)
