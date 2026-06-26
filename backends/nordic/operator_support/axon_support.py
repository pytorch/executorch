# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""AXON NPU operator support checks.

Defines which operations the AXON NPU can accelerate and their constraints.
Used by the partitioner to decide what gets delegated to AXON vs CPU.

Operations fall into three categories:

1. **AXON-accelerated**: Run entirely on the AXON NPU hardware.
2. **Fused activations**: Fused into the preceding compute layer at zero cost.
3. **Op extensions**: Hybrid AXON+CPU — the preceding layer runs on AXON with
   higher-precision output (INT16 q3.12 or INT32 q11.12), then a CPU callback
   completes the non-linear function.

AXON hardware constraints (from Nordic documentation):

- Tensors stored as ``int8_t tensor[channels][height][width]`` (CHW layout).
  The ARM TOSA pass pipeline handles NCHW → NHWC, and Nordic's compiler
  handles NHWC → CHW internally.
- Maximum tensor dimensions: 1024 height, 1024 width, 1024 channels.
- Maximum 2 inputs per node.
- Output rows aligned to 32-bit boundary (compiler handles padding).
- Most reshape operations are transparent to AXON (no data movement).
  Non-transparent reshapes fall back to CPU.
"""

from executorch.backends.nordic.axon.compile_spec import (
    AXON_MAX_CONV2D_FILTER,
    AXON_MAX_CONV_STRIDE,
    AXON_MAX_FC_INPUT,
    AXON_MAX_FC_OUTPUT,
    AXON_MAX_INPUTS_PER_NODE,
    AXON_MAX_POOL_FILTER,
    AXON_MAX_TENSOR_DIM,
)

# ── Category 1: AXON-accelerated operations ──────────────────────
# These run entirely on the AXON NPU hardware via command buffers.
AXON_SUPPORTED_OPS = {
    "fully_connected",
    "conv2d",
    "depthwise_conv2d",
    "pointwise_conv2d",
    "avg_pool2d",
    "max_pool2d",
    "add",
    "mul",
    "mean",
    "concatenate",
    "strided_slice",
    "channel_padding",
}

# ── Category 2: Fused activations (zero overhead) ────────────────
# These are fused into the preceding compute layer's command buffer.
# They don't create separate AXON layers — the activation is encoded
# in the preceding layer's configuration word.
AXON_FUSED_ACTIVATIONS = {"relu", "relu6", "leaky_relu"}

# ── Category 3: Op extensions (AXON+CPU hybrid) ──────────────────
# The preceding AXON layer outputs higher-precision data, then a CPU
# callback function completes the non-linear computation.
#   sigmoid (101): preceding layer → INT16 q3.12 → CPU expf sigmoid
#   tanh    (102): preceding layer → INT16 q3.12 → CPU expf tanh
#   softmax (100): preceding layer → INT32 q11.12 → CPU softmax
AXON_OP_EXTENSIONS = {"sigmoid", "tanh", "softmax"}

# ── Operations that fall back to ExecuTorch portable CPU kernels ──
# These are not accelerated by AXON in any way.
# Note: most reshape operations are transparent to AXON (the compiler
# handles them without data movement). Only non-transparent reshapes
# that require actual memory reorganization fall back to CPU.
AXON_CPU_ONLY_OPS = {
    "reshape",
}


# ── Global constraint checks ─────────────────────────────────────
# These apply to ALL AXON operations regardless of type.

def check_tensor_dimensions(
    height: int, width: int, channels: int
) -> tuple[bool, str]:
    """Check if tensor dimensions fit AXON global constraints.

    AXON supports max 1024 for height, width, and channels.
    """
    if height > AXON_MAX_TENSOR_DIM:
        return False, f"height {height} > max {AXON_MAX_TENSOR_DIM}"
    if width > AXON_MAX_TENSOR_DIM:
        return False, f"width {width} > max {AXON_MAX_TENSOR_DIM}"
    if channels > AXON_MAX_TENSOR_DIM:
        return False, f"channels {channels} > max {AXON_MAX_TENSOR_DIM}"
    return True, ""


def check_input_count(num_inputs: int) -> tuple[bool, str]:
    """Check if the number of inputs fits AXON constraints.

    AXON allows a maximum of 2 inputs per node.
    """
    if num_inputs > AXON_MAX_INPUTS_PER_NODE:
        return False, f"inputs {num_inputs} > max {AXON_MAX_INPUTS_PER_NODE}"
    return True, ""


# ── Per-operation constraint checks ──────────────────────────────

def check_fully_connected(input_size: int, output_size: int) -> tuple[bool, str]:
    """Check if a fully connected layer fits AXON constraints.

    FC max input/output: 2048 elements each.
    """
    if input_size > AXON_MAX_FC_INPUT:
        return False, f"FC input size {input_size} > max {AXON_MAX_FC_INPUT}"
    if output_size > AXON_MAX_FC_OUTPUT:
        return False, f"FC output size {output_size} > max {AXON_MAX_FC_OUTPUT}"
    return True, ""


def check_conv2d(
    filter_h: int, filter_w: int, stride_h: int, stride_w: int, channels: int
) -> tuple[bool, str]:
    """Check if a conv2d layer fits AXON constraints.

    Conv2D max filter: 16x16, max stride: 31, max channels: 1024.
    """
    if filter_h > AXON_MAX_CONV2D_FILTER or filter_w > AXON_MAX_CONV2D_FILTER:
        return False, f"Conv2d filter {filter_h}x{filter_w} > max {AXON_MAX_CONV2D_FILTER}"
    if stride_h > AXON_MAX_CONV_STRIDE or stride_w > AXON_MAX_CONV_STRIDE:
        return False, f"Conv2d stride > max {AXON_MAX_CONV_STRIDE}"
    if channels > AXON_MAX_TENSOR_DIM:
        return False, f"Conv2d channels {channels} > max {AXON_MAX_TENSOR_DIM}"
    return True, ""


def check_pooling(filter_h: int, filter_w: int) -> tuple[bool, str]:
    """Check if a pooling layer fits AXON constraints.

    Pooling max filter: 32x32.
    """
    if filter_h > AXON_MAX_POOL_FILTER or filter_w > AXON_MAX_POOL_FILTER:
        return False, f"Pool filter {filter_h}x{filter_w} > max {AXON_MAX_POOL_FILTER}"
    return True, ""
