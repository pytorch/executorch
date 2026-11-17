# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from executorch.exir.tensor import dim_order_from_stride, without_zero_strides
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind

logger: logging.Logger = logging.getLogger(__name__)


def _should_transform(tensor: torch.Tensor) -> bool:
    """
    Returns whether the given tensor should be transformed by the pass to
    contiguous dim order.
    """
    return not tensor.is_contiguous() and not tensor.is_contiguous(
        memory_format=torch.channels_last
    )


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Returns the tensor with default or channels-last dim order and no zero
    strides, copying it only if needed.
    """
    if _should_transform(tensor):
        return tensor.contiguous()
    tensor = without_zero_strides(tensor)
    ndim = tensor.dim()
    dim_order = dim_order_from_stride(tensor.stride(), tuple(tensor.shape))
    if dim_order == tuple(range(ndim)) or (
        ndim in (4, 5) and dim_order == (0, *range(2, ndim), 1)
    ):
        return tensor
    # PyTorch ignores the stride of a size-1 dim when it checks contiguity, so
    # sizes (8, 1, 4) with strides (4, 32, 1) are contiguous, yet give dim
    # order (1, 0, 2).
    memory_format = (
        torch.contiguous_format if tensor.is_contiguous() else torch.channels_last
    )
    return tensor.clone(memory_format=memory_format)


def _update_placeholder_meta(
    exported_program: ExportedProgram,
    target: str,
    kind: InputKind,
):
    input_spec = next(
        (
            spec
            for spec in exported_program.graph_signature.input_specs
            if spec.kind == kind and spec.target == target
        ),
        None,
    )
    if input_spec is None:
        raise RuntimeError(f"Missing input spec for lifted tensor {target}.")

    placeholder_node = next(
        (
            n
            for n in exported_program.graph.nodes
            if n.op == "placeholder" and n.name == input_spec.arg.name
        ),
        None,
    )
    if placeholder_node is None:
        raise RuntimeError(f"Missing placeholder for {input_spec.arg.name}.")

    val = placeholder_node.meta["val"]
    placeholder_node.meta["val"] = _normalize(val)
    # Fixing a stride that PyTorch ignores does not change the layout.
    if not _should_transform(val):
        return
    logger.warning(
        "%s '%s' has strides %s, which are neither contiguous nor channels-last. "
        "to_edge stores it contiguous, so partitioners and delegates see the "
        "contiguous layout, not the strides in the model. Make it contiguous "
        "where it is created to avoid this change.",
        kind.name.replace("_", " ").capitalize(),
        target,
        tuple(val.stride()),
    )


def convert_constant_dim_order_pass(
    exported_program: ExportedProgram,
) -> ExportedProgram:
    """
    Normalize lifted tensors to default or channels-last dim order.

    Tensor constants, buffers and parameters with other dim orders or striding
    are converted to contiguous tensors. Those that PyTorch calls contiguous or
    channels-last, but whose strides on size-1 dims give another dim order, are
    copied into that layout, and zero strides are removed. User inputs with a
    zero stride become contiguous, or stay channels-last; other inputs are not
    changed. This pass acts in-place on the unlifted exported program.

    Args:
        exported_program: The ExportedProgram to transform.

    Returns:
        The modified ExportedProgram with normalized lifted tensor dim order.
    """

    non_persistent_buffer_names = set(
        exported_program.graph_signature.non_persistent_buffers
    )

    for key, const in exported_program.constants.items():
        if not isinstance(const, torch.Tensor):
            continue
        normalized = _normalize(const)
        if normalized is not const:
            exported_program.constants[key] = normalized

            # Also update the corresponding placeholder node meta value. This doesn't
            # get automatically updated during retracing as ExportPass uses the placeholder
            # meta as the source of truth. TODO(?)
            # Non-persistent buffers are stored as constants but remain buffer inputs.
            kind = (
                InputKind.BUFFER
                if key in non_persistent_buffer_names
                else InputKind.CONSTANT_TENSOR
            )
            _update_placeholder_meta(exported_program, key, kind)

    for key, param in exported_program.named_parameters():
        normalized = _normalize(param)
        if normalized is not param:
            exported_program.state_dict[key] = torch.nn.Parameter(
                normalized, requires_grad=param.requires_grad
            )
            _update_placeholder_meta(exported_program, key, InputKind.PARAMETER)

    # Convert buffers.
    for key, buffer in exported_program.named_buffers():
        if key in non_persistent_buffer_names or not isinstance(buffer, torch.Tensor):
            continue
        normalized = _normalize(buffer)
        if normalized is not buffer:
            # Persistent buffers are stored in the state dict.
            exported_program.state_dict[key] = normalized

            _update_placeholder_meta(exported_program, key, InputKind.BUFFER)

    # An expanded example input has a zero stride, which ExecuTorch cannot
    # store either. Constants are already normalized, so this only changes inputs.
    for node in exported_program.graph.nodes:
        if node.op == "placeholder" and isinstance(node.meta.get("val"), torch.Tensor):
            node.meta["val"] = without_zero_strides(node.meta["val"])

    return exported_program
