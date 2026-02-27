# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Operator support checker for TensorRT backend."""

from typing import Set

import torch
from torch.fx.passes.operator_support import OperatorSupportBase


class TensorRTOperatorSupport(OperatorSupportBase):
    """Determines which nodes can be delegated to TensorRT.

    The partitioner uses this class to decide which operations can run on TensorRT.
    An operation is supported if:
    1. It's a "call_function" node (actual computation, not control flow)
    2. It's either a SPECIAL_OP (glue operations) or in SUPPORTED_OPS
    3. Its output dtype is in SUPPORTED_DTYPES
    """

    # Operations that have TensorRT converters (sorted alphabetically).
    SUPPORTED_OPS: Set[str] = {
        "_log_softmax.default",
        "_native_batch_norm_legit.default",
        "_native_batch_norm_legit_no_training.default",
        "_scaled_dot_product_efficient_attention.default",
        "_scaled_dot_product_flash_attention.default",
        "_softmax.default",
        "_unsafe_view.default",
        "adaptive_avg_pool2d.default",
        "add.Tensor",
        "add_.Tensor",
        "addmm.default",
        "alias_copy.default",
        "any.dim",
        "arange.start_step",
        "avg_pool2d.default",
        "batch_norm.default",
        "bmm.default",
        "cat.default",
        "chunk.default",
        "clamp.default",
        "clamp_max.default",
        "clamp_min.default",
        "clone.default",
        "constant_pad_nd.default",
        "contiguous.default",
        "copy.default",
        "conv2d.default",
        "convolution.default",
        "div.Tensor",
        "div.Tensor_mode",
        "dropout.default",
        "dropout_.default",
        "embedding.default",
        "eq.Scalar",
        "expand.default",
        "expand_copy.default",
        "flatten.using_ints",
        "full.default",
        "full_like.default",
        "ge.Scalar",
        "gelu.default",
        "gt.Scalar",
        "hardsigmoid.default",
        "hardswish.default",
        "hardswish_.default",
        "hardtanh.default",
        "hardtanh_.default",
        "index.Tensor",
        "layer_norm.default",
        "le.Scalar",
        "linear.default",
        "log_softmax.int",
        "logical_not.default",
        "lt.Scalar",
        "max_pool2d.default",
        "max_pool2d_with_indices.default",
        "mean.dim",
        "mm.default",
        "mul.Scalar",
        "mul.Tensor",
        "mul_.Tensor",
        "native_layer_norm.default",
        "ne.Scalar",
        "ones_like.default",
        "permute.default",
        "permute_copy.default",
        "pixel_shuffle.default",
        "relu.default",
        "relu_.default",
        "repeat.default",
        "reshape.default",
        "rsub.Scalar",
        "scalar_tensor.default",
        "scaled_dot_product_attention.default",
        "select.int",
        "select_copy.int",
        "sigmoid.default",
        "silu.default",
        "slice.Tensor",
        "slice_copy.Tensor",
        "softmax.int",
        "split.Tensor",
        "split_with_sizes.default",
        "split_with_sizes_copy.default",
        "squeeze.dim",
        "squeeze.dims",
        "squeeze_copy.dim",
        "squeeze_copy.dims",
        "stack.default",
        "sub.Tensor",
        "tanh.default",
        "transpose.int",
        "unflatten.int",
        "unsqueeze.default",
        "unsqueeze_copy.default",
        "upsample_bilinear2d.vec",
        "upsample_nearest2d.vec",
        "view.default",
        "view_copy.default",
        "where.ScalarSelf",
        "where.self",
        "zeros_like.default",
    }

    # Glue operations that don't compute but are needed to keep partitions connected.
    # These are always supported regardless of dtype.
    #
    # - getitem: Unpacks tuple outputs (e.g., from max_pool2d_with_indices).
    #   Without this, every tuple unpacking would break the partition.
    #
    # - dim_order_ops: ExecuTorch Edge dialect inserts these for memory layout
    #   conversions. Without these, layout ops would fragment the partition.
    SPECIAL_OPS: Set[str] = {
        "_clone_dim_order.default",
        "_operator.getitem",
        "_to_dim_order_copy.default",
        "dim_order_ops._clone_dim_order.default",
        "dim_order_ops._to_dim_order_copy.default",
        "exir_ops.edge.dim_order_ops._clone_dim_order.default",
        "exir_ops.edge.dim_order_ops._to_dim_order_copy.default",
        "getitem",
        "operator.getitem",
    }

    SUPPORTED_DTYPES: Set[torch.dtype] = {
        torch.bool,
        torch.bfloat16,
        torch.float32,
        torch.int64,
    }

    def is_node_supported(self, submodules: dict, node: torch.fx.Node) -> bool:
        if node.op != "call_function":
            return False

        op_name = self._get_op_name(node)
        target_name = self._remove_namespace(op_name)

        # Special ops are always supported (they're just glue)
        if op_name in self.SPECIAL_OPS or target_name in self.SPECIAL_OPS:
            return True

        # Check if we have a converter for this op
        if target_name not in self.SUPPORTED_OPS:
            return False

        # Check dtype compatibility
        if not self._is_dtype_supported(node):
            return False

        return True

    def _get_op_name(self, node: torch.fx.Node) -> str:
        """Extract operation name from node target.

        Returns format like "aten.add.Tensor" or "getitem".
        """
        target = node.target

        # torch.ops and Edge dialect ops have _schema
        if hasattr(target, "_schema"):
            schema = target._schema
            base_name = schema.name.replace("::", ".")
            # Note: For the "default" overload, overload_name is an empty string "",
            # so we need to check for that and use "default" as the overload name.
            if hasattr(schema, "overload_name"):
                overload_name = schema.overload_name
                if overload_name:
                    return f"{base_name}.{overload_name}"
                else:
                    # Empty overload_name means "default" overload
                    return f"{base_name}.default"
            return base_name

        # Callable with module info (e.g., operator.getitem)
        if hasattr(target, "__module__") and hasattr(target, "__name__"):
            module = target.__module__
            name = target.__name__
            if "aten" in module:
                return f"aten.{name}"
            return name

        # Fallback
        if hasattr(target, "__name__"):
            return target.__name__
        if hasattr(target, "name"):
            return target.name()
        return str(target)

    def _remove_namespace(self, op_name: str) -> str:
        """Remove dialect namespace prefix.

        "aten.add.Tensor" -> "add.Tensor"
        "add.Tensor" -> "add.Tensor"
        """
        parts = op_name.split(".")
        if len(parts) > 2:
            parts.pop(0)
        return ".".join(parts)

    def _is_dtype_supported(self, node: torch.fx.Node) -> bool:
        """Check if output dtype is supported.

        For tuple outputs (e.g., max_pool2d_with_indices returns (values, indices)),
        we allow int64 index tensors since we don't use them in converters anyway.
        """
        if "val" not in node.meta:
            return True

        val = node.meta["val"]

        if isinstance(val, torch.Tensor):
            return val.dtype in self.SUPPORTED_DTYPES

        if isinstance(val, (list, tuple)):
            for v in val:
                if isinstance(v, torch.Tensor):
                    # Allow int64 for index tensors
                    if v.dtype == torch.int64:
                        continue
                    if v.dtype not in self.SUPPORTED_DTYPES:
                        return False

        return True
