# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Operator support checker for TensorRT backend."""

import logging
from typing import Set

import torch

logger = logging.getLogger(__name__)
from torch.fx.passes.operator_support import OperatorSupportBase

from executorch.backends.nvidia.tensorrt import converter_registry


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
        "abs.default",
        "adaptive_avg_pool2d.default",
        "add",  # scalar add (operator.add, shape arithmetic)
        "add.default",  # aten scalar add
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
        "ceil.default",
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
        "cos.default",
        "div.Tensor",
        "div.Tensor_mode",
        "dropout.default",
        "dropout_.default",
        "embedding.default",
        "eq.Scalar",
        "erf.default",
        "exp.default",
        "expand.default",
        "expand_copy.default",
        "flatten.using_ints",
        "floordiv",  # scalar floordiv (operator.floordiv, shape arithmetic)
        "floordiv.default",
        "floor.default",
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
        "log.default",
        "log_softmax.int",
        "logical_not.default",
        "_to_copy.default",
        "argmax.default",
        "bitwise_not.default",
        "logical_and.default",
        "lt.Scalar",
        "lt.Tensor",
        "max_pool2d.default",
        "max_pool2d_with_indices.default",
        "mean.dim",
        "mm.default",
        "mul",  # scalar mul (operator.mul, shape arithmetic)
        "mul.default",
        "mul.Scalar",
        "mul.Tensor",
        "mul_.Tensor",
        "native_layer_norm.default",
        "ne.Scalar",
        "neg.default",
        "ones_like.default",
        "permute.default",
        "permute_copy.default",
        "pixel_shuffle.default",
        "pow.Tensor_Scalar",
        "pow.Tensor_Tensor",
        "reciprocal.default",
        "relu.default",
        "relu_.default",
        "repeat.default",
        "reshape.default",
        "rsqrt.default",
        "rsub.Scalar",
        "scalar_tensor.default",
        "scaled_dot_product_attention.default",
        "select.int",
        "select_copy.int",
        "sigmoid.default",
        "silu.default",
        "sin.default",
        "slice.Tensor",
        "slice_copy.Tensor",
        "softmax.int",
        "split.Tensor",
        "split_with_sizes.default",
        "split_with_sizes_copy.default",
        "sqrt.default",
        "squeeze.dim",
        "squeeze.dims",
        "squeeze_copy.dim",
        "squeeze_copy.dims",
        "stack.default",
        "sub",  # scalar sub (operator.sub, shape arithmetic)
        "sub.default",
        "sub.Tensor",
        "sum.dim_IntList",
        "sym_size.int",  # query tensor dimension (shape arithmetic)
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
        torch.float16,
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
            logger.debug(f"[TRT partitioner] REJECTED {node.name} ({target_name}): no converter")
            return False

        # Reject nodes that produce scalar (non-tensor) outputs IF they
        # are graph outputs. Intermediate scalar nodes (like sym_size,
        # operator.add for shape arithmetic) are fine — TRT handles
        # them as shape tensors internally. Only graph outputs must be
        # tensors for the ExecuTorch emitter.
        if not self._produces_tensor_output(node):
            # Check if any user is a graph output
            is_output = any(u.op == "output" for u in node.users)
            if is_output:
                logger.debug(f"[TRT partitioner] REJECTED {node.name} ({target_name}): non-tensor graph output")
                return False

        # Check dtype compatibility
        if not self._is_dtype_supported(node):
            logger.debug(f"[TRT partitioner] REJECTED {node.name} ({target_name}): unsupported dtype")
            return False

        # runtime-computed values (e.g. expand sizes from dynamic shapes)
        # that static converters cannot handle at engine-build time.
        if self._has_symbolic_scalar_args(node):
            # Try both the full op name and the namespace-stripped name
            # since converters register with "aten." prefix.
            if not (
                converter_registry.supports_dynamic_shapes(op_name)
                or converter_registry.supports_dynamic_shapes(target_name)
            ):
                logger.debug(f"[TRT partitioner] REJECTED {node.name} ({target_name}): symbolic scalar args, converter lacks supports_dynamic_shapes")
                return False

        # Reject view/reshape with dynamic output dims but all-concrete
        # target shape.  This means the concrete ints were captured at
        # trace time and won't adapt to different input sizes, causing
        # volume mismatches in the TRT engine at non-trace-time shapes.
        if self._is_stale_reshape(node, target_name):
            logger.debug(f"[TRT partitioner] REJECTED {node.name} ({target_name}): stale reshape")
            return False

        return True

    @staticmethod
    def _is_symbolic_scalar_node(n: torch.fx.Node) -> bool:
        """Return True if *n* represents a symbolic scalar (SymInt / SymFloat)."""
        if "val" not in n.meta:
            return True
        val = n.meta["val"]
        if isinstance(val, torch.Tensor) or hasattr(val, "shape"):
            return False
        return True

    def _has_symbolic_scalar_args(self, node: torch.fx.Node) -> bool:
        """Check if any non-tensor argument is a symbolic FX Node.

        Tensor arguments (other graph nodes whose ``val`` is a Tensor) are
        fine — they are actual data flowing through the graph.  Scalar
        arguments that are FX Nodes represent values computed at runtime
        (e.g. symbolic sizes from dynamic shapes) that TRT converters
        cannot evaluate at engine-build time.
        """
        for arg in node.args:
            if isinstance(arg, torch.fx.Node) and self._is_symbolic_scalar_node(arg):
                return True
            if isinstance(arg, (list, tuple)):
                for a in arg:
                    if isinstance(a, torch.fx.Node) and self._is_symbolic_scalar_node(a):
                        return True
        return False

    _RESHAPE_OPS: Set[str] = {
        "view.default",
        "view_copy.default",
        "_unsafe_view.default",
        "reshape.default",
    }

    def _is_stale_reshape(self, node: torch.fx.Node, target_name: str) -> bool:
        """Return True if a view/reshape has dynamic output dims but a
        fully-concrete target shape AND the converter can't handle it.

        When torch.export traces with dynamic shapes, derived sizes (like
        ``2 * seq_len - 1``) may be captured as concrete ints in the view
        target.  At engine build time, TRT bakes those ints into the
        shape tensor, causing volume mismatches at non-trace-time shapes.

        However, when at most 1 output dim is dynamic, the converter uses
        ``trt.Dims`` with a single -1 (inferred from input volume) and
        concrete values from the *metadata* — not from the stale target
        args — so the reshape adapts correctly at runtime.
        """
        if target_name not in self._RESHAPE_OPS:
            return False
        if len(node.args) < 2:
            return False

        # Check if the output shape has any dynamic (symbolic) dims.
        val = node.meta.get("val")
        if val is None or not hasattr(val, "shape"):
            return False

        num_sym = sum(1 for s in val.shape if type(s).__name__ == "SymInt")
        if num_sym == 0:
            return False  # No dynamic dims → not stale

        if num_sym <= 1:
            return False

        # Reject if target_shape is all concrete (stale trace-time values).
        target_shape = node.args[1]
        if not isinstance(target_shape, (list, tuple)):
            return False
        all_concrete = all(
            isinstance(d, int) for d in target_shape
        )
        return all_concrete

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

    @staticmethod
    def _produces_tensor_output(node: torch.fx.Node) -> bool:
        """Return True if the node produces tensor output(s).

        The ExecuTorch emitter requires all delegate outputs to be TensorSpecs.
        Nodes that produce plain scalars (int, float, SymInt) must stay outside
        the delegate to avoid emitter failures.
        """
        if "val" not in node.meta:
            return True
        val = node.meta["val"]
        if isinstance(val, torch.Tensor):
            return True
        if isinstance(val, (list, tuple)):
            return all(isinstance(v, torch.Tensor) for v in val)
        return False

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
