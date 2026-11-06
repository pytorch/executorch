# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.backends.qualcomm._passes.utils import find_patterns
from executorch.backends.qualcomm.builders.node_visitor import (
    dq_ops,
    QNN_TENSOR_TYPE_MAP,
)
from executorch.backends.qualcomm.builders.utils import (
    is_mutable_buffer_input,
    is_parameter,
)
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
    flatbuffer_to_option,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass


def _is_node(node):
    return isinstance(node, torch.fx.Node)


def _is_call(node):
    return _is_node(node) and node.op == "call_function"


def _is_placeholder(node):
    return _is_node(node) and node.op == "placeholder"


def _is_get_attr(node):
    return _is_node(node) and node.op == "get_attr"


def _is_add(node):
    return _is_call(node) and node.target in [
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.add.Scalar,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add.Scalar,
    ]


def _is_dq(node):
    return _is_call(node) and node.target in dq_ops


def _is_mean(node):
    return _is_call(node) and node.target in [
        exir_ops.edge.aten.mean.dim,
        torch.ops.aten.mean.dim,
    ]


def _is_mul(node):
    return _is_call(node) and node.target in [
        exir_ops.edge.aten.mul.Tensor,
        torch.ops.aten.mul.Tensor,
    ]


def _is_pow(node):
    return _is_call(node) and node.target in [
        exir_ops.edge.aten.pow.Tensor_Tensor,
        exir_ops.edge.aten.pow.Tensor_Scalar,
        torch.ops.aten.pow.Tensor_Scalar,
    ]


def _is_rsqrt(node):
    return _is_call(node) and node.target in [
        exir_ops.edge.aten.rsqrt.default,
        torch.ops.aten.rsqrt.default,
    ]


class RecomposeRmsNorm(ExportPass):
    """
    Merge decomposed operators back to one super node.
    """

    def __init__(
        self,
        quantization_capture=False,
        edge_program=None,
        compiler_specs=None,
        skip_node_id_set=None,
        skip_node_op_set=None,
    ):
        super(RecomposeRmsNorm, self).__init__()
        assert (
            quantization_capture or edge_program is not None
        ), "RecomposeRmsNorm needs edge_program to tell a weight from an input."
        self.edge_program = edge_program
        self.skip_node_id_set = skip_node_id_set or set()
        self.skip_node_op_set = skip_node_op_set or set()
        self.may_be_htp, self.htp_fp16 = True, False
        if compiler_specs is not None:
            # Imported here, since partition depends on _passes.
            from executorch.backends.qualcomm.partition.utils import (
                generate_qnn_executorch_option,
                is_htp_fp16,
            )

            options = flatbuffer_to_option(
                generate_qnn_executorch_option(compiler_specs)
            ).backend_options
            self.may_be_htp = (
                options.backend_type == QnnExecuTorchBackendType.kHtpBackend
            )
            self.htp_fp16 = is_htp_fp16(options)
        self.rms_norm_target = exir_ops.edge.aten.rms_norm.default
        self.skip_targets = [
            exir_ops.edge.aten.to.dtype,
            exir_ops.edge.aten._to_copy.default,
            exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
        ]
        self.quantization_capture = quantization_capture
        if quantization_capture:
            self.rms_norm_target = torch.ops.aten.rms_norm.default
            self.skip_targets = [
                torch.ops.aten.to.dtype,
                torch.ops.aten.to.device,
                torch.ops.aten.type_as.default,
            ]

    def _is_cast(self, node, dtype):
        """
        Whether the node is a cast that changes nothing but the dtype, to a
        dtype at least as wide as `dtype` and at most float32. A narrower cast
        rounds the values, which the rebuilt rms_norm in `dtype` would not do,
        and rms_norm computes in float32 at most.
        """
        if not _is_call(node) or node.target not in self.skip_targets:
            return False
        source, cast = node.args[0].meta["val"], node.meta["val"]
        return (
            torch.promote_types(cast.dtype, dtype) == cast.dtype
            and torch.promote_types(cast.dtype, torch.float32) == torch.float32
            and (cast.layout, cast.device, cast.dim_order())
            == (source.layout, source.device, source.dim_order())
        )

    def _strip_casts(self, node, dtype):
        while self._is_cast(node, dtype):
            node = node.args[0]
        return node

    def _squares_the_scaled_tensor(self, pattern, dtype):
        square, scale_mul = pattern[6], pattern[2]
        base = square.args[0]
        if _is_pow(square):
            exponent = square.args[1]
            if _is_node(exponent):
                # LiftConstantScalarOperands may have turned it into a constant.
                constant = exponent.meta["val"].constant
                exponent = None if constant is None else constant.item()
            squares = exponent == 2
        else:
            squares = self._strip_casts(square.args[1], dtype) is self._strip_casts(
                base, dtype
            )
        scaled = [arg for arg in scale_mul.args if arg is not pattern[3]]
        # T5's norm squares a float32 copy of the tensor it scales.
        return (
            squares
            and len(scaled) == 1
            and self._strip_casts(scaled[0], dtype) is self._strip_casts(base, dtype)
        )

    def _is_skipped(self, input_node, output_node, gamma_node):
        """
        Whether the partitioner would skip rms_norm, or a node that rebuilding
        `output_node` on `input_node` removes: one that only feeds it.
        """
        if self.rms_norm_target.__name__ in self.skip_node_op_set:
            return True
        replaced, nodes = set(), [output_node]
        while nodes:
            node = nodes.pop()
            if (
                node in replaced
                or node in (input_node, gamma_node)
                or node.op != "call_function"
                or not (node is output_node or node.users.keys() <= replaced)
            ):
                continue
            if (
                node.name in self.skip_node_id_set
                or node.target.__name__ in self.skip_node_op_set
            ):
                return True
            replaced.add(node)
            nodes.extend(node.all_input_nodes)
        return False

    def _get_input(self, pattern, dtype, output_node, gamma_node=None):
        """
        The tensor to rebuild `output_node` on as an rms_norm in `dtype`, or
        None if there is none. rms_norm returns its input's dtype, so the input
        has to be in `dtype`. Unless the norm is about to be quantized, QNN also
        needs a type for it, and for HTP the input has to read something at
        runtime: the pass cannot see the partitions, and HTP drops one that
        reads only constants. And the partitioner must not skip a node the
        rebuild removes. These last two hold because rms_norm has no CPU kernel
        to fall back on.
        """
        if not self._squares_the_scaled_tensor(pattern, dtype):
            return None
        input_node = pattern[6].args[0]
        while self._is_cast(input_node, dtype) and (
            input_node.meta["val"].dtype != dtype
            or input_node.args[0].meta["val"].dtype == dtype
        ):
            input_node = input_node.args[0]
        if (
            input_node.meta["val"].dtype != dtype
            or (
                not self.quantization_capture
                and (
                    dtype not in QNN_TENSOR_TYPE_MAP
                    or (self.may_be_htp and not self._reads_runtime_input(input_node))
                )
            )
            or self._is_skipped(input_node, output_node, gamma_node)
        ):
            return None
        return input_node

    def _reads_runtime_input(self, node):
        # Placeholders count as QNN's partitioner counts them. A captured graph
        # may come without the program, and then every placeholder counts.
        nodes, seen = [node], set()
        while nodes:
            node = nodes.pop()
            if node in seen:
                continue
            seen.add(node)
            if node.op == "placeholder" and (
                self.edge_program is None
                or not is_parameter(node, self.edge_program)
                or is_mutable_buffer_input(node, self.edge_program)
            ):
                return True
            nodes.extend(node.all_input_nodes)
        return False

    def _get_norm_input(self, pattern, gamma_node):
        last_mul_node = pattern[0]
        dtype = last_mul_node.meta["val"].dtype
        # The "*" in the patterns can match any op, but only casts may sit
        # between the normalization and the weight.
        normalized = next(
            (arg for arg in last_mul_node.args if arg is not gamma_node), None
        )
        if self._strip_casts(normalized, dtype) is not pattern[2]:
            return None
        input_node = self._get_input(pattern, dtype, last_mul_node, gamma_node)
        if (
            input_node is None
            or list(gamma_node.meta["val"].shape)
            != list(input_node.meta["val"].shape)[-1:]
        ):
            return None
        return input_node

    @staticmethod
    def _erase_unused(node):
        # Erased right away, so a node another norm shares counts its users
        # right in _is_skipped.
        nodes, erased = [node], set()
        while nodes:
            node = nodes.pop()
            if node in erased or node.users or node.op != "call_function":
                continue
            erased.add(node)
            nodes.extend(node.all_input_nodes)
            node.graph.erase_node(node)

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        # _is_skipped counts users, so dead ones must not be left over.
        graph.eliminate_dead_code()

        # Root Mean Square normalization math equivalent implementation
        patterns = [
            # transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm
            [_is_mul, "*", _is_mul, _is_rsqrt, _is_add, _is_mean, _is_pow],
            # executorch.examples.models.llama.norm.RMSNorm
            [_is_mul, "*", _is_mul, _is_rsqrt, _is_add, _is_mean, _is_mul],
        ]

        for node in graph.nodes:
            if not _is_mul(node):
                continue

            rms_norm_patterns = [
                pattern
                for pattern in find_patterns(node, patterns)
                if pattern is not None
            ]

            if len(rms_norm_patterns) > 0:
                # Use first matched pattern
                rms_norm_pattern = rms_norm_patterns[0][0]
                last_mul_node = rms_norm_pattern[0]
                gamma_node = None
                # weight should be a constant
                for arg in last_mul_node.args:
                    if (
                        _is_get_attr(arg)
                        or (_is_placeholder(arg) and not self._reads_runtime_input(arg))
                        or _is_dq(arg)
                    ):
                        gamma_node = arg

                input_node = (
                    self._get_norm_input(rms_norm_pattern, gamma_node)
                    if gamma_node is not None
                    else None
                )
                dtype = rms_norm_pattern[2].meta["val"].dtype
                if input_node is None and self.htp_fp16 and dtype == torch.float32:
                    # Rebuild only the normalization and keep the casts and the
                    # weight after it. rms_norm computes in float32, so this is
                    # exact. Left decomposed, HTP's float16 precision would
                    # square in float16, which overflows above 65504.
                    output_node, gamma_node = rms_norm_pattern[2], None
                    input_node = self._get_input(rms_norm_pattern, dtype, output_node)
                else:
                    output_node = last_mul_node
                if input_node is None:
                    continue
                # Only the form QNN's RmsNorm takes is recomposed: the last of
                # 1 to 4 dims, all with static sizes.
                input_shape = list(input_node.meta["val"].shape)
                rank = len(input_shape)
                mean_node = rms_norm_pattern[5]
                mean_dims = mean_node.args[1] or range(rank)
                keepdim = len(mean_node.args) > 2 and mean_node.args[2]
                if (
                    not 0 < rank <= 4
                    or not all(isinstance(dim, int) for dim in input_shape)
                    or [dim % rank for dim in mean_dims] != [rank - 1]
                    or not (keepdim or rank == 1)
                ):
                    continue

                eps = rms_norm_pattern[4].args[1]
                if isinstance(eps, torch.fx.Node):
                    eps = eps.meta["val"].constant
                    if eps is None:
                        continue
                    eps = eps.item()

                with graph.inserting_before(output_node):
                    # args schema
                    # (Tensor input, int[] normalized_shape, Tensor? weight=None, float? eps=None) -> Tensor
                    rms_node = graph.create_node(
                        "call_function",
                        self.rms_norm_target,
                        (
                            input_node,
                            input_shape[-1:],
                            gamma_node,
                            eps,
                        ),
                    )
                output_node.replace_all_uses_with(rms_node)
                rms_node.meta = output_node.meta
                self._erase_unused(output_node)

        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
