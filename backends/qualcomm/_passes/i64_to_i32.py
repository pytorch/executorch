# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import FrozenSet, Optional

import torch
from executorch.backends.qualcomm.builders.node_visitor_manager import (
    _node_visitor_dict,
)
from executorch.backends.qualcomm.builders.utils import (
    get_parameter,
    is_constant,
    is_graph_output,
)
from executorch.backends.qualcomm.partition.utils import must_stay_on_cpu
from executorch.backends.qualcomm.utils.constants import (
    QCOM_INDEX_CAST,
    QCOM_INSERTED_CAST,
    QCOM_KEEP_INT64,
    QCOM_ORIG_DTYPE,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, NodeMetadata, PassResult, ProxyValue
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils import _pytree as pytree


class I64toI32(ExportPass):
    """
    Insert a cast node to cast dtype from int64 to int32.
    This will be applied on operator and constant nodes such as weights.
    Every cast it inserts is marked with QCOM_INSERTED_CAST.
    An ATen or quantized_decomposed op that QNN has no builder for runs on
    CPU, and so do an op the partitioner skips and an op whose data types or
    shapes keep it off QNN. Their int64 results stay int64, and a QNN op
    reads them through an int32 cast.
    They read int64 inputs, and QNN results that were int64, as int64 too.
    """

    # This dict is to ensure that the input of the OPs are int64 due to PyTorch and CPU kernel restrictions.
    # For example, scatter op can only accept args[2], the index, as int64.
    # Key: Ops to cast input to i64
    # Value: The args' indices to add casting op
    I64_IN_OPS = {
        exir_ops.edge.aten.gather.default: [2],
        exir_ops.edge.aten.scatter.src: [2],
        exir_ops.edge.aten.scatter.value: [2],
        exir_ops.edge.aten.scatter_add.default: [2],
        exir_ops.edge.aten.scatter_reduce.two: [2],
        exir_ops.edge.quantized_decomposed.embedding_byte.default: [5],
        exir_ops.edge.quantized_decomposed.embedding_byte.dtype: [5],
        exir_ops.edge.aten.max_pool2d_with_indices_backward.default: [7],
    }
    copy_op = exir_ops.edge.aten._to_copy.default

    def __init__(
        self,
        edge_program,
        skip_node: FrozenSet[str] = frozenset(),
        skip_node_id_set: Optional[set] = None,
        skip_node_op_set: Optional[set] = None,
    ):
        super(I64toI32, self).__init__()
        self.edge_program = edge_program
        self.skip_node = skip_node
        self.skip_node_id_set = skip_node_id_set or set()
        self.skip_node_op_set = skip_node_op_set or set()

    # pyre-ignore[2]
    def _is_tensor_of_dtype(self, node_val, dtype: torch.dtype) -> bool:
        return isinstance(node_val, FakeTensor) and node_val.dtype == dtype

    @staticmethod
    def _runs_on_cpu(op, meta) -> bool:
        return QCOM_KEEP_INT64 in meta or (
            getattr(op, "namespace", None) in ("aten", "quantized_decomposed")
            and op.__name__ not in _node_visitor_dict
        )

    def _mark_ops_kept_on_cpu(self, graph_module: torch.fx.GraphModule):
        # The partitioner leaves these ops on CPU even when QNN has a builder.
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and (
                node.name in self.skip_node_id_set
                or getattr(node.target, "__name__", None) in self.skip_node_op_set
                or must_stay_on_cpu(node)
            ):
                node.meta[QCOM_KEEP_INT64] = True

    def _cast(self, value, dtype):
        # The cast takes the meta of the value it casts, not of the op that
        # reads it, since every later reader shares it.
        return super().call_operator(
            self.copy_op,
            (value,),
            {"dtype": dtype},
            NodeMetadata({**value.node.meta, QCOM_INSERTED_CAST: True}),
        )

    def _cast_to_int32(self, res, meta):
        # QNN builders handle integer tensors such as indices as int32, so no
        # int64 result may reach them: argmax, a sum of bools or topk indices.
        if self._is_tensor_of_dtype(res.data, torch.int64):
            res = self._cast(res, torch.int32)
        # An op that reads only int32 inputs gives int32 without a cast.
        if self._is_tensor_of_dtype(meta["val"], torch.int64):
            self.narrowed_results[res.node] = None
        return res

    def _read_int32(self, arg):
        if arg.node in self.cpu_results:
            if self.cpu_results[arg.node] is None:
                self.cpu_results[arg.node] = self._cast_to_int32(arg, arg.node.meta)
            return self.cpu_results[arg.node]
        return arg

    def _read_int64(self, arg):
        if arg.node in self.narrowed_results:
            if self.narrowed_results[arg.node] is None:
                self.narrowed_results[arg.node] = self._cast(arg, torch.int64)
            return self.narrowed_results[arg.node]
        return arg

    def _keep_cpu_result(self, res):
        if self._is_tensor_of_dtype(res.data, torch.int64):
            self.cpu_results[res.node] = None
        return res

    def call_operator(self, op, args, kwargs, meta):
        if self._runs_on_cpu(op, meta):
            # Casting a product, for example, to int32 would wrap it, and some
            # CPU kernels need int64, like the zero point of quantize_per_token.
            # A QNN result that was int64 is read cast back up, which loses
            # nothing.
            args, kwargs = pytree.tree_map_only(
                ProxyValue, self._read_int64, (args, kwargs)
            )
            return self._keep_cpu_result(super().call_operator(op, args, kwargs, meta))
        args, kwargs = pytree.tree_map_only(
            ProxyValue, self._read_int32, (args, kwargs)
        )
        res = super().call_operator(op, args, kwargs, meta)
        # The index casts added for I64_IN_OPS must stay int64.
        if QCOM_INDEX_CAST in meta:
            return res
        return self._cast_to_int32(res, meta)

    def call_getitem(self, value, key, meta):
        res = super().call_getitem(value, key, meta)
        if self._runs_on_cpu(value.node.target, value.node.meta):
            return self._keep_cpu_result(res)
        return self._cast_to_int32(res, meta)

    def _record_original_output_dtype(self, graph_module: torch.fx.GraphModule):
        for n in graph_module.graph.nodes:
            # Keep track of original output dtype so we ensure the dtype of the graph is consistent with nn.Module
            if is_graph_output(n):
                if isinstance(n.meta["val"], (tuple, list)):
                    dtype_list = [
                        tensor.dtype if tensor is not None else None
                        for tensor in n.meta["val"]
                    ]
                    n.meta[QCOM_ORIG_DTYPE] = dtype_list
                else:
                    n.meta[QCOM_ORIG_DTYPE] = n.meta["val"].dtype

    def _preserve_output_dtype(self, graph_module: torch.fx.GraphModule):
        for n in graph_module.graph.nodes:
            if is_graph_output(n) and QCOM_ORIG_DTYPE in n.meta:
                # The getitems of a multi-output node, such as topk, are graph
                # outputs too, and each output is cast back there.
                if isinstance(n.meta["val"], (tuple, list)):
                    continue
                if n.meta["val"].dtype != n.meta[QCOM_ORIG_DTYPE]:
                    if n.meta[QCOM_ORIG_DTYPE] != torch.int64:
                        logging.warning(
                            "This pass is intended to maintain output as int64 when nn.Module outputs int64. Other dtype modification is detected. Please ensure this is desired."
                        )
                    with graph_module.graph.inserting_after(n):
                        orig_dtype = n.meta[QCOM_ORIG_DTYPE]
                        node_val = n.meta["val"]
                        args = (n,)
                        users = list(n.users.keys())
                        output_users = [
                            user for user in users if user.target == "output"
                        ]
                        cast_node = graph_module.graph.create_node(
                            "call_function",
                            self.copy_op,
                            args,
                            {"dtype": orig_dtype},
                        )
                        cast_node.meta["val"] = node_val.to(orig_dtype)
                        cast_node.meta[QCOM_INSERTED_CAST] = True
                        cast_node.args = args
                        for user in output_users:
                            user.replace_input_with(n, cast_node)

    def _update_meta(self, node: torch.fx.node) -> None:
        meta_val = node.meta["val"]
        if isinstance(meta_val, tuple):
            node.meta["val"] = tuple(
                (
                    fake_tensor.to(torch.int32)
                    if fake_tensor is not None and fake_tensor.dtype == torch.int64
                    else fake_tensor
                )
                for fake_tensor in meta_val
            )
        else:
            if meta_val.dtype == torch.int64:
                # TODO This trick seems to use in mobilebert.
                # It would be better to convert to torch.int32
                node.meta["val"] = meta_val.to(torch.float)

    def _cast_constant_to_int32(self, graph_module: torch.fx.GraphModule):
        for n in graph_module.graph.nodes:
            if n.target in self.skip_node:
                continue
            if is_constant(n, self.edge_program):
                param = get_parameter(n, self.edge_program)
                if param.dtype == torch.int64:
                    # QNN does not support int64
                    self._update_meta(n)
            elif n.op == "placeholder" or n.op == "get_attr":
                # A `get_attr` node may be introduced after the `LiftConstantScalarOperands` pass.
                # If the scalar operand has an int64 data type, it should be explicitly cast to int32.
                node_val = n.meta["val"]
                if self._is_tensor_of_dtype(node_val, torch.int64):
                    # A get_attr constant can sit above the placeholders. Cast it
                    # right before its first user, so the cast stays below it once
                    # it is lifted to a placeholder.
                    anchor = n
                    if n.op == "get_attr" and n.users:
                        anchor = next(
                            user for user in graph_module.graph.nodes if user in n.users
                        ).prev
                    with graph_module.graph.inserting_after(anchor):
                        args = (n,)
                        to_dst_node = graph_module.graph.create_node(
                            "call_function",
                            self.copy_op,
                            args,
                            {"dtype": torch.int32},
                        )
                        to_dst_node.meta["val"] = node_val.to(torch.int32)
                        to_dst_node.meta[QCOM_INSERTED_CAST] = True
                        self.input_casts.append(to_dst_node)

                        # Replace usage of the src dtype result with the dst dtype result.
                        n.replace_all_uses_with(to_dst_node)
                        to_dst_node.args = (n,)

    def _read_int64_inputs_on_cpu(self, graph_module: torch.fx.GraphModule):
        # The int32 cast of an int64 input may already have wrapped it.
        for cast in self.input_casts:
            for user in list(cast.users):
                if self._runs_on_cpu(user.target, user.meta):
                    user.replace_input_with(cast, cast.args[0])
            if not cast.users:
                graph_module.graph.erase_node(cast)

    def _cast_op_args_to_i64(self, graph_module: torch.fx.GraphModule):
        # input will be cast to i32 during call_operator dtype propogation
        # insert i64 cast node to prevent PyTorch's operator validation failure
        for node in graph_module.graph.nodes:
            if node.target in self.I64_IN_OPS:
                with graph_module.graph.inserting_before(node):
                    arg_indices = self.I64_IN_OPS[node.target]
                    for arg_index in arg_indices:
                        input_node = node.args[arg_index]
                        # A skipped input is never cast, so it is still int64.
                        if input_node.target in self.skip_node:
                            continue
                        cast_i64_node = graph_module.graph.create_node(
                            "call_function",
                            self.copy_op,
                            (input_node,),
                            {"dtype": torch.int64},
                        )
                        cast_i64_node.meta[QCOM_INDEX_CAST] = True
                        cast_i64_node.meta[QCOM_INSERTED_CAST] = True
                        # This cast produces the *index* tensor, so its
                        # FakeTensor must be derived from the argument being
                        # cast, not from the op output: index.shape ==
                        # output.shape for gather, but for scatter* the output
                        # takes the shape of 'self', which may differ.
                        cast_i64_node.meta["val"] = input_node.meta["val"].to(
                            torch.int64
                        )
                        args_list = list(node.args)
                        args_list[arg_index] = cast_i64_node
                        node.args = tuple(args_list)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        # Record original output dtype to ensure that if user expects int64 as output,
        # convert the output back to int64 if it is casted from int64->int32.
        self._record_original_output_dtype(graph_module)
        self.input_casts = []
        self._cast_constant_to_int32(graph_module)
        self._cast_op_args_to_i64(graph_module)
        self._mark_ops_kept_on_cpu(graph_module)
        self._read_int64_inputs_on_cpu(graph_module)
        # The int32 cast of each int64 result of a CPU op, made when a QNN op
        # first reads it, and the int64 cast of each QNN result that was int64,
        # made when a CPU op first reads it.
        self.cpu_results, self.narrowed_results = {}, {}
        graph_module = super().call(graph_module).graph_module
        for node in graph_module.graph.nodes:
            node.meta.pop(QCOM_KEEP_INT64, None)
        self._preserve_output_dtype(graph_module)
        graph_module.recompile()
        return PassResult(graph_module, True)
