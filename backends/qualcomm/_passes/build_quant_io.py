# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_QUANTIZED_IO

from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.program._program import _get_updated_graph_signature
from executorch.exir.tensor import TensorSpec


class BuildQuantIo(ExportPass):
    """
    To make lowering process correct, the pass assign the correct quantized dtype to spec of call_delegate.
    """

    def __init__(self):
        super(BuildQuantIo, self).__init__()

    def _make_spec(self, x):
        if isinstance(x, torch.Tensor):
            return TensorSpec.from_tensor(x)
        elif isinstance(x, (int, bool, float)):
            return x
        else:
            return None

    def _build(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        # Forcedly update delegate node's meta['spec'] to get correct output
        # tensor size in runtime
        call_delegates = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function" and node.target == executorch_call_delegate
        ]
        for n in graph_module.graph.nodes:
            if QCOM_QUANTIZED_IO in n.meta:
                n.meta["val"] = n.meta["val"].to(dtype=n.meta[QCOM_QUANTIZED_IO])
                n.meta["spec"] = self._make_spec(n.meta["val"])

        for call_delegate in call_delegates:
            spec = []
            for user in list(call_delegate.users):
                spec.append(self._make_spec(user.meta["val"]))
            call_delegate.meta["spec"] = tuple(spec)
            self._retype_reference(graph_module, call_delegate)

    def _retype_reference(self, graph_module, call_delegate):
        """Give the delegate's reference module (which passes that retrace the
        graph run for its output types) the delegate's quantized inputs and
        outputs, by casts at its boundary."""
        lowered = getattr(graph_module, call_delegate.args[0].target)
        program = lowered.original_module
        graph = program.graph
        inputs = {
            s.arg.name: arg
            for s, arg in zip(
                [
                    s
                    for s in program.graph_signature.input_specs
                    if s.kind.name == "USER_INPUT"
                ],
                call_delegate.args[1:],
            )
        }
        for node in list(graph.nodes):
            arg = inputs.get(node.name) if node.op == "placeholder" else None
            dtype = arg.meta.get(QCOM_QUANTIZED_IO) if arg is not None else None
            if dtype is None or node.meta["val"].dtype == dtype:
                continue
            with graph.inserting_after(node):
                cast = graph.call_function(
                    exir_ops.edge.aten._to_copy.default,
                    (node,),
                    {"dtype": node.meta["val"].dtype},
                )
            cast.meta["val"] = node.meta["val"]
            node.replace_all_uses_with(
                cast, delete_user_cb=lambda u, cast=cast: u is not cast
            )
            node.meta["val"] = node.meta["val"].to(dtype)
        output = graph.output_node()
        outs = list(output.args[0])
        for user in call_delegate.users:
            dtype = user.meta.get(QCOM_QUANTIZED_IO)
            index = user.args[1]
            if dtype is None or outs[index].meta["val"].dtype == dtype:
                continue
            with graph.inserting_before(output):
                cast = graph.call_function(
                    exir_ops.edge.aten._to_copy.default,
                    (outs[index],),
                    {"dtype": dtype},
                )
            cast.meta["val"] = outs[index].meta["val"].to(dtype)
            outs[index] = cast
        output.args = (tuple(outs),)
        program.graph_module.recompile()
        program._graph_signature = _get_updated_graph_signature(
            program.graph_signature, program.graph_module
        )

    def call(self, graph_module: torch.fx.GraphModule):
        self._build(graph_module)
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
