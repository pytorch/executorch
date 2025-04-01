# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch._export.utils
from executorch.backends.arm._passes.arm_pass_utils import (
    get_constant_placeholder_kind,
    get_first_fake_tensor,
    get_param_tensor,
    is_persistent_buffer,
)
from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    delete_constant_placeholder,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export.graph_signature import InputKind

logger = logging.getLogger(__name__)


class FuseConstantArgsPass(ExportPass):
    """
    Fuses ops with only placeholder parameters into one placeholder parameter node with the op
    pre-calulcated on its data.

    Original:
        state_dict = {x_tensor_name : data}
        def f():
            return x.view(...)

    After pass:
        state_dict = {x_tensor_name_fused_const : data.view(...)}
        def f():
            return x
    """

    def __init__(self, exported_program: ExportedProgram) -> None:
        super().__init__()
        self.exported_program = exported_program

    def _fuse_nodes(self, node) -> bool:
        """
        Takes a node with only parameter inputs and replaces it with one constant tensor node with
        the operations already carried out on the data.
        """

        # Extract tensors and args from the node
        data_list = [
            get_param_tensor(self.exported_program, input_node)
            for input_node in node.all_input_nodes
        ]

        args = node.args[len(node.all_input_nodes) :]
        kwargs = node.kwargs

        if "input_qparams" in node.meta and len(node.meta["input_qparams"]) > 0:
            for i in range(len(node.all_input_nodes)):
                q_params = node.meta["input_qparams"][i]
                data_list[i] = q_params.dequantize_value(data_list[i])

        # Run the op on the extracted tensor
        data = node.target(*data_list, *args, **kwargs)

        # Only fuse if the tensor does not get bigger.
        if data.numel() > get_first_fake_tensor(node).numel():
            return False

        if "output_qparams" in node.meta and len(node.meta["output_qparams"]) > 0:
            q_params = node.meta["output_qparams"][0]
            data = q_params.quantize_value(data)

        insert_pos = list(node.all_input_nodes)[0]

        # Make new node the same kind as the first constant input
        input_kind = get_constant_placeholder_kind(self.exported_program, insert_pos)
        persistent_buffer = is_persistent_buffer(self.exported_program, insert_pos)

        # Create new node
        with node.graph.inserting_before(insert_pos):
            const_node = create_constant_placeholder(
                exp_program=self.exported_program,
                graph=node.graph,
                kind=input_kind,
                name=node.name + "_fused_const",
                data=data,
                persistent_buffer=persistent_buffer,
            )

        node.replace_all_uses_with(const_node)

        return True

    def call(self, graph_module):
        modified = False
        input_nodes_to_delete = []
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target == torch.ops.tosa._table.default:
                continue

            input_nodes = node.all_input_nodes
            if len(input_nodes) == 0:
                continue
            input_nodes_constant = (
                torch._export.utils.is_param(self.exported_program, input_node)
                or torch._export.utils.is_lifted_tensor_constant(
                    self.exported_program, input_node
                )
                or torch._export.utils.is_buffer(self.exported_program, input_node)
                for input_node in input_nodes
            )
            input_nodes_single_users = (
                len(input_node.users) == 1 for input_node in input_nodes
            )

            if all(input_nodes_constant) and all(input_nodes_single_users):
                try:
                    did_fuse = self._fuse_nodes(node)
                    modified |= did_fuse
                    if did_fuse:
                        graph_module.recompile()  # Recompile needed to catch chains of constant ops
                        input_nodes_to_delete.extend(input_nodes)
                except Exception as e:
                    logger.warning(
                        f"\nFailed to fuse constant op {node.name} due to exception:\n{str(e)}"
                    )

        if modified:
            graph_module.graph.eliminate_dead_code()
            for input_node in input_nodes_to_delete:
                delete_constant_placeholder(self.exported_program, input_node)

            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)


class ComputeConstantOpsAOT(ExportPass):
    """
    Evaluates call_functions that produce constant tensor outputs and replaces them with placeholders.

    Original:
        state_dict = {}
        def f():
            return torch.arange(0,10)
    After pass:
        state_dict = {node_name_pre_computed : torch.arange(0,10)}
        def f(node_name_pre_computed):
            return node_name_pre_computed
    """

    targeted_ops = [
        exir_ops.edge.aten.full.default,
        exir_ops.edge.aten.arange.start_step,
        exir_ops.edge.aten.eye.default,
        exir_ops.edge.aten.linspace.default,
        torch.ops.aten.scalar_tensor.default,
    ]

    def __init__(self, exported_program: ExportedProgram) -> None:
        super().__init__()
        self.exported_program = exported_program

    def compute_node_aot(self, node: torch.fx.Node) -> bool:
        """
        Takes a node with only parameter inputs and replaces it with one constant tensor node with
        the operations already carried out on the data.
        """

        # Create data from args
        output_qparams = node.meta.get("output_qparams", None)
        if output_qparams:
            # If we have output_qparams, compute data in fp and quantize
            data = node.target(*node.args)  #  type: ignore
            output_qparams = output_qparams[0]
            data = output_qparams.quantize_value(data)
        else:
            # If we don't have output_qparams, compute data using kwarg-specified dtype
            data = node.target(*node.args, **node.kwargs)  #  type: ignore

        # Create new node
        insert_pos = list(node.graph.nodes)[0]
        input_kind = InputKind.BUFFER
        persistent_buffer = True

        with node.graph.inserting_before(insert_pos):
            const_node = create_constant_placeholder(
                exp_program=self.exported_program,
                graph=node.graph,
                kind=input_kind,
                name=node.name + "_pre_computed",
                data=data,
                persistent_buffer=persistent_buffer,
            )
        node.replace_all_uses_with(const_node)

        return True

    def call(self, graph_module):
        modified = False
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target not in self.targeted_ops:
                continue
            try:
                modified |= self.compute_node_aot(node)
            except Exception as e:
                logger.warning(
                    f"\nFailed to pre-compute op {node.name} due to exception:\n{str(e)}"
                )

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
