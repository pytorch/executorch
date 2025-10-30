# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.annotate_decomposed_matmul import (
    AnnotateDecomposedMatmulPass,
)
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
    is_param_node,
)
from executorch.backends.arm.constants import (
    NCHW_ORDER,
    NHWC_INVERSE_ORDER,
    NHWC_ORDER,
    NNCHW_ORDER,
    NNHWC_INVERSE_ORDER,
    NNHWC_ORDER,
    NNNCHW_ORDER,
    NNNHWC_INVERSE_ORDER,
    NNNHWC_ORDER,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

logger = logging.getLogger(__name__)


def _is_input(node: torch.fx.Node, exported_program: ExportedProgram) -> bool:
    """
    Returns True if the node is an input node, i.e. a placeholder or a parameter.
    """
    return node.op == "placeholder" and not is_param_node(exported_program, node)


class ToTosaMemoryFormatPass(ArmPass):
    """
    Annotates each node with a tosa_dim_order. tosa_dim_order can be seen as a channels-last dim-order
    that in most cases will be (0, 2, 3, 1) for nodes with 4D-shapes. The pass also inserts backend.tosa.TRANSPOSE
    when a transition between 3D and 4D/5D tensors happen.
    The annotated tosa_dim_order is used to permute the node's shape such that it gives a TOSA-compliant shape.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(self, exported_program: ExportedProgram) -> None:
        super().__init__()
        self.exported_program = exported_program

    @staticmethod
    def memory_format_differs(shape):
        """Returns true if the shape will have a different memory layout in (N)NCHW and (N)NHWC format"""
        if len(shape) >= 6:
            C = shape[3]
            H = shape[4]
            W = shape[5]
        elif len(shape) == 5:
            C = shape[2]
            H = shape[3]
            W = shape[4]
        elif len(shape) == 4:
            C = shape[1]
            H = shape[2]
            W = shape[3]
        elif len(shape) == 3:
            C = shape[0]
            H = shape[1]
            W = shape[2]
        if len(shape) <= 2:
            return False

        return C > 1 and (H > 1 or W > 1)

    @staticmethod
    def is_channel_reshape(input_shape, output_shape):
        """Returns true if reshape changes the channel dimension or batch product dimension(s)"""

        valid_ranks = {4, 5, 6}

        if not (len(input_shape) in valid_ranks and len(output_shape) in valid_ranks):
            return False

        C_old = input_shape[-3]
        C_new = output_shape[-3]

        def get_batch_prod_dim(shape):
            product = 1

            for dim in shape[:-3]:
                product = product * dim

            return product

        N_old = get_batch_prod_dim(input_shape)
        N_new = get_batch_prod_dim(output_shape)

        return (N_old != N_new) or (C_old != C_new)

    @staticmethod
    def insert_input_transpose(node, input_node, graph_module):
        if input_node.target == exir_ops.backend.tosa.TRANSPOSE.default:
            pre_permute_node = input_node.all_input_nodes[0]
            node.replace_input_with(input_node, pre_permute_node)
            return

        if len(get_first_fake_tensor(input_node).size()) == 6:
            mem_format = NNNHWC_INVERSE_ORDER
        elif len(get_first_fake_tensor(input_node).size()) == 5:
            mem_format = NNHWC_INVERSE_ORDER
        else:
            mem_format = NHWC_INVERSE_ORDER
        # Guard: mem_format must be a true permutation for the current rank
        _rank_ = len(
            get_first_fake_tensor(input_node).size()
        )  # or (node) in output path
        assert sorted(mem_format) == list(
            range(_rank_)
        ), f"bad perm {mem_format} for rank {_rank_} in insert_input_transpose"

        with graph_module.graph.inserting_before(node):
            permute_node = create_node(
                graph_module.graph,
                exir_ops.backend.tosa.TRANSPOSE.default,
                args=(
                    input_node,
                    list(mem_format),
                ),
                from_node=node,
            )
            node.replace_input_with(input_node, permute_node)

            permute_node.meta["tosa_dim_order"] = tuple(
                range(len(input_node.meta["val"].size()))
            )

    @staticmethod
    def insert_output_transpose(node, graph_module):

        if len(get_first_fake_tensor(node).size()) == 6:
            mem_format = NNNHWC_ORDER
        elif len(get_first_fake_tensor(node).size()) == 5:
            mem_format = NNHWC_ORDER
        else:
            mem_format = NHWC_ORDER
        # Guard: mem_format must be a true permutation for the current rank
        _rank_ = len(get_first_fake_tensor(node).size())  # or (node) in output path
        assert sorted(mem_format) == list(
            range(_rank_)
        ), f"bad perm {mem_format} for rank {_rank_} in insert_input_transpose"

        with graph_module.graph.inserting_after(node):
            permute_node = create_node(
                graph_module.graph,
                exir_ops.backend.tosa.TRANSPOSE.default,
                args=(
                    node,
                    list(mem_format),
                ),
                from_node=node,
            )

            rank = len(get_first_fake_tensor(node).size())
            if rank == 6:
                permute_node.meta["tosa_dim_order"] = NNNHWC_ORDER
            elif rank == 5:
                permute_node.meta["tosa_dim_order"] = NNHWC_ORDER
            else:
                permute_node.meta["tosa_dim_order"] = NHWC_ORDER

            node.meta["tosa_dim_order"] = tuple(
                range(len(get_first_fake_tensor(node).size()))
            )

            users = [user for user in node.users if user != permute_node]
            for user in users:
                user.replace_input_with(node, permute_node)

    @staticmethod
    def _insert_view_transpose(
        input_shape, output_shape, node, input_node, graph_module
    ):
        nchw_to_nhwc = len(input_shape) < 4 and len(output_shape) >= 4
        nhwc_to_nchw = len(input_shape) >= 4 and len(output_shape) < 4
        channel_reshape = ToTosaMemoryFormatPass.is_channel_reshape(
            output_shape, input_shape
        )

        if (
            channel_reshape or nhwc_to_nchw
        ) and ToTosaMemoryFormatPass.memory_format_differs(input_shape):

            ToTosaMemoryFormatPass.insert_input_transpose(
                node, input_node, graph_module
            )

        if (
            channel_reshape or nchw_to_nhwc
        ) and ToTosaMemoryFormatPass.memory_format_differs(output_shape):

            ToTosaMemoryFormatPass.insert_output_transpose(node, graph_module)

    def insert_tosa_transposes(self, graph_module: torch.fx.GraphModule):
        """
        Transposes are needed for operators transforming the input to a different rank, as 4D and 5D-tensors are assumed to be in (N)NHWC-format, whereas all other are in (N)NCHW format.
        This is relevant for the following cases:
        - view:       <4D ->  >=4D
        - view:      >=4D ->   <4D
        Additionally, a 4D/5D->4D/5D view operation acting on the channel dimension currently needs to be performed in (N)NCHW format, leadning to one extra input and output transpose for this case.

        Transposes can be avoided for shapes where there is no difference in actual memory, e.g for
        - H == W == 1
        - C == 1
        - 1D/2D tensors
        """
        for node in graph_module.graph.nodes:
            # call_function and placeholder allowed due to
            # index.Tensor being able to come in as both
            if node.op != "call_function":
                continue

            # Transpose views
            elif node.target in (
                exir_ops.edge.aten.view_copy.default,
                exir_ops.edge.aten.index.Tensor,
            ):
                # For index.Tensor:
                #   If we want to support 4D indexing tensors this logic
                #   should be updated.
                input_node = node.args[0]
                input_shape = input_node.meta["val"].shape
                output_shape = node.meta["val"].shape
                self._insert_view_transpose(
                    input_shape,
                    output_shape,
                    node,
                    input_node,
                    graph_module,
                )

        output_node = graph_module.graph.output_node()

        # Transpose inputs if they are in (N)NCHW format
        inputs = [
            n for n in graph_module.graph.nodes if _is_input(n, self.exported_program)
        ]
        for input_node in inputs:
            input_dim_order = get_first_fake_tensor(input_node).dim_order()
            if input_dim_order in (NCHW_ORDER, NNCHW_ORDER, NNNCHW_ORDER):
                self.insert_output_transpose(input_node, graph_module)

        # Transpose outputs if they are in (N)NCHW format
        outputs = output_node.args[0]
        if not isinstance(outputs, (list, tuple)):
            raise TypeError(
                f"Expected output node args to be a list or tuple, got {type(outputs)}"
            )
        output_dim_orders = output_node.meta.get("original_dim_orders")
        if output_dim_orders is None:
            raise RuntimeError(
                f"{AnnotateDecomposedMatmulPass.__name__} is required to run at the beginning of the pass pipeline when using {ToTosaMemoryFormatPass.__name__}."
            )

        for output_node_input, output_dim_order in zip(
            outputs, output_dim_orders, strict=True
        ):
            if output_dim_order in (
                NCHW_ORDER,
                NNCHW_ORDER,
                NNNCHW_ORDER,
            ):
                self.insert_input_transpose(
                    output_node, output_node_input, graph_module
                )

    def remove_dim_order_kwargs(
        self, graph_module: torch.fx.GraphModule, node: torch.fx.Node
    ):
        if node.op != "call_function":
            return

        kwargs = dict(node.kwargs)

        if "dim_order" in kwargs:
            logger.warning(
                f"Ignoring dim_order kwarg '{kwargs['dim_order']}' for '{node.name}'."
            )
            del kwargs["dim_order"]

        node.kwargs = kwargs

    def call(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            node_data = get_first_fake_tensor(node).data

            self.remove_dim_order_kwargs(graph_module, node)
            # Inputs and outputs may vary in dim_order
            if _is_input(node, self.exported_program) or node.op == "output":
                dim_order = node_data.dim_order()
            elif node_data.dim() == 4:
                dim_order = NHWC_ORDER
            elif node_data.dim() == 5:
                dim_order = NNHWC_ORDER
            elif node_data.dim() == 6:
                dim_order = NNNHWC_ORDER
            else:
                dim_order = tuple(range(node_data.dim()))  # type: ignore[assignment]

            node.meta["tosa_dim_order"] = dim_order

        # Insert TOSA transposes to convert between (N)NCHW and (N)NHWC format.
        # See insert_tosa_transposes for insertion conditions.
        self.insert_tosa_transposes(graph_module)
        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
