# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
    is_param_node,
)
from executorch.backends.arm.constants import NCHW_ORDER, NNCHW_ORDER, NNNCHW_ORDER
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

logger = logging.getLogger(__name__)


def _is_input(node: torch.fx.Node, exported_program: ExportedProgram) -> bool:
    """
    Returns True if the node is an input node, i.e. a placeholder or a parameter.
    """
    return node.op == "placeholder" and not is_param_node(exported_program, node)


def _is_transpose_conv2d_weight(node: torch.fx.Node) -> bool:
    for user in node.users:
        if (
            user.op == "call_function"
            and user.target == exir_ops.backend.tosa.TRANSPOSE_CONV2D.default
            and len(user.args) > 1
            and user.args[1] is node
        ):
            return True
    return False


class ToTosaMemoryFormatPass(ArmPass):
    """
    Annotates each node with a tosa_dim_order. tosa_dim_order can be seen as a channels-last dim-order
    that in most cases will be (0, 2, 3, 1) for nodes with 4D-shapes. The pass also inserts backend.tosa.TRANSPOSE
    when a transition between 3D and 4D/5D tensors happen.
    The annotated tosa_dim_order is used to permute the node's shape such that it gives a TOSA-compliant shape.
    This pass also makes other values aware of spatial dimensions required by future operators by back propogating info as required.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(self, exported_program: ExportedProgram, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.exported_program = exported_program

    @staticmethod
    def _channels_last_order(rank: int, spatial_rank: int) -> tuple[int, ...]:
        """
        Compute the permutation of tensor dimensions corresponding to a
        "channels_last"-style memory layout for an arbitrary tensor rank.

        In standard PyTorch convention:
        - "channels_first" order is (N, C, H, W)
        - "channels_last" order is (N, H, W, C)
        This helper generalizes that concept beyond 4D tensors, producing an index
        ordering that moves the channel dimension to the end while preserving the
        relative order of batch and spatial dimensions.

        Args:
            rank (int): Total number of tensor dimensions (e.g. 4 for NCHW).
            spatial_rank (int): Number of spatial dimensions (e.g. 2 for HW, 3 for DHW).
                Values outside [0, rank - 2] are clamped to that range.

        Returns:
            tuple[int, ...]: A permutation of dimension indices that reorders the
            tensor into "channels_last" format. For example:
                - rank=4, spatial_rank=2 → (0, 2, 3, 1)  # NCHW → NHWC
                - rank=5, spatial_rank=3 → (0, 2, 3, 4, 1)  # NCDHW → NDHWC
                - rank=3, spatial_rank=1 → (0, 2, 1)

        Notes:
            If `rank <= 2`, the function returns the identity order since there
            are no distinct channel/spatial dimensions.
            In practice only rank 4+ tensors will reach this function as the dim order should be fixed for those.
        """
        if rank <= 2:
            return tuple(range(rank))
        spatial_rank = max(0, min(spatial_rank, rank - 2))
        channel_axis = rank - (spatial_rank + 1)
        batch_axes = list(range(channel_axis))
        spatial_axes = list(range(channel_axis + 1, rank))
        return tuple(batch_axes + spatial_axes + [channel_axis])

    @staticmethod
    def _channels_last_inverse_order(rank: int, spatial_rank: int) -> tuple[int, ...]:
        """
        Return the inverse permutation of `_channels_last_order`.

        This provides the axis order needed to map a tensor from
        "channels_last" layout back to its original layout.
        """
        order = ToTosaMemoryFormatPass._channels_last_order(rank, spatial_rank)
        inverse = [0] * rank
        for idx, axis in enumerate(order):
            inverse[axis] = idx
        return tuple(inverse)

    def _initial_spatial_rank(self, node: torch.fx.Node) -> int:
        """
        Infer the initial spatial rank based on the current rank, input node spatial
        ranks and node target. A spatial dimension includes Height, Width or Depth
        fields. In most operators this will only ever be Height and Width, but for 3D
        operators such as conv3d this would contain 3 spatial dims.

        Spatial rank is the max of any input node spatial ranks and the number of
        trailing spatial dims we need to preserve (rank - 2, capped at 3). This
        decides which axes must stay channels-last when inserting transposes.
        """
        tensor = get_first_fake_tensor(node).data
        # Start by assuming 2D when dealing with rank4+ to account for the base case
        # of an increasing amount of batch dimensions.
        rank = tensor.dim()
        if rank >= 4:
            spatial_rank = 2
        elif rank == 3:
            spatial_rank = 1
        else:
            spatial_rank = 0

        # Look for supported 3D ops and update spatial rank if relevent.
        # Currently only Conv3d is supported.
        if node.target == exir_ops.backend.tosa.CONV3D.default:
            spatial_rank = 3

        # Check input spatial ranks to know what the previous node spatial ranks were.
        input_ranks = [
            input_node.meta.get("tosa_spatial_rank", 0)
            for input_node in node.all_input_nodes
        ]
        if input_ranks:
            spatial_rank = max([spatial_rank, *input_ranks])

        # The max that spatial rank can be is 3. If the current rank not capable of holding
        # the current spatial rank, we clamp the max to Rank - (Channels and a singular batch dimension).
        # This ensures we revert back to lower spatial ranks after we are finished processing higher spatial ops.
        return min(spatial_rank, max(rank - 2, 0))

    @staticmethod
    def memory_format_differs(shape, spatial_rank):
        """
        Determine whether a tensor shape would be laid out differently in
        channels-first ((N)NCHW) versus channels-last ((N)NHWC) memory format.
        """
        if len(shape) <= 2 or spatial_rank <= 0:
            return False
        channel_idx = len(shape) - (spatial_rank + 1)
        channel_idx = max(0, min(channel_idx, len(shape) - 1))
        spatial_dims = shape[channel_idx + 1 :]
        if not spatial_dims:
            return False
        channel_dim = shape[channel_idx]
        return channel_dim > 1 and any(dim > 1 for dim in spatial_dims)

    @staticmethod
    def is_channel_reshape(
        input_shape, output_shape, input_spatial_rank, output_spatial_rank
    ):
        """
        Check whether a reshape touches the logical channel or consolidated
        batch dimensions, which would invalidate dim-order annotations.
        """

        valid_ranks = {4, 5, 6}

        if not (len(input_shape) in valid_ranks and len(output_shape) in valid_ranks):
            return False

        def channel_index(shape, spatial_rank):
            if len(shape) <= 2:
                return len(shape) - 1
            idx = len(shape) - (spatial_rank + 1)
            return max(0, min(idx, len(shape) - 1))

        C_old = input_shape[channel_index(input_shape, input_spatial_rank)]
        C_new = output_shape[channel_index(output_shape, output_spatial_rank)]

        def get_batch_prod_dim(shape, spatial_rank):
            product = 1

            for dim in shape[: channel_index(shape, spatial_rank)]:
                product = product * dim

            return product

        N_old = get_batch_prod_dim(input_shape, input_spatial_rank)
        N_new = get_batch_prod_dim(output_shape, output_spatial_rank)

        return (N_old != N_new) or (C_old != C_new)

    @staticmethod
    def insert_input_transpose(node, input_node, graph_module):
        """
        Ensure an input tensor is converted to channels-last ordering by
        inserting (or folding) a backend `TRANSPOSE` node.
        """
        if input_node.target == exir_ops.backend.tosa.TRANSPOSE.default:
            pre_permute_node = input_node.all_input_nodes[0]
            node.replace_input_with(input_node, pre_permute_node)
            return

        rank = len(get_first_fake_tensor(input_node).size())
        spatial_rank = input_node.meta["tosa_spatial_rank"]
        mem_format = ToTosaMemoryFormatPass._channels_last_inverse_order(
            rank, spatial_rank
        )
        # Guard: mem_format must be a true permutation for the current rank
        assert sorted(mem_format) == list(
            range(rank)
        ), f"bad perm {mem_format} for rank {rank} in insert_input_transpose"

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
            permute_node.meta["tosa_spatial_rank"] = spatial_rank

    @staticmethod
    def insert_output_transpose(node, graph_module):
        """
        Convert a producer's output to channels-last by appending a backend
        `TRANSPOSE` node and rewiring its users.
        """

        rank = len(get_first_fake_tensor(node).size())
        spatial_rank = node.meta["tosa_spatial_rank"]
        mem_format = ToTosaMemoryFormatPass._channels_last_order(rank, spatial_rank)
        # Guard: mem_format must be a true permutation for the current rank
        assert sorted(mem_format) == list(
            range(rank)
        ), f"bad perm {mem_format} for rank {rank} in insert_input_transpose"

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
            permute_node.meta["tosa_dim_order"] = mem_format

            node.meta["tosa_dim_order"] = tuple(
                range(len(get_first_fake_tensor(node).size()))
            )
            permute_node.meta["tosa_spatial_rank"] = spatial_rank

            users = [user for user in node.users if user != permute_node]
            for user in users:
                user.replace_input_with(node, permute_node)

    @staticmethod
    def _insert_view_transpose(
        input_shape, output_shape, node, input_node, graph_module
    ):
        """
        Insert the necessary input/output transposes around reshapes that cross
        the (N)NCHW -> (N)NHWC boundary or that touch channel dimensions.
        """
        nchw_to_nhwc = len(input_shape) < 4 and len(output_shape) >= 4
        nhwc_to_nchw = len(input_shape) >= 4 and len(output_shape) < 4

        input_sr = input_node.meta["tosa_spatial_rank"]
        output_sr = node.meta["tosa_spatial_rank"]

        channel_reshape = ToTosaMemoryFormatPass.is_channel_reshape(
            input_shape,
            output_shape,
            input_sr,
            output_sr,
        )

        if (
            channel_reshape or nhwc_to_nchw
        ) and ToTosaMemoryFormatPass.memory_format_differs(input_shape, input_sr):
            ToTosaMemoryFormatPass.insert_input_transpose(
                node, input_node, graph_module
            )

        if (
            channel_reshape or nchw_to_nhwc
        ) and ToTosaMemoryFormatPass.memory_format_differs(output_shape, output_sr):
            ToTosaMemoryFormatPass.insert_output_transpose(node, graph_module)

    def insert_tosa_transposes(self, graph_module: torch.fx.GraphModule):
        """
        Transposes are needed for operators transforming the input to a different rank, as 4D and 5D-tensors are assumed to be in (N)NHWC-format, whereas all other are in (N)NCHW format.
        This is relevant for the following cases:
        - view:       <4D ->  >=4D
        - view:      >=4D ->   <4D
        Additionally, a 4D/5D->4D/5D view operation acting on the channel dimension currently needs to be performed in (N)NCHW format, leading to one extra input and output transpose for this case.

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
            raise RuntimeError(f"{output_dim_orders=} is not supported.")

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
        """
        Drop any user-specified `dim_order` keyword arguments so the pass remains
        the single source of truth for dim-order annotations.
        """
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
        """
        Entry point for the pass: annotate spatial ranks, compute dim orders,
        insert bridging transposes, and forward to child passes.
        """
        nodes = list(graph_module.graph.nodes)
        for node in nodes:
            if "val" not in node.meta:
                continue
            node.meta["tosa_spatial_rank"] = self._initial_spatial_rank(node)
            self.remove_dim_order_kwargs(graph_module, node)

        self._propagate_spatial_ranks(nodes)

        for node in nodes:
            if "val" not in node.meta:
                continue
            node_data = get_first_fake_tensor(node).data
            spatial_rank = node.meta["tosa_spatial_rank"]
            if _is_input(node, self.exported_program) or node.op == "output":
                dim_order = node_data.dim_order()
            else:
                if node_data.dim() == 4 and _is_transpose_conv2d_weight(node):
                    dim_order = (1, 2, 3, 0)
                elif node_data.dim() >= 4:
                    dim_order = self._channels_last_order(node_data.dim(), spatial_rank)
                else:
                    dim_order = tuple(range(node_data.dim()))  # type: ignore[assignment]
            node.meta["tosa_dim_order"] = dim_order

        # Insert TOSA transposes to convert between (N)NCHW and (N)NHWC format.
        # See insert_tosa_transposes for insertion conditions.
        self.insert_tosa_transposes(graph_module)
        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)

    def _propagate_spatial_ranks(self, nodes):
        """
        Propagate `tosa_spatial_rank` metadata backwards so earlier nodes learn
        about upcoming spatial requirements from future ops.
        """
        changed = True
        while changed:
            changed = False
            for node in reversed(nodes):
                if "val" not in node.meta:
                    continue
                tensor = get_first_fake_tensor(node)
                limit = max(tensor.dim() - 2, 0)
                current = node.meta.get("tosa_spatial_rank")
                propagated = current
                for user in node.users:
                    user_rank = user.meta.get("tosa_spatial_rank")
                    if user_rank is None:
                        continue
                    propagated = max(propagated, min(user_rank, limit))
                if propagated != current:
                    node.meta["tosa_spatial_rank"] = propagated
                    changed = True
