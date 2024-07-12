# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import operator
from typing import Any, Callable, cast, Dict, List, Optional, Set, Union

import torch

from executorch.backends.xnnpack.partition.configs import (
    STATIC_MODULES,
    STATIC_OPS,
    SUPPORTED_DYN_QUANT_LINEAR_MODULES,
    SUPPORTED_DYN_QUANT_MODULES,
    SUPPORTED_MODULES,
    SUPPORTED_OPS,
    SUPPORTED_QUANT_MODULES,
    SUPPORTED_QUANT_OPS,
    UNSUPPORTED_QUANT_MODULES,
)
from executorch.backends.xnnpack.partition.graphs import bilinear_2d, sdpa
from executorch.backends.xnnpack.passes.fuse_batch_norm_with_conv import (
    FuseBatchNormWithConvPass,
)
from executorch.backends.xnnpack.utils.quant_utils import is_dequant
from executorch.backends.xnnpack.utils.utils import (
    get_input_node,
    get_source_fn,
    is_param_node,
)
from executorch.backends.xnnpack.xnnpack_preprocess import XnnpackBackend

from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_partitions_from_list_of_nodes,
    generate_pattern_op_partitions,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec

from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram
from torch.fx.passes.infra.partitioner import Partition
from torch.fx.passes.operator_support import OperatorSupportBase

from torch.fx.passes.utils.source_matcher_utils import (
    get_source_partitions,
    SourcePartition,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# TODO - Remove asserts - partitioner shouldn't assert, just not partition that part of the graph

"""
Per op Constraints
------------------

These constraints are used to filter out nodes from a partition when specific
conditions are not met. Indirectly, they specify constrains under which a node
should be lowerable to XNNPACK. If a constraint is not specified here, we will
always lower it. Nodes inside a decomposed subgraph i.e. linear subgraph will
also get test.

Interface: Callable[[torch.fx.Node], bool]

Note: Constraint fns are shared for both module based, op support based and
graph based (for now) partitioner implementations. Given that these stem from
XNNPACK limitations it should be ok to share the same constraint across both.

For module based partitioner - if any node fails to qualify, we discard that
instance of the module.

Don't update this global dict directly. It is updated through decorator
`XnnpackOperatorSupport._constraint`
"""
_OP_SUPPORT_CONSTRAINTS = {}


class XnnpackOperatorSupport(OperatorSupportBase):
    def __init__(
        self,
        ep: ExportedProgram,
        constraints_dict: Dict[
            Any, Callable[[torch.fx.Node], bool]
        ] = _OP_SUPPORT_CONSTRAINTS,
        supported_ops: Optional[List] = None,
        unsupported_modules: Optional[List] = None,
    ):
        """
        @Arg constraints_dict: Dict mapping each node to a lambda function that
             returns True if backend constraints are met for that instance of the
             node.
        @Arg supported_ops: List of supported operators for partitioning
        """
        self.unsupported_modules = unsupported_modules
        self.supported_ops = supported_ops
        self.constraints = constraints_dict
        self.ep = ep
        self.nodes_with_packed_weights = {
            exir_ops.edge.aten.convolution.default,
            exir_ops.edge.aten.addmm.default,
            exir_ops.edge.aten.mm.default,
            exir_ops.edge.aten.bmm.default,
        }
        assert len(self.constraints)

    def _check_inputs_are_valid_dtypes(self, node, valid_dtypes):
        # Check inputs are valid dtypes

        # Gather all args which are nodes
        args_to_check = []
        for arg in node.args:
            if isinstance(arg, list) or isinstance(arg, tuple):
                for item in arg:
                    if isinstance(item, torch.fx.Node):
                        args_to_check.append(item)

            if isinstance(arg, torch.fx.Node):
                args_to_check.append(arg)

        for arg in args_to_check:
            arg_val = arg.meta.get("val", None)

            if arg_val is None or isinstance(arg_val, tuple):
                continue

            # Being conservative for now, UX >> Perf
            # TODO: We need a pass to scrub these out.
            if not isinstance(arg_val, torch.Tensor):
                return False

            # XNNPACK does not support empty tensors
            if arg_val.numel() == 0:
                return False

            if arg_val.dtype not in valid_dtypes:
                return False

        return True

    def _check_outputs_are_valid_dtypes(self, node, valid_dtypes):
        # Check outputs are valid dtype
        node_val = node.meta.get("val", None)
        if node_val is None:
            return True

        if not isinstance(node_val, tuple):
            node_val = (node_val,)

        for val in node_val:
            if not isinstance(val, torch.Tensor):
                return False

            if val.dtype not in valid_dtypes:
                return False

        return True

    def check_node_has_valid_dtype(self, node):
        # max_pool2d_with_indicies returns indicies which is int64
        # this is supportable within XNNPACK
        if node.target in {exir_ops.edge.aten.max_pool2d_with_indices.default}:
            return True

        valid_dtypes = {
            torch.float32,
            torch.float16,
            torch.int8,
            torch.qint8,
        }
        if (
            node.op != "placeholder"
            and node.op != "call_function"
            and node.op != "get_attr"
        ):
            return False

        return self._check_inputs_are_valid_dtypes(
            node, valid_dtypes
        ) and self._check_outputs_are_valid_dtypes(node, valid_dtypes)

    def check_common_constraints(self, node) -> bool:
        has_valid_dtypes = self.check_node_has_valid_dtype(node)

        return has_valid_dtypes

    @staticmethod
    def check_constraint(node, ep) -> bool:
        """
        This node is from a partitioned subgraph by one of the partitioners so
        should be a valid node otherwise, let's make sure the constraint is met
        if specified
        """
        return _OP_SUPPORT_CONSTRAINTS.get(node.target, lambda node, ep: True)(node, ep)

    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        # Parameters are supported if any of their users are supported
        if is_param_node(self.ep, node):
            for user in node.users.keys():
                user_of_param = user
                if is_dequant(user):
                    user_of_param = list(user.users.keys())[0]
                if (
                    self.is_node_supported(submodules, user_of_param)
                    and user_of_param.target in self.nodes_with_packed_weights
                ):
                    return True

            return False

        # TODO - other ops?
        if node.op != "call_function":
            return False

        # Specifying supported ops is optional
        if self.supported_ops and node.target not in self.supported_ops:
            return False

        return self.check_constraint(node, self.ep) and self.check_common_constraints(
            node
        )

    @staticmethod
    def _constraint(target):  # noqa
        """
        Decorator to register a constraint fn for a node
        """

        def register(func: Callable[[torch.fx.Node, ExportedProgram], bool]):
            """
            Pass through registration for the constraint fn
            """
            _OP_SUPPORT_CONSTRAINTS[target] = func
            return staticmethod(func)

        return register

    """
    Define per op constraints functions below

    These constraint functions are staticmethods, which are registered through
    the decorator in a global dict. And called through `check_constraint()`
    method. These are not directly related to the class or the class instance
    but they are logically connected.

    Marked as `noqa` because Flake doesn't understand the staticmethod tag and
    complains about self not being the first arg.
    """

    @_constraint(exir_ops.edge.aten.mean.dim)
    def mean_dim(node: torch.fx.Node, ep: ExportedProgram) -> bool:  # noqa
        """
        Only select 2d cases are supported by XNNPACK
        """
        dims = node.args[1]
        return dims in ([-2, -1], [-1, -2])

    @_constraint(exir_ops.edge.aten.max_pool2d_with_indices.default)
    def maxpool2d_with_indices(
        node: torch.fx.Node, ep: ExportedProgram  # noqa
    ) -> bool:
        """
        Only if the first output value is consumed in the graph
        and it is not in ceil mode
        """
        users = list(node.users.keys())
        is_ceil_mode = len(node.args) >= 6 and node.args[5]
        return (
            True
            if len(users) == 1
            and users[0].target == operator.getitem
            and users[0].args[1] == 0
            and not is_ceil_mode
            else False
        )

    @_constraint(exir_ops.edge.quantized_decomposed.quantize_per_tensor.default)
    def quant_per_tensor_default(q: torch.fx.Node, ep: ExportedProgram) -> bool:  # noqa
        """
        Decide if we want to pull this q node or not in the partition.
        Given, op1 -> q -> dq -> op2
        For node q, if op1 or op2 is good, q should be good
        TODO: q -> op -> dq, real q not handled right now
        """
        if XnnpackOperatorSupport.check_constraint(q.args[0], ep):
            return True
        else:
            dq = list(q.users.keys())[0]
            op2 = list(dq.users.keys())[0]
            return XnnpackOperatorSupport.check_constraint(op2, ep)

    @_constraint(exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default)
    def dequant_per_tensor_default(
        dq: torch.fx.Node, ep: ExportedProgram  # noqa
    ) -> bool:
        """
        Decide if we want to pull this dq node or not.
        """
        return XnnpackOperatorSupport.check_constraint(dq.args[0], ep)

    @_constraint(exir_ops.edge.quantized_decomposed.quantize_per_channel.default)
    def quant_per_channel_default(
        q: torch.fx.Node, ep: ExportedProgram  # noqa
    ) -> bool:
        return XnnpackOperatorSupport.quant_per_tensor_default(q, ep)

    @_constraint(exir_ops.edge.quantized_decomposed.dequantize_per_channel.default)
    def dequant_per_channel_default(
        dq: torch.fx.Node, ep: ExportedProgram  # noqa
    ) -> bool:
        return XnnpackOperatorSupport.dequant_per_tensor_default(dq, ep)

    @_constraint(exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor)
    def quant_per_tensor_tensor(q: torch.fx.Node, ep: ExportedProgram) -> bool:  # noqa
        return XnnpackOperatorSupport.quant_per_tensor_default(q, ep)

    @_constraint(exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor)
    def dequant_per_tensor_tensor(
        dq: torch.fx.Node, ep: ExportedProgram  # noqa
    ) -> bool:
        return XnnpackOperatorSupport.dequant_per_tensor_default(dq, ep)

    @_constraint(exir_ops.edge.quantized_decomposed.choose_qparams.tensor)
    def choose_qparams_tensor(cqp: torch.fx.Node, ep: ExportedProgram) -> bool:  # noqa
        """
        Given, cqp -> getitem -> q -> dq -> op2
        Just check q, because it will check op2
        """
        getitem0 = list(cqp.users.keys())[0]
        q = list(getitem0.users.keys())[0]
        return XnnpackOperatorSupport.check_constraint(q, ep)

    @_constraint(exir_ops.edge.quantized_decomposed.dequantize_per_token.default)
    def dequant_per_token(dq: torch.fx.Node, ep: ExportedProgram) -> bool:  # noqa
        node = list(dq.users.keys())[0]
        assert isinstance(node, torch.fx.Node)
        return (
            node.target
            in [
                exir_ops.edge.aten.mm.default,
                exir_ops.edge.aten.addmm.default,
            ]
            or get_source_fn(node) in SUPPORTED_DYN_QUANT_LINEAR_MODULES
        )

    @_constraint(exir_ops.edge.quantized_decomposed.quantize_per_token.default)
    def quant_per_token(q: torch.fx.Node, ep: ExportedProgram) -> bool:  # noqa
        dq = list(q.users.keys())[0]
        return (
            dq.target == exir_ops.edge.quantized_decomposed.dequantize_per_token.default
            and XnnpackOperatorSupport.dequant_per_token(dq, ep)
        )

    @_constraint(
        exir_ops.edge.quantized_decomposed.choose_qparams_per_token_asymmetric.default
    )
    def choose_qparams_per_token_asymmetric(
        cqp: torch.fx.Node, ep: ExportedProgram  # noqa
    ) -> bool:
        """
        Given, cqp -> getitem -> q -> dq -> {mm, addmm}
        Just check q, because it will check dq
        """
        getitem0 = list(cqp.users.keys())[0]
        q = list(getitem0.users.keys())[0]
        return (
            q.target == exir_ops.edge.quantized_decomposed.quantize_per_token.default
            and XnnpackOperatorSupport.check_constraint(q, ep)
        )

    @_constraint(
        exir_ops.edge.quantized_decomposed.dequantize_per_channel_group.default
    )
    def dequant_per_channel_group_default(
        dq: torch.fx.Node, ep: ExportedProgram  # noqa
    ) -> bool:
        # Currently only supported by dqlinear weights
        permute_node = list(dq.users.keys())[0]
        assert isinstance(permute_node, torch.fx.Node)
        # We must have a transpose on [add]mm weights
        if permute_node.target != exir_ops.edge.aten.permute_copy.default:
            return False
        mm_node = list(permute_node.users.keys())[0]
        assert isinstance(mm_node, torch.fx.Node)
        return mm_node.target in [
            exir_ops.edge.aten.mm.default,
            exir_ops.edge.aten.addmm.default,
        ]

    @_constraint(exir_ops.edge.quantized_decomposed.quantize_per_channel_group.default)
    def quant_per_channel_group_default(
        q: torch.fx.Node, ep: ExportedProgram  # noqa
    ) -> bool:
        # we shouldn't have this with folded quant weights but doesn't hurt to lower it
        dq = list(q.users.keys())[0]
        assert isinstance(dq, torch.fx.Node)
        return (
            dq.target
            == exir_ops.edge.quantized_decomposed.dequantize_per_channel_group.default
            and XnnpackOperatorSupport.dequant_per_channel_default(dq, ep)
        )

    @_constraint(exir_ops.edge.aten.pow.Tensor_Scalar)
    def pow_tensor_scalar(node: torch.fx.Node, ep: ExportedProgram) -> bool:  # noqa
        """
        Only supports square, when args_2 = 2
        """
        power = node.args[1]
        return isinstance(power, int) and power == 2

    @_constraint(exir_ops.edge.aten.avg_pool2d.default)
    def avg_pool_2d(node: torch.fx.Node, ep: ExportedProgram) -> bool:  # noqa
        """
        Arguments to avg_pool2d.default node are as follows:
            - input,
            - kernel_size,
            - stride,
            - padding,
            - ceil_mode,
            - count_include_pad,
            - divisor_override,

        XNNPACK does not support ceil_mode = True and count_include_pad = True
        Additionally, we only support divisor_override if divisor_override = pooling region
        """
        args = node.args

        ceil_mode = False  # default is False
        if len(args) >= 5:
            ceil_mode = cast(bool, args[4])

        count_include_pad = True  # default is True
        if len(args) >= 6:
            count_include_pad = cast(bool, args[5])

        kernel_size = cast(List[int], args[1])
        pooling_region = kernel_size[0] * kernel_size[1]
        divisor_override = pooling_region  # Default divisor is pooling_region
        if len(args) >= 7:
            divisor_override = cast(int, args[6])

        return (
            not (ceil_mode or count_include_pad) and divisor_override == pooling_region
        )

    @_constraint(exir_ops.edge.aten._prelu_kernel.default)
    def prelu(node: torch.fx.Node, ep: ExportedProgram) -> bool:  # noqa
        """
        Input and Weight must be 4-dimensional
        """
        input_dim = cast(torch.fx.Node, node.args[0]).meta["val"].dim()
        weight_dim = cast(torch.fx.Node, node.args[1]).meta["val"].dim()
        return input_dim == 4 and weight_dim == 4

    @_constraint(exir_ops.edge.aten.cat.default)
    def cat(node: torch.fx.Node, ep: ExportedProgram) -> bool:  # noqa
        """
        Only support concatenation of 2 - 4 tensors
        """
        num_tensors = len(cast(List[torch.fx.Node], node.args[0]))
        return num_tensors >= 2 and num_tensors <= 4

    @_constraint(exir_ops.edge.aten.slice_copy.Tensor)
    def slice_copy(node: torch.fx.Node, ep: ExportedProgram) -> bool:  # noqa
        """
        Support slicing with stride = 1, no zero-dim tensors
        """
        stride = 1
        if len(node.args) > 4:
            stride = cast(int, node.args[4])

        if stride != 1:
            return False

        input_node = get_input_node(node, 0)
        output_node = node

        input_shape = list(input_node.meta["val"].shape)
        output_shape = list(output_node.meta["val"].shape)

        for dim in input_shape:
            if dim == 0:
                return False

        for dim in output_shape:
            if dim == 0:
                return False

        return True

    @_constraint(exir_ops.edge.aten.amax.default)
    def amax(node: torch.fx.Node, ep: ExportedProgram) -> bool:  # noqa
        """
        A: Only with keep_dim == True
        B: Only support with dim == 2 or dim == 3
        valid iff, A && B
        """
        is_keep_dim = (len(node.args) == 3) and (cast(bool, node.args[3]) is True)
        dim_arg_val = cast(int, node.args[1])
        return is_keep_dim and (dim_arg_val == 2 or dim_arg_val == 3)

    @_constraint(exir_ops.edge.aten._native_batch_norm_legit_no_training.default)
    def batch_norm(node: torch.fx.Node, ep: ExportedProgram) -> bool:  # noqa
        """
        Only support batch norms that can be fused with convolutions.
        This will be removed once standalone batch norm is supported.
        """

        # TODO(gjcomer) Remove after standalone batch norm (T171796544).

        conv_node = node.args[0]
        assert isinstance(conv_node, torch.fx.Node)

        if conv_node.target != exir_ops.edge.aten.convolution.default:
            return False

        return FuseBatchNormWithConvPass.can_fuse(conv_node, node, ep)


class XnnpackFloatingPointPartitioner(Partitioner):
    """
    Module and Opname based partitioner for FP32 modules/ops listed in
    SUPPORTED_MODULES and SUPPORTED_OPS.
    """

    def __init__(
        self,
        supported_modules: List[Callable] = SUPPORTED_MODULES,
        supported_ops: Optional[List[Callable]] = SUPPORTED_OPS,
        unsupported_modules: Optional[List[Callable]] = None,
    ):
        super().__init__()
        self.supported_modules = set(supported_modules)
        self.unsupported_modules = unsupported_modules
        self.supported_ops = set(supported_ops or [])

        self.delegation_spec = DelegationSpec(XnnpackBackend.__name__, [])

    @staticmethod
    def check_partitions(partitions: Union[dict, list]) -> bool:
        """
        Warn users if there aren't any matches

        TODO: convert this into a stronger validation, may need a flag in
        `to_backend()` or partitioner __init__()
        """
        pl = len(partitions)
        if pl == 0:
            log.warning("Nothing can be partitioned!")
        else:
            log.info(f"Found {pl} subgraphs to be partitioned.")
        return pl != 0

    def get_input_deps(  # noqa
        self, input_nodes: List[torch.fx.Node], ep: ExportedProgram
    ) -> List[torch.fx.Node]:
        """
        For each input node, walk up and pull necessary param/attr nodes in the partition
        """
        nodes = set()

        def is_param(ep: ExportedProgram, node) -> bool:
            return isinstance(node, torch.fx.Node) and is_param_node(ep, node)

        for inp in input_nodes:
            if is_param(ep, inp):
                nodes.add(inp)

        return list(nodes)

    def get_nodes(
        self, src_partition: SourcePartition, ep: ExportedProgram
    ) -> List[torch.fx.Node]:
        """
        Return nodes from the source partition.

        This is a wrapper to allow derived classes to add their own custom
        logic to extend the src_partition nodes list.
        """
        return src_partition.nodes + self.get_input_deps(src_partition.input_nodes, ep)

    def qualify_nodes(
        self, input_nodes: List[torch.fx.Node], ep: ExportedProgram
    ) -> bool:
        """
        Each node in the module (post decomposition) must satisfy the
        constraints specified for XNNPACK.

        Disqualify the whole module if one of the nodes fails to satisfy.
        """
        return all(
            XnnpackOperatorSupport.check_constraint(node, ep) for node in input_nodes
        )

    def get_module_partitions(self, ep: ExportedProgram) -> List[List[torch.fx.Node]]:
        """
        Get all partitions in the torch.fx.GraphModule for the supported
        modules.
        """
        graph_module = ep.graph_module
        src_partition_dict = get_source_partitions(
            graph_module.graph, self.supported_modules
        )
        all_partitions = src_partition_dict.values()

        module_partitions = []
        for src_partitions in all_partitions:
            for src_partition in src_partitions:
                partition_nodes = self.get_nodes(src_partition, ep)
                if self.qualify_nodes(partition_nodes, ep):
                    module_partitions.append(partition_nodes)

        return module_partitions

    def generate_partitions(self, ep: ExportedProgram) -> List[Any]:
        """
        Generate a list of partitions for an torch.fx.GraphModule.
        Also pass the supported ops to match.
        """
        graph_module = ep.graph_module
        matched_module_nodes = self.get_module_partitions(ep)
        return generate_partitions_from_list_of_nodes(
            graph_module,
            matched_module_nodes,
            XnnpackOperatorSupport(
                ep=ep,
                supported_ops=self.supported_ops,
                unsupported_modules=self.unsupported_modules,
            ),
        )

    def tag_nodes(self, partitions: List[Partition]) -> Dict[str, DelegationSpec]:
        """
        Tag each partition in the list with its delegation tag.
        """
        partition_tags: Dict[str, DelegationSpec] = {}
        for partition in partitions:
            # Add delegation tags
            for node in partition.nodes:
                delegation_tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                partition_tags[delegation_tag] = self.delegation_spec
        return partition_tags

    # override
    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        """
        Run the partitioner on the given graph module, then tag each partition
        with its delegation tag (and partition id)
        """
        partitions = self.generate_partitions(exported_program)
        partition_tags: Dict[str, DelegationSpec] = {}
        if self.check_partitions(partitions):
            partition_tags = self.tag_nodes(partitions)
        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )


# TODO: Merge XnnpackQuantizedPartitioner and XnnpackFloatingPointPartitioner
class XnnpackQuantizedPartitioner(XnnpackFloatingPointPartitioner):
    """
    Module and Opname based partitioner for statically quantized modules/ops listed in SUPPORTED_QUANT_MODULES and SUPPORTED_QUANT_OPS.
    """

    _Q_OPS = [
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
        exir_ops.edge.quantized_decomposed.quantize_per_channel_group.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor,
        exir_ops.edge.quantized_decomposed.quantize_per_token.default,
    ]

    _DQ_OPS = [
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_channel_group.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
        exir_ops.edge.quantized_decomposed.dequantize_per_token.default,
    ]

    _QPARAM_OPS = [
        exir_ops.edge.quantized_decomposed.choose_qparams.tensor,
        exir_ops.edge.quantized_decomposed.choose_qparams_per_token_asymmetric.default,
    ]

    _QUANT_OPS = _Q_OPS + _DQ_OPS + _QPARAM_OPS

    def __init__(
        self,
        supported_modules=SUPPORTED_QUANT_MODULES,
        supported_ops=SUPPORTED_QUANT_OPS,
        unsupported_modules=UNSUPPORTED_QUANT_MODULES,
    ):
        supported_ops = supported_ops or []
        super().__init__(
            supported_modules, supported_ops + self._QUANT_OPS, unsupported_modules
        )

    def get_input_deps(  # noqa
        self, input_nodes: List[torch.fx.Node], ep: ExportedProgram
    ) -> List[torch.fx.Node]:
        """
        For each input node, walk up and pull necessary quant/attr nodes in the partition
        """
        nodes = set()

        def is_param(ep: ExportedProgram, node) -> bool:
            return isinstance(node, torch.fx.Node) and is_param_node(ep, node)

        def is_q(ep: ExportedProgram, node) -> bool:
            return isinstance(node, torch.fx.Node) and node.target in self._Q_OPS

        def is_dq(ep: ExportedProgram, node) -> bool:
            return isinstance(node, torch.fx.Node) and node.target in self._DQ_OPS

        def is_qparam(node) -> bool:
            return isinstance(node, torch.fx.Node) and node.target in self._QPARAM_OPS

        def is_getitem(node) -> bool:
            return (
                isinstance(node, torch.fx.Node)
                and node.op == "call_function"
                and node.target == operator.getitem
            )

        for inp in input_nodes:
            if is_dq(ep, inp):
                dq = inp

                # Possible graph we want to partition
                #                  op(...)
                #                     ^
                #                     |
                #                     dq(0,   1, 2)
                #                        ^    ^  ^
                #                        |    |  |
                #                        q(0, 1, 2) # optional, only when not folded by the quantizer
                #                          ^  ^  ^
                #                          |  |  |
                # parameter ---------------'  |  |
                #                  [choose_qparams --> get_item(s)]  # optional, only with dynamic quant
                # per_channel_zp* ------------'  |
                # per_channel_scale* ------------'

                # The dequant node
                nodes.add(dq)

                # possible per_channel scale/zp for the dequant node args{1, 2}
                for i in [1, 2]:
                    node = dq.args[i]
                    if is_param(ep, node):
                        nodes.add(node)

                # is it quant or param node?
                prod = dq.args[0]

                assert is_q(ep, prod) or is_param(
                    ep, prod
                ), f"Expecting quant or param node as an input to a dq node, but got {prod.target} for {prod} node"

                nodes.add(prod)

                if is_q(ep, prod):
                    # possible nodes for quant node args{0, 1, 2}: 0: weight, 1: scale, 2: zero_point
                    for i in [0, 1, 2]:
                        node = prod.args[i]  # pyre-ignore

                        # possible choose_qparam
                        if is_getitem(node) and is_qparam(node.args[0]):
                            nodes.add(node)
                            nodes.add(node.args[0])

                        # weights or possible per_channel scale/zp for the quant node
                        elif is_param(ep, node):
                            nodes.add(node)
        return list(nodes)

    def get_output_deps(
        self, output_nodes: List[torch.fx.Node], exported_program
    ) -> List[torch.fx.Node]:
        """
        For each output node, check all the users and insert them into the partition if needed
        """
        nodes = []
        for output in output_nodes:
            for node in output.users:
                if node.target in self._Q_OPS:
                    nodes.append(node)
                    users = list(node.users.keys())
                    for dq_user in users:
                        assert (
                            dq_user.target in self._DQ_OPS
                        ), "Expecting a dq node(s) after a q node, but got target {dq_user.target} for {dq_user} node"
                        nodes.append(dq_user)
        return nodes

    # override
    def get_nodes(
        self, src_partition: SourcePartition, ep: ExportedProgram
    ) -> List[torch.fx.Node]:  # noqa
        """
        Insert quantization ops into src_partition by following the input, output node.
        """
        return (
            src_partition.nodes
            + self.get_input_deps(src_partition.input_nodes, ep)
            + self.get_output_deps(src_partition.output_nodes, ep)
        )


class XnnpackPartitioner(Partitioner):
    """
    Module and Opname based partitioner for FP32 modules/ops listed in
    SUPPORTED_MODULES and SUPPORTED_OPS and statically quantized modules/ops listed in
    SUPPORTED_QUANT_MODULES and SUPPORTED_QUANT_OPS.
    """

    _Q_OPS = [
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
        exir_ops.edge.quantized_decomposed.quantize_per_channel_group.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor,
        exir_ops.edge.quantized_decomposed.quantize_per_token.default,
    ]

    _DQ_OPS = [
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
        exir_ops.edge.quantized_decomposed.quantize_per_channel_group.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
        exir_ops.edge.quantized_decomposed.dequantize_per_token.default,
    ]

    _QPARAM_OPS = [
        exir_ops.edge.quantized_decomposed.choose_qparams.tensor,
    ]

    _QUANT_OPS = _Q_OPS + _DQ_OPS + _QPARAM_OPS

    def __init__(
        self,
        *,
        supported_modules: List[Callable] = SUPPORTED_MODULES,
        supported_ops: Optional[List[Callable]] = SUPPORTED_OPS,
        supported_quant_modules: List[Callable] = SUPPORTED_QUANT_MODULES,
        supported_quant_ops: Optional[List[Callable]] = SUPPORTED_QUANT_OPS,
        quant: Optional[bool] = None,
        has_dynamic_shapes: bool = False,
        _lower_recomposed_sdpa: Optional[bool] = True,
    ):
        super().__init__()
        self.supported_modules = set(supported_modules)
        self.supported_ops = set(supported_ops or [])
        self.supported_quant_modules = set(supported_quant_modules)

        supported_quant_ops = supported_quant_ops or []
        self.supported_quant_ops = set(supported_quant_ops + self._QUANT_OPS)

        self.quant = quant

        # TODO(T174256335) - remove this once we have a better way to handle >2d Mask
        self._lower_recomposed_sdpa: bool = (
            _lower_recomposed_sdpa if _lower_recomposed_sdpa is not None else True
        )

        self.delegation_spec = DelegationSpec(XnnpackBackend.__name__, [])
        self.partition_tags: Dict[str, DelegationSpec] = {}

        self.has_dynamic_shapes = has_dynamic_shapes
        if has_dynamic_shapes:
            self.supported_ops = self.supported_ops - set(STATIC_OPS)
            self.supported_modules = self.supported_modules - set(STATIC_MODULES)
            self.supported_quant_ops = self.supported_quant_ops - set(STATIC_OPS)
            self.supported_quant_modules = self.supported_quant_modules - set(
                STATIC_MODULES
            )

    def get_supported_modules(self, quant: bool) -> Set[Callable]:
        """
        Get supported modules
        """
        if quant is True:
            return self.supported_quant_modules
        elif quant is False:
            return self.supported_modules
        else:
            return self.supported_modules | self.supported_quant_modules

    def get_supported_ops(self, quant: Optional[bool]) -> Set[Callable]:
        """
        Get supported ops
        """
        if quant is True:
            return self.supported_quant_ops
        elif quant is False:
            return self.supported_ops
        else:
            return self.supported_ops | self.supported_quant_ops

    @staticmethod
    def check_partitions(partitions: Union[dict, list]) -> bool:
        """
        Warn users if there aren't any matches

        TODO: convert this into a stronger validation, may need a flag in
        `to_backend()` or partitioner __init__()
        """
        pl = len(partitions)
        if pl == 0:
            log.warning("Nothing can be partitioned!")
        else:
            log.info(f"Found {pl} subgraphs to be partitioned.")
        return pl != 0

    def get_input_deps(  # noqa
        self, input_nodes: List[torch.fx.Node], ep: ExportedProgram
    ) -> List[torch.fx.Node]:
        """
        For each input node, walk up and pull necessary quant/attr nodes in the partition
        """
        nodes = set()

        def is_param(ep: ExportedProgram, node) -> bool:
            return isinstance(node, torch.fx.Node) and is_param_node(ep, node)

        def is_q(ep: ExportedProgram, node) -> bool:
            return isinstance(node, torch.fx.Node) and node.target in self._Q_OPS

        def is_dq(ep: ExportedProgram, node) -> bool:
            return isinstance(node, torch.fx.Node) and node.target in self._DQ_OPS

        def is_qparam(node) -> bool:
            return isinstance(node, torch.fx.Node) and node.target in self._QPARAM_OPS

        def is_getitem(node) -> bool:
            return (
                isinstance(node, torch.fx.Node)
                and node.op == "call_function"
                and node.target == operator.getitem
            )

        for inp in input_nodes:
            if is_dq(ep, inp):
                dq = inp

                # Possible graph we want to partition
                #                  op(...)
                #                     ^
                #                     |
                #                     dq(0,   1, 2)
                #                        ^    ^  ^
                #                        |    |  |
                #                        q(0, 1, 2) # optional, only when not folded by the quantizer
                #                          ^  ^  ^
                #                          |  |  |
                # parameter ---------------'  |  |
                #                  [choose_qparams --> get_item(s)]  # optional, only with dynamic quant
                # per_channel_zp* ------------'  |
                # per_channel_scale* ------------'

                # The dequant node
                nodes.add(dq)

                # possible per_channel scale/zp for the dequant node args{1, 2}
                for i in [1, 2]:
                    node = dq.args[i]
                    if is_param(ep, node):
                        nodes.add(node)

                # is it quant or param node?
                prod = dq.args[0]

                assert is_q(ep, prod) or is_param(
                    ep, prod
                ), f"Expecting quant or param node as an input to a dq node, but got {prod.target} for {prod} node"

                nodes.add(prod)

                if is_q(ep, prod):
                    # possible nodes for quant node args{0, 1, 2}: 0: weight, 1: scale, 2: zero_point
                    for i in [0, 1, 2]:
                        node = prod.args[i]  # pyre-ignore

                        # possible choose_qparam
                        if is_getitem(node) and is_qparam(node.args[0]):
                            nodes.add(node)
                            nodes.add(node.args[0])

                        # weights or possible per_channel scale/zp for the quant node
                        elif is_param(ep, node):
                            nodes.add(node)
        return list(nodes)

    def get_output_deps(
        self, output_nodes: List[torch.fx.Node], ep: ExportedProgram
    ) -> List[torch.fx.Node]:
        """
        For each output node, check all the users and insert them into the partition if needed
        """
        nodes = []
        for output in output_nodes:
            for node in output.users:
                if node.target in self._Q_OPS:
                    nodes.append(node)
                    users = list(node.users.keys())
                    for dq_user in users:
                        assert (
                            dq_user.target in self._DQ_OPS
                        ), "Expecting a dq node(s) after a q node, but got target {dq_user.target} for {dq_user} node"
                        nodes.append(dq_user)
        return nodes

    def get_nodes(
        self, src_partition: SourcePartition, ep: ExportedProgram, quant: bool
    ) -> List[torch.fx.Node]:
        """
        Return nodes from the source partition.
        """
        if quant:
            # Insert quantization ops into src_partition by following the input, output node.
            return (
                src_partition.nodes
                + self.get_input_deps(src_partition.input_nodes, ep)
                + self.get_output_deps(src_partition.output_nodes, ep)
            )
        else:
            return src_partition.nodes

    def qualify_nodes(
        self, input_nodes: List[torch.fx.Node], ep: ExportedProgram
    ) -> bool:
        """
        Each node in the module (post decomposition) must satisfy the
        constraints specified for XNNPACK.

        Disqualify the whole module if one of the nodes fails to satisfy.
        """
        return all(
            XnnpackOperatorSupport.check_constraint(node, ep) for node in input_nodes
        )

    def get_module_partitions(
        self,
        ep: ExportedProgram,
        quant: Optional[bool],
    ) -> List[List[torch.fx.Node]]:
        """
        Get all partitions in the torch.fx.GraphModule for the supported
        modules.
        """
        graph_module = ep.graph_module
        if quant is None:
            module_partitions = self.get_module_partitions(ep, True)
            for node_list in module_partitions:
                for node in node_list:
                    node.meta["quant_match"] = True
            fp32_module_partitions = self.get_module_partitions(ep, False)
            for node_list in fp32_module_partitions:
                for node in node_list:
                    if node.meta.get("quant_match", False):
                        break
                else:
                    module_partitions.append(node_list)
            for node_list in module_partitions:
                for node in node_list:
                    node.meta.pop("quant_match", False)
            return module_partitions

        src_partition_dict = get_source_partitions(
            graph_module.graph, self.get_supported_modules(quant)
        )
        all_partitions = src_partition_dict.values()

        module_partitions = []
        for src_partitions in all_partitions:
            for src_partition in src_partitions:
                partition_nodes = self.get_nodes(src_partition, ep, quant)
                if self.qualify_nodes(partition_nodes, ep):
                    module_partitions.append(partition_nodes)

        return module_partitions

    def get_graph_partitions(
        self, ep, quant: Optional[bool]
    ) -> List[List[torch.fx.Node]]:
        graph_module = ep.graph_module
        graphs = bilinear_2d.get_graphs()

        # Temporary for lowering SDPA
        if self._lower_recomposed_sdpa:
            graphs += sdpa.get_graphs()

        graph_patterns = [gm_pattern.graph for gm_pattern in graphs]
        partitions = generate_pattern_op_partitions(
            graph_module, graph_patterns, ignore_literals=True
        )
        graph_partitions = []
        for src_partition in partitions:
            graph_partitions.append(src_partition.nodes)

        return graph_partitions

    def generate_partitions(
        self, ep: ExportedProgram, quant: Optional[bool]
    ) -> List[Any]:
        """
        Generate a list of partitions for an torch.fx.GraphModule.
        Also pass the supported ops to match.
        """
        graph_module = ep.graph_module
        matched_module_nodes = self.get_module_partitions(ep, quant)
        matched_graph_nodes = self.get_graph_partitions(ep, quant)
        return generate_partitions_from_list_of_nodes(
            graph_module,
            matched_module_nodes + matched_graph_nodes,
            XnnpackOperatorSupport(
                ep=ep, supported_ops=list(self.get_supported_ops(quant))
            ),
        )

    def tag_nodes(self, partitions: List[Partition]) -> Dict[str, DelegationSpec]:
        """
        Tag each partition in the list with its delegation tag.
        """
        partition_tags: Dict[str, DelegationSpec] = {}
        for partition in partitions:
            # Add delegation tags
            skip = False
            for node in partition.nodes:
                if "delegation_tag" in node.meta:
                    skip = True
            if skip:
                continue
            for node in partition.nodes:
                delegation_tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                partition_tags[delegation_tag] = self.delegation_spec
        return partition_tags

    # override
    def _partition(
        self, exported_program: ExportedProgram, quant: Optional[bool]
    ) -> PartitionResult:
        """
        Run the partitioner on the given graph module, then tag each partition
        with its delegation tag (and partition id)
        """
        partitions = self.generate_partitions(exported_program, quant)
        partition_tags: Dict[str, DelegationSpec] = {}
        if self.check_partitions(partitions):
            partition_tags = self.tag_nodes(partitions)
        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        ret: PartitionResult = self._partition(exported_program, self.quant)
        return ret


class XnnpackDynamicallyQuantizedPartitioner(XnnpackQuantizedPartitioner):
    def __init__(
        self,
        supported_modules=SUPPORTED_DYN_QUANT_MODULES,
        supported_ops=None,  # no other ops are supported
    ):
        super().__init__(supported_modules, supported_ops)

    # override
    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        """
        Run the partitioner on the given graph module, then tag each partition with its delegegation tag (and partition id)

        We don't want to use `generate_*_partitions` helpers because we don't want these modules to fuse in the same delegate.
        """
        partition_id = itertools.count()
        partitions = [
            Partition(
                id=next(partition_id),
                nodes=set(
                    filter(lambda x: x.target != torch.ops.aten.sym_size.int, match)
                ),
            )
            for match in self.get_module_partitions(exported_program)
        ]
        partition_tags: Dict[str, DelegationSpec] = {}
        self.delegation_spec = DelegationSpec(
            XnnpackBackend.__name__, [CompileSpec("dqlinear_partitioner", bytes())]
        )

        if self.check_partitions(partitions):
            partition_tags = self.tag_nodes(partitions)
        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )
