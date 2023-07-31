# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import operator
from typing import Any, Callable, cast, Dict, List, Optional, Union

import torch

from executorch.backends.canonical_partitioners.pattern_op_partitioner import (
    generate_partitions_from_list_of_nodes,
    generate_pattern_op_partitions,
)

from executorch.backends.partitioner import DelegationSpec, Partitioner
from executorch.backends.xnnpack.partition.support_patterns import (
    get_add_graphs,
    get_all_dynamically_quantized_linear_pattern,
    get_all_fp_linear_pattern,
    get_all_quantized_linear_pattern,
    get_batch_norm_graphs,
    get_clamp_graph,
    get_conv2d_graphs,
    get_div_graph,
    get_floor_graph,
    get_hardtanh_graph,
    get_max_dim_graph,
    get_max_pool2d_graph,
    get_mean_dim_graphs,
    get_minimum_graph,
    get_multiply_graph,
    get_quantized_add_graphs,
    get_quantized_add_relu_graphs,
    get_quantized_conv_graphs,
    get_quantized_conv_relu_graphs,
    get_quantized_hardtanh_graphs,
    get_quantized_max_pool_2d_graphs,
    get_quantized_mean_dim_graphs,
    get_relu_graph,
    get_sigmoid_graph,
    get_softmax_graph,
    get_static_constant_pad_graph,
    get_sub_graph,
)
from executorch.backends.xnnpack.utils.utils import get_input_node
from executorch.backends.xnnpack.xnnpack_preprocess import XnnpackBackend
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.passes.infra.partitioner import Partition
from torch.fx.passes.operator_support import chain, OperatorSupportBase

from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
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
        constraints_dict: Dict[
            Any, Callable[[torch.fx.Node], bool]
        ] = _OP_SUPPORT_CONSTRAINTS,
        supported_ops: Optional[List] = None,
    ):
        """
        @Arg constraints_dict: Dict mapping each node to a lambda function that
             returns True if backend constraints are met for that instance of the
             node.
        @Arg supported_ops: List of supported operators for partitioning
        """
        self.supported_ops = supported_ops
        self.constraints = constraints_dict
        assert len(self.constraints)

    @staticmethod
    def check_constraint(node) -> bool:
        """
        This node is from a partitioned subgraph by one of the partitioners so
        should be a valid node otherwise, let's make sure the constraint is met
        if specified
        """
        return _OP_SUPPORT_CONSTRAINTS.get(node.target, lambda _: True)(node)

    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        # TODO - other ops?
        if node.op != "call_function":
            return False

        # Specifying supported ops is optional
        if self.supported_ops and node.target not in self.supported_ops:
            return False

        return self.check_constraint(node)

    def _constraint(target):  # noqa
        """
        Decorator to register a constraint fn for a node
        """

        def register(func: Callable[[torch.fx.Node], bool]):
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
    def mean_dim(node: torch.fx.Node) -> bool:  # noqa
        """
        Only select 2d cases are supported by XNNPACK
        """
        dims = node.args[1]
        return dims in ([-2, -1], [-1, -2])

    @_constraint(exir_ops.edge.aten.max_pool2d_with_indices.default)
    def maxpool2d_with_indices(node: torch.fx.Node) -> bool:  # noqa
        """
        Only if the first output value is consumed in the graph
        """
        users = list(node.users.keys())
        return (
            True
            if len(users) == 1
            and users[0].target == operator.getitem
            and users[0].args[1] == 0
            else False
        )

    @_constraint(exir_ops.edge.quantized_decomposed.quantize_per_tensor.default)
    def quant_per_tensor_default(q: torch.fx.Node) -> bool:  # noqa
        """
        Decide if we want to pull this q node or not in the partition.
        Given, op1 -> q -> dq -> op2
        For node q, if op1 or op2 is good, q should be good
        TODO: q -> op -> dq, real q not handled right now
        """
        first = XnnpackOperatorSupport.check_constraint(q.args[0])
        dq = list(q.users.keys())[0]
        op2 = list(dq.users.keys())[0]
        return first or XnnpackOperatorSupport.check_constraint(op2)

    @_constraint(exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default)
    def dequant_per_tensor_default(dq: torch.fx.Node) -> bool:  # noqa
        """
        Decide if we want to pull this dq node or not.
        """
        return XnnpackOperatorSupport.check_constraint(dq.args[0])

    @_constraint(exir_ops.edge.quantized_decomposed.quantize_per_channel.default)
    def quant_per_channel_default(q: torch.fx.Node) -> bool:  # noqa
        return XnnpackOperatorSupport.quant_per_tensor_default(q)

    @_constraint(exir_ops.edge.quantized_decomposed.dequantize_per_channel.default)
    def dequant_per_channel_default(dq: torch.fx.Node) -> bool:  # noqa
        return XnnpackOperatorSupport.dequant_per_tensor_default(dq)

    @_constraint(exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor)
    def quant_per_tensor_tensor(q: torch.fx.Node) -> bool:  # noqa
        return XnnpackOperatorSupport.quant_per_tensor_default(q)

    @_constraint(exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor)
    def dequant_per_tensor_tensor(dq: torch.fx.Node) -> bool:  # noqa
        return XnnpackOperatorSupport.dequant_per_tensor_default(dq)

    @_constraint(exir_ops.edge.quantized_decomposed.choose_qparams.tensor)
    def choose_qparams_tensor(cqp: torch.fx.Node) -> bool:  # noqa
        """
        Given, cqp -> getitem -> q -> dq -> op2
        Just check q, because it will check op2
        """
        getitem0 = list(cqp.users.keys())[0]
        q = list(getitem0.users.keys())[0]
        return XnnpackOperatorSupport.check_constraint(q)

    @_constraint(exir_ops.edge.aten.pow.Tensor_Scalar)
    def pow_tensor_scalar(node: torch.fx.Node) -> bool:  # noqa
        """
        Only supports square, when args_2 = 2
        """
        power = node.args[1]
        return isinstance(power, int) and power == 2

    @_constraint(exir_ops.edge.aten.avg_pool2d.default)
    def avg_pool_2d(node: torch.fx.Node) -> bool:  # noqa
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
    def prelu(node: torch.fx.Node) -> bool:  # noqa
        """
        Input and Weight must be 4-dimensional
        """
        input_dim = cast(torch.fx.Node, node.args[0]).meta["val"].dim()
        weight_dim = cast(torch.fx.Node, node.args[1]).meta["val"].dim()
        return input_dim == 4 and weight_dim == 4

    @_constraint(exir_ops.edge.aten.cat.default)
    def cat(node: torch.fx.Node) -> bool:  # noqa
        """
        Only support concatenation of 2 - 4 tensors
        """
        num_tensors = len(cast(List[torch.fx.Node], node.args[0]))
        return num_tensors >= 2 and num_tensors <= 4

    @_constraint(exir_ops.edge.aten.slice_copy.Tensor)
    def slice_copy(node: torch.fx.Node) -> bool:  # noqa
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


###
### Graph pattern based partitioners
###

FLOATING_POINT_PATTERNS = (
    [
        get_div_graph(),
        get_sigmoid_graph(),
        get_softmax_graph(),
        get_hardtanh_graph(),
        get_relu_graph(),
        get_static_constant_pad_graph(),
        get_clamp_graph(),
        get_minimum_graph(),
        get_max_pool2d_graph(),
        get_max_dim_graph(),
        get_multiply_graph(),
        get_sub_graph(),
        get_floor_graph(),
    ]
    + get_add_graphs()
    + get_all_fp_linear_pattern()
    + get_conv2d_graphs()
    # + get_static_resize_bilinear_2d_graphs() TODO(T148779166) recompose bilinear
    + get_batch_norm_graphs()
    + get_mean_dim_graphs()
)

QUANTIZED_PATTERNS = (
    get_all_quantized_linear_pattern()
    + get_quantized_conv_graphs()
    + get_quantized_hardtanh_graphs()
    + get_quantized_mean_dim_graphs()
    + get_quantized_add_graphs()
    + get_quantized_max_pool_2d_graphs()
    + get_quantized_conv_relu_graphs()
    + get_quantized_add_relu_graphs()
)


class _BasePartitioner(Partitioner):
    """
    Graph based partitioner base for on XNNPACK backend.
    """

    def __init__(self, delegate_name, patterns):
        self.patterns = patterns

        self.delegation_spec = DelegationSpec(delegate_name, [])
        self.partition_tags: Dict[str, DelegationSpec] = {}

    @staticmethod
    def check_partitions(partitions: Union[dict, list]) -> None:
        """
        Warn users if there aren't any matches
        """
        pl = len(partitions)
        if pl == 0:
            log.warning("Nothing can be partitioned!")
        else:
            log.info(f"Found {pl} subgraphs to be partitioned.")

    def partition(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        class MatchTag(OperatorSupportBase):
            def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
                return node.meta.get("match", False)

        partition_list = generate_pattern_op_partitions(
            graph_module,
            self.patterns,
            op_support=chain(XnnpackOperatorSupport(), MatchTag()),
            ignore_literals=True,
        )

        self.check_partitions(partition_list)

        for partition in partition_list:
            # Add delegation tags
            for node in partition.nodes:
                delegation_tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                self.partition_tags[delegation_tag] = self.delegation_spec
        return graph_module


# TODO - Update pye/.../test_xnnpack_asr.py to use dqlinear
class XnnpackPartitioner(_BasePartitioner):
    """
    Graph based partitioner base for on XNNPACK backend given patterns
    """

    def __init__(self, patterns):
        super().__init__(XnnpackBackend.__name__, patterns)


class XnnpackModelPartitioner(_BasePartitioner):
    """
    Graph based partitioner base for on XNNPACK backend for quantized as well as floating point patterns
    listed in QUANTIZED_PATTERNS and FLOATING_POINT_PATTERNS.
    """

    def __init__(self):
        super().__init__(
            XnnpackBackend.__name__, QUANTIZED_PATTERNS + FLOATING_POINT_PATTERNS
        )


# TODO(T143912091): Merge XnnpackQuantizedPartitioner and XnnpackFloatingPointPartitioner
# when capturing quantized model is faster. Right now it's too slow, so we separate them
# as we mainly focus on fp operators right now.
class XnnpackQuantizedPartitioner(_BasePartitioner):
    """
    Graph based partitioner base for XNNPACK backend for quantize ops only.
    """

    def __init__(self):
        super().__init__(XnnpackBackend.__name__, QUANTIZED_PATTERNS)


class _SingleOpDelegatePartitioner(_BasePartitioner):
    """
    Graph based partitioner base for a single "op" or "node" or a pattern match for XNNPACK backend.
    This is tailored for DQLinear where XNNPACK (and also QNNPACK) delegates prefers to have a single DQLinear node in the graph.
    This is a base class given XNNPACK and QNNPACK currently share this.
    """

    def __init__(
        self,
        delegate_name,
        patterns,
        transforms: Optional[List[Callable[[torch.fx.Graph], torch.fx.Graph]]] = None,
    ):
        """
        @param transforms: Optional list of transforms that will be applied to the graph before running the partitioner.
        """
        super().__init__(delegate_name, patterns)
        self.transforms = transforms

    # override
    def partition(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        # TODO delete this since we are not allowed to do this
        if self.transforms is not None:
            for transform in self.transforms:  # pyre-ignore
                graph_module.graph = transform(graph_module.graph)

        matches = [
            match
            for matches in (
                SubgraphMatcher(pattern, ignore_literals=True).match(graph_module.graph)
                for pattern in self.patterns
            )
            for match in matches
        ]

        match_sets = [
            {
                node_in_graph
                for (node_in_pattern, node_in_graph) in match.nodes_map.items()
                if (
                    node_in_pattern.op != "placeholder"
                    and node_in_graph.op != "placeholder"
                )
            }
            for match in matches
        ]

        # Sort match sets in descending order of length so that any match sets
        # which are supersets of other match sets are processed first
        match_sets = sorted(match_sets, key=len, reverse=True)

        self.check_partitions(match_sets)

        # Mapping from delegation tag to match set
        tag_mapping = {}

        for (partition_id, match_set) in enumerate(match_sets):
            delegation_tag = f"tag{partition_id}"
            for node in match_set:
                if "delegation_tag" in node.meta:
                    # This node already has delegation tag assigned.
                    # Check that the current match set is a subset of the one
                    # used to assign its delegation tag, then skip this match
                    # set. We have this check to ensure there are no pairs of
                    # match sets where they are overlapping but neither is a
                    # subset of the other.
                    if not match_set.issubset(tag_mapping[node.meta["delegation_tag"]]):
                        raise AssertionError(
                            f"Found match sets which are overlapping but neither is a subset of the other: {match_set}, {tag_mapping[node.meta['delegation_tag']]}"
                        )
                    break
                node.meta["delegation_tag"] = delegation_tag
            self.partition_tags[delegation_tag] = self.delegation_spec
            tag_mapping[delegation_tag] = match_set

        return graph_module


class XnnpackDynamicallyQuantizedPartitioner(_SingleOpDelegatePartitioner):
    """
    For XnnpackDynamicallyQuantizedPartitioner, we want to manually
    use SubgraphMatcher to assign partitions. This is because the default
    partitioning process causes addmm/mm partitions within close proximity
    of each other to be merged together. We don't want this merging to
    happening because there is only support for having exactly one addmm
    or mm per dqlinear.
    """

    def __init__(self):
        super().__init__(
            XnnpackBackend.__name__, get_all_dynamically_quantized_linear_pattern()
        )


###
### Module based partitioners
###

SUPPORTED_OPS = [
    exir_ops.edge.aten.div.Tensor,
    exir_ops.edge.aten.add.Tensor,
    exir_ops.edge.aten.clamp.default,
    exir_ops.edge.aten.sub.Tensor,
    exir_ops.edge.aten.floor.default,
    exir_ops.edge.aten.maximum.default,
    exir_ops.edge.aten.minimum.default,
    exir_ops.edge.aten.mul.Tensor,
    exir_ops.edge.aten.constant_pad_nd.default,
    exir_ops.edge.aten.upsample_bilinear2d.default,
    exir_ops.edge.aten.mean.dim,
    exir_ops.edge.aten.max.dim,
    exir_ops.edge.aten.hardtanh.default,
    exir_ops.edge.aten.sqrt.default,
    exir_ops.edge.aten.ceil.default,
    exir_ops.edge.aten.hardswish.default,
    exir_ops.edge.aten.neg.default,
    exir_ops.edge.aten.pow.Tensor_Scalar,
    exir_ops.edge.aten.abs.default,
    exir_ops.edge.aten._prelu_kernel.default,
    exir_ops.edge.aten.slice_copy.Tensor,
]

SUPPORTED_MODULES = [
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.ReLU,
    torch.nn.Sigmoid,
    torch.nn.Softmax,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.Linear,
    torch.nn.functional.linear,
    torch.nn.Hardtanh,
    torch.nn.MaxPool2d,
    torch.nn.LeakyReLU,
    torch.nn.ELU,
    torch.nn.AvgPool2d,
    torch.nn.PReLU,  # Without this, the PReLU weight becomes not a get_attr
    torch.cat,
    torch.concat,
    torch.concatenate,
]

# TODO delete this and should use SUPPORTED_OPS instead once we align fp32 and quant support
SUPPORTED_QUANT_OPS = [
    exir_ops.edge.aten.add.Tensor,
    exir_ops.edge.aten.sub.Tensor,
    exir_ops.edge.aten.mul.Tensor,
    exir_ops.edge.aten.mean.dim,
    exir_ops.edge.aten.hardtanh.default,  # TODO - which one module or op or both?
    exir_ops.edge.aten.slice_copy.Tensor,
]

# TODO delete this and should use SUPPORTED_MODULES instead once we align fp32 and quant support
SUPPORTED_QUANT_MODULES = [
    torch.clamp,
    torch.mean,
    torch.permute,
    torch.permute_copy,
    torch.cat,
    torch.concat,
    torch.concatenate,
    torch.nn.Linear,
    torch.nn.functional.linear,
    # TODO - T158982884
    # torch.ao.nn.quantized.reference.modules.linear.Linear,
    torch.nn.MaxPool2d,
    torch.nn.Conv1d,
    torch.nn.functional.conv1d,
    torch.ao.nn.quantized.reference.modules.conv.Conv1d,
    torch.nn.Conv2d,
    torch.nn.functional.conv2d,
    torch.nn.functional.pad,
    torch.nn.functional.elu,
    torch.ao.nn.quantized.reference.modules.conv.Conv2d,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.ConstantPad2d,
    torch.nn.ELU,
    torch.nn.Hardtanh,
    torch.nn.ReLU,
    torch.nn.functional.relu,
    torch.nn.functional.relu_,
    torch.nn.functional.leaky_relu,
    torch.nn.functional.leaky_relu_,
    torch.nn.LeakyReLU,
]

# Modules which support dynamic quantization
SUPPORTED_DYN_QUANT_MODULES = [
    torch.nn.Linear,
    torch.nn.functional.linear,
]


class XnnpackFloatingPointPartitioner(Partitioner):
    """
    Module and Opname based partitioner for FP32 modules/ops listed in
    SUPPORTED_MODULES and SUPPORTED_OPS.
    """

    def __init__(
        self,
        supported_modules: List[Callable] = SUPPORTED_MODULES,
        supported_ops: Optional[List[Callable]] = SUPPORTED_OPS,
    ):
        super().__init__()
        self.supported_modules = set(supported_modules)

        self.supported_ops = set(supported_ops or [])

        self.delegation_spec = DelegationSpec(XnnpackBackend.__name__, [])
        self.partition_tags: Dict[str, DelegationSpec] = {}

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

    def get_nodes(self, src_partition: SourcePartition) -> List[torch.fx.Node]:
        """
        Return nodes from the source partition.

        This is a wrapper to allow derived classes to add their own custom
        logic to extend the src_partition nodes list.
        """
        return src_partition.nodes

    def qualify_nodes(self, input_nodes: List[torch.fx.Node]) -> bool:
        """
        Each node in the module (post decomposition) must satisfy the
        constraints specified for XNNPACK.

        Disqualify the whole module if one of the nodes fails to satisfy.
        """
        return all(
            [XnnpackOperatorSupport.check_constraint(node) for node in input_nodes]
        )

    def get_module_partitions(
        self, graph_module: torch.fx.GraphModule
    ) -> List[List[torch.fx.Node]]:
        """
        Get all partitions in the torch.fx.GraphModule for the supported
        modules.
        """
        src_partition_dict = get_source_partitions(
            graph_module.graph, self.supported_modules
        )
        all_partitions = src_partition_dict.values()

        module_partitions = []
        for src_partitions in all_partitions:
            for src_partition in src_partitions:
                partition_nodes = self.get_nodes(src_partition)
                if self.qualify_nodes(partition_nodes):
                    module_partitions.append(partition_nodes)

        return module_partitions

    def generate_partitions(self, graph_module: torch.fx.GraphModule) -> List[Any]:
        """
        Generate a list of partitions for an torch.fx.GraphModule.
        Also pass the supported ops to match.
        """
        matched_module_nodes = self.get_module_partitions(graph_module)
        return generate_partitions_from_list_of_nodes(
            graph_module,
            matched_module_nodes,
            XnnpackOperatorSupport(supported_ops=self.supported_ops),
        )

    def tag_nodes(self, partitions: List[Partition]) -> None:
        """
        Tag each partition in the list with its delegation tag.
        """
        for partition in partitions:
            # Add delegation tags
            for node in partition.nodes:
                delegation_tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                self.partition_tags[delegation_tag] = self.delegation_spec

    # override
    def partition(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """
        Run the partitioner on the given graph module, then tag each partition
        with its delegation tag (and partition id)
        """
        partitions = self.generate_partitions(graph_module)
        if self.check_partitions(partitions):
            self.tag_nodes(partitions)
        return graph_module


# TODO: Merge XnnpackQuantizedPartitioner and XnnpackFloatingPointPartitioner
class XnnpackQuantizedPartitioner2(XnnpackFloatingPointPartitioner):
    """
    Module and Opname based partitioner for statically quantized modules/ops listed in SUPPORTED_QUANT_MODULES and SUPPORTED_QUANT_OPS.
    """

    _Q_OPS = [
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor,
    ]

    _DQ_OPS = [
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
    ]

    _QPARAM_OPS = [
        exir_ops.edge.quantized_decomposed.choose_qparams.tensor,
    ]

    _QUANT_OPS = _Q_OPS + _DQ_OPS + _QPARAM_OPS

    def __init__(
        self,
        supported_modules=SUPPORTED_QUANT_MODULES,
        supported_ops=SUPPORTED_QUANT_OPS,
    ):
        supported_ops = supported_ops or []
        super().__init__(supported_modules, supported_ops + self._QUANT_OPS)

    # TODO Refactor this
    # TODO Don't be greedy when pulling q->dq pairs for a given op, add convert tracker pass
    def get_input_deps(  # noqa
        self, input_nodes: List[torch.fx.Node]
    ) -> List[torch.fx.Node]:
        """
        For each input node, walk up and pull necessary quant/attr nodes in the partition
        """
        nodes = set()
        for inp in input_nodes:
            if inp.target in self._DQ_OPS:
                # dequant node
                nodes.add(inp)

                # possible per_channel scale/zp for the dequant node args{1, 2}
                for i in [1, 2]:
                    node = inp.args[i]
                    if isinstance(node, torch.fx.Node) and node.op == "get_attr":
                        nodes.add(node)

                # quant node
                q_prod = inp.args[0]
                assert (
                    isinstance(q_prod, torch.fx.Node) and q_prod.target in self._Q_OPS
                )
                nodes.add(q_prod)

                # possible weight for the quant node arg{0}
                node = q_prod.args[0]
                if isinstance(node, torch.fx.Node) and node.op == "get_attr":
                    nodes.add(node)

                # possible nodes for quant node args{1, 2}
                for i in [1, 2]:
                    node = q_prod.args[i]
                    # possible choose_qparam
                    if (
                        isinstance(node, torch.fx.Node)
                        and node.op == "call_function"
                        and node.target == operator.getitem
                    ):
                        parent = node.args[0]
                        if (
                            isinstance(parent, torch.fx.Node)
                            and parent.op == "call_function"
                            and parent.target in self._QPARAM_OPS
                        ):
                            nodes.add(node)
                            nodes.add(parent)

                    # possible per_channel scale/zp for the quant node
                    elif isinstance(node, torch.fx.Node) and node.op == "get_attr":
                        nodes.add(node)
        return list(nodes)

    def get_output_deps(self, output_nodes: List[torch.fx.Node]) -> List[torch.fx.Node]:
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
    def get_nodes(self, src_partition: SourcePartition) -> List[torch.fx.Node]:  # noqa
        """
        Insert quantization ops into src_partition by following the input, output node.
        """
        return (
            src_partition.nodes
            + self.get_input_deps(src_partition.input_nodes)
            + self.get_output_deps(src_partition.output_nodes)
        )


class XnnpackDynamicallyQuantizedPartitioner2(XnnpackQuantizedPartitioner2):
    def __init__(
        self,
        supported_modules=SUPPORTED_DYN_QUANT_MODULES,
        supported_ops=None,  # no other ops are supported
    ):
        super().__init__(supported_modules, supported_ops)

    # override
    def partition(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """
        Run the partitioner on the given graph module, then tag each partition with its delegegation tag (and partition id)

        We don't want to use `generate_*_partitions` helpers because we don't want these modules to fuse in the same delegate.
        """
        partition_id = itertools.count()
        partitions = [
            Partition(
                id=next(partition_id),
                nodes=set(match),
            )
            for match in self.get_module_partitions(graph_module)
        ]

        if self.check_partitions(partitions):
            self.tag_nodes(partitions)
        return graph_module
