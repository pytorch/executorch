# Copyright © 2023 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import inspect
import logging
import operator
import weakref
from typing import Any, Callable, Collection, Dict, List, Optional, Set, Tuple

import coremltools as ct

import torch

from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.compiler.coreml_preprocess import (
    COMPILE_SPEC_KEYS,
)

from executorch.backends.apple.coreml.logging import get_coreml_log_level
from executorch.exir.backend.compile_spec_schema import CompileSpec

from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data, tag_mutated_buffer
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.dim_order_ops_registry import DimOrderOpsMap
from executorch.exir.passes.replace_aten_with_edge_pass import aten_to_edge
from executorch.exir.tracer import _default_decomposition_table
from torch._export.error import InternalError
from torch._export.passes.replace_view_ops_with_view_copy_ops_pass import (
    get_view_copy_of_view_op,
)
from torch.export.exported_program import ExportedProgram, OutputKind
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
    is_concrete_int,
    statically_known_true,
    sym_eq,
)
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition
from torch.fx.passes.operator_support import (
    chain,
    create_op_support,
    OperatorSupportBase,
)

logger = logging.getLogger(__name__)
logger.setLevel(get_coreml_log_level(default_level=logging.INFO))


# Ops that the CoreML partitioner must always reject regardless of their
# arguments.  Each entry is annotated with the upstream issue that motivates
# it so future readers can tell when an entry is safe to drop.
_UNSUPPORTED_OP_TARGETS = frozenset(
    [
        # https://github.com/apple/coremltools/issues/2565 — diagonal has a
        # CoreML correctness bug.
        torch.ops.aten.diagonal.default,
        torch.ops.aten.diagonal_copy.default,
        exir_ops.edge.aten.diagonal.default,
        exir_ops.edge.aten.diagonal_copy.default,
        # https://github.com/apple/coremltools/issues/2569 — acosh / asinh
        # are not implemented in coremltools.
        torch.ops.aten.acosh.default,
        exir_ops.edge.aten.acosh.default,
        torch.ops.aten.asinh.default,
        exir_ops.edge.aten.asinh.default,
        # https://github.com/pytorch/executorch/issues/11722 — only
        # ``aten.rand.default`` actually reaches an unimplemented branch in
        # coremltools 9.0 ("not enough values to unpack (expected 5, got 1)"
        # raised from the rand handler).  ``randn`` / ``rand_like`` /
        # ``randn_like`` / ``randint`` lower cleanly today, so we leave them
        # delegated.  Verified locally against coremltools 9.0 / Python 3.10
        # by lowering each op in isolation.
        torch.ops.aten.rand.default,
        exir_ops.edge.aten.rand.default,
    ]
)


# Ops that coremltools can carry through a tensor above rank 5, by rewriting a
# reshape -> transpose -> reshape chain into rank 4 steps.
_HIGH_RANK_LAYOUT_TARGETS = frozenset(
    [
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.aten.permute_copy.default,
        exir_ops.edge.aten.clone.default,
        exir_ops.edge.dim_order_ops._clone_dim_order.default,
    ]
)


def _produces_tensor_above_rank_5(node: torch.fx.Node) -> bool:
    val = node.meta.get("val")
    vals = list(val) if isinstance(val, (list, tuple)) else [val]
    return any(isinstance(v, torch.Tensor) and v.dim() > 5 for v in vals)


def _touches_unsupported_high_rank_tensor(
    node: torch.fx.Node, is_skipped: Callable[[torch.fx.Node], bool]
) -> bool:
    """Core ML tensors have at most rank 5.

    coremltools 9.0 raises "Core ML only supports tensors with rank <= 5" for
    any tensor above rank 5 that it cannot rewrite away. It only rewrites a
    static, unbranched layout chain, so such a tensor is fine only when every
    node connected to it through tensors above rank 5 is a delegated view,
    permute or clone, and every node producing such a tensor has a static shape
    and a single user.
    """
    seen, todo = set(), [node]
    while todo:
        n = todo.pop()
        if n in seen:
            continue
        seen.add(n)
        high_rank_inputs = [
            arg for arg in n.all_input_nodes if _produces_tensor_above_rank_5(arg)
        ]
        high_rank_output = _produces_tensor_above_rank_5(n)
        if not high_rank_inputs and not high_rank_output:
            continue
        if (
            n.op != "call_function"
            or n.target not in _HIGH_RANK_LAYOUT_TARGETS
            or is_skipped(n)
        ):
            return True
        if high_rank_output and (
            len(n.users) != 1
            or any(isinstance(d, torch.SymInt) for d in n.meta["val"].shape)
        ):
            return True
        todo += high_rank_inputs
        if high_rank_output:
            todo += list(n.users)
    return False


_ARG_MIN_MAX_TARGETS = (
    torch.ops.aten.argmax.default,
    torch.ops.aten.argmin.default,
    exir_ops.edge.aten.argmax.default,
    exir_ops.edge.aten.argmin.default,
)


def _is_arg_min_max_over_flattened_input(node: torch.fx.Node) -> bool:
    """``argmin``/``argmax`` with ``dim=None`` reduces over the flattened input.

    CoreML doesn't support that reduction shape and intermittently crashes
    the process at runtime — see pytorch/executorch#11715.
    """
    if node.target not in _ARG_MIN_MAX_TARGETS:
        return False
    dim = node.args[1] if len(node.args) >= 2 else node.kwargs.get("dim", None)
    return dim is None


def _reads_or_produces_bfloat16(node: torch.fx.Node) -> bool:
    """coremltools 9.0 has no bfloat16 type.

    It raises ``KeyError: torch.bfloat16`` on any op that reads or produces a
    bfloat16 tensor, instead of reporting the op as unsupported.
    """
    val = node.meta.get("val")
    vals = list(val) if isinstance(val, (list, tuple)) else [val]
    vals += [arg.meta.get("val") for arg in node.all_input_nodes]
    return any(isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16 for v in vals)


# No upstream issue yet. For example, Core ML fails to load x.sum().sum(): it gets
# the second sum as a reduce_sum of a 0-d tensor, "Param 'x' has incorrect type
# for operator 'reduce_sum'". A std or vector norm of a 0-d tensor fails too, but
# stays: it has no CPU kernel, and a refused call can decompose into an op that
# another call keeps whole.
_REDUCTION_TARGETS = frozenset(
    target
    for aten in (torch.ops.aten, exir_ops.edge.aten)
    for target in (
        aten.sum.default,
        aten.sum.dim_IntList,
        aten.mean.default,
        aten.mean.dim,
        aten.amax.default,
        aten.amin.default,
        aten.max.default,
        aten.max.dim,
        aten.min.default,
        aten.min.dim,
        aten.argmax.default,
        aten.argmin.default,
        aten.any.default,
        aten.any.dim,
        aten.any.dims,
        aten.all.default,
        aten.all.dim,
        aten.all.dims,
        aten.var.default,
        aten.var.dim,
        aten.var.correction,
    )
)
_SUM_OVER_DIMS = (torch.ops.aten.sum.dim_IntList, exir_ops.edge.aten.sum.dim_IntList)


def _reduces_0d_tensor(node: torch.fx.Node) -> bool:
    """Core ML cannot run the reductions above on a 0-d tensor: its reduce ops
    need rank 1 or more.

    A sum over a given dim, such as x.sum().sum(0), stays and still fails to
    lower, since ExecuTorch's optimized sum kernel aborts on it at run time.
    """
    if node.target not in _REDUCTION_TARGETS or (
        node.target in _SUM_OVER_DIMS and len(node.args) > 1 and node.args[1]
    ):
        return False
    val = node.args[0].meta.get("val")
    return isinstance(val, torch.Tensor) and val.dim() == 0


def _makes_tensor(
    node: torch.fx.Node, dtypes: Optional[Collection[torch.dtype]] = None
) -> bool:
    """Whether node makes a tensor, of one of dtypes if given."""
    val = node.meta.get("val")
    vals = val if isinstance(val, (list, tuple)) else [val]
    return any(
        isinstance(v, torch.Tensor) and (dtypes is None or v.dtype in dtypes)
        for v in vals
    )


def _makes_symbolic_value(node: torch.fx.Node) -> bool:
    """Whether node makes a symbolic value, such as a size known only at run
    time."""
    return isinstance(
        node.meta.get("val"), (torch.SymInt, torch.SymFloat, torch.SymBool)
    )


def _dynamic_size_names(ep: ExportedProgram) -> Set[str]:
    """The names of the dynamic sizes ep has range constraints for, which are
    the sizes coremltools can give a Core ML input besides numbers."""
    return {str(symbol) for symbol in ep.range_constraints}


# https://github.com/pytorch/executorch/issues/11686
def _cannot_be_coreml_input(node: torch.fx.Node, sizes: Collection[str]) -> bool:
    """Whether node makes a tensor Core ML cannot take as a program input: a
    uint8 or int8 tensor, or one with a size that is neither a number nor in
    sizes, such as s0 + 1.

    coremltools types each program input through ``TORCH_DTYPE_TO_MIL_DTYPE``
    and raises ``KeyError`` for a dtype missing from it instead of reporting the
    op as unsupported. Only uint8 and int8 move to the CPU: the portable kernels
    handle them, while other missing dtypes, such as uint16, would only fail
    later, at run time. It looks each dynamic size up by name in the range
    constraints of the program, and raises ``ValueError`` for one it cannot
    find.
    """
    val = node.meta.get("val")
    vals = val if isinstance(val, (list, tuple)) else [val]
    return any(
        isinstance(v, torch.Tensor)
        and (
            v.dtype in (torch.uint8, torch.int8)
            or any(
                not is_concrete_int(size) and str(size) not in sizes for size in v.shape
            )
        )
        for v in vals
    )


def _cannot_be_read_from_outside(
    arg: torch.fx.Node,
    kept_inside: Collection[str],
    sizes: Collection[str],
    outside_coreml: Callable[[torch.fx.Node], bool],
) -> bool:
    """Whether a Core ML op cannot read arg from outside its partition: a
    tensor Core ML cannot take as an input, or a symbolic value computed
    outside of Core ML, which it cannot take either. A weight, constant or
    state the partitioner moves into Core ML is not an input.
    """
    return arg.name not in kept_inside and (
        _cannot_be_coreml_input(arg, sizes)
        or (_makes_symbolic_value(arg) and outside_coreml(arg))
    )


def _find_readers_of_unsupported_inputs(
    partitions: List[Partition], kept_inside: Collection[str], sizes: Collection[str]
) -> Set[torch.fx.Node]:
    """The nodes that read a tensor Core ML cannot take as an input from outside
    their partition, or a symbolic value computed outside of Core ML.
    """
    inside = {node for partition in partitions for node in partition.nodes}
    return {
        node
        for partition in partitions
        for node in partition.nodes
        if any(
            arg not in partition.nodes
            and _cannot_be_read_from_outside(
                arg, kept_inside, sizes, lambda arg: arg not in inside
            )
            for arg in node.all_input_nodes
        )
    }


def _cannot_be_coreml_output(node: torch.fx.Node) -> bool:
    """Whether node makes a value the Core ML delegate cannot return.

    The delegate has no int8 data type to hand an int8 tensor back to
    ExecuTorch, and it returns only tensors, so a symbolic value cannot be an
    output either.
    """
    return _makes_tensor(node, (torch.int8,)) or _makes_symbolic_value(node)


def _find_makers_of_unsupported_outputs(
    partitions: List[Partition], state_writes: Collection[torch.fx.Node]
) -> Set[torch.fx.Node]:
    """The nodes that make a value the Core ML delegate cannot return, for use
    outside their partition. A write of state that Core ML keeps is no output.

    An op with several outputs hands each one out through a getitem, which
    always stays with the op, so the op itself is returned.
    """
    return {
        node.args[0] if node.target is operator.getitem else node
        for partition in partitions
        for node in partition.nodes
        if _cannot_be_coreml_output(node)
        and any(
            user not in partition.nodes
            and not (user.op == "output" and node in state_writes)
            for user in node.users
        )
    }


def _find_unsupported_inputs_of_left_out_ops(
    left_out: Set[torch.fx.Node],
    inside: Set[torch.fx.Node],
    kept_inside: Collection[str],
    sizes: Collection[str],
) -> Set[torch.fx.Node]:
    """The tensors Core ML cannot take as an input that explain why the ops in
    left_out run outside of Core ML."""
    read: Set[torch.fx.Node] = set()
    for node in left_out:
        args = [
            arg
            for arg in node.all_input_nodes
            if arg.name not in kept_inside
            and isinstance(arg.meta.get("val"), torch.Tensor)
            and _cannot_be_coreml_input(arg, sizes)
        ]
        # A value the delegate cannot return, made by an op left out of Core
        # ML, is no reason for its readers: that op may be out only for them.
        outside = [
            arg
            for arg in args
            if arg not in inside
            and not (arg in left_out and _cannot_be_coreml_output(arg))
        ]
        if outside:
            read.update(arg for arg in outside if arg not in left_out)
        # A tensor that a Core ML partition makes explains the op only when no
        # tensor from outside does, and the op is not left out for making a
        # value the delegate cannot return.
        elif not (
            _cannot_be_coreml_output(node)
            and any(user not in left_out for user in node.users)
        ):
            read.update(arg for arg in args if arg in inside)
    return read


def _find_nodes_of_partitions_without_tensor_inputs(
    partitions: List[Partition],
    kept_inside: Collection[str],
    allow_partitions_without_inputs: bool,
) -> Set[torch.fx.Node]:
    """The nodes of the partitions that read no tensor from outside, except the
    weights, constants and state the partitioner moves into Core ML.

    Core ML cannot take a symbolic size as an input, and below iOS 18 it
    rejects a model with no inputs, so such a partition cannot be delegated.
    """
    nodes = set()
    for partition in partitions:
        inputs = [
            arg
            for node in partition.nodes
            for arg in node.all_input_nodes
            if arg not in partition.nodes and arg.name not in kept_inside
        ]
        reads_tensor = any(_makes_tensor(arg) for arg in inputs)
        if not reads_tensor and (inputs or not allow_partitions_without_inputs):
            nodes.update(partition.nodes)
    return nodes


def _find_ops_left_out_of_any_partition(
    graph_module: torch.fx.GraphModule,
    op_support: OperatorSupportBase,
    kept_inside: Collection[str],
    sizes: Collection[str],
    state_writes: Collection[torch.fx.Node],
) -> Set[torch.fx.Node]:
    """The ops _propose_partitions leaves out however it partitions the rest:
    each op that reads a tensor Core ML cannot take as an input, or a symbolic
    value, from an op outside of Core ML, and each op that makes an int8 tensor
    or a symbolic value an op outside of Core ML uses, other than to write
    state Core ML may keep.
    """
    left_out: Set[torch.fx.Node] = set()
    supported = {
        node
        for node in graph_module.graph.nodes
        if op_support.is_node_supported(None, node)
    }

    def outside(node: torch.fx.Node) -> bool:
        return node in left_out or node not in supported

    # Leaving an op out can make an op before or after it read or make such a
    # tensor across the edge of Core ML, so repeat until nothing changes.
    changed = True
    while changed:
        changed = False
        for node in graph_module.graph.nodes:
            if outside(node):
                continue
            if any(
                outside(arg)
                and _cannot_be_read_from_outside(arg, kept_inside, sizes, outside)
                for arg in node.all_input_nodes
            ):
                left_out.add(node)
                changed = True
            elif _cannot_be_coreml_output(node) and any(
                outside(user) and not (user.op == "output" and node in state_writes)
                for user in node.users
            ):
                maker = node.args[0] if node.target is operator.getitem else node
                changed |= maker not in left_out
                left_out.add(maker)
    return left_out


def _computed_from_kept_inside(
    graph_module: torch.fx.GraphModule, kept_inside: Collection[str]
) -> Set[torch.fx.Node]:
    """The nodes that read no tensor but the weights, constants and state the
    partitioner moves into Core ML, or nodes computed from them."""
    nodes: Set[torch.fx.Node] = set()
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            if node.name in kept_inside:
                nodes.add(node)
        elif all(arg in nodes for arg in node.all_input_nodes if _makes_tensor(arg)):
            nodes.add(node)
    return nodes


def _find_ops_partitioning_may_leave_out(
    graph_module: torch.fx.GraphModule,
    kept_inside: Collection[str],
    sizes: Collection[str],
    is_supported: Callable[[torch.fx.Node], bool],
    states: Collection[str],
) -> Set[torch.fx.Node]:
    """The ops _propose_partitions may leave out of Core ML.

    Those that read a tensor Core ML cannot take as an input, make a value the
    delegate cannot return, or read a symbolic value from an op that may be
    outside of Core ML set it off. State can stay in the program, so reading it
    can be reading an input. Once it leaves out one, a partition left without
    tensor inputs moves out too, and such a partition holds only ops that
    compute from the weights, constants and state the partitioner moves into
    Core ML.
    """
    weights_and_constants = set(kept_inside).difference(states)
    set_off: Set[torch.fx.Node] = set()
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            continue
        if _cannot_be_coreml_output(node) or any(
            _cannot_be_read_from_outside(
                arg,
                weights_and_constants,
                sizes,
                lambda arg: arg in set_off or not is_supported(arg),
            )
            for arg in node.all_input_nodes
        ):
            set_off.add(node)
    if not set_off:
        return set()
    return set_off | _computed_from_kept_inside(graph_module, kept_inside)


def _states_left_in_program(
    partitions: List[Partition],
    state_writers: Dict[torch.fx.Node, List[torch.fx.Node]],
) -> Set[str]:
    """The state tag_mutated_buffer leaves in the program for these partitions:
    state that an op outside one partition reads or writes, also through its
    new value.
    """
    partition_of = {
        node: partition.id for partition in partitions for node in partition.nodes
    }
    left = set()
    for state, writers in state_writers.items():
        users = [*state.users, *writers]
        users += [
            user for writer in writers for user in writer.users if user.op != "output"
        ]
        ids = {partition_of.get(user) for user in users}
        if None in ids or len(ids) > 1:
            left.add(state.name)
    return left


def _propose_partitions(
    graph_module: torch.fx.GraphModule,
    op_support: OperatorSupportBase,
    kept_inside: Collection[str],
    sizes: Collection[str],
    allow_partitions_without_inputs: bool,
    state_writers: Dict[torch.fx.Node, List[torch.fx.Node]],
) -> Tuple[List[Partition], Set[torch.fx.Node]]:
    """Propose Core ML partitions, leaving out every op that reads a value
    Core ML cannot take as an input from outside its partition or makes one
    the delegate cannot return for use outside, and then every partition this
    leaves without a tensor input. State the partitions leave in the program
    is an input.

    Returns the partitions and the ops left out.
    """
    # Searching for partitions is slow on a large graph, so first leave out
    # the ops that do not depend on how the search goes. Most graphs then need
    # a single search.
    left_out = _find_ops_left_out_of_any_partition(
        graph_module,
        op_support,
        kept_inside,
        sizes,
        {writer for writers in state_writers.values() for writer in writers},
    )
    support = chain(create_op_support(lambda _, node: node not in left_out), op_support)
    # Each op moved out of Core ML can hand such a tensor to the next op, so
    # repeat until no partition reads one.
    while True:
        partitions = CapabilityBasedPartitioner(
            graph_module, support, allows_single_node_partition=True
        ).propose_partitions()
        left_in_program = _states_left_in_program(partitions, state_writers)
        kept = set(kept_inside).difference(left_in_program)
        state_writes = {
            writer
            for state, writers in state_writers.items()
            if state.name not in left_in_program
            for writer in writers
        }
        new = _find_readers_of_unsupported_inputs(
            partitions, kept, sizes
        ) | _find_makers_of_unsupported_outputs(partitions, state_writes)
        if new or left_out:
            # Moving those ops out can leave a partition that reads no tensor,
            # such as the index math of a gather. So once any op moves out,
            # every such partition that Core ML cannot take moves out too. If
            # none does, they stay: below iOS 18, lowering then fails on a
            # partition with no inputs and asks for iOS 18.
            # _find_ops_partitioning_may_leave_out relies on this.
            new |= _find_nodes_of_partitions_without_tensor_inputs(
                partitions, kept, allow_partitions_without_inputs
            )
        # Stop once a round leaves out nothing new, so the loop always ends.
        new -= left_out
        if not new:
            return partitions, left_out
        left_out |= new


def _decompose_alone(
    node: torch.fx.Node, decompositions: Dict[torch._ops.OperatorBase, Callable]
) -> torch.fx.GraphModule:
    """Traces the decomposition of node by itself, from the values of its
    inputs."""
    inputs = node.all_input_nodes

    def call(*args):
        env = dict(zip(inputs, args))
        return node.target(
            *torch.fx.node.map_arg(node.args, env.__getitem__),
            **torch.fx.node.map_arg(node.kwargs, env.__getitem__),
        )

    # pre_dispatch keeps whole a composite op the table leaves out, such as
    # matmul, as lowering does.
    return make_fx(
        call,
        decomposition_table=decompositions,
        tracing_mode="fake",
        pre_dispatch=True,
    )(*(arg.meta["val"] for arg in inputs))


def _decomposing_may_leave_out(
    graph_module: torch.fx.GraphModule,
    decompositions: Dict[torch._ops.OperatorBase, Callable],
    not_kept: Collection[torch._ops.OpOverload],
    kept: Collection[torch._ops.OpOverload],
    from_kept_inside: Collection[torch.fx.Node],
    sizes: Collection[str],
    is_supported: Callable[[torch.fx.Node], bool],
    may_leave_out: bool,
    log: Callable[[str], None],
) -> bool:
    """Whether decomposing the calls of the ops lowering decomposes may make a
    call of a kept op that Core ML rejects, or make partition() leave out a
    call of a kept op it does not leave out in graph_module. may_leave_out
    says whether partition() may leave out some op there.

    That needs a decomposition of a not_kept op that makes a call of a kept op
    that Core ML rejects, or that reads or makes a value Core ML cannot take or
    return. A decomposed call of any op that reads or makes such a value may
    also be the first op partition() moves out, after which a partition left
    without tensor inputs moves out too. So it matters once a kept call can end
    up in such a partition: a kept call that reads only the weights, constants
    and state the partitioner moves into Core ML, or a call of a not_kept op
    that reads only those and makes a tensor or a call of a kept op. If
    may_leave_out, such a call of a not_kept op matters by itself. Each call is
    traced by itself, in milliseconds, while decomposing a large program takes
    minutes. A call that cannot be traced by itself may matter.
    """

    def crosses(call: torch.fx.Node) -> bool:
        return _cannot_be_coreml_output(call) or any(
            _cannot_be_coreml_input(arg, sizes) or _makes_symbolic_value(arg)
            for arg in call.all_input_nodes
        )

    crossing = False
    without_inputs = any(node.target in kept for node in from_kept_inside)
    for node in graph_module.graph.nodes:
        if node.target not in decompositions:
            continue
        not_kept_call = node.target in not_kept
        reads_kept_inside = not_kept_call and all(
            arg in from_kept_inside
            for arg in node.all_input_nodes
            if _makes_tensor(arg)
        )
        from_inside = may_leave_out and reads_kept_inside
        if from_inside and _makes_tensor(node):
            return True
        try:
            calls = list(_decompose_alone(node, decompositions).graph.nodes)
        except Exception as e:
            log(
                f"Decomposing the program to find the ops that run outside of "
                f"CoreML, because {node.target} cannot be decomposed by itself: {e}"
            )
            return True
        if not_kept_call and any(
            call.target in kept
            and (from_inside or not is_supported(call) or crosses(call))
            for call in calls
        ):
            return True
        crossing = crossing or any(crosses(call) for call in calls)
        without_inputs = without_inputs or (
            reads_kept_inside
            and (_makes_tensor(node) or any(call.target in kept for call in calls))
        )
    return crossing and without_inputs


def _edge_op_names(target: Any) -> Set[str]:
    """The lowercase names target can have once lowered to the edge dialect,
    which partition() matches skip names against: an ATen op gains its
    namespace, a view op becomes its copy, and some ops become dim order ops.
    """
    if not isinstance(target, torch._ops.OpOverload):
        return {getattr(target, "__name__", "").lower()}
    try:
        view_copy = get_view_copy_of_view_op(target._schema)
    except InternalError:
        # A view op without a copy, such as reshape in a program as exported,
        # is turned into views by lowering before partition() sees it.
        view_copy = None
    edge = aten_to_edge(view_copy or target)
    return {edge.__name__.lower(), DimOrderOpsMap.get(edge, edge).__name__.lower()}


def _is_view_op(op: torch._ops.OpOverload) -> bool:
    schema = op._schema
    if len(schema.arguments) == 0:
        return False
    alias_info = schema.arguments[0].alias_info
    return (alias_info is not None) and (not alias_info.is_write)


# The partitioner picks the ops to keep whole from torch ops. Refusing min.other
# and max.other there lets lowering turn them into minimum and maximum. The
# others stay whole, as core ATen ops or ops the partitioner keeps, so only their
# edge ops reach it.
_MIN_MAX_TARGETS = (
    torch.ops.aten.min.other,
    torch.ops.aten.max.other,
    exir_ops.edge.aten.min.default,
    exir_ops.edge.aten.min.dim,
    exir_ops.edge.aten.max.default,
    exir_ops.edge.aten.max.dim,
    exir_ops.edge.aten.amin.default,
    exir_ops.edge.aten.amax.default,
    exir_ops.edge.aten.minimum.default,
    exir_ops.edge.aten.maximum.default,
)


# No upstream issue yet. For example, converting x.amax() of a bool x raises
# 'Op "amax" (op_type: reduce_max) Input x="x" expects tensor or scalar of
# dtype from type domain ['fp16', 'fp32', 'int32'] but got tensor[3,2,bool]'.
def _is_min_max_of_bool(node: torch.fx.Node) -> bool:
    """coremltools 9.0 cannot compute a min or max of bool tensors.

    It raises ``ValueError`` when every input is bool, so that the result is
    bool, such as for the all-true check that torch.distributions exports,
    instead of reporting the op as unsupported. A bool next to a float or int
    tensor works, and so do int tensors.
    """
    if node.target not in _MIN_MAX_TARGETS:
        return False
    val = node.meta.get("val")
    if isinstance(val, (list, tuple)):
        val = val[0]
    return isinstance(val, torch.Tensor) and val.dtype == torch.bool


_INDEX_PUT_TARGETS = (
    torch.ops.aten.index_put.default,
    exir_ops.edge.aten.index_put.default,
)


# https://github.com/apple/coremltools/issues/2141 covers a bool index, which
# coremltools 9.0 converts only on its own. No upstream issue covers the other
# forms yet. For example, y[:, idx] = v with idx = torch.tensor([0, 2]) raises
# "Input squeeze_0 has rank 1 != other inputs rank 0".
def _index_put_refusal(
    node: torch.fx.Node, minimum_deployment_target: Optional[ct.target]
) -> Optional[str]:
    """The form of an ``index_put`` that coremltools 9.0 cannot convert, such
    as "with accumulate=True next to a slice", or None for any other node.

    With a whole slice, such as ``y[:, idx] = v``, it writes ``v`` into the
    part of ``y`` the indices select, keeping each indexed dim as size 1. That
    needs ``accumulate=False``, int indices of shape ``()`` or ``(1,)``, a
    static size in every other dim, and a non-bool ``y`` below iOS 18. For a
    size known only at run time, it raises "Cannot add const". With a 0-d
    index or several indices, torch lays out that part differently, so ``v``
    may only vary in the dims after the last index. Without a whole slice,
    several indices need the same shape and no bool mask, and a bool ``y``
    comes back as int32. It raises or writes the wrong place for any other
    form, such as ``idx`` with two entries, instead of reporting the op as
    unsupported.
    """
    if node.target not in _INDEX_PUT_TARGETS:
        return None
    y, values = node.args[0].meta.get("val"), node.args[2].meta.get("val")
    indices = [i if i is None else i.meta.get("val") for i in node.args[1]]
    given = {dim: indices[dim] for dim, i in enumerate(node.args[1]) if i is not None}
    # Export always sets val, so an index_put without it is left to coremltools.
    if not given or not all(
        isinstance(t, torch.Tensor) for t in (y, values, *given.values())
    ):
        return None
    if len(given) == len(indices):
        if y.dtype == torch.bool:
            return "on a bool tensor without a slice"
        if len(indices) > 1 and any(
            index.dtype == torch.bool
            or not statically_known_true(sym_eq(index.shape, indices[0].shape))
            for index in indices
        ):
            return "with indices that differ in shape or include a bool mask"
        return None
    accumulate = (
        node.args[3] if len(node.args) > 3 else node.kwargs.get("accumulate", False)
    )
    if accumulate:
        return "with accumulate=True next to a slice"
    if any(
        index.dtype == torch.bool
        or index.dim() > 1
        or not statically_known_true(index.numel() == 1)
        for index in given.values()
    ):
        return (
            "with an index other than an int tensor of shape () or (1,) next to a slice"
        )
    selected = [1 if dim in given else size for dim, size in enumerate(y.shape)]
    if any(isinstance(size, torch.SymInt) for size in selected):
        return "with a size known only at run time next to a slice"
    reorders = len(given) > 1 or any(index.dim() == 0 for index in given.values())
    after = len(selected) - 1 - max(given)
    if reorders and not all(
        statically_known_true(v == 1)
        for v in values.shape[: max(0, values.dim() - after)]
    ):
        return (
            "with a 0-d index or several indices, and values that vary before "
            "the last index"
        )
    if y.dtype == torch.bool and (
        minimum_deployment_target is None or minimum_deployment_target < ct.target.iOS18
    ):
        return "on a bool tensor below minimum_deployment_target=ct.target.iOS18"
    return None


class _OperatorsSupportedForCoreMLBackend(OperatorSupportBase):
    def __init__(
        self,
        skip_ops_for_coreml_delegation: Optional[List[str]] = None,
        lower_full_graph: bool = False,
        log: bool = False,
        minimum_deployment_target: Optional[ct.target] = None,
    ) -> None:
        if skip_ops_for_coreml_delegation is None:
            skip_ops_for_coreml_delegation = []
        super().__init__()
        self.skip_ops_for_coreml_delegation = skip_ops_for_coreml_delegation
        self.lower_full_graph = lower_full_graph
        self.minimum_deployment_target = minimum_deployment_target
        self._logged_msgs = set()
        self._log = log

    def log_once(self, msg: str) -> None:
        if self._log and msg not in self._logged_msgs:
            logger.info(msg)
            self._logged_msgs.add(msg)

    def should_skip_op_for_delegation(self, node_target_name: str) -> bool:
        skipped_ops = self.skip_ops_for_coreml_delegation or []
        if node_target_name in skipped_ops:
            assert (
                not self.lower_full_graph
            ), f"Cannot skip {node_target_name} because lower_full_graph is True.  Please set skip_ops_for_coreml_delegation=None or lower_full_graph=False in the CoreMLPartitioner"
            self.log_once(
                "Skipping op for CoreML delegation because it is in skip_ops_for_coreml_delegation: "
                + node_target_name
            )
            return True
        return False

    def override_reason(self, node) -> Optional[str]:
        """Why CoreML cannot take node, even when coremltools converts its op."""
        name = getattr(node.target, "__name__", node.target)
        # https://github.com/apple/coremltools/issues/2573
        if (
            node.target
            in [
                torch.ops.aten.sub.Tensor,
                exir_ops.edge.aten.sub.Tensor,
                torch.ops.aten.add.Tensor,
                exir_ops.edge.aten.add.Tensor,
            ]
            and "alpha" in node.kwargs
            and node.kwargs["alpha"] != 1
        ):
            return "torch.ops.aten.{sub, add}.Tensor with alpha != 1 is not supported by CoreML."

        # Ops that are unsupported by CoreML purely on the basis of their
        # target — no per-arg conditions to check.  Grouped by upstream issue
        # so the comment trail still points at the underlying coremltools /
        # executorch bug for each entry.
        if node.target in _UNSUPPORTED_OP_TARGETS:
            return f"{name} is not supported by CoreML."

        if _touches_unsupported_high_rank_tensor(
            node,
            lambda n: self.should_skip_op_for_delegation(
                getattr(n.target, "__name__", "").lower()
            ),
        ):
            return f"{name} on a tensor above rank 5 is not supported by CoreML."

        if _is_arg_min_max_over_flattened_input(node):
            return (
                "torch.ops.aten.{argmax, argmin}.default with dim=None is "
                "not supported by CoreML."
            )

        if _reduces_0d_tensor(node):
            return f"{name} of a 0-d tensor is not supported by CoreML."

        if _reads_or_produces_bfloat16(node):
            return f"{name} on a bfloat16 tensor is not supported by CoreML."

        if _is_min_max_of_bool(node):
            return f"{name} on bool tensors is not supported by CoreML."

        refusal = _index_put_refusal(node, self.minimum_deployment_target)
        if refusal is not None:
            return f"{name} {refusal} is not supported by CoreML."

        return None

    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        # get_attr node can always be supported on any backend
        if node.op == "get_attr":
            return True
        # check if the PyTorch op get called is supported in Core ML
        elif node.op == "call_function":
            # skip ops if specified by user
            node_target_name = getattr(node.target, "__name__", "").lower()

            if self.should_skip_op_for_delegation(node_target_name):
                return False

            # query coremltools to see if node is supported
            is_supported = ct.converters.mil.frontend.torch.is_torch_fx_node_supported(
                node
            )
            reason = self.override_reason(node) if is_supported else None
            if reason is not None:
                self.log_once(f"{reason}  Overriding op support.")
                is_supported = False

            if not is_supported:
                if self.lower_full_graph:
                    unsupported = (
                        f"CoreML does not support the op {node_target_name}"
                        if reason is None
                        else reason.rstrip(".")
                    )
                    raise NotImplementedError(
                        f"""{unsupported}, but you have set lower_full_graph=True in the CoreMLPartitioner.

Please set lower_full_graph=False in the CoreMLPartitioner to allow running unsupported ops outside of CoreML.  Note that setting lower_full_graph=False may affect performance of CoreML and the available features.
As an alternative to setting lower_full_graph=False, you can try rewriting your model to avoid using this op.

Also consider filing an issue with Apple's coremltools repo to request support for the op: https://github.com/apple/coremltools/issues
Do not file an issue with ExecuTorch for op support.
"""
                    )
                self.log_once(
                    "Skipping op for CoreML delegation because it is not supported by CoreML: "
                    + node_target_name
                )
            return is_supported
        # cowardly refuse to support all other types of node:
        # 1. placeholder / output nodes should not be tagged
        #    reference: https://github.com/pytorch/executorch/pull/1398
        # 2. call_module / call_method should have been replaced with call_function?
        else:
            self.log_once(
                "Skipping op for CoreML delegation because it is not get_attr or call_function: "
                + node.op
            )
            return False


class CoreMLPartitioner(Partitioner):
    def __init__(
        self,
        *,
        skip_ops_for_coreml_delegation: Optional[List[str]] = None,
        compile_specs: Optional[List[CompileSpec]] = None,
        take_over_mutable_buffer: Optional[bool] = True,
        lower_full_graph: bool = False,
        take_over_constant_data: bool = True,
    ) -> None:
        if skip_ops_for_coreml_delegation is None:
            skip_ops_for_coreml_delegation = []
        self.skip_ops_for_coreml_delegation = skip_ops_for_coreml_delegation

        for compile_spec in compile_specs or []:
            if compile_spec.key == COMPILE_SPEC_KEYS.ENUMERATED_SHAPES.value:
                assert (
                    lower_full_graph
                ), "lower_full_graph must be True in the CoreMLPartitioner when using an enumerated shape compile spec"

        self.delegation_spec = DelegationSpec(
            backend_id=CoreMLBackend.__name__,
            compile_specs=compile_specs if compile_specs is not None else [],
        )
        self.take_over_mutable_buffer = take_over_mutable_buffer
        self.lower_full_graph = lower_full_graph
        self.take_over_constant_data = take_over_constant_data
        self._minimum_deployment_target = (
            CoreMLBackend.min_deployment_target_from_compile_specs(
                self.delegation_spec.compile_specs
            )
        )
        self._logged_msgs = set()
        self._ops_to_not_decompose_cache = None

        if self.lower_full_graph:
            assert (
                len(self.skip_ops_for_coreml_delegation) == 0
            ), "When lower_full_graph=True, you cannot set skip_ops_for_coreml_delegation"
            assert (
                self.take_over_constant_data
            ), "When lower_full_graph=True, you must set take_over_constant_data=True"
            assert (
                self.take_over_mutable_buffer
            ), "When lower_full_graph=True, you must set take_over_mutable_buffer=True"

    def _check_if_called_from_to_backend(self) -> bool:
        """
        Check if the partition method is being called from the deprecated to_backend workflow.
        Returns True if called from deprecated direct to_backend, False if called from to_edge_transform_and_lower.
        """
        stack = inspect.stack()

        for frame_info in stack:
            if frame_info.function == "to_edge_transform_and_lower":
                return False

        for frame_info in stack:
            if frame_info.function == "to_backend":
                filename = frame_info.filename
                if "program/_program.py" in filename:
                    return True
        return False

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        """
        Override partition to add deprecation warning when called from to_backend.
        """
        # Check if we're being called from the deprecated to_backend workflow
        if self._check_if_called_from_to_backend():
            logger.warning(
                "\nDEPRECATION WARNING: You are using the deprecated 'to_edge() + to_backend()' workflow. "
                "This may result in decreased performance because ExecuTorch decomposes ops (e.g., SDPA) "
                "that CoreML has optimized implementations for. "
                "Please consider migrating to 'to_edge_transform_and_lower()' for better performance. "
                "See: https://docs.pytorch.org/executorch/main/backends/coreml/coreml-overview.html#using-the-core-ml-backend"
            )

        # Run the CapabilityBasedPartitioner to return the largest possible
        # subgraphs containing the nodes with the tags
        logger.info("CoreMLPartitioner::partition")
        partition_tags = {}

        state_writers = self._state_writers(exported_program)
        kept_inside = self._placeholders_kept_inside(exported_program, state_writers)
        sizes = _dynamic_size_names(exported_program)
        partition_list, left_out = _propose_partitions(
            exported_program.graph_module,
            _OperatorsSupportedForCoreMLBackend(
                self.skip_ops_for_coreml_delegation,
                self.lower_full_graph,
                log=True,
                minimum_deployment_target=self._minimum_deployment_target,
            ),
            kept_inside,
            sizes,
            self._targets_ios18(),
            state_writers,
        )
        if left_out:
            kept = set(kept_inside).difference(
                _states_left_in_program(partition_list, state_writers)
            )
            inside = {node for partition in partition_list for node in partition.nodes}
            inputs = sorted(
                f"{arg.name} ({arg.meta['val'].dtype}, {tuple(arg.meta['val'].shape)})"
                for arg in _find_unsupported_inputs_of_left_out_ops(
                    left_out, inside, kept, sizes
                )
            )
            outputs = sorted(
                node.name
                for node in left_out
                if _makes_tensor(node, (torch.int8,))
                and any(user not in left_out for user in node.users)
            )
            symbols = sorted(
                node.name
                for node in left_out
                if (
                    _makes_symbolic_value(node)
                    and any(user not in left_out for user in node.users)
                )
                or any(
                    _makes_symbolic_value(arg) and arg not in inside
                    for arg in node.all_input_nodes
                )
            )
            reasons = []
            if inputs:
                reasons.append(
                    "reads a tensor CoreML cannot take as an input, of uint8 or "
                    "int8 or with a size computed from a dynamic size, from "
                    "outside a CoreML partition: " + ", ".join(inputs)
                )
            if outputs:
                reasons.append(
                    "makes an int8 tensor, which the CoreML delegate cannot "
                    "return, for use outside of CoreML: " + ", ".join(outputs)
                )
            if symbols:
                reasons.append(
                    "makes or reads a symbolic value, which CoreML can neither "
                    "take as an input nor return, outside of CoreML: "
                    + ", ".join(symbols)
                )
            msg = (
                f"The model {', and '.join(reasons)}."
                if reasons
                else "The model uses a tensor or a symbolic value CoreML cannot handle."
            )
            if self.lower_full_graph:
                raise NotImplementedError(
                    f"{msg}  Set lower_full_graph=False in the CoreMLPartitioner "
                    "to run these ops outside of CoreML."
                )
            self.log_once(
                f"{msg}  Running these ops outside of CoreML: "
                + ", ".join(
                    sorted(
                        {
                            getattr(node.target, "__name__", str(node.target))
                            for node in left_out
                        }
                    )
                )
            )
        for partition in partition_list:
            for node in partition.nodes:
                tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = tag
                partition_tags[tag] = self.delegation_spec

        if self.take_over_constant_data:
            tag_constant_data(exported_program)
        if self.take_over_mutable_buffer:
            logger.info(
                "Core ML partitioner will take over torch mutable buffer as Core ML state, "
                "so if your model contains mutable buffer, "
                "then you will need MacOS15+/iOS18+ to execute. "
                "If you want your mutable buffer model to be compatible with older OS, "
                "then please set `take_over_mutable_buffer=False`"
            )
            tag_mutated_buffer(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )

    def log_once(self, msg: str) -> None:
        if msg not in self._logged_msgs:
            logger.info(msg)
            self._logged_msgs.add(msg)

    def _targets_ios18(self) -> bool:
        """Whether the deployment target is iOS 18 or later."""
        target = self._minimum_deployment_target
        return target is not None and target >= ct.target.iOS18

    def _placeholders_kept_inside(
        self,
        ep: ExportedProgram,
        state_writers: Dict[torch.fx.Node, List[torch.fx.Node]],
    ) -> Set[str]:
        """The weights, constants and state this partitioner moves into Core ML."""
        signature = ep.graph_signature
        mutated = set(signature.buffers_to_mutate.values())
        kept = {state.name for state in state_writers}
        if self.take_over_constant_data:
            kept |= set(signature.inputs_to_parameters)
            kept |= set(signature.inputs_to_lifted_tensor_constants)
            kept |= {
                name
                for name, buffer in signature.inputs_to_buffers.items()
                if buffer not in mutated
            }
        return kept

    def _state_writers(
        self, ep: ExportedProgram
    ) -> Dict[torch.fx.Node, List[torch.fx.Node]]:
        """The nodes that write each mutable buffer this partitioner may move
        into Core ML as state, by the placeholder that reads it. A buffer that
        no op reads, or whose new value is also returned, stays in the program.
        """
        if not self.take_over_mutable_buffer:
            return {}
        signature = ep.graph_signature
        nodes = {node.name: node for node in ep.graph.nodes}
        returned = {
            spec.arg.name
            for spec in signature.output_specs
            if spec.kind == OutputKind.USER_OUTPUT
        }
        writers: Dict[str, List[torch.fx.Node]] = {}
        for name, buffer in signature.buffers_to_mutate.items():
            writers.setdefault(buffer, []).append(nodes[name])
        return {
            nodes[name]: writers[buffer]
            for name, buffer in signature.inputs_to_buffers.items()
            if buffer in writers
            and nodes[name].users
            and not any(writer.name in returned for writer in writers[buffer])
        }

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        # to_edge_transform_and_lower asks twice in a row for the same program,
        # and the dry run in _drop_ops_left_outside can decompose all of it.
        cached = self._ops_to_not_decompose_cache
        if cached is not None and cached[0]() is ep:
            return list(cached[1]), None
        do_not_decompose = []
        op_support = _OperatorsSupportedForCoreMLBackend(
            self.skip_ops_for_coreml_delegation,
            self.lower_full_graph,
            log=False,
            minimum_deployment_target=self._minimum_deployment_target,
        )

        # CoreML prevents certain ops (like triu) from lowering to CoreML when put in the ExecuTorch op namespace
        # TODO: upstream fixes, but pending ET consuming a new published version of coremltools with the
        # desired changes, we need to manually block them here
        do_not_decompose_blocklist = [
            # https://github.com/apple/coremltools/blob/release/8.3/coremltools/converters/mil/frontend/torch/ops.py#L6965-L6966
            torch.ops.aten.triu.default,
            # https://github.com/apple/coremltools/blob/release/8.3/coremltools/converters/mil/frontend/torch/ops.py#L6997-L6998
            torch.ops.aten.tril.default,
            # CoreML's translation of repeat_interleave has poor perf
            torch.ops.aten.repeat_interleave.self_int,
            torch.ops.aten.repeat_interleave.self_Tensor,
        ]

        def is_supported(submodules, node: torch.fx.Node) -> bool:
            try:
                return op_support.is_node_supported(submodules, node)
            except Exception as e:
                # CoreML's op_support.is_node_supported will sometimes throw
                # for unsupported ops, rather than returning False
                self.log_once(
                    f"Encountered exception when checking node support, treating node as unsupported: {e}"
                )
                return False

        rejected = set()
        for node in ep.graph.nodes:
            if node.op == "call_function" and isinstance(
                node.target, torch._ops.OpOverload
            ):
                if not is_supported(None, node):
                    rejected.add(node.target)
                elif node.target not in do_not_decompose_blocklist and not _is_view_op(
                    node.target
                ):
                    do_not_decompose.append(node.target)
        # Without a per-node filter, a preserved op stays whole at every call, so
        # one rejected call would be left with no kernel.
        keep = [op for op in do_not_decompose if op not in rejected]
        keep = self._drop_ops_left_outside(ep, keep, rejected, is_supported)
        # A weak reference, so the partitioner does not keep the program alive.
        self._ops_to_not_decompose_cache = (weakref.ref(ep), keep)
        return list(keep), None

    def _drop_ops_left_outside(
        self,
        ep: ExportedProgram,
        keep: List[torch._ops.OpOverload],
        rejected: Set[torch._ops.OpOverload],
        is_supported: Callable[[Any, torch.fx.Node], bool],
    ) -> List[torch._ops.OpOverload]:
        """Drops every kept op that partition() moves out of CoreML at some
        call, for a value CoreML cannot take or return, or that CoreML rejects
        once the program is decomposed, since the portable kernels may have no
        kernel for it. An op skipped by name stays kept.

        partition() sees the program once lowering decomposes the ops that are
        not kept, which can make new calls of a kept op that CoreML rejects,
        such as the matmul of a bfloat16 linear, or change the calls it moves
        out. So when that may matter, this decomposes the program the same way
        and finds the calls partition() leaves out, again after each drop that
        changes the decomposition. It does not see transform passes, or the
        ops an earlier partitioner takes.
        """
        states = self._state_writers(ep)
        kept_inside = self._placeholders_kept_inside(ep, states)
        skip = set(self.skip_ops_for_coreml_delegation)
        support = create_op_support(
            lambda submodules, node: is_supported(submodules, node)
            and not (skip and _edge_op_names(node.target) & skip)
        )
        from_kept_inside = _computed_from_kept_inside(ep.graph_module, kept_inside)
        may_leave_out = bool(
            _find_ops_partitioning_may_leave_out(
                ep.graph_module,
                kept_inside,
                _dynamic_size_names(ep),
                lambda node: support.is_node_supported(None, node),
                {state.name for state in states},
            )
        )
        not_kept = set(rejected)
        decompositions = _default_decomposition_table()
        while keep:
            table = dict(decompositions)
            for op in keep:
                table.pop(op, None)
            kept = set(keep)
            decomposed = _decomposing_may_leave_out(
                ep.graph_module,
                table,
                not_kept,
                kept,
                from_kept_inside,
                _dynamic_size_names(ep),
                lambda node: is_supported(None, node),
                may_leave_out,
                self.log_once,
            )
            program = ep.run_decompositions(table) if decomposed else ep
            graph_module = program.graph_module
            sizes = _dynamic_size_names(program)
            state_writers = self._state_writers(program)
            calls = [node for node in graph_module.graph.nodes if node.target in kept]
            left_out = _find_ops_left_out_of_any_partition(
                graph_module,
                support,
                kept_inside,
                sizes,
                {writer for writers in state_writers.values() for writer in writers},
            )
            # The search is slow on a large graph, so run it only when it may
            # leave out a call of an op lowering would otherwise decompose.
            # Lowering keeps any other op whole either way, so whether such an
            # op is kept changes nothing.
            uncertain = _find_ops_partitioning_may_leave_out(
                graph_module,
                kept_inside,
                sizes,
                lambda node: support.is_node_supported(None, node),
                {state.name for state in state_writers},
            ).difference(left_out)
            if any(
                node.target in decompositions and node in uncertain for node in calls
            ):
                _, left_out = _propose_partitions(
                    graph_module,
                    support,
                    kept_inside,
                    sizes,
                    self._targets_ios18(),
                    state_writers,
                )
            else:
                left_out |= uncertain
            late = {
                node.target
                for node in calls
                if node in left_out or (decomposed and not is_supported(None, node))
            }
            if not late:
                break
            for op in late & decompositions.keys():
                self.log_once(
                    f"Not keeping {op} whole, because some call of it runs outside "
                    "of CoreML."
                )
            not_kept |= late
            keep = [op for op in keep if op not in late]
            if not any(op in decompositions for op in late):
                break
        return keep
