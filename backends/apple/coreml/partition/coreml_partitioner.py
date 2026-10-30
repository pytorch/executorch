# Copyright © 2023 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import inspect
import logging
from typing import Callable, Collection, List, Optional, Set, Tuple

import coremltools as ct

import torch
from coremltools.converters.mil.frontend.torch.utils import TORCH_DTYPE_TO_MIL_DTYPE

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
from torch.export.exported_program import ExportedProgram
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


def _cannot_be_coreml_input(node: torch.fx.Node) -> bool:
    """Whether node makes a tensor that Core ML cannot take as a program input.

    coremltools types each program input through ``TORCH_DTYPE_TO_MIL_DTYPE``
    and raises ``KeyError`` for a dtype missing from it, such as uint8 or int8,
    instead of reporting the op as unsupported.
    """
    val = node.meta.get("val")
    vals = val if isinstance(val, (list, tuple)) else [val]
    return any(
        isinstance(v, torch.Tensor) and v.dtype not in TORCH_DTYPE_TO_MIL_DTYPE
        for v in vals
    )


def _find_readers_of_unsupported_inputs(
    partitions: List[Partition],
    kept_inside: Collection[str],
    left_out: Collection[torch.fx.Node],
) -> Set[torch.fx.Node]:
    """The nodes that read a tensor Core ML cannot take as an input from outside
    their partition, or a symbolic value computed by an op left out of Core ML.

    A tensor made inside the partition, or a weight, constant or state the
    partitioner moves into Core ML, is not an input.
    """
    return {
        node
        for partition in partitions
        for node in partition.nodes
        if any(
            arg not in partition.nodes
            and arg.name not in kept_inside
            and (
                _cannot_be_coreml_input(arg)
                or (
                    arg in left_out
                    and isinstance(
                        arg.meta.get("val"),
                        (torch.SymInt, torch.SymFloat, torch.SymBool),
                    )
                )
            )
            for arg in node.all_input_nodes
        )
    }


def _makes_tensor(node: torch.fx.Node) -> bool:
    val = node.meta.get("val")
    vals = val if isinstance(val, (list, tuple)) else [val]
    return any(isinstance(v, torch.Tensor) for v in vals)


def _find_nodes_of_partitions_without_tensor_inputs(
    partitions: List[Partition],
    kept_inside: Collection[str],
    state: Collection[str],
    allow_partitions_without_inputs: bool,
) -> Set[torch.fx.Node]:
    """The nodes of the partitions that read no tensor from outside, except the
    weights, constants and state the partitioner moves into Core ML.

    Core ML cannot take a symbolic size as an input, and below iOS 18 it
    rejects a model with no inputs, so such a partition cannot be delegated.
    A partition that uses state stays: moving it to the CPU would leave the
    state's other readers in Core ML a stale copy, and Core ML state needs
    iOS 18 anyway.
    """
    nodes = set()
    for partition in partitions:
        outside = [
            arg
            for node in partition.nodes
            for arg in node.all_input_nodes
            if arg not in partition.nodes
        ]
        if any(arg.name in state for arg in outside):
            continue
        inputs = [arg for arg in outside if arg.name not in kept_inside]
        reads_tensor = any(_makes_tensor(arg) for arg in inputs)
        if not reads_tensor and (inputs or not allow_partitions_without_inputs):
            nodes.update(partition.nodes)
    return nodes


def _propose_partitions(
    graph_module: torch.fx.GraphModule,
    op_support: OperatorSupportBase,
    kept_inside: Collection[str],
    state: Collection[str],
    allow_partitions_without_inputs: bool,
) -> Tuple[List[Partition], Set[torch.fx.Node]]:
    """Propose Core ML partitions, leaving out every op that reads a tensor
    Core ML cannot take as an input from outside its partition, and then every
    partition this leaves without a tensor input.

    Returns the partitions and the ops left out.
    """
    left_out: Set[torch.fx.Node] = set()
    support = chain(create_op_support(lambda _, node: node not in left_out), op_support)
    # Each op moved out of Core ML can hand such a tensor to the next op, so
    # repeat until no partition reads one.
    while True:
        partitions = CapabilityBasedPartitioner(
            graph_module, support, allows_single_node_partition=True
        ).propose_partitions()
        new = _find_readers_of_unsupported_inputs(partitions, kept_inside, left_out)
        if new or left_out:
            # Moving those ops out can leave a partition that reads no tensor,
            # such as the index math of a gather.
            new |= _find_nodes_of_partitions_without_tensor_inputs(
                partitions, kept_inside, state, allow_partitions_without_inputs
            )
        if not new:
            return partitions, left_out
        left_out |= new


def _is_view_op(op: torch._ops.OpOverload) -> bool:
    schema = op._schema
    if len(schema.arguments) == 0:
        return False
    alias_info = schema.arguments[0].alias_info
    return (alias_info is not None) and (not alias_info.is_write)


class _OperatorsSupportedForCoreMLBackend(OperatorSupportBase):
    def __init__(
        self,
        skip_ops_for_coreml_delegation: Optional[List[str]] = None,
        lower_full_graph: bool = False,
        log: bool = False,
    ) -> None:
        if skip_ops_for_coreml_delegation is None:
            skip_ops_for_coreml_delegation = []
        super().__init__()
        self.skip_ops_for_coreml_delegation = skip_ops_for_coreml_delegation
        self.lower_full_graph = lower_full_graph
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

    def should_override_support(self, node) -> bool:
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
            self.log_once(
                "torch.ops.aten.{sub, add}.Tensor with alpha != 1 is not supported by CoreML.  Overriding support."
            )
            return True

        # Ops that are unsupported by CoreML purely on the basis of their
        # target — no per-arg conditions to check.  Grouped by upstream issue
        # so the comment trail still points at the underlying coremltools /
        # executorch bug for each entry.
        if node.target in _UNSUPPORTED_OP_TARGETS:
            self.log_once(
                f"{node.target} is not supported by CoreML.  Overriding op support."
            )
            return True

        if _touches_unsupported_high_rank_tensor(
            node,
            lambda n: self.should_skip_op_for_delegation(
                getattr(n.target, "__name__", "").lower()
            ),
        ):
            self.log_once(
                f"{node.target} on a tensor above rank 5 is not supported by "
                "CoreML.  Overriding op support."
            )
            return True

        if _is_arg_min_max_over_flattened_input(node):
            self.log_once(
                "torch.ops.aten.{argmax, argmin}.default with dim=None is "
                "not supported by CoreML.  Overriding op support."
            )
            return True

        if _reads_or_produces_bfloat16(node):
            self.log_once(
                f"{node.target} on a bfloat16 tensor is not supported by CoreML.  "
                "Overriding op support."
            )
            return True

        # TODO: enable this after bugs in ExecuTorch's partitioner are fixed
        # # If lower_full_graph=False, do not partition nodes with symbolic args because it can result in symbolic args
        # # in the placeholders due to partitioning, which CoreML does not support
        # if not self.lower_full_graph and any(
        #     isinstance(arg, torch.fx.Node)
        #     and isinstance(
        #         arg.meta.get("val", None),
        #         (torch.SymInt, torch.SymBool, torch.SymFloat),
        #     )
        #     for arg in node.args
        # ):
        #     self.log_once(
        #         "Skipping op for CoreML delegation because it contains symbolic args: "
        #         + node_target_name
        #     )
        #     return True

        return False

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
            if self.should_override_support(node):
                is_supported = False

            if not is_supported:
                if self.lower_full_graph:
                    raise NotImplementedError(
                        f"""CoreML does not support the op {node_target_name}, but you have set lower_full_graph=True in the CoreMLPartitioner.

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
        self._logged_msgs = set()

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

        kept_inside = self._placeholders_kept_inside(exported_program)
        state = self._state_kept_inside(exported_program)
        partition_list, left_out = _propose_partitions(
            exported_program.graph_module,
            _OperatorsSupportedForCoreMLBackend(
                self.skip_ops_for_coreml_delegation,
                self.lower_full_graph,
                log=True,
            ),
            kept_inside,
            state,
            self._allows_partitions_without_inputs(),
        )
        for node in left_out:
            read = [arg.name for arg in node.all_input_nodes if arg.name in state]
            if read:
                raise NotImplementedError(
                    f"{node} has to run outside of CoreML, but it reads "
                    f"{', '.join(read)}, which CoreML takes over as state, so it "
                    "would read a stale copy.  Set take_over_mutable_buffer=False "
                    "in the CoreMLPartitioner to keep the state outside of CoreML."
                )
        if left_out:
            inputs = sorted(
                {
                    f"{arg.name} ({arg.meta['val'].dtype})"
                    for node in left_out
                    for arg in node.all_input_nodes
                    if arg not in left_out
                    and arg.name not in kept_inside
                    and isinstance(arg.meta.get("val"), torch.Tensor)
                    and _cannot_be_coreml_input(arg)
                }
            )
            msg = (
                "The model reads a tensor whose dtype CoreML cannot take as an "
                "input, such as uint8 or int8, from outside a CoreML partition: "
                + ", ".join(inputs)
                + "."
            )
            if self.lower_full_graph:
                raise NotImplementedError(
                    f"{msg}  Convert the tensor to a supported dtype before it "
                    "enters the model, or set lower_full_graph=False in the "
                    "CoreMLPartitioner to run the ops that read it outside of "
                    "CoreML."
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

    def _allows_partitions_without_inputs(self) -> bool:
        """Whether Core ML accepts a partition with no inputs, which needs iOS 18."""
        target = CoreMLBackend.min_deployment_target_from_compile_specs(
            self.delegation_spec.compile_specs
        )
        return target is not None and target >= ct.target.iOS18

    def _placeholders_kept_inside(self, ep: ExportedProgram) -> Set[str]:
        """The weights, constants and state this partitioner moves into Core ML."""
        signature = ep.graph_signature
        mutated = set(signature.buffers_to_mutate.values())
        kept = self._state_kept_inside(ep)
        if self.take_over_constant_data:
            kept |= set(signature.inputs_to_parameters)
            kept |= set(signature.inputs_to_lifted_tensor_constants)
            kept |= {
                name
                for name, buffer in signature.inputs_to_buffers.items()
                if buffer not in mutated
            }
        return kept

    def _state_kept_inside(self, ep: ExportedProgram) -> Set[str]:
        """The mutable buffers this partitioner moves into Core ML as state."""
        if not self.take_over_mutable_buffer:
            return set()
        signature = ep.graph_signature
        mutated = set(signature.buffers_to_mutate.values())
        return {
            name
            for name, buffer in signature.inputs_to_buffers.items()
            if buffer in mutated
        }

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        do_not_decompose = []
        op_support = _OperatorsSupportedForCoreMLBackend(
            self.skip_ops_for_coreml_delegation,
            self.lower_full_graph,
            log=False,
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

        # An op that partition() leaves out runs on the CPU, so it is never
        # preserved: a core ATen op stays whole anyway, and any other op is
        # decomposed into core ATen ops, the op set the portable kernels target.
        _, left_out = _propose_partitions(
            ep.graph_module,
            create_op_support(is_supported),
            self._placeholders_kept_inside(ep),
            self._state_kept_inside(ep),
            self._allows_partitions_without_inputs(),
        )
        rejected = set()
        for node in ep.graph.nodes:
            if node.op == "call_function" and isinstance(
                node.target, torch._ops.OpOverload
            ):
                if not is_supported(None, node):
                    rejected.add(node.target)
                elif node in left_out:
                    if torch.Tag.core not in node.target.tags:
                        self.log_once(
                            f"{node.target} runs outside of CoreML here because "
                            "of a tensor CoreML cannot take as an input, so every "
                            "call of it is decomposed."
                        )
                    rejected.add(node.target)
                elif node.target not in do_not_decompose_blocklist and not _is_view_op(
                    node.target
                ):
                    do_not_decompose.append(node.target)
        # Without a per-node filter, a preserved op stays whole at every call, so
        # one rejected call would be left with no kernel.
        return [op for op in do_not_decompose if op not in rejected], None
