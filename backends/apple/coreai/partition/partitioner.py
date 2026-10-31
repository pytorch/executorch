# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import json
import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from executorch.backends.apple.coreai.compiler.enumerated_shapes import (
    resolve_input_enumerations,
)

from executorch.backends.apple.coreai.compiler.preprocess import (
    AOTCompileConfig,
    COMPILE_SPEC_KEYS,
    CoreAIBackend,
)
from executorch.backends.apple.coreai.passes.normalize_op_args import coreai_aten_op
from executorch.backends.apple.coreai.passes.replace_copy_ops import functional_aten_op
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data, tag_mutated_buffer
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from torch.export.exported_program import ExportedProgram
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Nodes carrying this meta key are never claimed by the delegate, even if Core AI
# could otherwise lower them.  Set it (e.g. via :func:`do_not_delegate`) on any
# node you want to keep running outside Core AI.
DO_NOT_DELEGATE_TAG = "coreai_do_not_delegate"


def do_not_delegate(node: torch.fx.Node) -> None:
    """Mark a node so the CoreAIPartitioner leaves it out of the delegate."""
    node.meta[DO_NOT_DELEGATE_TAG] = True


def _resolvers() -> Tuple[dict, dict]:
    """Return ``coreai-torch``'s (aten, higher-order) lowering tables.

    These dicts are the source of truth for which ops the Core AI converter
    can lower; the partitioner mirrors them so a tagged subgraph is guaranteed
    to be convertible.  Imported lazily so this module is importable in
    environments without ``coreai-torch`` installed.
    """
    from coreai_torch._aten_to_core import (
        _aten_to_core_resolver,
        _higher_order_resolver,
    )

    return _aten_to_core_resolver, _higher_order_resolver


def _underlying_target(target):
    """Unwrap an ExecuTorch ``EdgeOpOverload`` to its plain ATen overload.

    ExecuTorch edge ops wrap the ATen overload in an ``EdgeOpOverload`` whose
    ``__name__`` is prefixed (``"aten.view.default"``); the overload at
    ``target._op`` has the bare name (``"view.default"``) that the converter
    keys on.  Plain ``OpOverload``s also expose a ``_op`` attribute, but its
    ``__name__`` is empty, so we must only unwrap genuine edge ops.
    """
    if isinstance(target, EdgeOpOverload):
        return target._op
    return target


def _coreai_op_name(target) -> Optional[str]:
    """Return the ``coreai-torch`` resolver key for an fx call_function target."""
    return getattr(_underlying_target(target), "__name__", None)


def _coreai_namespace(target) -> Optional[str]:
    return getattr(_underlying_target(target), "namespace", None)


def is_coreai_supported_target(target) -> bool:
    """Whether ``coreai-torch`` has a lowering for this fx target."""
    name = _coreai_op_name(target)
    if name is None:
        return False

    aten_resolver, higher_order_resolver = _resolvers()
    namespace = _coreai_namespace(target)

    if namespace in ("coreai", "coreaix"):
        return True
    if namespace == "higher_order":
        return name in higher_order_resolver
    # ATen ops (namespace "aten") and namespace-less targets such as
    # operator.getitem / sym_* are all keyed in the aten resolver.
    if name in aten_resolver:
        return True
    # Edge ``*_copy`` variants (e.g. permute_copy) are not in the resolver, but
    # their functional forms (permute) are.  Claim them here so they land in the
    # delegate; CoreAIBackend.preprocess remaps them via
    # ReplaceCopyOpsWithFunctionalPass before conversion.
    func_op = functional_aten_op(target)
    if func_op is not None:
        return getattr(func_op, "__name__", None) in aten_resolver
    return False


def _index_put_supported(node: torch.fx.Node) -> bool:
    accumulate = node.kwargs.get(
        "accumulate", node.args[3] if len(node.args) > 3 else False
    )
    if accumulate:
        return False
    # Only the batch dim of the input and values may be dynamic.
    if not all(
        isinstance(d, int)
        for arg in (node.args[0], node.args[2])
        for d in arg.meta["val"].shape[1:]
    ):
        return False
    return not _index_put_has_partial_integer_indexing(node)


def _index_put_has_partial_integer_indexing(node: torch.fx.Node) -> bool:
    # TODO: report to Apple. Integer indices that leave input dims unindexed
    # (e.g. x[idx] = rows on a 2D x) abort the process in MPS scatter_nd with a
    # rank assertion ("2 != 1 + 0"). This is a conservative workaround, not a
    # proven support boundary, so some working slice updates also fall back.
    has_integer_index = False
    indexed_dims = 0
    for index in node.args[1]:
        if index is None:  # A full slice; consumes a dim without indexing it.
            continue
        val = index.meta["val"]
        if val.dtype in (torch.bool, torch.uint8):
            indexed_dims += val.dim()  # One mask covers as many dims as its rank.
        else:
            has_integer_index = True
            indexed_dims += 1
    return has_integer_index and indexed_dims < node.args[0].meta["val"].dim()


def _convolution_supported(node: torch.fx.Node) -> bool:
    transposed = node.args[6]
    return not (transposed and node.args[0].meta["val"].dim() == 5)


def _add_sub_supported(node: torch.fx.Node) -> bool:
    # DecomposeAddSubAlphaPass (in get_default_passes) removes non-unit alpha.
    alpha = node.kwargs.get("alpha", 1)
    return not isinstance(alpha, torch.fx.Node) and alpha == 1


# The rest are rewritten by NormalizeCoreAIOpArgsPass (in get_default_passes).


def _reduce_dims_supported(node: torch.fx.Node) -> bool:
    return len(node.args) > 1 and bool(node.args[1])


def _avg_pool_supported(node: torch.fx.Node) -> bool:
    args = node.args
    # The converter reads count_include_pad only if ceil_mode is present too,
    # and doesn't expand an empty stride to kernel_size.
    if len(args) == 5 or (len(args) > 2 and args[2] == []):
        return False
    # It also always pools as if ceil_mode=False and count_include_pad=True.
    ceil_mode = len(args) > 4 and args[4]
    padded = len(args) > 3 and any(args[3])
    exclude_pad = len(args) > 5 and not args[5]
    return not ceil_mode and not (padded and exclude_pad)


def _squeeze_dims_supported(node: torch.fx.Node) -> bool:
    # shrink_dims requires size-1 dims, and aborts the process on an empty list.
    shape = node.args[0].meta["val"].shape
    dims = node.args[1]
    return bool(dims) and all(shape[d] == 1 for d in dims if isinstance(shape[d], int))


# coreai-torch handlers that raise on, or silently mis-lower, some argument
# combinations of an op they otherwise support. Keyed by the ATen overload the
# converter sees (edge and ``*_copy`` variants unwrapped).
_VARIANT_CHECKS: Dict[object, Callable[[torch.fx.Node], bool]] = {
    torch.ops.aten.index_put.default: _index_put_supported,
    torch.ops.aten.convolution.default: _convolution_supported,
    # The converter ignores alpha.
    torch.ops.aten.add.Tensor: _add_sub_supported,
    torch.ops.aten.add.Scalar: _add_sub_supported,
    torch.ops.aten.sub.Tensor: _add_sub_supported,
    torch.ops.aten.sub.Scalar: _add_sub_supported,
    torch.ops.aten.mean.dim: _reduce_dims_supported,
    torch.ops.aten.amax.default: _reduce_dims_supported,
    torch.ops.aten.amin.default: _reduce_dims_supported,
    torch.ops.aten.avg_pool2d.default: _avg_pool_supported,
    torch.ops.aten.avg_pool3d.default: _avg_pool_supported,
    torch.ops.aten.squeeze.dims: _squeeze_dims_supported,
}


def is_coreai_supported_variant(node: torch.fx.Node) -> bool:
    """Whether ``coreai-torch`` can lower this node's specific arguments.

    Complements :func:`is_coreai_supported_target`, which only checks the op.
    Rejecting these here keeps the node on CPU instead of failing lowering or
    producing wrong results.
    """
    check = _VARIANT_CHECKS.get(coreai_aten_op(node.target))
    return check is None or check(node)


def _node_tensor_vals(node: torch.fx.Node):
    """Yield the meta ``val``s for a node's inputs and its own result.

    Flattens tuple/list results so multi-output ops are covered.
    """
    for arg in node.all_input_nodes:
        yield arg.meta.get("val")
    out = node.meta.get("val")
    if isinstance(out, (tuple, list)):
        yield from out
    else:
        yield out


class _OperatorsSupportedForCoreAIBackend(OperatorSupportBase):
    def __init__(self, log: bool = False) -> None:
        super().__init__()
        self._log = log
        self._logged_msgs = set()

    def log_once(self, msg: str) -> None:
        if self._log and msg not in self._logged_msgs:
            logger.info(msg)
            self._logged_msgs.add(msg)

    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        # get_attr nodes (e.g. subgraphs referenced by higher-order ops) can
        # always ride along with the delegate.
        if node.op == "get_attr":
            return True
        if node.op != "call_function":
            return False

        # Respect an explicit user opt-out on the node.
        if node.meta.get(DO_NOT_DELEGATE_TAG, False):
            self.log_once(
                f"Node {node.name} tagged {DO_NOT_DELEGATE_TAG}; leaving out of delegate"
            )
            return False

        name = _coreai_op_name(node.target) or ""
        if not is_coreai_supported_target(node.target):
            self.log_once(f"Core AI cannot lower op, leaving out of delegate: {name}")
            return False

        if not is_coreai_supported_variant(node):
            self.log_once(
                f"Core AI cannot lower these arguments of {name} ({node.name}); "
                "leaving out of delegate"
            )
            return False

        # The delegate boundary cannot carry Python or symbolic scalars, and
        # Core AI narrows i64/f64 tensors to 32-bit. Reject nodes with scalar
        # operands/results or i64/f64 tensor operands/results; the default
        # NarrowToCoreAIDtypesPass casts i64/f64 to 32-bit at the EP boundary so
        # only the boundary cast (which keeps an i64/f64 tensor) is left out.
        for val in _node_tensor_vals(node):
            if isinstance(
                val, (int, float, bool, torch.SymInt, torch.SymFloat, torch.SymBool)
            ):
                self.log_once(
                    f"Node {node.name} has a scalar operand or result; "
                    "leaving out of delegate"
                )
                return False
            if isinstance(val, torch.Tensor) and val.dtype in (
                torch.int64,
                torch.float64,
            ):
                self.log_once(
                    f"Node {node.name} has a {val.dtype} tensor; Core AI narrows "
                    "64-bit dtypes and can't carry them across the delegate "
                    "boundary. Run NarrowToCoreAIDtypesPass (included in "
                    "coreai.get_default_passes()) to cast int64/float64 inputs to "
                    "32-bit at the boundary."
                )
                return False
        return True


class CoreAIPartitioner(Partitioner):
    """Partition the largest subgraphs that Core AI can lower.

    Op support is derived directly from ``coreai-torch``'s lowering tables, and
    :meth:`ops_to_not_decompose` is derived from ``coreai-torch``'s own
    decomposition table, so ExecuTorch preserves exactly the ops the converter
    expects to see (e.g. fused SDPA) rather than decomposing them.
    """

    def __init__(
        self,
        *,
        aot_compile_config: Optional[AOTCompileConfig] = None,
        min_deployment_version: Optional[str] = None,
        input_enumerations: Optional[
            Sequence[Optional[Dict[int, Sequence[int]]]]
        ] = None,
        take_over_constant_data: bool = True,
        take_over_mutable_buffer: bool = True,
    ) -> None:
        # aot_compile_config requests ahead-of-time ``xcrun coreai-build
        # compile`` in preprocess, emitting per-architecture ``.aimodelc``
        # bundles instead of the portable ``.aimodel``. It is serialized as a
        # single JSON AOT_COMPILE_CONFIG spec (presence implies AOT).
        # min_deployment_version is a general spec (it also sets the portable
        # .aimodel's OS floor), so it is emitted separately.
        #
        # input_enumerations declares enumerated shapes on the ET model inputs
        # ({dim_index: [values]}, aligned to inputs); it is resolved to export
        # symbols in :meth:`partition` and propagated to each subgraph boundary
        # in preprocess, so it is not built into a compile spec here.
        specs = []
        if min_deployment_version is not None:
            specs.append(
                CompileSpec(
                    COMPILE_SPEC_KEYS.MIN_DEPLOYMENT_VERSION.value,
                    str(min_deployment_version).encode(),
                )
            )
        if aot_compile_config is not None:
            specs.append(
                CompileSpec(
                    COMPILE_SPEC_KEYS.AOT_COMPILE_CONFIG.value,
                    aot_compile_config.to_json().encode(),
                )
            )

        self.delegation_spec = DelegationSpec(
            backend_id=CoreAIBackend.__name__,
            compile_specs=specs,
        )
        self._base_compile_specs = specs
        self._input_enumerations = input_enumerations
        self.take_over_constant_data = take_over_constant_data
        self.take_over_mutable_buffer = take_over_mutable_buffer

    @staticmethod
    def _is_to_edge_transform_and_lower() -> bool:
        """Whether we are being called from ``to_edge_transform_and_lower``."""
        return any(
            frame.function == "to_edge_transform_and_lower" for frame in inspect.stack()
        )

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # Core AI derives op support and ops_to_not_decompose from coreai-torch's
        # tables; the deprecated to_edge() + to_backend() flow decomposes ops
        # (e.g. fused SDPA) before the partitioner runs, breaking that contract.
        if not self._is_to_edge_transform_and_lower():
            raise RuntimeError(
                "CoreAIPartitioner must be used with to_edge_transform_and_lower(). "
                "The to_edge() + to_backend() workflow is not supported because it "
                "decomposes ops that Core AI lowers directly. Please use:\n"
                "    exir.to_edge_transform_and_lower(\n"
                '        {"forward": exported_program},\n'
                "        partitioner=[CoreAIPartitioner()],\n"
                "        compile_config=get_default_compile_config(),\n"
                "    )"
            )
        logger.info("CoreAIPartitioner::partition")
        partition_tags = {}

        # Resolve ET-input enumerations to export symbols and attach as a compile
        # spec; preprocess propagates them to each subgraph's boundary.
        delegation_spec = self.delegation_spec
        resolved = resolve_input_enumerations(
            exported_program, self._input_enumerations
        )
        if resolved:
            delegation_spec = DelegationSpec(
                backend_id=CoreAIBackend.__name__,
                compile_specs=self._base_compile_specs
                + [
                    CompileSpec(
                        COMPILE_SPEC_KEYS.INPUT_ENUMERATIONS.value,
                        json.dumps(resolved).encode(),
                    )
                ],
            )

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            _OperatorsSupportedForCoreAIBackend(log=True),
            allows_single_node_partition=True,
        )
        partition_list = capability_partitioner.propose_partitions()
        for partition in partition_list:
            for node in partition.nodes:
                tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = tag
                partition_tags[tag] = delegation_spec

        if self.take_over_constant_data:
            tag_constant_data(exported_program)
        if self.take_over_mutable_buffer:
            tag_mutated_buffer(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags=partition_tags,
        )

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        # Preserve exactly the ops that Core AI removes from the default
        # decomposition table (e.g. scaled_dot_product_attention, silu) so they
        # reach the converter in the fused form it lowers optimally.
        from coreai_torch import get_decomp_table

        default_table = torch.export.default_decompositions()
        coreai_table = get_decomp_table()
        do_not_decompose = [
            op
            for op in default_table
            if op not in coreai_table and isinstance(op, torch._ops.OpOverload)
        ]
        return do_not_decompose, None
