# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import logging
from typing import Callable, List, Optional, Tuple

import torch

from executorch.backends.apple.coreai.compiler.preprocess import (
    AOTCompileConfig,
    COMPILE_SPEC_KEYS,
    CoreAIBackend,
)
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

        # Core AI has no scalar-symint graph input, and it narrows i64/f64 to
        # 32-bit, so it can't faithfully carry an i64/f64 tensor across the
        # delegate boundary. Reject any node with a SymInt/SymFloat/SymBool
        # operand or an i64/f64 tensor operand/result; the default
        # NarrowToCoreAIDtypesPass casts i64/f64 to 32-bit at the EP boundary so
        # only the boundary cast (which keeps an i64/f64 tensor) is left out.
        for val in _node_tensor_vals(node):
            if isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                self.log_once(
                    f"Node {node.name} has a symbolic scalar operand; "
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
        uses_sidecar: bool = False,
        aot_compile_config: Optional[AOTCompileConfig] = None,
        min_deployment_version: Optional[str] = None,
        take_over_constant_data: bool = True,
        take_over_mutable_buffer: bool = True,
    ) -> None:
        # uses_sidecar selects sidecar delivery (vs inline). It is embedded as a
        # compile spec because the runtime needs to know how to load the asset,
        # but it carries only the mode, no path. The build-time output directory
        # comes from the COREAI_SIDECAR_DIR env var (see preprocess.py /
        # coreai_sidecar_dir), never a compile spec, so no build-machine path is
        # serialized.
        #
        # aot_compile_config requests ahead-of-time ``xcrun coreai-build
        # compile`` in preprocess, emitting per-architecture ``.aimodelc``
        # bundles instead of the portable ``.aimodel``. It is serialized as a
        # single JSON AOT_COMPILE_CONFIG spec (presence implies AOT).
        # min_deployment_version is a general spec (it also sets the portable
        # .aimodel's OS floor), so it is emitted separately.
        specs = []
        if uses_sidecar:
            specs.append(CompileSpec(COMPILE_SPEC_KEYS.USES_SIDECAR.value, b"1"))
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
        delegation_spec = self.delegation_spec

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
