# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Provide operator-support checks and registries for TOSA delegation.

Define a base check class, a registry/dispatcher, and several generic checks
used by the TOSA partitioner to decide if FX nodes are eligible for delegation.

"""


import itertools
import operator
import typing
from typing import final, Optional, Sequence, Type

import torch
import torch.fx as fx

from executorch.backends.arm._passes.arm_pass_utils import (
    get_first_fake_tensor,
    is_submodule_node,
)
from executorch.backends.arm._passes.fuse_constant_ops_pass import ComputeConstantOpsAOT
from executorch.backends.arm._passes.fuse_quantized_activation_pass import (
    FuseQuantizedActivationPass,
)
from executorch.backends.arm._passes.insert_table_ops import TableOps
from executorch.backends.arm.common.annotation_meta import ArmAnnotationInfo
from executorch.backends.arm.constants import DQ_OPS, MAX_RANK, Q_OPS
from executorch.backends.arm.operator_support.control_flow_support import (
    ControlFlowOpSupported,
    ControlFlowSubmoduleSupported,
)
from executorch.backends.arm.operator_support.ethos_u55_support import (
    EthosU55CastCheck,
    EthosU55DtypeSupport,
    EthosU55NotSupported,
    EthosU55TransposeCheck,
    EthosU55ViewCheck,
)
from executorch.backends.arm.operator_support.tosa_profile_supported_op_lists import (
    TOSA_PRO_FP_SupportList,
    TOSA_PRO_INT_SupportList,
)
from executorch.backends.arm.tosa.specification import (
    TosaSpecification,
    TosaSpecMapping,
)
from executorch.exir import ExportedProgram
from executorch.exir.backend.utils import WhyNoPartitionReporter
from executorch.exir.dialects._ops import ops as exir_ops

from torch._subclasses.fake_tensor import FakeTensor
from torch.export.graph_signature import InputKind
from torch.fx.passes.operator_support import any_chain, chain, OperatorSupportBase
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


class SupportedTOSAOperatorCheck(OperatorSupportBase):
    """Provide a base operator-support check for TOSA lowering.

    Subclasses should implement :py:meth:`is_node_tosa_supported` and declare
    the class attributes below to indicate what they support.

    Attributes:
        targets (list[OpOverload]): Operator overloads supported by this
            check.
        tosa_specs (list[TosaSpecification]): TOSA specs where the check is
            applicable.

    """

    def __init__(self, tosa_spec: TosaSpecification, reporter: WhyNoPartitionReporter):
        """Initialize the check with a TOSA spec and reporter.

        Args:
            tosa_spec (TosaSpecification): Active TOSA specification.
            reporter (WhyNoPartitionReporter): Reporter for rejection reasons.

        """
        self.tosa_spec = tosa_spec
        self.reporter = reporter

    # Class attributes populated by subclasses
    tosa_specs: list[TosaSpecification] = []
    targets: list[str] = []

    @final
    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        """Return True if the node matches targets and subclass-specific checks.

        Args:
            submodules (typing.Mapping[str, torch.nn.Module]): Exported program
                modules.
            node (fx.Node): Node to evaluate.

        Returns:
            bool: True if both the target and TOSA-specific checks pass.

        """
        if node.target not in self.targets:
            return False
        return self.is_node_tosa_supported(node, self.tosa_spec)

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:
        """Check if the node is lowerable under the given TOSA spec.

        Args:
            node (fx.Node): FX node to check.
            tosa_spec (TosaSpecification): Active TOSA specification.

        Returns:
            bool: True if supported; otherwise, False.

        """
        raise NotImplementedError("SupportedTOSAOperatorCheck must be extended.")


# container for all SupportedTosaOperatorCheck classes
_tosa_spec_support: TosaSpecMapping[Type[SupportedTOSAOperatorCheck]] = (
    TosaSpecMapping()
)


def register_tosa_support_check(checker: Type[SupportedTOSAOperatorCheck]):
    """Register an operator-support checker for one or more TOSA specs.

    Decorate subclasses of :py:class:`SupportedTOSAOperatorCheck` so they are
    picked up by the factory and partitioner for the specs declared in their
    ``tosa_specs`` class attribute.

    Args:
        checker (Type[SupportedTOSAOperatorCheck]): Checker class to register.

    """
    for tosa_spec in checker.tosa_specs:
        _tosa_spec_support.add(tosa_spec, checker)
    return checker


def get_registered_tosa_support_checks(
    tosa_spec: TosaSpecification,
) -> list[Type[SupportedTOSAOperatorCheck]]:
    """Get all registered operator-support checkers for a given spec.

    Args:
        tosa_spec (TosaSpecification): TOSA spec to query.

    Returns:
        list[Type[SupportedTOSAOperatorCheck]]: Registered checker classes.

    """
    checks = _tosa_spec_support.get(tosa_spec)
    if not checks:
        raise RuntimeError(
            f"TOSA specification not valid: {tosa_spec} not in {list(_tosa_spec_support._mapping.keys())}"
        )
    return checks


def tosa_support_factory(
    tosa_spec: TosaSpecification,
    exported_program: ExportedProgram,
    reporter: WhyNoPartitionReporter,
    additional_checks: Optional[Sequence[OperatorSupportBase]] = None,
) -> OperatorSupportBase:
    """Create an OperatorSupport composite for a TOSA spec.

    Combine profile-specific positive checks, registered operator checks, and
    negative checks into a single :py:class:`OperatorSupportBase` chain.

    Args:
        tosa_spec (TosaSpecification): Active TOSA specification.
        exported_program (ExportedProgram): Program context for checks.
        reporter (WhyNoPartitionReporter): Reporter for rejections.
        additional_checks (Optional[Sequence[OperatorSupportBase]]): Extra
            negative checks to apply.

    Returns:
        OperatorSupportBase: Composite checker for the given spec.

    """
    # Postive checks: Add nodes to partitioning
    positive_checks: list[OperatorSupportBase] = [
        ControlFlowSubmoduleSupported(exported_program, tosa_spec, reporter),
        ControlFlowOpSupported(exported_program, tosa_spec, reporter),
    ]

    if tosa_spec.support_integer():
        positive_checks.append(TOSAProINTSupportList())
    if tosa_spec.support_float():
        positive_checks.append(TOSAProFPSupportList())
    # TODO: Refactor to use TOSAProSupportLists + negtive checks
    positive_checks += [
        check(tosa_spec, reporter)
        for check in get_registered_tosa_support_checks(tosa_spec)
    ]

    # Negative checks: Remove nodes from partitioning
    negative_checks: list[OperatorSupportBase] = [
        CheckInt64InputsAndOutputs(exported_program, reporter),
        CheckFloat64Inputs(exported_program, reporter),
        RankCheck(reporter, max_rank=MAX_RANK),
        *[
            reporter.wrap_check(check, f"Rejected by {check.__class__.__name__}")
            for check in (additional_checks if additional_checks else [])
        ],
    ]

    if not tosa_spec.support_float():
        negative_checks.append(CheckArmQuantized(reporter))
        negative_checks.append(CheckProperQuantization(reporter))
    if tosa_spec.is_U55_subset:
        negative_checks.append(EthosU55NotSupported(reporter))
        negative_checks.append(EthosU55DtypeSupport(reporter))
        negative_checks.append(EthosU55TransposeCheck(reporter))
        negative_checks.append(EthosU55ViewCheck(reporter))
        negative_checks.append(EthosU55CastCheck(reporter))

    return chain(
        reporter.wrap_check(
            any_chain(*positive_checks),
            "Not included in BaseTOSASupportList or a registered tosa_support_check",
        ),
        *negative_checks,
    )


class TOSAProINTSupportList(OperatorSupportBase):
    """Provide the INT profile support list for TOSA.

    TOSA_PRO_INT_SupportList enumerates ops supported in the INT profile via
    native TOSA ops, decompositions, pre-compute steps, or TableOps.

    Note:
        Ops supported via pre-quantization decompositions are not included
        here.

    """

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        """Return True if the node is in the INT profile support list."""
        return node.op == "call_function" and node.target in TOSA_PRO_INT_SupportList


class TOSAProFPSupportList(OperatorSupportBase):
    """Provide the FP profile support list for TOSA.

    Includes ops supported natively, via decomposition/transformation, and pre-
    compute.

    """

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        """Return True if the node is in the FP profile support list."""
        return node.op == "call_function" and node.target in TOSA_PRO_FP_SupportList


class CheckArmQuantized(OperatorSupportBase):
    """
    Check if the node was marked as quantized in the Arm backend.
    This is used to ensure that nodes that were quantized in the Arm backend
    are only partitioned if they are supported by the TOSA backend.
    """

    def __init__(self, reporter: WhyNoPartitionReporter):
        self.reporter = reporter

    def _is_quantized(self, node: torch.fx.Node) -> bool:
        """Checks if the node is quantized.

        A node is considered quantized if at least one criteria is met:
        - Its dtype is not floating point or complex => integer
        - It is one of the special cases where the node has been created in to_edge, e.g.
          .Scalar operations that have been promoted .Tensor operations
          where the scalar is replaced by a full op.
        - It has been marked as quantized in the ArmAnnotationInfo custom meta.

        Args:
            node (torch.fx.Node): The FX node to check.

        Returns:
            bool: True if the node is quantized, False otherwise.
        """
        node_dtype = get_first_fake_tensor(node).dtype
        if not node_dtype.is_complex and not node_dtype.is_floating_point:
            return True
        if node.target in (
            exir_ops.edge.aten.full_like.default,
            *ComputeConstantOpsAOT.targeted_ops,
        ):
            # Special cases where nodes have been created in to_edge, e.g.
            # .Scalar operations that have been promoted .Tensor operations
            # where the scalar is replaced by a full op.
            if all(user.target in Q_OPS for user in node.users):
                return True
            for user in node.users:
                if (
                    user.target
                    == exir_ops.edge.dim_order_ops._to_dim_order_copy.default
                ):
                    dim_order_dtype = get_first_fake_tensor(user).dtype
                    if dim_order_dtype.is_complex or dim_order_dtype.is_floating_point:
                        return False
                else:
                    return False
            return True
        return (
            ArmAnnotationInfo.CUSTOM_META_KEY in node.meta.get("custom", {})
            and ArmAnnotationInfo(
                node.meta["custom"][ArmAnnotationInfo.CUSTOM_META_KEY]
            ).quantized
        )

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:

        if node.target in (*DQ_OPS, *Q_OPS):
            return True

        if not self._is_quantized(node):
            self.reporter.report_reject(
                node, "Node was not marked as quantized in the Arm backend."
            )
            return False
        return True


class CheckProperQuantization(OperatorSupportBase):
    """Ensure targeted nodes are properly quantized.

    Verify that a pair of quantize/dequantize nodes surrounds targeted ops so
    rescaling and table operators behave correctly.

    """

    targeted_ops = (
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.avg_pool2d.default,
        exir_ops.edge.aten.bmm.default,
        exir_ops.edge.aten.convolution.default,
        exir_ops.edge.aten.full.default,
        exir_ops.edge.aten.full_like.default,
        exir_ops.edge.aten.hardtanh.default,
        exir_ops.edge.aten.linear.default,
        exir_ops.edge.aten.max_pool2d_with_indices.default,
        exir_ops.edge.aten.mm.default,
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.neg.default,
        exir_ops.edge.aten.relu.default,
        exir_ops.edge.aten.sub.Tensor,
        exir_ops.edge.aten.upsample_bilinear2d.vec,
        exir_ops.edge.aten.upsample_nearest2d.vec,
        torch.ops.aten.scalar_tensor.default,
        exir_ops.edge.aten.mean.dim,
        *TableOps.included_ops(),
    )

    def __init__(self, reporter: WhyNoPartitionReporter):
        """Initialize the check with a reporter."""
        self.reporter = reporter

    def _is_matmul_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ):
        """Check quantization for decomposed matmul partitions.

        Handles an edge case where the quantized pipeline
        `dq -> torch.matmul/operator.matmul -> q` decomposes into
        `dq -> expand -> view -> aten.mm -> view -> q`.

        Args:
            submodules (Mapping[str, torch.nn.Module]): Map of child modules to
                inspect for matmul partitions.
            node (fx.Node): Node that should belong to a quantized matmul
                partition.

        Returns:
            bool: True if the matched partition uses quantized inputs and
                outputs.

        """
        for graph_module in submodules.values():
            graph_module = typing.cast(fx.GraphModule, graph_module)
            matmul_partitions_map = get_source_partitions(
                graph_module.graph,
                [
                    torch.matmul,
                    operator.matmul,
                ],
                None,
            )
            matmul_partitions = list(
                itertools.chain.from_iterable(matmul_partitions_map.values())
            )
            matched_partition = None
            for partition in matmul_partitions:
                if node in partition.nodes:
                    matched_partition = partition
            if matched_partition is not None:
                input_quantized = all(
                    input_node.target in DQ_OPS
                    for input_node in matched_partition.input_nodes
                )
                if not input_quantized:
                    self.reporter.report_reject(
                        node, "One or more matmul inputs were not quantized."
                    )
                    return False
                output_quantized = all(
                    output_node_user.target in Q_OPS
                    for output_node_user in matched_partition.output_nodes[0].users
                )
                if not output_quantized:
                    self.reporter.report_reject(
                        node, "One or more matmul outputs were not quantized."
                    )
                    return False
            else:
                self.reporter.report_reject(
                    node, "Node did not match any matmul source partition."
                )
                return False

        return True

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        """Return True if the node passes constant-cast and multi-output checks.

        Ensures decomposition-specific matmul partitions keep quantized inputs
        and outputs.

        """
        output_quantized = False
        input_quantized = False
        if node.target not in self.targeted_ops:
            return True
        elif node.target in (
            exir_ops.edge.aten.bmm.default,
            exir_ops.edge.aten.mm.default,
        ):
            source_fn_stack: tuple[typing.Any] = node.meta.get("source_fn_stack", [])
            if len(source_fn_stack) > 0:
                if source_fn_stack[-1][1] in (torch.matmul, operator.matmul):
                    return self._is_matmul_node_supported(submodules, node)

        elif node.target in (exir_ops.edge.aten.max_pool2d_with_indices.default,):
            users = node.users
            output_quantized = all(
                user.target == operator.getitem
                and all(user_user.target in Q_OPS for user_user in user.users)
                for user in users
            )
        elif FuseQuantizedActivationPass._is_fuseable_input(node):
            users = node.users
            output_quantized = all(
                FuseQuantizedActivationPass._is_fuseable_quantized_activation(user)
                for user in users
            )
        elif FuseQuantizedActivationPass._is_fuseable_quantized_activation(node):
            input_node = node.all_input_nodes[0]
            input_quantized = FuseQuantizedActivationPass._is_fuseable_input(input_node)

        input_quantized = input_quantized or all(
            (input_node.target in DQ_OPS)
            or (not get_first_fake_tensor(input_node).dtype.is_floating_point)
            for input_node in node.all_input_nodes
        )

        if not input_quantized:
            self.reporter.report_reject(node, "One or more inputs were not quantized.")
            return False

        all_q_users = all((output_node.target in Q_OPS) for output_node in node.users)
        is_floating_point = get_first_fake_tensor(node).dtype.is_floating_point
        output_quantized = output_quantized or all_q_users or not is_floating_point

        if not output_quantized:
            self.reporter.report_reject(node, "One or more outputs were not quantized.")
            return False
        return True


class CheckInt64InputsAndOutputs(OperatorSupportBase):
    """Reject general int64 tensors while allowing safe exceptions.

    Exceptions are:
        - Nodes with contant int64 output within int32 range that are cast away
          from int64 by all users.
        - Int64 output where all users are getitem nodes with non-int64 outputs.
          In this case there are multiple outputs and the int64 output is unused.
        - Nodes where all inputs are int64 constant placeholders or constant ops
          that fulfill the above exceptions.

    """

    def __init__(
        self, exported_program: ExportedProgram, reporter: WhyNoPartitionReporter
    ):
        """Initialize the check with program context and reporter."""
        self.input_names = [
            spec.arg.name
            for spec in exported_program.graph_signature.input_specs
            if spec.kind == InputKind.USER_INPUT
        ]
        self.reporter = reporter
        self.int32_min = torch.iinfo(torch.int32).min
        self.int32_max = torch.iinfo(torch.int32).max
        super().__init__()

    def inside_int32_bounds(self, node: torch.fx.Node) -> bool:
        """Node is assumed to be call_function with int64 output."""
        if isinstance(node.target, str):
            return False
        data = node.target(*node.args, **node.kwargs)
        min_val, max_val = int(torch.min(data)), int(torch.max(data))
        return min_val >= self.int32_min and max_val <= self.int32_max

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        """Return True when int64 use is absent or safe per exceptions."""
        if is_submodule_node(node):
            return True
        vals = node.meta["val"]
        tensor_list = vals if isinstance(vals, (list, tuple)) else [vals]

        any_int64 = any(tensor.dtype == torch.int64 for tensor in tensor_list)
        # Don't partition nodes with int64 output...
        if any_int64:
            # ... Except for constant ops that are directly cast to something non-int64.
            # This could be an explicit cast, or something like a less than that outputs a different dtype than the input.
            users_output_non_int64 = all(
                get_first_fake_tensor(output_node).dtype != torch.int64
                for output_node in node.users
            )
            if (
                node.target in ComputeConstantOpsAOT.targeted_ops
                and users_output_non_int64
            ):
                if not self.inside_int32_bounds(node):
                    self.reporter.report_reject(
                        node, "Constant node outside int32 range."
                    )
                    return False
                # Will never have input nodes, safe to return True
                return True

            # ... Or ops with multiple outputs where only non-int64 are used.
            users_are_getitem = all(
                user.target == operator.getitem for user in node.users
            )
            if users_are_getitem and users_output_non_int64:
                # Passed output check, go to input check.
                pass
            else:
                self.reporter.report_reject(
                    node, "Non-constant node with int64 output."
                )
                return False

        # Ops with int64 inputs are only partitioned if input nodes are constant and will be partitioned.
        # If it is not partitioned, the partition will get an int64 input and fail.
        for input_node in (
            input_node
            for input_node in node.all_input_nodes
            if input_node.op != "get_attr"
        ):
            tensor_in = get_first_fake_tensor(input_node)
            if tensor_in.dtype != torch.int64:
                continue
            # Constant placeholder
            if (
                input_node.op != "call_function"
                and input_node.name not in self.input_names
            ):
                continue
            # Constant operator
            if input_node.op == "call_function":
                if input_node.target in ComputeConstantOpsAOT.targeted_ops:
                    # This is not perfect since the input_node can still be rejected by other checks but
                    # this should cover the majority of cases.
                    if self.is_node_supported({}, input_node):
                        continue
            self.reporter.report_reject(
                node, f"Non-constant int64 input {input_node.name}"
            )
            return False

        return True


class CheckFloat64Inputs(OperatorSupportBase):
    """Reject nodes with float64 inputs.

    Useful as a negative check for specs that do not allow float64.

    """

    def __init__(
        self, exported_program: ExportedProgram, reporter: WhyNoPartitionReporter
    ):
        """Initialize the check with program context and reporter."""
        self.reporter = reporter
        super().__init__()

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        """Return True if no float64 inputs are present."""
        if is_submodule_node(node):
            return True
        for input_node in (
            input_node
            for input_node in node.all_input_nodes
            if input_node.op != "get_attr"
        ):
            tensor = get_first_fake_tensor(input_node)
            if tensor.dtype == torch.float64:
                self.reporter.report_reject(
                    node,
                    f"Had float64 input {input_node.name} that couldn't be handled.",
                )
                return False
        return True


class RankCheck(OperatorSupportBase):
    """Reject nodes with rank greater than ``max_rank``."""

    def __init__(self, reporter: WhyNoPartitionReporter, max_rank: int):
        """Initialize the check with a reporter and maximum rank."""
        self.reporter = reporter
        self.max_rank = max_rank
        super().__init__()

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        """Return True if input/output tensor ranks are within the limit."""
        if is_submodule_node(node):
            return True
        input_nodes = (
            input_node
            for input_node in node.all_input_nodes
            if input_node.op != "get_attr"
        )
        # check if any input node has an unsupported rank
        for input_node in input_nodes:
            input_node_shape = get_first_fake_tensor(input_node).shape
            if len(input_node_shape) > self.max_rank:
                self.reporter.report_reject(
                    node,
                    f"{node.name} has input_node {input_node.name} with shape {input_node_shape}, "
                    f"rank {len(input_node_shape)} which is unsupported. "
                    f"Max supported rank is {self.max_rank}.",
                )
                return False

        meta_val = node.meta["val"]
        if isinstance(
            meta_val, (Sequence, torch.fx.immutable_collections.immutable_list)
        ):
            for val in meta_val:
                if isinstance(val, FakeTensor):
                    if len(val.shape) > self.max_rank:
                        self.reporter.report_reject(
                            node,
                            f"{node.name} has a shape {val.shape}, rank {len(val.shape)} which is unsupported."
                            f"Max supported rank is {self.max_rank}.",
                        )
                        return False
        elif isinstance(meta_val, FakeTensor):
            if len(meta_val.shape) > self.max_rank:
                self.reporter.report_reject(
                    node,
                    f"{node.name} has shape {meta_val.shape}, rank={len(meta_val.shape)} which is unsupported."
                    f"Max supported rank is {self.max_rank}.",
                )
                return False
        return True
