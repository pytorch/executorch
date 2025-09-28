# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import itertools
import operator
import typing
from typing import final, Optional, Sequence, Type

import torch
import torch.fx as fx

from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm._passes.fuse_constant_ops_pass import ComputeConstantOpsAOT
from executorch.backends.arm._passes.fuse_quantized_activation_pass import (
    FuseQuantizedActivationPass,
)
from executorch.backends.arm._passes.insert_table_ops import TableOps
from executorch.backends.arm.constants import DQ_OPS, Q_OPS
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
from executorch.backends.arm.tosa import TosaSpecification
from executorch.exir import ExportedProgram
from executorch.exir.backend.utils import WhyNoPartitionReporter
from executorch.exir.dialects._ops import ops as exir_ops

from torch._subclasses.fake_tensor import FakeTensor
from torch.export.graph_signature import InputKind
from torch.fx.passes.operator_support import any_chain, chain, OperatorSupportBase
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


class SupportedTOSAOperatorCheck(OperatorSupportBase):
    """
    Supported OP for TOSA lowering
    """

    def __init__(self, tosa_spec: TosaSpecification, reporter: WhyNoPartitionReporter):
        self.tosa_spec = tosa_spec
        self.reporter = reporter

    # Should be populated by subclass implementation
    tosa_specs: list[TosaSpecification] = []
    targets: list[str] = []

    @final
    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        if node.target not in self.targets:
            return False
        return self.is_node_tosa_supported(node, self.tosa_spec)

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:
        """
        Checks if the fx.Node node is lowerable using the TOSA specification defined by tosa_spec.
        """
        raise NotImplementedError("SupportedTOSAOperatorCheck must be extended.")


# container for all SupportedTosaOperatorCheck classes
_tosa_spec_support: dict[TosaSpecification, list[Type[SupportedTOSAOperatorCheck]]] = {
    TosaSpecification.create_from_string("TOSA-1.0+INT"): [],
    TosaSpecification.create_from_string("TOSA-1.0+FP"): [],
}


def register_tosa_support_check(checker: Type[SupportedTOSAOperatorCheck]):
    """
    Decorator to mark a subclass implmentation of SupportedTosaOperatorCheck
    to be registered for checking if a torch.fx.Node is lowerable given
    a TOSA specification.
    """
    for tosa_spec in checker.tosa_specs:
        _tosa_spec_support[tosa_spec].append(checker)
    return checker


def get_registered_tosa_support_checks(
    tosa_spec: TosaSpecification,
) -> list[Type[SupportedTOSAOperatorCheck]]:
    if tosa_spec not in _tosa_spec_support:
        raise RuntimeError(
            f"TOSA specification not valid: {tosa_spec} not in {list(_tosa_spec_support.keys())}"
        )

    return _tosa_spec_support[tosa_spec]


def tosa_support_factory(
    tosa_spec: TosaSpecification,
    exported_program: ExportedProgram,
    reporter: WhyNoPartitionReporter,
    additional_checks: Optional[Sequence[OperatorSupportBase]] = None,
) -> OperatorSupportBase:
    """Generates an OperatorSupport class depending on the given `tosa_spec`.
    Additional checks can be supplied to avoid partitioning additional nodes.
    """
    # Postive checks: Add nodes to partitioning
    positive_checks: list[OperatorSupportBase] = []

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
        RankCheck(reporter, max_rank=5),
        *[
            reporter.wrap_check(check, f"Rejected by {check.__class__.__name__}")
            for check in (additional_checks if additional_checks else [])
        ],
    ]

    if not tosa_spec.support_float():
        negative_checks.append(NeedsDecompositionCheck(reporter))
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
    """
    TOSA_PRO_INT_SupportList:
        Ops supported in INT profile via native TOSA ops, decomposition/transformation, pre-compute, or TableOps
    """

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:

        return node.op == "call_function" and node.target in TOSA_PRO_INT_SupportList


class TOSAProFPSupportList(OperatorSupportBase):
    """
    TOSA_PRO_FP_SupportList:
        Ops supported in FP profile via native TOSA ops, decomposition/transformation, pre-compute
    """

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:

        return node.op == "call_function" and node.target in TOSA_PRO_FP_SupportList


class NeedsDecompositionCheck(OperatorSupportBase):
    """
    Targeted operators need to be decomposed prior to quantization in order to get a pair of q-dq-nodes surrounding
    the operator, and to get optimal quantization parameters for each operator. This check will reject operators
    that need to be decomposed.
    """

    def __init__(self, reporter: WhyNoPartitionReporter):
        self.reporter = reporter

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:

        if node.op != "call_function":
            return True

        needs_decomp_dict = {
            exir_ops.edge.aten.div.Tensor: None,
            exir_ops.edge.aten._native_batch_norm_legit_no_training.default: "BatchNorm2D with track_running_stats==True not immediately following a convolution is not supported for quantized TOSA backends.",
            exir_ops.edge.aten.native_layer_norm.default: None,
            exir_ops.edge.aten.native_group_norm.default: None,
            exir_ops.edge.aten._softmax.default: None,
            exir_ops.edge.aten._log_softmax.default: None,
            exir_ops.edge.aten.var.correction: None,
            exir_ops.edge.aten.var.dim: None,
            exir_ops.edge.aten.add.Scalar: None,
            exir_ops.edge.aten.sqrt.default: None,
            exir_ops.edge.aten.sub.Scalar: None,
            exir_ops.edge.aten.mul.Scalar: None,
            exir_ops.edge.aten.ne.Tensor: None,
            exir_ops.edge.aten.ne.Scalar: None,
            exir_ops.edge.aten.div.Scalar: None,
            exir_ops.edge.aten.leaky_relu.default: None,
            exir_ops.edge.aten.round.default: None,
            exir_ops.edge.aten.addmm.default: None,
            exir_ops.edge.aten.glu.default: None,
            exir_ops.edge.aten.logit.default: None,
        }

        if node.target in needs_decomp_dict:
            reject_message = needs_decomp_dict[node.target]
            if reject_message is None:
                reject_message = "Op needs to be decomposed into other ops before quantization to get quantized properly."

            self.reporter.report_reject(node, reject_message)
            return False
        else:
            return True


class CheckProperQuantization(OperatorSupportBase):
    """
    For targeted nodes, check that it has been quantized as expected. In most cases this means that a pair of quantize
    and dequantize nodes surrounds the node. This is neccessary for table operators and operators that need to rescale
    activations.
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
        self.reporter = reporter

    def _is_matmul_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ):
        """
        Find the matmul source partition containing this node and check that all its inputs and outputs are quantized.
        """
        for graph_module in submodules.values():
            graph_module = typing.cast(fx.GraphModule, graph_module)
            matmul_partitions = get_source_partitions(
                graph_module.graph,
                [
                    torch.matmul,
                    operator.matmul,
                ],
                None,
            )
            matmul_partitions = list(
                itertools.chain.from_iterable(matmul_partitions.values())
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
    """TOSA does not support int64 tensors so in general, ops with int64 inputs or outputs should not be partitioned.
    There are however some exceptions:
        - Nodes with int64 output can be partitioned if they are constant, within int32,
            and all users cast to something else. In this case, the int64 tensor can safely be cast to int32 AOT.
        - Nodes with int64 output can be partitioned if all users are getitem with non-int64 output.
            In this case, there are multiple outputs and the int64 ones are not used.
        - Nodes with int64 inputs can be partitioned if the inputs are constant placeholders, or constant
            ops fulfilling the criteria above.
    Note that we don't check placeholders here, they are partitioned based on whether their users are partitioned
    or not.
    """

    def __init__(
        self, exported_program: ExportedProgram, reporter: WhyNoPartitionReporter
    ):
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
        for input_node in node.all_input_nodes:
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
                    if self.is_node_supported(
                        None, input_node  # type: ignore[arg-type] #(we don't use 'submodules')
                    ):
                        continue
            self.reporter.report_reject(
                node, f"Non-constant int64 input {input_node.name}"
            )
            return False

        return True


class CheckFloat64Inputs(OperatorSupportBase):

    def __init__(
        self, exported_program: ExportedProgram, reporter: WhyNoPartitionReporter
    ):
        self.reporter = reporter
        super().__init__()

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:

        for input_node in node.all_input_nodes:
            tensor = get_first_fake_tensor(input_node)
            if tensor.dtype == torch.float64:
                self.reporter.report_reject(
                    node,
                    f"Had float64 input {input_node.name} that couldn't be handled.",
                )
                return False
        return True


class RankCheck(OperatorSupportBase):
    """Makes sure that nodes with input or output tensors with rank > max_rank are not partitioned"""

    def __init__(self, reporter: WhyNoPartitionReporter, max_rank: int):
        self.reporter = reporter
        self.max_rank = max_rank
        super().__init__()

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        input_nodes = node.all_input_nodes
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
