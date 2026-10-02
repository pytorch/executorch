# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Provide operator-support checks and registries for TOSA delegation.

Define a base check class, a registry/dispatcher, and several generic checks
used by the TOSA partitioner to decide if FX nodes are eligible for delegation.

"""

import math
import operator
import typing
from typing import final, Optional, Sequence, Type

# Register Arm-specific torch.library ops and MXFP transforms at package
# import time.
import executorch.backends.arm.ao_ext  # noqa: F401

import torch
import torch.fx as fx
from executorch.backends.arm._passes.arm_pass_utils import (
    get_first_fake_tensor,
    is_submodule_node,
)
from executorch.backends.arm._passes.fuse_constant_ops_pass import (
    ComputeConstantOpsAOTPass,
)
from executorch.backends.arm._passes.fuse_quantized_activation_pass import (
    FuseQuantizedActivationPass,
)
from executorch.backends.arm._passes.insert_table_ops import TableOps
from executorch.backends.arm._passes.prepare_gather_indices_pass import (
    is_safe_int32_to_int64_gather_boundary,
)

from executorch.backends.arm._passes.size_adjust_input_pass import (
    get_slices_convolution,
    get_slices_pooling,
    has_dynamic_conv_padding,
    has_dynamic_pooling_padding,
)
from executorch.backends.arm.common.annotation_meta import ArmAnnotationInfo
from executorch.backends.arm.constants import DQ_OPS, MAX_RANK, Q_OPS
from executorch.backends.arm.operator_support.control_flow_support import (
    ControlFlowOpSupported,
    ControlFlowSubmoduleSupported,
    ControlFlowSubmoduleSupportList,
)
from executorch.backends.arm.operator_support.ethos_u55_support import (
    EthosU55CastCheck,
    EthosU55DtypeSupport,
    EthosU55IndexSelectCheck,
    EthosU55IndexTensorCheck,
    EthosU55NotSupported,
    EthosU55ResizeCheck,
    EthosU55ReverseCheck,
    EthosU55UnfoldCopyCheck,
)
from executorch.backends.arm.operator_support.tosa_profile_supported_op_lists import (
    TOSA_EXT_CONTROL_FLOW_SupportList,
    TOSA_EXT_MXFP_SupportList,
    TOSA_EXT_SHAPE_SupportList,
    TOSA_PRO_FP_SupportList,
    TOSA_PRO_INT_SupportList,
    TOSA_PRO_MIXED_INT_SupportList,
)
from executorch.backends.arm.tosa.specification import (
    get_context_shape_env,
    TosaSpecification,
    TosaSpecMapping,
)
from executorch.exir import ExportedProgram
from executorch.exir.backend.utils import WhyNoPartitionReporter
from executorch.exir.dialects._ops import ops as exir_ops

from torch._subclasses.fake_tensor import FakeTensor
from torch.export.graph_signature import InputKind
from torch.fx.passes.operator_support import any_chain, chain, OperatorSupportBase


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
    tosa_specs: list[TosaSpecification] = TosaSpecification.all_versions_and_profiles()
    targets: list[object] = []

    @final
    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        """Apply the subclass-specific check to matching targets.

        Args:
            submodules (typing.Mapping[str, torch.nn.Module]): Exported program
                modules.
            node (fx.Node): Node to evaluate.

        Returns:
            bool: True for unrelated nodes or when the TOSA-specific check passes.

        """
        if node.target not in self.targets:
            return True
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


def _is_integer_dtype(dtype: torch.dtype) -> bool:
    return not dtype.is_floating_point and not dtype.is_complex


@register_tosa_support_check
class ProductSupported(SupportedTOSAOperatorCheck):
    """Provide TOSA support check for product reductions."""

    # TOSA REDUCE_PRODUCT is only available to the floating-point path used by
    # this checker. Do not register prod.dim_int as positive support for an
    # INT-only specification such as Ethos-U55.
    tosa_specs = TosaSpecification.all_versions_for_profile("FP")
    targets = [exir_ops.edge.aten.prod.dim_int]

    @staticmethod
    def _supported_dtypes(tosa_spec: TosaSpecification) -> list[torch.dtype]:
        if not tosa_spec.support_float():
            return []

        supported_dtypes = [torch.float16, torch.float32]
        if tosa_spec.support_extension("bf16"):
            supported_dtypes.append(torch.bfloat16)
        return supported_dtypes

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:
        """Return True if product reduction input dtype is supported."""
        supported_dtypes = self._supported_dtypes(tosa_spec)
        if not supported_dtypes:
            self.reporter.report_reject(
                node,
                f"TOSA spec {tosa_spec} does not support REDUCE_PRODUCT.",
            )
            return False

        input_dtype = get_first_fake_tensor(node.all_input_nodes[0]).dtype
        if input_dtype in supported_dtypes:
            return True

        self.reporter.report_reject(
            node,
            (
                f"Input dtype {input_dtype} is not supported for {node.target}; "
                f"expected one of {supported_dtypes}."
            ),
        )
        return False


def _is_quantized_constant(node: torch.fx.Node) -> bool:
    if node.target not in (
        exir_ops.edge.aten.full_like.default,
        *ComputeConstantOpsAOTPass.targeted_ops,
    ):
        return False

    users = tuple(node.users)
    if users and all(user.target in Q_OPS for user in users):
        # The node feeds directly into only quantized ops.
        return True

    for user in users:
        if user.target == exir_ops.edge.dim_order_ops._to_dim_order_copy.default:
            dim_order_dtype = get_first_fake_tensor(user).dtype
            if not _is_integer_dtype(dim_order_dtype):
                return False
        else:
            return False

    return len(users) > 0


def _floating_profile_negative_checks(
    tosa_spec: TosaSpecification, reporter: WhyNoPartitionReporter
) -> list[OperatorSupportBase]:
    checks: list[OperatorSupportBase] = [CheckMixedFloatingInputs(reporter)]
    if not tosa_spec.support_integer():
        checks.append(CheckFPComparisonInputs(reporter))
    return checks


def is_quantized(node: torch.fx.Node) -> bool:
    """Checks if the node is quantized.

    A node is considered quantized if any of the following is true:
    - Its output dtype is not floating point or complex => integer
    - It is an op that produces a constant that in turn feeds only quantized users
    - It has been marked as quantized in the ArmAnnotationInfo custom meta.

    Args:
        node (torch.fx.Node): The FX node to check.

    Returns:
        bool: True if the node is quantized, False otherwise.

    """

    try:
        node_dtype = get_first_fake_tensor(node).dtype
        # Integer-like dtype implies the node is already quantized as long
        # as inputs are not floating-point.
        if _is_integer_dtype(node_dtype):
            input_nodes = node.all_input_nodes
            input_nodes_dtypes = [
                get_first_fake_tensor(input_node).dtype for input_node in input_nodes
            ]
            if all(
                _is_integer_dtype(input_node_dtype)
                for input_node_dtype in input_nodes_dtypes
            ):
                return True

    except TypeError:
        # Could not determine dtype, fall back to other checks.
        pass

    # Nodes introduced during lowering that exclusively feed quantized users.
    if _is_quantized_constant(node):
        return True

    # Finally, fall back to the explicit annotation emitted by Arm passes.
    custom_meta = node.meta.get("custom", {})
    if ArmAnnotationInfo.CUSTOM_META_KEY in custom_meta:
        return custom_meta[ArmAnnotationInfo.CUSTOM_META_KEY]["quantized"]

    return False


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


class TOSAExtensionSupportList(OperatorSupportBase):
    """Accept operators belonging to an enabled TOSA extension."""

    def __init__(self, targets: typing.Container[object]) -> None:
        self.targets = targets

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        return node.op == "call_function" and node.target in self.targets


class DynamicW8A8QParamsSupport(OperatorSupportBase):
    """Admit the canonical dynamic INT8 activation helper chain.

    Partition-time support is intentionally limited to the helper chain itself:

        choose_qparams_symmetric.tensor
          -> getitem(0/1)
          -> quantize_per_tensor.tensor
          -> dequantize_per_tensor.tensor

    The backend detector remains responsible for validating the destination
    Linear, static weight representation, shapes, scales, zero points, and
    dtypes before replacing the Linear with INT8 MATMUL.

    """

    def __init__(self, tosa_spec: TosaSpecification):
        self.tosa_spec = tosa_spec

    @staticmethod
    def _choose_target():
        return exir_ops.edge.quantized_decomposed.choose_qparams_symmetric.tensor

    @staticmethod
    def _q_target():
        return exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor

    @staticmethod
    def _dq_target():
        return exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor

    @staticmethod
    def _is_int_literal(value: object, expected: int) -> bool:
        return (
            isinstance(value, int) and not isinstance(value, bool) and value == expected
        )

    @staticmethod
    def _node_dtype(node: object) -> torch.dtype | None:
        if not isinstance(node, fx.Node):
            return None
        return getattr(node.meta.get("val"), "dtype", None)

    @staticmethod
    def _same_arg(lhs: object, rhs: object) -> bool:
        if isinstance(lhs, fx.Node) or isinstance(rhs, fx.Node):
            return lhs is rhs
        if isinstance(lhs, torch.dtype) or isinstance(rhs, torch.dtype):
            return lhs is rhs
        return lhs == rhs

    @classmethod
    def _same_args(cls, lhs: tuple[object, ...], rhs: tuple[object, ...]) -> bool:
        return len(lhs) == len(rhs) and all(
            cls._same_arg(a, b) for a, b in zip(lhs, rhs)
        )

    @staticmethod
    def _node_rank(node: object) -> int | None:
        if not isinstance(node, fx.Node):
            return None
        shape = getattr(node.meta.get("val"), "shape", None)
        return None if shape is None else len(shape)

    @staticmethod
    def _epsilon_is_valid(value: object) -> bool:
        # Keep partition-time support deliberately conservative. The choose
        # decomposition can resolve a few additional static forms, but the
        # partitioner must not bypass normal support checks unless it can
        # guarantee that the node will be decomposed later.
        return (
            isinstance(value, (float, int))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0
        )

    @classmethod
    def _choose_signature_is_valid(cls, node: fx.Node) -> bool:
        if (
            node.op != "call_function"
            or node.target != cls._choose_target()
            or len(node.args) < 5
        ):
            return False

        x, qmin, qmax, eps, dtype = node.args[:5]
        rank = cls._node_rank(x)
        return (
            cls._is_int_literal(qmin, -127)
            and cls._is_int_literal(qmax, 127)
            and dtype is torch.int8
            and cls._node_dtype(x) is torch.float32
            and rank is not None
            and rank > 0
            and cls._epsilon_is_valid(eps)
        )

    @classmethod
    def _getitem_info(cls, node: fx.Node) -> tuple[fx.Node, int] | None:
        if (
            node.op != "call_function"
            or node.target is not operator.getitem
            or len(node.args) < 2
            or not isinstance(node.args[0], fx.Node)
            or not cls._choose_signature_is_valid(node.args[0])
        ):
            return None
        if cls._is_int_literal(node.args[1], 0):
            return node.args[0], 0
        if cls._is_int_literal(node.args[1], 1):
            return node.args[0], 1
        return None

    @classmethod
    def _q_signature_is_valid(cls, node: fx.Node) -> bool:
        if (
            node.op != "call_function"
            or node.target != cls._q_target()
            or len(node.args) < 6
        ):
            return False

        scale = node.args[1]
        zero_point = node.args[2]
        if not isinstance(scale, fx.Node) or not isinstance(zero_point, fx.Node):
            return False

        scale_info = cls._getitem_info(scale)
        zero_point_info = cls._getitem_info(zero_point)
        return (
            scale_info is not None
            and zero_point_info is not None
            and scale_info[0] is zero_point_info[0]
            and scale_info[1] == 0
            and zero_point_info[1] == 1
            and cls._is_int_literal(node.args[3], -127)
            and cls._is_int_literal(node.args[4], 127)
            and node.args[5] is torch.int8
            and cls._node_dtype(node.args[0]) is torch.float32
            and cls._node_dtype(node) is torch.int8
        )

    @classmethod
    def _dq_matches_q(cls, dq: fx.Node, q: fx.Node) -> bool:
        if (
            dq.op != "call_function"
            or dq.target != cls._dq_target()
            or len(dq.args) < 6
            or dq.args[0] is not q
        ):
            return False
        return (
            cls._same_args(tuple(dq.args[1:6]), tuple(q.args[1:6]))
            and dq.kwargs.get("out_dtype") in (None, torch.float32)
            and cls._node_dtype(dq) is torch.float32
        )

    @classmethod
    def _is_dynamic_q(cls, node: fx.Node) -> bool:
        if not cls._q_signature_is_valid(node):
            return False
        return any(
            isinstance(user, fx.Node) and cls._dq_matches_q(user, node)
            for user in node.users
        )

    @classmethod
    def _is_dynamic_dq(cls, node: fx.Node) -> bool:
        if not node.args or not isinstance(node.args[0], fx.Node):
            return False
        q = node.args[0]
        return cls._q_signature_is_valid(q) and cls._dq_matches_q(node, q)

    @classmethod
    def _getitem_participates(cls, node: fx.Node) -> bool:
        if cls._getitem_info(node) is None:
            return False
        return any(
            (user.target == cls._q_target() and cls._is_dynamic_q(user))
            or (user.target == cls._dq_target() and cls._is_dynamic_dq(user))
            for user in node.users
            if user.op == "call_function"
        )

    @classmethod
    def _choose_participates(cls, node: fx.Node) -> bool:
        if not cls._choose_signature_is_valid(node) or not node.users:
            return False

        indices: set[int] = set()
        for user in node.users:
            info = cls._getitem_info(user)
            if info is None or not cls._getitem_participates(user):
                return False
            indices.add(info[1])
        return indices == {0, 1}

    @classmethod
    def matches(cls, node: fx.Node) -> bool:
        if node.target == cls._choose_target():
            return cls._choose_participates(node)
        if node.target is operator.getitem:
            return cls._getitem_participates(node)
        if node.target == cls._q_target():
            return cls._is_dynamic_q(node)
        if node.target == cls._dq_target():
            return cls._is_dynamic_dq(node)
        return False

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        del submodules
        return (
            self.tosa_spec.support_integer()
            and self.tosa_spec.support_float()
            and self.matches(node)
        )


class _AllowDynamicW8A8QParamsOr(OperatorSupportBase):
    """Bypass dtype/profile checks only for the validated qparam helper
    nodes.
    """

    def __init__(self, wrapped: OperatorSupportBase, tosa_spec: TosaSpecification):
        self.wrapped = wrapped
        self.dynamic = DynamicW8A8QParamsSupport(tosa_spec)

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        return self.dynamic.is_node_supported(
            submodules, node
        ) or self.wrapped.is_node_supported(submodules, node)


def _registered_negative_checks(
    tosa_spec: TosaSpecification,
    reporter: WhyNoPartitionReporter,
) -> list[OperatorSupportBase]:
    check_types = dict.fromkeys(get_registered_tosa_support_checks(tosa_spec))
    return [check(tosa_spec, reporter) for check in check_types]


def _positive_checks(
    tosa_spec: TosaSpecification,
) -> list[OperatorSupportBase]:
    checks: list[OperatorSupportBase] = []

    if tosa_spec.support_integer() and tosa_spec.support_float():
        checks.append(
            TOSAExtensionSupportList(
                TOSA_PRO_MIXED_INT_SupportList | TOSA_PRO_FP_SupportList
            )
        )
    elif tosa_spec.support_integer():
        checks.append(TOSAExtensionSupportList(TOSA_PRO_INT_SupportList))
    elif tosa_spec.support_float():
        checks.append(TOSAExtensionSupportList(TOSA_PRO_FP_SupportList))

    checks.append(DynamicW8A8QParamsSupport(tosa_spec))

    if tosa_spec.support_extension("cf"):
        checks.append(TOSAExtensionSupportList(TOSA_EXT_CONTROL_FLOW_SupportList))
        checks.append(ControlFlowSubmoduleSupportList())

    if tosa_spec.support_extension("mxfp"):
        checks.append(TOSAExtensionSupportList(TOSA_EXT_MXFP_SupportList))

    if tosa_spec.support_extension("shape"):
        checks.append(TOSAExtensionSupportList(TOSA_EXT_SHAPE_SupportList))

    return checks


def _disallowed_dtypes(tosa_spec: TosaSpecification) -> list[torch.dtype]:
    dtypes = [torch.float64, torch.complex32, torch.complex64, torch.complex128]
    if not tosa_spec.support_extension("bf16"):
        dtypes.append(torch.bfloat16)
    if not (
        tosa_spec.support_extension("fp8e4m3") or tosa_spec.support_extension("mxfp")
    ):
        dtypes.append(torch.float8_e4m3fn)
    if not (
        tosa_spec.support_extension("fp8e5m2") or tosa_spec.support_extension("mxfp")
    ):
        dtypes.append(torch.float8_e5m2)
    if tosa_spec.is_U55_subset:
        dtypes.append(torch.bool)
    return dtypes


def _negative_checks(
    tosa_spec: TosaSpecification,
    exported_program: ExportedProgram,
    reporter: WhyNoPartitionReporter,
    additional_checks: Optional[Sequence[OperatorSupportBase]],
) -> list[OperatorSupportBase]:
    checks: list[OperatorSupportBase] = [RankCheck(reporter, MAX_RANK)]
    checks.append(CheckKnownUnsupportedTOSASemantics(reporter))
    checks.append(ControlFlowSubmoduleSupported(exported_program, tosa_spec, reporter))
    checks.append(ControlFlowOpSupported(exported_program, tosa_spec, reporter))
    checks.extend(_registered_negative_checks(tosa_spec, reporter))

    if tosa_spec.support_integer() and tosa_spec.support_float():
        checks.append(CheckMixedProfileOperatorSupport(reporter))

    if not tosa_spec.support_extension("int64"):
        checks.append(
            _AllowDynamicW8A8QParamsOr(
                CheckInt64InputsAndOutputs(exported_program, reporter, tosa_spec),
                tosa_spec,
            )
        )

    checks.append(CheckScalarReductionInputs(reporter))

    checks.extend(additional_checks or ())

    if tosa_spec.support_float():
        checks.extend(
            _AllowDynamicW8A8QParamsOr(check, tosa_spec)
            for check in _floating_profile_negative_checks(tosa_spec, reporter)
        )
    else:
        checks.append(CheckArmQuantized(reporter))
        checks.append(CheckProperQuantization(reporter))

    checks.append(
        _AllowDynamicW8A8QParamsOr(
            CheckDtypeInputsAndOutputs(
                exported_program, reporter, _disallowed_dtypes(tosa_spec), tosa_spec
            ),
            tosa_spec,
        )
    )

    if tosa_spec.is_U55_subset:
        checks.append(EthosU55NotSupported(reporter, tosa_spec))
        checks.append(EthosU55ResizeCheck(reporter))
        checks.append(EthosU55ReverseCheck(reporter))
        checks.append(EthosU55UnfoldCopyCheck(reporter))
        checks.append(EthosU55IndexTensorCheck(exported_program, reporter))
        checks.append(EthosU55IndexSelectCheck(exported_program, reporter))
        checks.append(EthosU55DtypeSupport(reporter))
        checks.append(EthosU55CastCheck(reporter))

    if not tosa_spec.support_extension("shape"):
        checks.append(SymbolicShapeSupportCheck(reporter))

    return [
        reporter.wrap_check(check, f"Rejected by {check.__class__.__name__}")
        for check in checks
    ]


_ARGMAX_OPS = (
    torch.ops.aten.argmax.default,
    exir_ops.edge.aten.argmax.default,
)

_TO_DIM_ORDER_COPY_OPS = (
    torch.ops.dim_order_ops._to_dim_order_copy.default,
    exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
)


def _argmax_all_users_cast_to_int32(node: fx.Node) -> bool:
    # TOSA ARGMAX produces int32 indices. This is only a faithful lowering
    # when every user immediately narrows aten.argmax's int64 result:
    #
    #   aten.argmax -> _to_dim_order_copy(dtype=int32) -> users
    if node.target not in _ARGMAX_OPS or not node.users:
        return False

    return all(
        user.target in _TO_DIM_ORDER_COPY_OPS
        and user.kwargs.get("dtype") == torch.int32
        for user in node.users
    )


class CheckKnownUnsupportedTOSASemantics(OperatorSupportBase):
    """Reject ops whose TOSA lowering is known to differ from PyTorch."""

    def __init__(self, reporter: WhyNoPartitionReporter):
        self.reporter = reporter

    def _check_argmax(self, node: fx.Node) -> bool:
        if _argmax_all_users_cast_to_int32(node):
            return True

        self.reporter.report_reject(
            node,
            "TOSA ARGMAX produces int32 but aten.argmax returns int64.",
        )
        return False

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        if node.target in (
            torch.ops.aten.argmax.default,
            exir_ops.edge.aten.argmax.default,
        ):
            return self._check_argmax(node)

        return True


def tosa_support_factory(
    tosa_spec: TosaSpecification,
    exported_program: ExportedProgram,
    reporter: WhyNoPartitionReporter,
    additional_checks: Optional[Sequence[OperatorSupportBase]] = None,
    additional_positive_checks: Optional[Sequence[OperatorSupportBase]] = None,
    additional_positive_overrides: Optional[Sequence[OperatorSupportBase]] = None,
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
        additional_positive_checks (Optional[Sequence[OperatorSupportBase]]):
            Extra positive checks to add to the support list.
        additional_positive_overrides (Optional[Sequence[OperatorSupportBase]]):
            Extra positive checks, overriding any later negative checks. Use with
            caution!

    Returns:
        OperatorSupportBase: Composite checker for the given spec.

    """
    positive_checks = _positive_checks(tosa_spec)
    if additional_positive_checks:
        positive_checks.extend(additional_positive_checks)
    negative_checks = _negative_checks(
        tosa_spec,
        exported_program,
        reporter,
        additional_checks,
    )

    # An op must be accepted by at least one postitive check, and not rejected by any
    # negative checks
    default_checks = chain(
        reporter.wrap_check(
            any_chain(*positive_checks),
            "Not included in BaseTOSASupportList or a registered tosa_support_check",
        ),
        *negative_checks,
    )

    # Let postitive overrides accept an op regardless of regular checks
    return any_chain(*additional_positive_overrides or (), default_checks)


def _has_symbolic_shape(node: fx.Node) -> bool:
    val = node.meta.get("val")
    vals = val if isinstance(val, (list, tuple)) else (val,)
    for node_val in vals:
        if isinstance(node_val, torch.SymInt):
            return True

        shape = getattr(node_val, "shape", None)
        if shape is not None and any(isinstance(dim, torch.SymInt) for dim in shape):
            return True

    return False


class SymbolicShapeSupportCheck(OperatorSupportBase):
    """Reject symbolic shape constructs that require the TOSA shape
    extension.
    """

    _SYMBOLIC_SPATIAL_DIM_TARGETS = (
        exir_ops.edge.aten.convolution.default,
        exir_ops.edge.aten.avg_pool2d.default,
        exir_ops.edge.aten.max_pool2d.default,
        exir_ops.edge.aten.max_pool2d_with_indices.default,
        exir_ops.edge.aten._adaptive_avg_pool2d.default,
        torch.ops.aten.conv_transpose2d.input,
    )
    _SYMBOLIC_MEAN_TARGETS = (
        exir_ops.edge.aten.mean.dim,
        exir_ops.edge.aten.mean.default,
    )
    _SYMBOLIC_VIEW_SHAPE_TARGETS = (
        exir_ops.edge.aten.squeeze_copy.dim,
        exir_ops.edge.aten.squeeze_copy.dims,
        exir_ops.edge.aten.unsqueeze_copy.default,
    )
    _SYMBOLIC_PAD_TARGETS = (exir_ops.edge.aten.constant_pad_nd.default,)
    _SYMBOLIC_SLICE_TARGETS = (exir_ops.edge.aten.slice_copy.Tensor,)

    def __init__(self, reporter: WhyNoPartitionReporter):
        """Initialize the check with a reporter.

        Args:
            reporter (WhyNoPartitionReporter): Reporter for rejection reasons.

        """
        self.reporter = reporter

    @staticmethod
    def _has_symbolic_shape_argument(arg: object) -> bool:
        if isinstance(arg, torch.SymInt):
            return True

        if isinstance(arg, fx.Node):
            return SymbolicShapeSupportCheck._has_symbolic_shape_argument(
                arg.meta.get("val")
            )

        if isinstance(arg, (list, tuple)):
            return any(
                SymbolicShapeSupportCheck._has_symbolic_shape_argument(item)
                for item in arg
            )

        return False

    @staticmethod
    def _get_mean_reduction_dims(node: fx.Node, input_rank: int) -> tuple[int, ...]:
        if node.target == exir_ops.edge.aten.mean.default:
            return tuple(range(input_rank))

        dims = node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else None)
        if dims is None:
            return tuple(range(input_rank))
        if isinstance(dims, int):
            return (dims % input_rank,)
        return tuple(dim % input_rank for dim in typing.cast(Sequence[int], dims))

    @staticmethod
    def _symbolic_spatial_op_requires_shape_extension(node: fx.Node) -> bool:
        """Return whether a symbolic spatial operation needs TOSA shape
        operations.

        Args:
            node (fx.Node): Spatial operation node to inspect.

        Returns:
            bool: Whether the operation cannot use static input adjustment and
                padding.

        """
        try:
            get_context_shape_env()
        except RuntimeError:
            return True

        if node.target == exir_ops.edge.aten.convolution.default:
            return (
                bool(node.args[6])
                or bool(get_slices_convolution(node))
                or has_dynamic_conv_padding(node)
            )
        if node.target in (
            exir_ops.edge.aten.avg_pool2d.default,
            exir_ops.edge.aten.max_pool2d.default,
            exir_ops.edge.aten.max_pool2d_with_indices.default,
        ):
            if node.target == exir_ops.edge.aten.max_pool2d_with_indices.default:
                users = list(node.users)
                if (
                    len(users) != 1
                    or users[0].target != operator.getitem
                    or users[0].args[1] != 0
                ):
                    return True
            return bool(get_slices_pooling(node)) or has_dynamic_pooling_padding(node)
        return True

    def _has_unsupported_symbolic_tensor_shape(self, node: fx.Node) -> bool:
        if node.target not in (
            *self._SYMBOLIC_SPATIAL_DIM_TARGETS,
            *self._SYMBOLIC_MEAN_TARGETS,
            *self._SYMBOLIC_VIEW_SHAPE_TARGETS,
            *self._SYMBOLIC_PAD_TARGETS,
            *self._SYMBOLIC_SLICE_TARGETS,
        ):
            return False
        if not node.all_input_nodes:
            return False

        input_node = node.all_input_nodes[0]
        input_fake_tensor = get_first_fake_tensor(input_node)
        if not any(isinstance(s, torch.SymInt) for s in input_fake_tensor.shape):
            return False

        if node.target in self._SYMBOLIC_SPATIAL_DIM_TARGETS:
            if any(isinstance(s, torch.SymInt) for s in input_fake_tensor.shape[2:]):
                if not self._symbolic_spatial_op_requires_shape_extension(node):
                    return False
                self.reporter.report_reject(node, "Symbolic spatial dims unsupported")
                return True

        if node.target in self._SYMBOLIC_MEAN_TARGETS:
            if any(
                isinstance(input_fake_tensor.shape[dim], torch.SymInt)
                for dim in self._get_mean_reduction_dims(
                    node, len(input_fake_tensor.shape)
                )
            ):
                self.reporter.report_reject(node, "Symbolic mean dims unsupported")
                return True

        if node.target in self._SYMBOLIC_VIEW_SHAPE_TARGETS:
            self.reporter.report_reject(node, "Symbolic view dims unsupported")
            return True

        if node.target in self._SYMBOLIC_PAD_TARGETS:
            self.reporter.report_reject(node, "Symbolic pad dims unsupported")
            return True

        if node.target in self._SYMBOLIC_SLICE_TARGETS:
            self.reporter.report_reject(node, "Symbolic slices unsupported")
            return True

        return False

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        """Return False for symbolic shape uses needing shape extension.

        Without TOSA shape extension, symbolic input/output tensor dimensions
        are generally allowed because they are tensor metadata. Symbolic shape
        arguments and known shape-materialization edge cases are rejected.

        Args:
            submodules (typing.Mapping[str, torch.nn.Module]): Exported modules.
            node (fx.Node): FX node to check.

        Returns:
            bool: False if rejected by constraints; otherwise, True.

        """
        del submodules
        if node.op in ("placeholder", "output"):
            return True
        if node.op == "call_function" and node.target in (*Q_OPS, *DQ_OPS):
            return True

        if self._has_symbolic_shape_argument(node.args):
            self.reporter.report_reject(
                node,
                "Node has symbolic shape arguments, has the TOSA spec shape extension support?",
            )
            return False

        if self._has_unsupported_symbolic_tensor_shape(node):
            return False

        return True


class CheckResolvedTensorShapes(OperatorSupportBase):
    """Reject nodes with unresolved tensor input or output shapes."""

    def __init__(self, reporter: WhyNoPartitionReporter):
        """Initialize the check with a reporter.

        Args:
            reporter (WhyNoPartitionReporter): Reporter for rejection reasons.

        """
        self.reporter = reporter

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        """Return False when the node depends on unresolved tensor shapes."""
        del submodules
        if node.op in ("placeholder", "output"):
            return True
        if node.op == "call_function" and node.target in (*Q_OPS, *DQ_OPS):
            return True

        if _has_symbolic_shape(node) or any(
            _has_symbolic_shape(input_node) for input_node in node.all_input_nodes
        ):
            self.reporter.report_reject(
                node,
                "Node has unresolved tensor shapes, which are not supported by this target.",
            )
            return False

        return True


class CheckMixedProfileOperatorSupport(OperatorSupportBase):
    """Constrain mixed-profile operators to their quantization-side list."""

    def __init__(self, reporter: WhyNoPartitionReporter) -> None:
        self.reporter = reporter

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        if node.op != "call_function":
            return True

        is_int_node = is_quantized(node) or node.target in (*Q_OPS, *DQ_OPS)
        support_list = (
            TOSA_PRO_MIXED_INT_SupportList if is_int_node else TOSA_PRO_FP_SupportList
        )
        if node.target in support_list:
            return True

        combined_support_list = TOSA_PRO_MIXED_INT_SupportList | TOSA_PRO_FP_SupportList
        if node.target not in combined_support_list:
            return True

        profile = "INT" if is_int_node else "FP"
        self.reporter.report_reject(
            node,
            f"Operator {node.target} is not supported on the {profile} side of the mixed INT+FP profile.",
        )
        return False


class CheckArmQuantized(OperatorSupportBase):
    """Check if the node was marked as quantized in the Arm backend.

    This is used to ensure that nodes that were quantized in the Arm backend are
    only partitioned if they are supported by the TOSA backend.

    """

    def __init__(self, reporter: WhyNoPartitionReporter):
        self.reporter = reporter

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:

        if node.target in (*DQ_OPS, *Q_OPS):
            return True

        if not is_quantized(node):
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

        if any(
            isinstance(input_node.meta["val"], torch.SymInt)
            for input_node in node.all_input_nodes
        ):
            self.reporter.report_reject(
                node, "Symbolic scalar inputs cannot be delegated."
            )
            return False

        input_quantized = input_quantized or all(
            (input_node.target in DQ_OPS)
            or _is_integer_dtype(get_first_fake_tensor(input_node).dtype)
            for input_node in node.all_input_nodes
        )

        if not input_quantized:
            self.reporter.report_reject(node, "One or more inputs were not quantized.")
            return False

        all_q_users = all(
            output_node.target in (*Q_OPS, torch.ops.aten.sym_size.int)
            for output_node in node.users
        )
        output_dtype = get_first_fake_tensor(node).dtype
        output_quantized = (
            output_quantized or all_q_users or _is_integer_dtype(output_dtype)
        )

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
        self,
        exported_program: ExportedProgram,
        reporter: WhyNoPartitionReporter,
        tosa_spec: TosaSpecification,
    ):
        """Initialize the check with program context and reporter."""
        self.input_names = [
            spec.arg.name
            for spec in exported_program.graph_signature.input_specs
            if spec.kind == InputKind.USER_INPUT
        ]
        self.reporter = reporter
        self.tosa_spec = tosa_spec
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

    def has_rejected_int64_output(
        self, node: torch.fx.Node, tensor_list: Sequence[typing.Any]
    ) -> bool:
        if is_safe_int32_to_int64_gather_boundary(node):
            return False
        if node.target in _ARGMAX_OPS:
            return not self._is_tosa_argmax_supported(node)

        return any(
            tensor.dtype == torch.int64
            for tensor in tensor_list
            if isinstance(tensor, FakeTensor)
        )

    def _is_argmax_int32_cast(
        self,
        node: torch.fx.Node,
        input_node: torch.fx.Node,
    ) -> bool:
        if node.target not in _TO_DIM_ORDER_COPY_OPS:
            return False

        if node.kwargs.get("dtype") != torch.int32:
            return False

        if input_node.target not in _ARGMAX_OPS:
            return False

        if not _argmax_all_users_cast_to_int32(input_node):
            return False

        return self._is_tosa_argmax_supported(input_node)

    def _is_tosa_argmax_dtype_supported(
        self, node: torch.fx.Node, input_dtype: torch.dtype
    ) -> bool:
        if input_dtype == torch.int8:
            if not self.tosa_spec.support_integer():
                self.reporter.report_reject(
                    node, "TOSA ARGMAX requires PRO-INT for int8 input."
                )
                return False
        elif input_dtype == torch.int16:
            if not (
                self.tosa_spec.support_integer()
                and self.tosa_spec.support_extension("int16")
            ):
                self.reporter.report_reject(
                    node, "TOSA ARGMAX requires EXT-INT16 for int16 input."
                )
                return False
        elif input_dtype in (torch.float16, torch.float32):
            if not self.tosa_spec.support_float():
                self.reporter.report_reject(
                    node, f"TOSA ARGMAX requires PRO-FP for {input_dtype} input."
                )
                return False
        elif input_dtype == torch.bfloat16:
            if not (
                self.tosa_spec.support_float()
                and self.tosa_spec.support_extension("bf16")
            ):
                self.reporter.report_reject(
                    node, "TOSA ARGMAX requires EXT-BF16 for bfloat16 input."
                )
                return False
        else:
            self.reporter.report_reject(
                node, f"TOSA ARGMAX does not support {input_dtype} input."
            )
            return False
        return True

    def _is_tosa_argmax_supported(self, node: torch.fx.Node) -> bool:
        dim = node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else None)
        if dim is None:
            self.reporter.report_reject(
                node, "TOSA ARGMAX requires an explicit reduction dimension."
            )
            return False
        if not isinstance(dim, int):
            self.reporter.report_reject(
                node, "TOSA ARGMAX requires a statically known reduction dimension."
            )
            return False

        input_node = typing.cast(torch.fx.Node, node.args[0])
        input_tensor = get_first_fake_tensor(input_node)
        if not self._is_tosa_argmax_dtype_supported(node, input_tensor.dtype):
            return False

        input_rank = len(input_tensor.shape)
        if input_rank == 0:
            self.reporter.report_reject(
                node, "TOSA ARGMAX requires an input with rank at least 1."
            )
            return False

        axis = dim + input_rank if dim < 0 else dim
        if axis < 0 or axis >= input_rank:
            self.reporter.report_reject(
                node,
                f"TOSA ARGMAX axis must be in [0, {input_rank - 1}] but got {dim}.",
            )
            return False

        keepdim = node.kwargs.get(
            "keepdim", node.args[2] if len(node.args) > 2 else False
        )
        if keepdim:
            self.reporter.report_reject(
                node, "TOSA ARGMAX does not support keepdim=True."
            )
            return False

        return True

    def _check_int64_input_nodes(self, node: torch.fx.Node) -> bool:
        """Check if all int64 input nodes are constant and will be
        partitioned.
        """
        for input_node in (
            input_node
            for input_node in node.all_input_nodes
            if input_node.op != "get_attr"
        ):
            if isinstance(input_node.meta["val"], torch.SymInt):
                continue
            tensor_in = get_first_fake_tensor(input_node)
            if tensor_in.dtype != torch.int64:
                continue

            if (
                node.target == exir_ops.edge.aten.gather.default
                and is_safe_int32_to_int64_gather_boundary(input_node)
            ):
                continue

            # aten.argmax is nominally int64, but TOSA ARGMAX produces int32.
            # Allow the explicit argmax -> int32 narrowing pattern so both nodes
            # can be placed in the same delegate.
            if self._is_argmax_int32_cast(node, input_node):
                continue

            # Constant placeholder
            if (
                input_node.op != "call_function"
                and input_node.name not in self.input_names
            ):
                continue
            # Constant operator
            if input_node.op == "call_function":
                if input_node.target in ComputeConstantOpsAOTPass.targeted_ops:
                    # This is not perfect since the input_node can still be rejected by other checks but
                    # this should cover the majority of cases.
                    if self.is_node_supported({}, input_node):
                        continue
            self.reporter.report_reject(
                node, f"Non-constant int64 input {input_node.name}"
            )
            return False

        return True

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        """Return True when int64 use is absent or safe per exceptions."""
        if is_submodule_node(node):
            return True
        vals = node.meta["val"]
        tensor_list = vals if isinstance(vals, (list, tuple)) else [vals]

        any_int64 = self.has_rejected_int64_output(node, tensor_list)
        # Don't partition nodes with int64 output...
        if any_int64:
            # ... Except for constant ops that are directly cast to something non-int64.
            # This could be an explicit cast, or something like a less than that outputs a different dtype than the input.
            users_output_non_int64 = all(
                get_first_fake_tensor(output_node).dtype != torch.int64
                for output_node in node.users
            )
            if (
                node.target in ComputeConstantOpsAOTPass.targeted_ops
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

        return self._check_int64_input_nodes(node)


class CheckDtypeInputsAndOutputs(OperatorSupportBase):
    """Reject nodes with at least one disallowed dtype on inputs or outputs."""

    def __init__(
        self,
        exported_program: ExportedProgram,
        reporter: WhyNoPartitionReporter,
        disallowed_dtypes: list[torch.dtype],
        tosa_spec: TosaSpecification,
    ):
        """Initialize the check with program context and reporter."""
        self.reporter = reporter
        self.disallowed_dtypes = disallowed_dtypes
        self.tosa_spec = tosa_spec
        super().__init__()

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        """Return True if no disallowed dtypes are present on inputs or
        outputs.
        """
        if is_submodule_node(node):
            return True
        for input_node in (
            input_node
            for input_node in node.all_input_nodes
            if input_node.op != "get_attr"
        ):
            if isinstance(input_node.meta["val"], torch.SymInt):
                continue

            tensor = get_first_fake_tensor(input_node)
            if tensor.dtype in self.disallowed_dtypes:
                self.reporter.report_reject(
                    node,
                    f"Had {tensor.dtype} input {input_node.name} that is not supported by {self.tosa_spec}.",
                )
                return False

        meta_val = node.meta["val"]
        if isinstance(
            meta_val, (Sequence, torch.fx.immutable_collections.immutable_list)
        ):
            outputs = meta_val
        else:
            outputs = (meta_val,)

        for output in outputs:
            if (
                isinstance(output, FakeTensor)
                and output.dtype in self.disallowed_dtypes
            ):
                self.reporter.report_reject(
                    node,
                    f"Had {output.dtype} output that is not supported by {self.tosa_spec}.",
                )
                return False
        return True


class CheckMixedFloatingInputs(OperatorSupportBase):
    """Reject nodes with mixed floating-point input dtypes."""

    def __init__(self, reporter: WhyNoPartitionReporter):
        self.reporter = reporter
        super().__init__()

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        """Return True if floating inputs are either absent or of a single
        dtype.
        """
        if is_submodule_node(node):
            return True

        if node.target in (
            torch.ops.higher_order.while_loop,
            torch.ops.higher_order.cond,
        ):
            return True

        if node.target in TOSA_EXT_MXFP_SupportList:
            return True

        floating_dtypes = set()
        for input_node in (
            input_node
            for input_node in node.all_input_nodes
            if input_node.op != "get_attr"
        ):
            if isinstance(input_node.meta["val"], torch.SymInt):
                continue
            dtype = get_first_fake_tensor(input_node).dtype
            if dtype.is_floating_point:
                floating_dtypes.add(dtype)

        if len(floating_dtypes) > 1:
            self.reporter.report_reject(
                node,
                f"Mixed floating-point input dtypes {floating_dtypes} are not supported by TOSA."
                " Did you call model.to(dtype=...) or cast properly?",
            )
            return False

        return True


class CheckFPComparisonInputs(OperatorSupportBase):
    """Reject unsupported comparison inputs under the FP profile."""

    comparison_ops = {
        exir_ops.edge.aten.eq.Tensor,
        exir_ops.edge.aten.eq.Scalar,
        exir_ops.edge.aten.ne.Tensor,
        exir_ops.edge.aten.ne.Scalar,
        exir_ops.edge.aten.ge.Tensor,
        exir_ops.edge.aten.ge.Scalar,
        exir_ops.edge.aten.gt.Tensor,
        exir_ops.edge.aten.gt.Scalar,
        exir_ops.edge.aten.le.Tensor,
        exir_ops.edge.aten.le.Scalar,
        exir_ops.edge.aten.lt.Tensor,
        exir_ops.edge.aten.lt.Scalar,
    }
    target_ops = comparison_ops | {
        exir_ops.edge.aten.isinf.default,
        exir_ops.edge.aten.isnan.default,
    }
    supported_dtypes = {torch.float16, torch.float32, torch.bfloat16}
    castable_comparison_dtypes = {torch.int8, torch.int16}

    def __init__(self, reporter: WhyNoPartitionReporter) -> None:
        self.reporter = reporter
        super().__init__()

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        if node.target not in self.target_ops:
            return True

        input_dtypes = [
            get_first_fake_tensor(input_node).dtype
            for input_node in node.all_input_nodes
            if input_node.op != "get_attr"
        ]
        if all(dtype in self.supported_dtypes for dtype in input_dtypes):
            return True

        if node.target in self.comparison_ops and all(
            dtype in self.castable_comparison_dtypes for dtype in input_dtypes
        ):
            return True

        unsupported_dtype = next(
            dtype for dtype in input_dtypes if dtype not in self.supported_dtypes
        )
        self.reporter.report_reject(
            node,
            f"FP profile does not support {unsupported_dtype} comparison inputs.",
        )
        return False


class CheckScalarReductionInputs(OperatorSupportBase):
    """Reject scalar inputs for reductions that require a TOSA axis."""

    reduction_targets = {
        exir_ops.edge.aten.all.dim,
        exir_ops.edge.aten.all.dims,
        exir_ops.edge.aten.amax.default,
        exir_ops.edge.aten.amin.default,
        exir_ops.edge.aten.any.dim,
        exir_ops.edge.aten.any.dims,
        exir_ops.edge.aten.prod.dim_int,
        exir_ops.edge.aten.sum.dim_IntList,
    }

    def __init__(self, reporter: WhyNoPartitionReporter):
        """Initialize the check with a reporter."""
        self.reporter = reporter

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        """Return False for scalar inputs to axis-based reductions."""
        if node.target not in self.reduction_targets:
            return True
        if not node.all_input_nodes:
            return True
        input_shape = get_first_fake_tensor(node.all_input_nodes[0]).shape
        if len(input_shape) != 0:
            return True
        self.reporter.report_reject(
            node,
            f"{node.name} reduces a scalar input, but TOSA reduction ops "
            "require a valid input axis.",
        )
        return False


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
            if isinstance(input_node.meta["val"], torch.SymInt):
                continue
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
