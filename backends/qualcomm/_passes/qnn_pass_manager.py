# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import inspect
from collections import OrderedDict
from typing import Dict

from executorch.backends.qualcomm._passes import (
    AnnotateAvgPool1D,
    AnnotateQuantAttrs,
    AnnotateStack,
    AnnotateUnbind,
    CanonicalizeConv,
    ConvertBmmToMatmul,
    ConvertLinearToConv2d,
    ConvertMhaToSha,
    ConvertSquareToPow,
    DecomposeAcos,
    DecomposeAny,
    DecomposeAtan2,
    DecomposeBinaryAlpha,
    DecomposeCDist,
    DecomposeColIm,
    DecomposeDivMode,
    DecomposeEinsum,
    DecomposeExpM1,
    DecomposeFill,
    DecomposeFloorDivide,
    DecomposeGlu,
    DecomposeLinalgVectorNorm,
    DecomposeLogVariants,
    DecomposeMaxPool3d,
    DecomposeMinMaxDim,
    DecomposePad,
    DecomposeRemainder,
    DecomposeRoll,
    DecomposeSelectScatter,
    DecomposeSilu,
    DecomposeTan,
    DecomposeThreshold,
    DecomposeTriu,
    DecomposeTrunc,
    DecomposeVar,
    DecomposeWrapWithAutocast,
    ExpandBroadcastTensorShape,
    FixedLinearKeepDim,
    FoldQDQ,
    FuseConsecutiveCast,
    FuseConsecutiveTranspose,
    I64toI32,
    InsertCastForFpActQuantizedWeight,
    InsertIOQDQ,
    InsertRequantize,
    InsertReshapeForReduceOps,
    LayoutTransform,
    LiftConstantScalarOperands,
    RecomposePadMaxPool2d,
    RecomposePixelUnshuffle,
    RecomposeRmsNorm,
    ReduceDynamicRange,
    Remove0DTensor,
    RemoveRedundancy,
    ReplaceArangeArgs,
    ReplaceInfValues,
    ResolveDebugHandle,
    TagQuantIO,
)
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.utils.constants import (
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
)
from executorch.backends.transforms.decompose_sdpa import (
    DecomposeScaledDotProductAttention,
)
from executorch.exir import ExportedProgram
from executorch.exir.pass_manager import PassManager
from executorch.exir.program._program import (
    _get_updated_graph_signature,
    lift_constant_tensor_pass,
)
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_manager import this_before_that_pass_constraint


class QnnPassManager(PassManager):

    def _transform(self, graph_module: GraphModule):
        return self(graph_module).graph_module

    def _reset(self):
        """Reset to avoid accumulation when the same pass manager instance is reused."""
        self.passes = []
        self.constraints = []

    @classmethod
    def get_default_pass_activations(cls):
        """Return default pass classes and their activation status.

        This is a classmethod that can be invoked without instantiating the
        pass manager, e.g. ``QnnHtpPassManager.get_default_pass_activations()``.

        Returns:
            list[tuple[type[ExportPass], bool]]: Each tuple is
                ``(PassClass, is_active)``. Active passes run by default in
                :meth:`get_capture_program_passes`; inactive ones (e.g.
                ``ConvertBmmToMatmul``, ``TagQuantIO``) are registered but
                skipped unless explicitly enabled via a *passes_job* override.

        Note:
            Subclasses should override this method to add backend-specific
            passes via ``super().get_default_pass_activations()`` + extend.
        """
        return [
            (AnnotateAvgPool1D, True),
            (AnnotateQuantAttrs, True),
            (AnnotateStack, True),
            (AnnotateUnbind, True),
            (ConvertBmmToMatmul, False),
            (DecomposeAcos, True),
            (DecomposeAny, True),
            (DecomposeAtan2, True),
            (DecomposeColIm, True),
            (DecomposeCDist, True),
            (DecomposeDivMode, True),
            (DecomposeFill, True),
            (DecomposeLogVariants, True),
            (DecomposeMaxPool3d, True),
            (DecomposeMinMaxDim, True),
            (DecomposePad, True),
            (DecomposeRemainder, True),
            (DecomposeTan, True),
            (DecomposeTrunc, True),
            (DecomposeVar, True),
            (ExpandBroadcastTensorShape, True),
            (FixedLinearKeepDim, True),
            (FoldQDQ, True),
            (I64toI32, True),
            (InsertCastForFpActQuantizedWeight, True),
            (LayoutTransform, True),
            (RecomposePadMaxPool2d, True),
            (RecomposePixelUnshuffle, True),
            (RecomposeRmsNorm, True),
            (Remove0DTensor, True),
            (RemoveRedundancy, True),
            (TagQuantIO, False),
            (ResolveDebugHandle, True),
        ]

    @classmethod
    def get_annotation_passes(cls):
        """Return annotation pipeline pass classes. Override in subclasses to add backend-specific passes."""
        return [
            RemoveRedundancy,
            ReduceDynamicRange,
            RecomposePixelUnshuffle,
            RecomposeRmsNorm,
            ReplaceArangeArgs,
            DecomposeAcos,
            DecomposeAtan2,
            DecomposeBinaryAlpha,
            DecomposeCDist,
            DecomposeDivMode,
            DecomposeMaxPool3d,
            DecomposePad,
            DecomposeScaledDotProductAttention,
            DecomposeRoll,
            DecomposeSilu,
            DecomposeTan,
            DecomposeThreshold,
            DecomposeTriu,
            DecomposeTrunc,
            DecomposeVar,
            DecomposeWrapWithAutocast,
            DecomposeEinsum,
            DecomposeExpM1,
            DecomposeFill,
            DecomposeGlu,
            DecomposeRemainder,
            DecomposeSelectScatter,
            DecomposeLinalgVectorNorm,
            DecomposeLogVariants,
            ReplaceInfValues,
            LiftConstantScalarOperands,
            InsertReshapeForReduceOps,
        ]

    @classmethod
    def get_export_passes(
        cls,
        convert_linear_to_conv2d: bool = False,
    ):
        """Return export pipeline pass classes. Override in subclasses to add backend-specific passes."""
        passes = [
            DecomposeBinaryAlpha,
            DecomposeCDist,
            DecomposePad,
            DecomposeScaledDotProductAttention,
            DecomposeRoll,
            DecomposeSelectScatter,
            DecomposeThreshold,
            DecomposeTriu,
            DecomposeLinalgVectorNorm,
            DecomposeExpM1,
            DecomposeFill,
            DecomposeVar,
            # DecomposeFloorDivide does not apply to the annotation pipeline,
            # since the CPU QDQ model would reduce accuracy.
            # We keep div and floor operations in floating-point to maintain precision.
            # This pass is needed before to_edge pipeline to avoid mixed type for div operator with RemoveMixedTypeOperators pass.
            DecomposeFloorDivide,
            DecomposeWrapWithAutocast,
            # this pass will rewrite state_dict, it needs to be accomplished before
            # to_edge_transform_and_lower
            CanonicalizeConv,
            ConvertLinearToConv2d,
            ConvertSquareToPow,
            LiftConstantScalarOperands,
            InsertReshapeForReduceOps,
        ]
        if not convert_linear_to_conv2d:
            passes.remove(ConvertLinearToConv2d)
        return passes

    @classmethod
    def get_preprocess_passes(
        cls,
        use_mha2sha: bool = False,
    ):
        """Return preprocess pipeline pass classes. Override in subclasses to add backend-specific passes."""
        passes = [
            FoldQDQ,
            ConvertMhaToSha,
            InsertRequantize,
            InsertIOQDQ,
            LayoutTransform,
            FuseConsecutiveCast,
            FuseConsecutiveTranspose,
        ]
        if not use_mha2sha:
            passes.remove(ConvertMhaToSha)
        return passes

    @classmethod
    def get_passes_dependency_for_capture_program(cls):
        """Return ordering constraints between capture-program passes.

        This is a classmethod that can be invoked without instantiating the
        pass manager, e.g. ``QnnHtpPassManager.get_passes_dependency_for_capture_program()``.

        Each entry maps a pass class to the list of passes that must run
        **before** it. These constraints are resolved by
        :meth:`get_to_edge_transform_passes` via
        ``PassManager.solve_constraints()``.

        Returns:
            dict[type[ExportPass], list[type[ExportPass]]]: Mapping from a
                pass to its prerequisite passes.

        Note:
            Subclasses should override this method to add backend-specific
            dependencies via
            ``super().get_passes_dependency_for_capture_program()`` + update.
        """
        return {
            AnnotateAvgPool1D: [RemoveRedundancy],
            AnnotateQuantAttrs: [
                ConvertBmmToMatmul,
                RecomposePixelUnshuffle,
                RemoveRedundancy,
            ],
            AnnotateStack: [RemoveRedundancy],
            AnnotateUnbind: [RemoveRedundancy],
            ConvertBmmToMatmul: [RecomposePixelUnshuffle],
            DecomposeAcos: [RemoveRedundancy],
            DecomposeAny: [RemoveRedundancy],
            DecomposeAtan2: [RemoveRedundancy],
            DecomposeColIm: [FoldQDQ],
            DecomposeCDist: [RemoveRedundancy],
            DecomposeDivMode: [RemoveRedundancy],
            DecomposeFill: [RemoveRedundancy],
            DecomposeLinalgVectorNorm: [RemoveRedundancy],
            DecomposeLogVariants: [RemoveRedundancy],
            DecomposeMaxPool3d: [RemoveRedundancy],
            DecomposePad: [RemoveRedundancy],
            DecomposeRemainder: [RemoveRedundancy],
            DecomposeTan: [RemoveRedundancy],
            DecomposeTrunc: [RemoveRedundancy],
            DecomposeVar: [RemoveRedundancy],
            ExpandBroadcastTensorShape: [FoldQDQ],
            FixedLinearKeepDim: [FoldQDQ],
            FoldQDQ: [AnnotateQuantAttrs, AnnotateStack, AnnotateUnbind],
            I64toI32: [RemoveRedundancy],
            InsertCastForFpActQuantizedWeight: [FoldQDQ, LayoutTransform],
            LayoutTransform: [
                AnnotateQuantAttrs,
                ExpandBroadcastTensorShape,
                FixedLinearKeepDim,
            ],
            RecomposePadMaxPool2d: [DecomposeMaxPool3d, FoldQDQ],
            RecomposePixelUnshuffle: [RemoveRedundancy],
            RecomposeRmsNorm: [RemoveRedundancy],
            TagQuantIO: [LayoutTransform],
            ResolveDebugHandle: [
                TagQuantIO
            ],  # IMPORTANT: Please always ensure ResolveDebugHandle is the last executed pass.
        }

    @classmethod
    def get_capture_program_passes(cls):
        """Build an ordered mapping of passes with activation flags and init defaults.

        This is a classmethod that can be invoked without instantiating the
        pass manager, e.g. ``QnnHtpPassManager.get_capture_program_passes()``.

        Introspects each pass's ``__init__`` signature to extract default
        keyword arguments, which are later used by
        :meth:`get_to_edge_transform_passes` to instantiate active passes.

        Returns:
            OrderedDict[type[ExportPass], dict]: Keys are pass classes; values
                contain ``QCOM_PASS_ACTIVATE_KEY`` (bool) and
                ``QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY`` (dict of param defaults).
        """
        passes = OrderedDict()
        for p, act in cls.get_default_pass_activations():
            init_signature = inspect.signature(p.__init__)

            args_kwargs_defaults = {
                k: v.default if v.default is not inspect.Parameter.empty else None
                for k, v in init_signature.parameters.items()
                if k != "self"
            }

            passes[p] = {
                QCOM_PASS_ACTIVATE_KEY: act,
                QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY: args_kwargs_defaults,
            }

        return passes

    # TODO: Move these passes into qnn_partitioner and qnn_preprocess to
    # prevent users from needing to call custom APIs like capture_program
    def get_to_edge_transform_passes(
        self,
        exported_program: ExportedProgram,
        passes_job: OrderedDict = None,
        dep_table: Dict = None,
    ):
        # TODO: remove this workaround when target could be correctly detected
        from executorch.backends.qualcomm.builders import node_visitor
        from executorch.exir.dialects._ops import ops as exir_ops

        node_visitor.q_ops.add(exir_ops.edge.torchao.quantize_affine.default)
        node_visitor.dq_ops.add(exir_ops.edge.torchao.dequantize_affine.default)

        self._reset()
        passes_job = (
            passes_job if passes_job is not None else self.get_capture_program_passes()
        )
        dep_table = (
            dep_table
            if dep_table is not None
            else self.get_passes_dependency_for_capture_program()
        )
        for that, these in dep_table.items():
            for this in these:
                self.add_constraint(this_before_that_pass_constraint(this, that))
        for p in passes_job:
            self.add_pass(p)
        self.solve_constraints()

        sorted_passes = self.passes
        self._reset()
        for p in sorted_passes:
            if not passes_job[p][QCOM_PASS_ACTIVATE_KEY]:
                continue

            kwargs = passes_job[p][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY]
            if "edge_program" in kwargs:
                kwargs["edge_program"] = exported_program
            self.add_pass(p(**kwargs))
        assert isinstance(
            self.passes[-1], ResolveDebugHandle
        ), "Please ensure ResolveDebugHandle is the last executed edge pass."
        return self.passes

    def _instantiate_passes(self, pass_classes, **available_kwargs):
        """Instantiate pass classes, injecting only kwargs each __init__ accepts."""
        self._reset()
        for p_cls in pass_classes:
            init_params = inspect.signature(p_cls.__init__).parameters
            kwargs = {k: v for k, v in available_kwargs.items() if k in init_params}
            self.add_pass(p_cls(**kwargs))

    def transform_for_annotation_pipeline(
        self,
        graph_module: GraphModule,
    ):
        self._instantiate_passes(
            self.get_annotation_passes(),
            quantization_capture=True,
        )
        return self._transform(graph_module)

    def transform_for_export_pipeline(
        self,
        exported_program: ExportedProgram,
        convert_linear_to_conv2d: bool = False,
    ):
        self._instantiate_passes(
            self.get_export_passes(convert_linear_to_conv2d),
            edge_program=exported_program,
            quantization_capture=True,
        )
        self._transform(exported_program.graph_module)
        ep = lift_constant_tensor_pass(exported_program)
        return ep

    def transform_for_to_edge_pipeline(
        self,
        exported_program: ExportedProgram,
        passes_job: OrderedDict = None,
        dep_table: Dict = None,
    ):
        transform_passes = self.get_to_edge_transform_passes(
            exported_program, passes_job=passes_job, dep_table=dep_table
        )
        for p in transform_passes:
            p(exported_program.graph_module)
        exported_program._graph_signature = _get_updated_graph_signature(
            exported_program.graph_signature,
            exported_program.graph_module,
        )
        exported_program._validate()

        return exported_program

    def transform_for_preprocess_pipeline(
        self, exported_program: ExportedProgram, use_mha2sha=False
    ):
        self._instantiate_passes(
            self.get_preprocess_passes(use_mha2sha),
            edge_program=exported_program,
            force_fold=True,
            insert_permute=True,
        )
        self._transform(exported_program.graph_module)
        # Update inputs_to_buffers and buffers_to_mutate in graph signature for mutable buffer
        # Since I/O will be inserted Q/DQ, it results in failed to mapping output node names and buffer
        exported_program._graph_signature = _get_updated_graph_signature(
            exported_program.graph_signature,
            exported_program.graph_module,
        )
        return exported_program.graph_module


@functools.lru_cache(maxsize=1)
def _get_backend_pass_manager_map():
    """Lazy import to avoid circular dependencies with backend subclasses."""
    from executorch.backends.qualcomm._passes.backends.gpu.qnn_gpu_pass_manager import (
        QnnGpuPassManager,
    )
    from executorch.backends.qualcomm._passes.backends.htp.qnn_htp_pass_manager import (
        QnnHtpPassManager,
    )
    from executorch.backends.qualcomm._passes.backends.lpai.qnn_lpai_pass_manager import (
        QnnLpaiPassManager,
    )

    return {
        QnnExecuTorchBackendType.kGpuBackend: QnnGpuPassManager,
        QnnExecuTorchBackendType.kHtpBackend: QnnHtpPassManager,
        QnnExecuTorchBackendType.kLpaiBackend: QnnLpaiPassManager,
    }


def get_qnn_pass_manager_cls(
    backend_type: QnnExecuTorchBackendType = QnnExecuTorchBackendType.kHtpBackend,
) -> type[QnnPassManager]:
    """Return the QnnPassManager subclass for the given backend type.

    Use this to call classmethods (e.g. ``get_capture_program_passes``,
    ``get_passes_dependency_for_capture_program``) without instantiation.

    Args:
        backend_type: The QNN backend to target. Defaults to kHtpBackend.

    Returns:
        The QnnPassManager subclass (not an instance) for the requested
        backend. Unrecognized backend types fall back to the base
        QnnPassManager.
    """
    return _get_backend_pass_manager_map().get(backend_type, QnnPassManager)
