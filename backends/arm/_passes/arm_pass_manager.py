# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from executorch.backends.arm._passes import (
    AccumulateIndexPutPass,
    BroadcastArgsPass,
    CanonicalizeGatherPass,
    CastInt64BuffersToInt32Pass,
    CastToInt32Pass,
    ComputeConstantOpsAOTPass,
    ConstantFoldingPass,
    ControlFlowConstInlinePass,
    Conv1dUnsqueezePass,
    ConvertEluFamilyToEluPass,
    ConvertELUParamsPass,
    ConvertExpandCopyToRepeatPass,
    ConvertFullLikeToFullPass,
    ConvertInt64ConstOpsToInt32Pass,
    ConvertInt64OutputOpsToInt32Pass,
    ConvertMinMaxPass,
    ConvertMmToBmmPass,
    ConvertPermuteSingletonToViewPass,
    ConvertSplitToSlicePass,
    ConvertSqueezesToViewPass,
    ConvertToClampPass,
    DecomposeAcoshPass,
    DecomposeAdaptiveAvgPool2dPass,
    DecomposeAddmmPass,
    DecomposeAddSubAlphaPass,
    DecomposeAnyPass,
    DecomposeAsinAndAcosPass,
    DecomposeAsinhPass,
    DecomposeAsStridedCopyPass,
    DecomposeAtanhPass,
    DecomposeAtanPass,
    DecomposeAvgPool2dPass,
    DecomposeBatchNormNoStatsPass,
    DecomposeCoshPass,
    DecomposeCosineSimilarityPass,
    DecomposeCumsumPass,
    DecomposeDivPass,
    DecomposeDivTensorModePass,
    DecomposeDynamicFullPass,
    DecomposeEinsumPass,
    DecomposeEluPass,
    DecomposeEmbeddingPass,
    DecomposeErfinvPass,
    DecomposeExpm1Pass,
    DecomposeFloorDividePass,
    DecomposeGeluPass,
    DecomposeGluPass,
    DecomposeGroupedConvPass,
    DecomposeGroupNormPass,
    DecomposeGruPass,
    DecomposeIndexCopyPass,
    DecomposeIndexSelectToGatherPass,
    DecomposeIndexTensorToGatherPass,
    DecomposeIntPowPass,
    DecomposeLayerNormPass,
    DecomposeLeakyReLUPass,
    DecomposeLinalgVectorNormPass,
    DecomposeLinearPass,
    DecomposeLog1pPass,
    DecomposeLogitPass,
    DecomposeLstmPass,
    DecomposeMaskedFillPass,
    DecomposeMatmulPass,
    DecomposeMaxPool2dPass,
    DecomposeMeanDimPass,
    DecomposeNotEqualPass,
    DecomposePermuteForU55Pass,
    DecomposeQuantNodesPass,
    DecomposeRemainderPass,
    DecomposeRnnPass,
    DecomposeRoundPass,
    DecomposeScaledDotProductAttentionPass,
    DecomposeSelectPass,
    DecomposeSelectScatterPass,
    DecomposeSignPass,
    DecomposeSinhPass,
    DecomposeSliceScatterPass,
    DecomposeSoftmaxPass,
    DecomposeSqrtPass,
    DecomposeStridedSliceCopyPass,
    DecomposeSumPass,
    DecomposeTanPass,
    DecomposeTOSAUnsupportedClampPass,
    DecomposeTrilPass,
    DecomposeUnfoldToGatherPass,
    DecomposeVarPass,
    DecomposeWhereScalarOtherPass,
    DecorateFp32toInt32CastingPass,
    DeduplicateGetAttrPass,
    EnsureUniqueOutputNodesPass,
    FoldAndAnnotateQParamsPass,
    FuseBatchNorm2dPass,
    FuseConsecutiveConcatShapesPass,
    FuseConsecutiveRescalesPass,
    FuseConstantArgsPass,
    FuseDuplicateUsersPass,
    FuseEqualPlaceholdersPass,
    FuseQuantizedActivationPass,
    FuseViewCopyTransformPass,
    InsertConstShapesPass,
    InsertControlFlowRescalesPass,
    InsertDataLayoutCastsPass,
    InsertInt32CastsAfterInt64PlaceholdersPass,
    InsertRescaleInt32Pass,
    InsertRescalePass,
    InsertTableOpsPass,
    MatchArgDtypePass,
    MatchArgRanksPass,
    NormalizeDelegateIOLayoutPass,
    NormalizeIndexPutBoolIndexTensorPass,
    NormalizeIndexPutNoneIndicesPass,
    NormalizeWhileInitialArgsPass,
    PromoteBoolOperandsPass,
    QuantizeClampArgumentsPass,
    RemoveGetItemPass,
    RemoveGraphAssertsPass,
    RemoveNoopPass,
    RemovePermutesAroundElementwiseTosaOps,
    ReplaceInfAndLimitValuesPass,
    ReplaceScalarWithTensorByProfilePass,
    RewriteAvgPool2dPass,
    RewriteBoolBitwiseToLogicalPass,
    RewriteBoolToFp32CastViaInt8Pass,
    RewriteConvPass,
    RewriteHighRankSingletonPermutePass,
    RewriteIndexPutPass,
    RewriteInplaceArithmeticPass,
    RewriteLeLtToGeGtPass,
    RewriteMatmulPass,
    RewriteMaxPool2dPass,
    RewriteMXFPLinearPass,
    RewritePadPass,
    RewriteSlicePass,
    RewriteUpsamplePass,
    ScalarsToAttributePass,
    SizeAdjustInputPass,
    UnsqueezeBeforeRepeatPass,
    UnsqueezeScalarPlaceholdersPass,
)
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
from executorch.backends.arm.common.pipeline_config import SoftmaxDecompositionConfig
from executorch.backends.arm.tosa.specification import (
    tosa_spec_in_set,
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.backends.transforms.fuse_cascaded_transpose_or_permute_ops import (
    FuseCascadedTransposeOrPermuteOps,
)
from executorch.backends.transforms.postpone_permute_below_squeeze_view import (
    PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView,
)

from executorch.exir import ExportedProgram
from executorch.exir._program_utils import _get_updated_graph_signature
from executorch.exir.pass_base import (
    ExportedProgramPassBase,
    ExportedProgramPassResult,
    ExportPass,
)
from executorch.exir.pass_manager import ExportedProgramPassManager
from torch._export.utils import _get_shape_env_from_gm
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager as GraphModulePassManager

logger = logging.getLogger(__name__)


@dataclass
class PassInsertions:
    """Holds lists of passes to be inserted before and after a target pass."""

    before_passes: list = field(default_factory=list)
    after_passes: list = field(default_factory=list)


_registered_pass_insertions: dict[type, PassInsertions] = {}


def _graph_pass_name(graph_pass: Callable[[GraphModule], PassResult | None]) -> str:
    if isinstance(graph_pass, ExportPass):
        return ArmPass.get_name(graph_pass)
    if hasattr(graph_pass, "__name__"):
        return graph_pass.__name__
    return type(graph_pass).__name__


class _ExportedProgramGraphPassAdapter(ExportedProgramPassBase):
    def __init__(self, graph_pass: Callable[[GraphModule], PassResult | None]) -> None:
        self.graph_pass = graph_pass

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        graph_pass = cast(Any, self.graph_pass)
        pass_exported_program = getattr(graph_pass, "exported_program", None)
        if pass_exported_program is not None:
            # ExportedProgramPassManager works on a shallow copy; Arm graph
            # passes that store an ExportedProgram must update that copy.
            graph_pass.exported_program = exported_program

        try:
            result = self.graph_pass(exported_program.graph_module)
        finally:
            if pass_exported_program is not None:
                graph_pass.exported_program = pass_exported_program

        if result is None:
            raise TypeError(
                f"The result of pass {_graph_pass_name(self.graph_pass)} should be type PassResult."
            )

        if result.modified:
            result.graph_module.recompile()
            exported_program._graph_module = result.graph_module
            exported_program._graph_signature = _get_updated_graph_signature(
                exported_program.graph_signature,
                result.graph_module,
            )
            # Arm graph passes do not change symbolic shape constraints, and
            # metadata-only fake modes may differ after propagation.

        return ExportedProgramPassResult(exported_program, result.modified)


def register_pass_insertions_before(
    target_pass_type: type, passes: list[ExportPass]
) -> None:
    """Register passes to be inserted before a target pass for all pipelines."""
    if target_pass_type not in _registered_pass_insertions:
        _registered_pass_insertions[target_pass_type] = PassInsertions()
    _registered_pass_insertions[target_pass_type].before_passes.extend(passes)


def register_pass_insertions_after(
    target_pass_type: type, passes: list[ExportPass]
) -> None:
    """Register passes to be inserted after a target pass for all pipelines."""
    if target_pass_type not in _registered_pass_insertions:
        _registered_pass_insertions[target_pass_type] = PassInsertions()
    _registered_pass_insertions[target_pass_type].after_passes.extend(passes)


def clear_registered_pass_insertions() -> None:
    """Clear all globally registered pass insertions."""
    _registered_pass_insertions.clear()


class ArmPassManager(ExportedProgramPassManager):
    def __init__(self, compile_spec: ArmCompileSpec) -> None:
        self.compile_spec = compile_spec
        self.tosa_spec = compile_spec.tosa_spec
        self._skip_pass_types: tuple[type, ...] = ()
        self._pass_insertions: dict[type, PassInsertions] = {}
        self._insertions_applied = False
        super().__init__()
        self.configure_skip_passes()

    def configure_skip_passes(self) -> tuple[type, ...]:
        """Configures the pass manager to skip certain passes based on the
        ArmPassPipelineConfig class found in the compile spec.
        """
        skip_set: set[type] = set()

        config = self.compile_spec._get_pass_pipeline_config()
        logger.debug(f"Skip Config: {config}")

        match config.softmax:
            case SoftmaxDecompositionConfig.MASKED:
                pass
            case SoftmaxDecompositionConfig.STABLE:
                skip_set.add(DecomposeMaskedFillPass)

        self._skip_pass_types = tuple(skip_set)
        skip_names = [skipped_pass.__name__ for skipped_pass in self._skip_pass_types]
        logger.debug(f"Passes in skip list: {skip_names}")

        return self._skip_pass_types

    def validate_constraints_mandatory(self):
        """Validates that necessary passes have run before transforming to
        backend.

        Note that this differs from the original validate_constraints function,
        which only checks the order of passes.

        """
        passes_to_run = defaultdict(list)

        for current_pass in self.passes:
            current_pass_name = ArmPass.get_name(current_pass)
            for required_pass_name in ArmPass.get_required_passes(current_pass):
                passes_to_run[required_pass_name].append(current_pass_name)

            passes_to_run.pop(current_pass_name, None)

        if len(passes_to_run) > 0:
            error_msg = "The following constraints for passes are not met:\n"
            for required_pass, requiring_passes in passes_to_run.items():
                for requiring_pass in requiring_passes:
                    error_msg += (
                        f"  - {required_pass} must run after {requiring_pass}\n"
                    )

            raise RuntimeError(error_msg)

    def insert_passes_before(
        self, target_pass_type: type, passes: list[ExportPass]
    ) -> None:
        """Register passes to be inserted before instances of target_pass_type.
        Insertions are deferred and applied via _apply_pass_insertions().

        Args:
            target_pass_type: The pass class to insert before (e.g., InsertTableOpsPass)
            passes: List of pass instances to insert

        """
        self._pass_insertions.setdefault(
            target_pass_type, PassInsertions()
        ).before_passes.extend(passes)

    def insert_passes_after(
        self, target_pass_type: type, passes: list[ExportPass]
    ) -> None:
        """Register passes to be inserted after instances of target_pass_type.
        Insertions are deferred and applied via _apply_pass_insertions().

        Args:
            target_pass_type: The pass class to insert after
            passes: List of pass instances to insert

        """
        self._pass_insertions.setdefault(
            target_pass_type, PassInsertions()
        ).after_passes.extend(passes)

    def _apply_pass_insertions(self) -> None:
        """Apply all registered pass insertions to the collected passes.

        Called ONCE after all add_passes() calls are complete, before execution.

        Raises:
            ValueError: If any registered target pass type is not found in the pipeline.

        """
        if self._insertions_applied or not self._pass_insertions:
            return

        # Fail fast if any target pass type is missing from the pipeline
        existing_pass_types = {type(p) for p in self.passes}
        for target_type in self._pass_insertions:
            if target_type not in existing_pass_types:
                available = [type(p).__name__ for p in self.passes]
                raise ValueError(
                    f"Target pass {target_type.__name__} not found in the pass "
                    f"pipeline. Available passes: {available}"
                )

        # Build new pass list with insertions applied
        new_passes = []
        for pass_obj in self.passes:
            pass_type = type(pass_obj)

            # Insert passes BEFORE this pass
            if pass_type in self._pass_insertions:
                insertions = self._pass_insertions[pass_type]
                for before_pass in insertions.before_passes:
                    # Check if we should skip this inserted pass
                    if type(before_pass) not in self._skip_pass_types:
                        new_passes.append(before_pass)

            # Add the original pass
            new_passes.append(pass_obj)

            # Insert passes AFTER this pass
            if pass_type in self._pass_insertions:
                insertions = self._pass_insertions[pass_type]
                for after_pass in insertions.after_passes:
                    # Check if we should skip this inserted pass
                    if type(after_pass) not in self._skip_pass_types:
                        new_passes.append(after_pass)

        # Replace the passes list
        self.passes = new_passes
        self._insertions_applied = True

    def _configure_pass_insertions(self, exported_program: ExportedProgram) -> None:
        """Hook for subclasses to configure pass insertions. Called at the START
        of pipeline construction, before any passes are added.

        Subclasses can override this to call insert_passes_before/after.

        Args:
            exported_program: The exported program being transformed

        """
        for pass_type, insertions in _registered_pass_insertions.items():
            if insertions.before_passes:
                self.insert_passes_before(pass_type, list(insertions.before_passes))
            if insertions.after_passes:
                self.insert_passes_after(pass_type, list(insertions.after_passes))

    def add_passes(self, passes: Sequence[ExportPass | None]):
        for p in passes:
            if p is not None:
                self.add_pass(p)

    def _tosa_context(self, graph_module: GraphModule) -> TosaLoweringContext:
        shape_env = _get_shape_env_from_gm(graph_module)
        return TosaLoweringContext(self.tosa_spec, shape_env)

    def _transform_graph_module(self, graph_module: GraphModule):
        # TFA and control-flow submodule paths operate on bare GraphModules
        # without a standalone ExportedProgram to keep in sync.
        return GraphModulePassManager(self.passes)(graph_module).graph_module

    def __call__(  # type: ignore[override]
        self,
        module: ExportedProgram | GraphModule,
        override_verifiers: Any | None = None,
    ) -> ExportedProgramPassResult | PassResult:
        if isinstance(module, GraphModule):
            if override_verifiers is not None:
                raise ValueError("override_verifiers is only valid for ExportedProgram")
            return GraphModulePassManager(self.passes)(module)
        return super().__call__(module, override_verifiers)

    def _transform(
        self,
        exported_program: ExportedProgram,
        graph_module: GraphModule,
    ) -> GraphModule:
        if graph_module is exported_program.graph_module:
            passes: list[
                ExportedProgramPassBase | Callable[[GraphModule], PassResult | None]
            ] = [_ExportedProgramGraphPassAdapter(p) for p in self.passes]
            transformed_program = ExportedProgramPassManager(passes)(
                exported_program
            ).exported_program
            exported_program._graph_module = transformed_program.graph_module
            exported_program._graph_signature = transformed_program.graph_signature
            exported_program._range_constraints = transformed_program.range_constraints
            return exported_program.graph_module
        return self._transform_graph_module(graph_module)

    def add_pass(self, pipeline_pass):
        if type(pipeline_pass) in self._skip_pass_types:
            return
        super().add_pass(pipeline_pass)

    def _tosa_pipeline(
        self, exported_program: ExportedProgram, graph_module: GraphModule
    ) -> GraphModule:
        # Allow subclasses to configure pass insertions before building pipeline
        self._configure_pass_insertions(exported_program)

        # Node transformation passes (pre q/dq folding)
        self.add_passes(
            [
                NormalizeDelegateIOLayoutPass(exported_program),
                FuseQuantizedActivationPass(),
                RewriteBoolToFp32CastViaInt8Pass(),
                CanonicalizeGatherPass(),
                ConvertToClampPass(),
                DecomposeTOSAUnsupportedClampPass(),
                DecomposeGroupNormPass(),
                DecomposeLayerNormPass(),
                DecomposeVarPass(),
                DecomposeMeanDimPass(exported_program.graph_module, self.tosa_spec),
                ConvertEluFamilyToEluPass(),
                ConvertELUParamsPass(),
                ControlFlowConstInlinePass(),
                NormalizeWhileInitialArgsPass(use_exir_clone=True),
            ]
        )

        # Fold Q/DQ nodes, insert INT8/INT32 rescales, decompose quantization nodes.
        self.add_passes(
            [
                FoldAndAnnotateQParamsPass(exported_program),
                FuseDuplicateUsersPass(),
                # TODO: DecomposeLinearPass should run after InsertRescaleInt32Pass or
                # before FoldAndAnnotateQParamsPass but is unable to at the moment.
                # Ticket: MLETORCH-1539
                DecomposeLinearPass(),
                InsertRescaleInt32Pass(),
                FuseConsecutiveRescalesPass(),
                InsertControlFlowRescalesPass(),
                DecomposeQuantNodesPass(),
            ]
        )

        # Node transformation passes (post q/dq folding)
        self.add_passes(
            [
                ConvertSplitToSlicePass(),
                QuantizeClampArgumentsPass(),
                RemoveGetItemPass(),
                FuseBatchNorm2dPass(exported_program),
                DecomposeBatchNormNoStatsPass(),
                DecomposeLogitPass(),
                DecomposeMaskedFillPass(),
                DecomposeRoundPass(),
                DecomposeAcoshPass(),
                DecomposeAsinhPass(),
                DecomposeCoshPass(),
                DecomposeAsinAndAcosPass(),
                DecomposeErfinvPass(),
                DecomposeSqrtPass(),
                DecomposeAtanPass(),
                DecomposeAtanhPass(),
                DecomposeTanPass(),
                DecomposeAddmmPass(),
                DecomposeEluPass(),
                DecomposeExpm1Pass(),
                DecomposeIntPowPass(),
                DecomposeLog1pPass(),
                PromoteBoolOperandsPass(),
                DecomposeSinhPass(),
                DecomposeSignPass(),
                DecomposeFloorDividePass(),
                DecomposeGeluPass(),
                DecomposeAddSubAlphaPass(),
                DecomposeGroupedConvPass(),
                DecomposeUnfoldToGatherPass(),
                DecomposeEmbeddingPass(),
                DecomposeIndexSelectToGatherPass(),
                DecomposeStridedSliceCopyPass(),
                DecomposeSliceScatterPass(),
                AccumulateIndexPutPass(),
                DecomposeIndexTensorToGatherPass(),
                DecomposeAdaptiveAvgPool2dPass(),
                DecomposeAvgPool2dPass(),
                Conv1dUnsqueezePass(),
            ]
        )

        # Scalars -> tensors, match tensor dtypes and ranks.
        self.add_passes(
            [
                ReplaceScalarWithTensorByProfilePass(),
                RewriteLeLtToGeGtPass(),
                DecomposeLeakyReLUPass(),  # Emits full_like so before ConvertFullLikeToFullPass
                ConvertFullLikeToFullPass(),
                MatchArgDtypePass(),
                UnsqueezeScalarPlaceholdersPass(exported_program),
                MatchArgRanksPass(exported_program),
            ]
        )

        # Node transformation passes (post scalar-removal)
        self.add_passes(
            [
                DecomposeNotEqualPass(),
                NormalizeIndexPutNoneIndicesPass(),
                NormalizeIndexPutBoolIndexTensorPass(),
                RewriteIndexPutPass(),
                RewriteBoolBitwiseToLogicalPass(),
                DecomposeRemainderPass(),
                DecomposeDivTensorModePass(),
                ConvertMmToBmmPass(),
                DecomposeGluPass(),
                DecomposeDivPass(),
                DecomposeSoftmaxPass(),
                ConvertMinMaxPass(),
                DecomposeAnyPass(),
                DecorateFp32toInt32CastingPass(),
                DecomposeDynamicFullPass(),
                ConvertExpandCopyToRepeatPass(),
                UnsqueezeBeforeRepeatPass(),
                DecomposeCumsumPass(exported_program),
                DecomposeAsStridedCopyPass(),
                DecomposeMaxPool2dPass(),
                SizeAdjustInputPass(),
                RewriteAvgPool2dPass(),
                ComputeConstantOpsAOTPass(exported_program),
                FuseConstantArgsPass(exported_program),
                DecomposeSelectPass(),
                ConvertSqueezesToViewPass(),
                CastToInt32Pass(),
                BroadcastArgsPass(),
                DecomposeSumPass(),
                InsertTableOpsPass(exported_program),
                RemoveNoopPass(),
                InsertDataLayoutCastsPass(),
            ]
        )

        # Aten -> TOSA transformation passes
        self.add_passes(
            [
                RewriteUpsamplePass(),
                RewriteMaxPool2dPass(),
                RewriteConvPass(exported_program),
                RewriteMXFPLinearPass(exported_program),
                RewriteMatmulPass(),
                RewritePadPass(),
                FuseViewCopyTransformPass(),
                RemovePermutesAroundElementwiseTosaOps(),
                PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView(),
                FuseCascadedTransposeOrPermuteOps(),
                ConvertPermuteSingletonToViewPass(),
                RewriteHighRankSingletonPermutePass(),
                DecomposePermuteForU55Pass(),
                RewriteSlicePass(),
                InsertConstShapesPass(),
            ]
        )

        # Postprocessing/cleanup passes
        self.add_passes(
            [
                CastInt64BuffersToInt32Pass(exported_program),
                FuseEqualPlaceholdersPass(exported_program),
                FuseConsecutiveConcatShapesPass(),
                EnsureUniqueOutputNodesPass(),
                RemoveNoopPass(),
                InsertRescalePass(),
            ]
        )

        # Apply all pass insertions once after all passes are collected
        self._apply_pass_insertions()

        self.validate_constraints_mandatory()
        return self._transform(exported_program, graph_module)

    def transform_to_backend_pipeline(
        self, exported_program: ExportedProgram, graph_module: GraphModule
    ):
        """Apply passes before transforming program to backend."""

        if not tosa_spec_in_set(
            self.tosa_spec,
            set(TosaSpecification.all_versions_and_profiles()),
        ):
            raise RuntimeError(
                f"No pass pipeline found for TOSA specification: {self.tosa_spec}"
            )

        with self._tosa_context(graph_module):
            return self._tosa_pipeline(exported_program, graph_module)

    def transform_for_annotation_pipeline(self, graph_module: GraphModule):
        with self._tosa_context(graph_module):
            # Preprocessing passes
            self.add_pass(RemoveGraphAssertsPass(tfa_pass=True))
            self.add_pass(ConstantFoldingPass())

            # Transformation passes (pre scalar -> tensor)
            self.add_passes(
                [
                    DecomposeIndexCopyPass(tfa_pass=True),
                    DecomposeSelectScatterPass(tfa_pass=True),
                    DecomposeSliceScatterPass(tfa_pass=True),
                    DecomposeDynamicFullPass(tfa_pass=True),
                    ConvertInt64ConstOpsToInt32Pass(tfa_pass=True),
                    ConvertInt64OutputOpsToInt32Pass(tfa_pass=True),
                    InsertInt32CastsAfterInt64PlaceholdersPass(tfa_pass=True),
                    DecomposeEmbeddingPass(tfa_pass=True),
                    DecomposeScaledDotProductAttentionPass(tfa_pass=True),
                    DecomposeRoundPass(tfa_pass=True),
                    DecomposeLogitPass(tfa_pass=True),
                    PromoteBoolOperandsPass(tfa_pass=True),
                    DecomposeSignPass(tfa_pass=True),
                    DecomposeTrilPass(tfa_pass=True),
                    DecomposeAddmmPass(tfa_pass=True),
                    DecomposeRemainderPass(tfa_pass=True),
                    DecomposeFloorDividePass(tfa_pass=True),
                    DecomposeDivTensorModePass(tfa_pass=True),
                    DecomposeWhereScalarOtherPass(tfa_pass=True),
                    DecomposeEinsumPass(tfa_pass=True),
                    RewriteInplaceArithmeticPass(tfa_pass=True),
                    DecomposeAddSubAlphaPass(tfa_pass=True),
                    DecomposeLeakyReLUPass(tfa_pass=True),
                    ConvertEluFamilyToEluPass(tfa_pass=True),
                    DecomposeGroupNormPass(tfa_pass=True),
                    DecomposeLayerNormPass(tfa_pass=True),
                    DecomposeVarPass(tfa_pass=True),
                    DecomposeMeanDimPass(graph_module, self.tosa_spec, tfa_pass=True),
                    DecomposeAdaptiveAvgPool2dPass(tfa_pass=True),
                    DecomposeAvgPool2dPass(tfa_pass=True),
                ]
            )

            # Scalars -> tensors
            self.add_passes(
                [
                    ReplaceScalarWithTensorByProfilePass(tfa_pass=True),
                    ScalarsToAttributePass(tfa_pass=True),
                    ControlFlowConstInlinePass(tfa_pass=True),
                ]
            )

            # Transformation passes (post scalar removal)
            self.add_passes(
                [
                    NormalizeWhileInitialArgsPass(use_exir_clone=False, tfa_pass=True),
                    DecomposeGruPass(tfa_pass=True),
                    DecomposeLstmPass(tfa_pass=True),
                    DecomposeRnnPass(tfa_pass=True),
                    DecomposeNotEqualPass(tfa_pass=True),
                    DecomposeCosineSimilarityPass(tfa_pass=True),
                    DecomposeGluPass(tfa_pass=True),
                    DecomposeDivPass(tfa_pass=True),
                    DecomposeLinalgVectorNormPass(tfa_pass=True),
                    DecomposeSqrtPass(tfa_pass=True),
                    DecomposeSoftmaxPass(
                        tfa_pass=True,
                    ),
                    ConvertMinMaxPass(tfa_pass=True),
                    AccumulateIndexPutPass(tfa_pass=True),
                    DecomposeMatmulPass(tfa_pass=True),
                ]
            )

            # Postprocessing passes
            quant_inf_cfg = self.compile_spec._get_pass_pipeline_config().quantize_inf
            self.add_passes(
                [
                    ReplaceInfAndLimitValuesPass(
                        quant_inf_cfg.neg_inf,
                        quant_inf_cfg.pos_inf,
                        tfa_pass=True,
                    ),
                    DecomposeMaskedFillPass(tfa_pass=True),
                    DeduplicateGetAttrPass(tfa_pass=True),
                ]
            )

            return self._transform_graph_module(graph_module)
