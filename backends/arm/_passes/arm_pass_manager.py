# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from collections import defaultdict
from collections.abc import Sequence

import executorch.backends.arm.tosa.dialect  # noqa: unused
from executorch.backends.arm._passes import (
    AnnotateDecomposedMatmulPass,
    AnnotateOutputDimOrderPass,
    BroadcastArgsPass,
    CastInt64BuffersToInt32Pass,
    CastToInt32Pass,
    ComputeConstantOpsAOTPass,
    Conv1dUnsqueezePass,
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
    DecomposeAtanhPass,
    DecomposeAtanPass,
    DecomposeAvgPool2dPass,
    DecomposeBatchNormNoStatsPass,
    DecomposeConv2dWithInt16ActivationPass,
    DecomposeCoshPass,
    DecomposeCosineSimilarityPass,
    DecomposeCumsumPass,
    DecomposeDivPass,
    DecomposeDivTensorModePass,
    DecomposeEluPass,
    DecomposeEmbeddingPass,
    DecomposeExpm1Pass,
    DecomposeFloorDividePass,
    DecomposeGeluPass,
    DecomposeGluPass,
    DecomposeGroupedConvPass,
    DecomposeGroupNormPass,
    DecomposeInt32ClampPass,
    DecomposeIntPowPass,
    DecomposeLayerNormPass,
    DecomposeLeakyReLUPass,
    DecomposeLinalgVectorNormPass,
    DecomposeLinearPass,
    DecomposeLogitPass,
    DecomposeMaskedFillPass,
    DecomposeMaxPool2dPass,
    DecomposeMeanDimPass,
    DecomposeNotEqualPass,
    DecomposeQuantNodesPass,
    DecomposeRemainderPass,
    DecomposeRoundPass,
    DecomposeScaledDotProductAttentionPass,
    DecomposeSelectPass,
    DecomposeSelectScatterPass,
    DecomposeSignPass,
    DecomposeSiluPass,
    DecomposeSinhPass,
    DecomposeSoftmaxPass,
    DecomposeSoftmaxUnstablePass,
    DecomposeSqrtPass,
    DecomposeSumPass,
    DecomposeVarPass,
    DecorateFp32toInt32CastingPass,
    FoldAndAnnotateQParamsPass,
    FuseBatchNorm2dPass,
    FuseConstantArgsPass,
    FuseDuplicateUsersPass,
    FuseEqualPlaceholdersPass,
    FuseQuantizedActivationPass,
    FuseViewCopyTransformPass,
    InsertControlFlowRescalesPass,
    InsertInt32CastsAfterInt64PlaceholdersPass,
    InsertRescaleInt32Pass,
    InsertRescalePass,
    InsertTableOpsPass,
    MatchArgDtypePass,
    MatchArgRanksPass,
    PromoteBoolOperandsPass,
    QuantizeClampArgumentsPass,
    RemoveGetItemPass,
    RemoveGraphAssertsPass,
    RemoveNoopPass,
    ReplaceInfValuesPass,
    ReplaceScalarWithTensorByProfilePass,
    RewriteConv2dPass,
    RewriteMatmulPass,
    RewriteUpsamplePass,
    ScalarsToAttributePass,
    SizeAdjustInputPass,
    ToTosaMemoryFormatPass,
    UnsqueezeBeforeRepeatPass,
    UnsqueezeScalarPlaceholdersPass,
)

from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm.tosa.specification import (
    tosa_spec_in_set,
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir import ExportedProgram
from executorch.exir.pass_base import ExportPass
from executorch.exir.pass_manager import PassManager
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassResult
from torch.nn.modules import Module


class ArmPassManager(PassManager):
    def __init__(self, tosa_spec: TosaSpecification) -> None:
        self.tosa_spec = tosa_spec
        super().__init__()

    def validate_constraints_mandatory(self):
        """
        Validates that necessary passes have run before transforming to backend.

        Note that this differs from the original validate_constraints function, which
        only checks the order of passes.
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

    def add_passes(self, passes: Sequence[ExportPass | None]):
        for p in passes:
            if p is not None:
                self.add_pass(p)

    def _transform(self, graph_module: GraphModule):
        with TosaLoweringContext(self.tosa_spec):
            return self(graph_module).graph_module

    def _tosa_pipeline(
        self, exported_program: ExportedProgram, graph_module: GraphModule
    ) -> GraphModule:
        # Preprocessing passes
        self.add_pass(AnnotateOutputDimOrderPass())

        # Node transformation passes (pre q/dq folding)
        self.add_passes(
            [
                FuseQuantizedActivationPass(),
                RemoveGetItemPass(),
                ConvertToClampPass(),
                DecomposeInt32ClampPass(),
                DecomposeGroupNormPass(),
                DecomposeLayerNormPass(),
                DecomposeBatchNormNoStatsPass(),
                DecomposeVarPass(),
                DecomposeMeanDimPass(exported_program.graph_module, self.tosa_spec),
                AnnotateDecomposedMatmulPass(),
                ConvertELUParamsPass(),
                ConvertSplitToSlicePass(),
                QuantizeClampArgumentsPass(),
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
                InsertControlFlowRescalesPass(),
                DecomposeQuantNodesPass(),
            ]
        )

        # Node transformation passes (post q/dq folding)
        self.add_passes(
            [
                DecomposeLogitPass(),
                DecomposeMaskedFillPass(),
                DecomposeRoundPass(),
                DecomposeAcoshPass(),
                DecomposeAsinhPass(),
                DecomposeCoshPass(),
                DecomposeAsinAndAcosPass(),
                DecomposeSqrtPass(),
                DecomposeAtanPass(),
                DecomposeAtanhPass(),
                DecomposeAddmmPass(),
                DecomposeEluPass(),
                DecomposeExpm1Pass(),
                DecomposeIntPowPass(),
                PromoteBoolOperandsPass(),
                DecomposeSinhPass(),
                DecomposeSignPass(),
                DecomposeFloorDividePass(),
                DecomposeGeluPass(),
                DecomposeAddSubAlphaPass(),
                DecomposeGroupedConvPass(),
                Conv1dUnsqueezePass(),
            ]
        )

        # Scalars -> tensors, match tensor dtypes and ranks.
        self.add_passes(
            [
                ReplaceScalarWithTensorByProfilePass(),
                ConvertFullLikeToFullPass(),
                MatchArgDtypePass(),
                UnsqueezeScalarPlaceholdersPass(exported_program),
                # TODO: Move DecomposeNotEqualPass to before or after this block of
                # passes. Ticket: MLETORCH-1540
                DecomposeNotEqualPass(),
                MatchArgRanksPass(exported_program),
                FuseConstantArgsPass(exported_program),
            ]
        )

        # Node transformation passes (post scalar-removal)
        self.add_passes(
            [
                DecomposeRemainderPass(),
                DecomposeDivTensorModePass(),
                DecomposeEmbeddingPass(),
                FuseBatchNorm2dPass(exported_program),
                ConvertMmToBmmPass(),
                DecomposeGluPass(),
                DecomposeLeakyReLUPass(),
                DecomposeDivPass(),
                DecomposeSoftmaxPass(),
                ConvertMinMaxPass(),
                DecomposeAnyPass(),
                DecomposeAdaptiveAvgPool2dPass(),
                DecomposeAvgPool2dPass(),
                DecorateFp32toInt32CastingPass(),
                ComputeConstantOpsAOTPass(exported_program),
                ConvertExpandCopyToRepeatPass(),
                UnsqueezeBeforeRepeatPass(),
                DecomposeCumsumPass(exported_program),
                DecomposeMaxPool2dPass(),
                SizeAdjustInputPass(),
                DecomposeSelectPass(),
                ConvertSqueezesToViewPass(),
                CastToInt32Pass(),
                BroadcastArgsPass(),
                ConvertPermuteSingletonToViewPass(),
                FuseViewCopyTransformPass(),
                DecomposeConv2dWithInt16ActivationPass(),
                DecomposeSumPass(),
                InsertTableOpsPass(exported_program),
            ]
        )

        # Aten -> TOSA transformation passes
        self.add_passes(
            [
                RewriteUpsamplePass(),
                RewriteConv2dPass(exported_program),
                RewriteMatmulPass(),
            ]
        )

        # Postprocessing/cleanup passes
        self.add_passes(
            [
                CastInt64BuffersToInt32Pass(exported_program),
                FuseEqualPlaceholdersPass(exported_program),
                ToTosaMemoryFormatPass(exported_program),
                RemoveNoopPass(),
                InsertRescalePass(),
            ]
        )

        self.validate_constraints_mandatory()
        return self._transform(graph_module)

    def transform_to_backend_pipeline(
        self, exported_program: ExportedProgram, graph_module: GraphModule
    ):
        """Apply passes before transforming program to backend"""

        if not tosa_spec_in_set(
            self.tosa_spec,
            {
                TosaSpecification.create_from_string("TOSA-1.0+FP"),
                TosaSpecification.create_from_string("TOSA-1.0+INT"),
            },
        ):
            raise RuntimeError(
                f"No pass pipeline found for TOSA specification: {self.tosa_spec}"
            )

        return self._tosa_pipeline(exported_program, graph_module)

    def transform_for_annotation_pipeline(self, graph_module: GraphModule):
        # Preprocessing passes
        self.add_pass(RemoveGraphAssertsPass())

        # Transformation passes (pre scalar -> tensor)
        self.add_passes(
            [
                DecomposeSelectScatterPass(),
                ConvertInt64ConstOpsToInt32Pass(),
                ConvertInt64OutputOpsToInt32Pass(),
                InsertInt32CastsAfterInt64PlaceholdersPass(),
                DecomposeEmbeddingPass(),
                DecomposeScaledDotProductAttentionPass(),
                DecomposeRoundPass(),
                DecomposeLogitPass(),
                PromoteBoolOperandsPass(),
                DecomposeSignPass(),
                DecomposeAddmmPass(),
                DecomposeRemainderPass(),
                DecomposeFloorDividePass(),
                DecomposeDivTensorModePass(),
            ]
        )

        # Scalars -> tensors
        self.add_passes(
            [
                ReplaceScalarWithTensorByProfilePass(),
                ScalarsToAttributePass(),
            ]
        )

        # Transformation passes (post scalar removal)
        self.add_passes(
            [
                DecomposeAddSubAlphaPass(),
                DecomposeGroupNormPass(),
                DecomposeLayerNormPass(),
                DecomposeVarPass(),
                DecomposeMeanDimPass(graph_module, self.tosa_spec),
                DecomposeNotEqualPass(),
                DecomposeCosineSimilarityPass(),
                DecomposeGluPass(),
                DecomposeDivPass(),
                DecomposeLeakyReLUPass(),
                DecomposeLinalgVectorNormPass(),
                DecomposeSqrtPass(),
                DecomposeSiluPass(),
                DecomposeAvgPool2dPass(),
                (
                    DecomposeSoftmaxUnstablePass()
                    if self.tosa_spec.is_U55_subset
                    else DecomposeSoftmaxPass()
                ),
                ConvertMinMaxPass(),
            ]
        )

        # Postprocessing passes
        self.add_passes(
            [
                ReplaceInfValuesPass(),
                DecomposeMaskedFillPass() if not self.tosa_spec.is_U55_subset else None,
            ]
        )

        return self._transform(graph_module)

    def __call__(self, module: Module) -> PassResult:
        try:
            return super().__call__(module)
        except Exception as e:
            first_exception = e.__cause__ or e.__context__ or e
            import re

            message = e.args[0]
            m = re.search(r"An error occurred when running the '([^']+)' pass", message)
            if m:
                pass_name = m.group(1)
                first_exception.args = (
                    f"{pass_name}: {first_exception.args[0]}",
                    *first_exception.args[1:],
                )
            raise first_exception
