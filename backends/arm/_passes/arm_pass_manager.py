# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from collections import defaultdict

import executorch.backends.arm.tosa.dialect  # noqa: unused
from executorch.backends.arm._passes import (
    AnnotateDecomposedMatmulPass,
    AnnotateOutputDimOrderPass,
    BroadcastArgsPass,
    CastBoolToInt8Pass,
    CastInt64BuffersToInt32Pass,
    CastToInt32Pass,
    ComputeConstantOpsAOT,
    Conv1dUnsqueezePass,
    ConvertELUParamsPass,
    ConvertExpandCopyToRepeatPass,
    ConvertFullLikeToFullPass,
    ConvertInt64ConstOpsToInt32Pass,
    ConvertInt64OutputOpsToInt32Pass,
    ConvertIntPowToMuls,
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
    DecomposeAvgPool2d,
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
    DecomposeGroupedConv,
    DecomposeGroupNormPass,
    DecomposeLayerNormPass,
    DecomposeLeakyReLUPass,
    DecomposeLinearPass,
    DecomposeLinearVectorNormPass,
    DecomposeLogitPass,
    DecomposeMaskedFill,
    DecomposeMaxPool2DPass,
    DecomposeMeanDimPass,
    DecomposeNotEqualPass,
    DecomposeRemainderPass,
    DecomposeRoundPass,
    DecomposeScaledDotProductAttention,
    DecomposeSelectPass,
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
    FuseBatchnorm2DPass,
    FuseConstantArgsPass,
    FuseDuplicateUsersPass,
    FuseEqualPlaceholdersPass,
    FuseQuantizedActivationPass,
    FuseViewCopyTransformPass,
    InsertInt32CastsAfterInt64PlaceholdersPass,
    InsertRescaleInt32Pass,
    InsertRescalePass,
    InsertTableOpsPass,
    MatchArgDtypePass,
    MatchArgRanksPass,
    QuantizeOperatorArguments,
    RemoveGetItemPass,
    RemoveGraphAssertsPass,
    RemoveNoopPass,
    ReplaceInfValues,
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
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir import ExportedProgram
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

    def _transform(self, graph_module: GraphModule):
        with TosaLoweringContext(self.tosa_spec):
            return self(graph_module).graph_module

    def _tosa_pipeline(
        self, exported_program: ExportedProgram, graph_module: GraphModule
    ) -> GraphModule:
        # Preprocessing passes

        self.add_pass(AnnotateOutputDimOrderPass())

        # Node transformation passes (pre q/dq folding)

        self.add_pass(FuseQuantizedActivationPass())
        self.add_pass(RemoveGetItemPass())
        self.add_pass(ConvertToClampPass())
        self.add_pass(DecomposeGroupNormPass())
        self.add_pass(DecomposeLayerNormPass())
        self.add_pass(DecomposeBatchNormNoStatsPass())
        self.add_pass(DecomposeVarPass())
        self.add_pass(
            DecomposeMeanDimPass(exported_program.graph_module, self.tosa_spec)
        )
        self.add_pass(AnnotateDecomposedMatmulPass())
        self.add_pass(ConvertELUParamsPass())
        self.add_pass(ConvertSplitToSlicePass())
        self.add_pass(QuantizeOperatorArguments())

        # Fold Q/DQ nodes, insert INT8/INT32 rescales.

        self.add_pass(FoldAndAnnotateQParamsPass(exported_program))  # type: ignore[call-arg]
        self.add_pass(FuseDuplicateUsersPass())
        # TODO: DecomposeLinearPass should run after InsertRescaleInt32Pass or
        # before FoldAndAnnotateQParamsPass but is unable to at the moment.
        # Ticket: MLETORCH-1539
        self.add_pass(DecomposeLinearPass())
        self.add_pass(InsertRescaleInt32Pass())

        # Node transformation passes (post q/dq folding)

        self.add_pass(DecomposeLogitPass())
        self.add_pass(DecomposeMaskedFill())
        self.add_pass(DecomposeRoundPass())
        self.add_pass(DecomposeAcoshPass())
        self.add_pass(DecomposeAsinhPass())
        self.add_pass(DecomposeCoshPass())
        self.add_pass(DecomposeAsinAndAcosPass())
        self.add_pass(DecomposeSqrtPass())
        self.add_pass(DecomposeAtanPass())
        self.add_pass(DecomposeAtanhPass())
        self.add_pass(DecomposeAddmmPass())
        self.add_pass(DecomposeEluPass())
        self.add_pass(DecomposeExpm1Pass())
        self.add_pass(ConvertIntPowToMuls())
        self.add_pass(CastBoolToInt8Pass())
        self.add_pass(DecomposeSinhPass())
        self.add_pass(DecomposeSignPass())
        self.add_pass(DecomposeFloorDividePass())
        self.add_pass(DecomposeGeluPass())
        self.add_pass(DecomposeAddSubAlphaPass())
        self.add_pass(DecomposeGroupedConv())
        self.add_pass(Conv1dUnsqueezePass())

        # Scalars -> tensors, match tensor dtypes and ranks.

        self.add_pass(ReplaceScalarWithTensorByProfilePass())
        self.add_pass(ConvertFullLikeToFullPass())
        self.add_pass(MatchArgDtypePass())
        self.add_pass(UnsqueezeScalarPlaceholdersPass(exported_program))
        # TODO: Move DecomposeNotEqualPass to before or after this block of
        # passes. Ticket: MLETORCH-1540
        self.add_pass(DecomposeNotEqualPass())
        self.add_pass(MatchArgRanksPass(exported_program))
        self.add_pass(FuseConstantArgsPass(exported_program))

        # Node transformation passes (post scalar-removal)

        self.add_pass(DecomposeRemainderPass())
        self.add_pass(DecomposeDivTensorModePass())
        self.add_pass(DecomposeEmbeddingPass())
        self.add_pass(FuseBatchnorm2DPass(exported_program))
        self.add_pass(ConvertMmToBmmPass())
        self.add_pass(DecomposeGluPass())
        self.add_pass(DecomposeLeakyReLUPass())
        self.add_pass(DecomposeDivPass())
        self.add_pass(DecomposeSoftmaxPass())
        self.add_pass(ConvertMinMaxPass())
        self.add_pass(DecomposeAnyPass())
        self.add_pass(DecomposeAdaptiveAvgPool2dPass())
        self.add_pass(DecomposeAvgPool2d())
        self.add_pass(
            DecorateFp32toInt32CastingPass()
        )  # Require that no new fp32->int32 is introduced after this pass
        self.add_pass(ComputeConstantOpsAOT(exported_program))
        self.add_pass(ConvertExpandCopyToRepeatPass())
        self.add_pass(UnsqueezeBeforeRepeatPass())
        self.add_pass(DecomposeCumsumPass(exported_program))
        self.add_pass(DecomposeMaxPool2DPass())
        self.add_pass(SizeAdjustInputPass())
        self.add_pass(DecomposeSelectPass())
        self.add_pass(ConvertSqueezesToViewPass())
        self.add_pass(CastToInt32Pass())
        self.add_pass(BroadcastArgsPass())
        self.add_pass(ConvertPermuteSingletonToViewPass())
        self.add_pass(FuseViewCopyTransformPass())
        self.add_pass(DecomposeConv2dWithInt16ActivationPass())
        self.add_pass(DecomposeSumPass())
        self.add_pass(InsertTableOpsPass(exported_program))

        # Aten -> TOSA transformation passes

        self.add_pass(RewriteUpsamplePass())
        self.add_pass(RewriteConv2dPass(exported_program))
        self.add_pass(RewriteMatmulPass())

        # Postprocessing/cleanup passes

        self.add_pass(CastInt64BuffersToInt32Pass(exported_program))
        self.add_pass(FuseEqualPlaceholdersPass(exported_program))
        self.add_pass(ToTosaMemoryFormatPass(exported_program))
        self.add_pass(RemoveNoopPass())
        self.add_pass(InsertRescalePass())

        self.validate_constraints_mandatory()
        return self._transform(graph_module)

    def transform_to_backend_pipeline(
        self, exported_program: ExportedProgram, graph_module: GraphModule
    ):
        """Apply passes before transforming program to backend"""
        if self.tosa_spec in (
            TosaSpecification.create_from_string("TOSA-1.0+FP"),
            TosaSpecification.create_from_string("TOSA-1.0+INT"),
        ):
            return self._tosa_pipeline(exported_program, graph_module)
        else:
            raise NotImplementedError(
                f"No pass pipeline implemented for {self.tosa_spec=}"
            )

    def transform_for_annotation_pipeline(self, graph_module: GraphModule):
        self.add_pass(
            RemoveGraphAssertsPass()
        )  # ConvertInt64ConstOpsToInt32Pass requires this pass to remove the assertation in Graph
        self.add_pass(ConvertInt64ConstOpsToInt32Pass())
        self.add_pass(ConvertInt64OutputOpsToInt32Pass())
        self.add_pass(InsertInt32CastsAfterInt64PlaceholdersPass())
        self.add_pass(DecomposeEmbeddingPass())
        self.add_pass(DecomposeScaledDotProductAttention())
        self.add_pass(DecomposeRoundPass())
        self.add_pass(DecomposeLogitPass())
        self.add_pass(CastBoolToInt8Pass())
        self.add_pass(DecomposeSignPass())
        self.add_pass(DecomposeAddmmPass())
        self.add_pass(ReplaceScalarWithTensorByProfilePass())
        self.add_pass(DecomposeRemainderPass())
        self.add_pass(DecomposeFloorDividePass())
        self.add_pass(DecomposeDivTensorModePass())
        self.add_pass(DecomposeAddSubAlphaPass())
        self.add_pass(ScalarsToAttributePass())
        self.add_pass(DecomposeGroupNormPass())
        self.add_pass(DecomposeLayerNormPass())
        self.add_pass(DecomposeVarPass())
        self.add_pass(DecomposeMeanDimPass(graph_module, self.tosa_spec))
        self.add_pass(DecomposeNotEqualPass())
        self.add_pass(DecomposeCosineSimilarityPass())
        self.add_pass(DecomposeGluPass())
        self.add_pass(DecomposeDivPass())
        self.add_pass(DecomposeLeakyReLUPass())
        self.add_pass(DecomposeLinearVectorNormPass())
        self.add_pass(DecomposeSqrtPass())
        self.add_pass(DecomposeSiluPass())
        self.add_pass(DecomposeAvgPool2d())

        if self.tosa_spec.is_U55_subset:
            # Numerically stable softmax uses amax which is not supported on Ethos-U55
            self.add_pass(DecomposeSoftmaxUnstablePass())
        else:
            self.add_pass(DecomposeSoftmaxPass())

        self.add_pass(ConvertMinMaxPass())
        self.add_pass(ReplaceInfValues())

        if not self.tosa_spec.is_U55_subset:
            # Uses where which is not supported on Ethos-U55
            self.add_pass(DecomposeMaskedFill())

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
