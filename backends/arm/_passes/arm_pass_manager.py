# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from executorch.backends.arm._passes import (
    AnnotateChannelsLastDimOrder,
    AnnotateDecomposedMatmulPass,
    CastInt64ToInt32Pass,
    Conv1dUnsqueezePass,
    ConvertExpandCopyToRepeatPass,
    ConvertFullLikeToFullPass,
    ConvertMeanDimToAveragePoolPass,
    ConvertMinMaxPass,
    ConvertMmToBmmPass,
    ConvertSplitToSlicePass,
    ConvertSqueezesToViewPass,
    ConvertToClampPass,
    DecomposeBatchNormPass,
    DecomposeDivPass,
    DecomposeLayerNormPass,
    DecomposeLinearPass,
    DecomposeMeanDimPass,
    DecomposeSelectPass,
    DecomposeSoftmaxesPass,
    DecomposeVarPass,
    FoldAndAnnotateQParamsPass,
    FuseBatchnorm2DPass,
    FuseConstantOpsPass,
    FuseQuantizedActivationPass,
    InsertRescalePass,
    InsertTableOpsPass,
    KeepDimsFalseToSqueezePass,
    MatchArgRanksPass,
    QuantizeOperatorArguments,
    RemoveClonePass,
    RetraceFoldedDtypesPass,
    ScalarsToAttributePass,
    SizeAdjustConv2DPass,
    UnsqueezeBeforeRepeatPass,
    UnsqueezeScalarPlaceholdersPass,
)
from executorch.backends.arm.tosa_specification import TosaSpecification
from executorch.backends.transforms.fuse_view_copy import FuseViewCopyTransform

from executorch.backends.transforms.replace_scalar_with_tensor import (
    ReplaceScalarWithTensorArgPass,
)
from executorch.backends.xnnpack._passes.remove_getitem_op import RemoveGetItemPass
from executorch.exir import ExportedProgram
from executorch.exir.pass_manager import PassManager
from torch.fx import GraphModule


class ArmPassManager(PassManager):

    def __init__(self, tosa_spec: TosaSpecification) -> None:
        self.tosa_spec = tosa_spec
        super().__init__()

    def _transform(self, graph_module: GraphModule):
        return self(graph_module).graph_module

    def _tosa_080_BI_pipeline(self, exported_program: ExportedProgram) -> GraphModule:
        self.add_pass(FuseQuantizedActivationPass())
        self.add_pass(RemoveGetItemPass())
        self.add_pass(DecomposeBatchNormPass())
        self.add_pass(ConvertSplitToSlicePass())
        self.add_pass(ConvertMmToBmmPass())
        self.add_pass(DecomposeLinearPass())
        self.add_pass(ConvertMeanDimToAveragePoolPass())
        self.add_pass(ConvertFullLikeToFullPass())
        self.add_pass(ConvertToClampPass())
        self.add_pass(ConvertMinMaxPass())

        self.add_pass(ReplaceScalarWithTensorArgPass())
        self.add_pass(AnnotateDecomposedMatmulPass())
        self.add_pass(QuantizeOperatorArguments())
        self.add_pass(FoldAndAnnotateQParamsPass())  # type: ignore[call-arg]
        self.add_pass(RetraceFoldedDtypesPass())

        self.add_pass(RemoveClonePass())
        self.add_pass(SizeAdjustConv2DPass())
        self.add_pass(ConvertExpandCopyToRepeatPass())
        self.add_pass(UnsqueezeBeforeRepeatPass())
        self.add_pass(UnsqueezeScalarPlaceholdersPass(exported_program))
        self.add_pass(CastInt64ToInt32Pass(exported_program))
        self.add_pass(MatchArgRanksPass(exported_program))
        self.add_pass(KeepDimsFalseToSqueezePass())
        self.add_pass(Conv1dUnsqueezePass(exported_program))
        self.add_pass(DecomposeSelectPass())
        self.add_pass(ConvertSqueezesToViewPass())

        self.add_pass(FuseViewCopyTransform())
        self.add_pass(FuseConstantOpsPass(exported_program))
        self.add_pass(InsertTableOpsPass(exported_program))
        self.add_pass(AnnotateChannelsLastDimOrder())
        self.add_pass(InsertRescalePass())

        return self._transform(exported_program.graph_module)

    def _tosa_080_MI_pipeline(self, exported_program: ExportedProgram) -> GraphModule:
        self.add_pass(ReplaceScalarWithTensorArgPass())
        self.add_pass(FuseQuantizedActivationPass())
        self.add_pass(RemoveGetItemPass())
        self.add_pass(ConvertSplitToSlicePass())
        self.add_pass(FuseBatchnorm2DPass(exported_program))
        self.add_pass(ConvertMmToBmmPass())
        self.add_pass(DecomposeLinearPass())
        self.add_pass(DecomposeBatchNormPass())
        self.add_pass(DecomposeLayerNormPass())
        self.add_pass(DecomposeVarPass())
        self.add_pass(DecomposeMeanDimPass())
        self.add_pass(ConvertMeanDimToAveragePoolPass())
        self.add_pass(DecomposeDivPass())
        self.add_pass(DecomposeSoftmaxesPass())
        self.add_pass(ConvertFullLikeToFullPass())
        self.add_pass(ConvertToClampPass())
        self.add_pass(ConvertMinMaxPass())

        self.add_pass(AnnotateDecomposedMatmulPass())
        self.add_pass(QuantizeOperatorArguments())
        self.add_pass(FoldAndAnnotateQParamsPass())  # type: ignore[call-arg]
        self.add_pass(RetraceFoldedDtypesPass())

        self.add_pass(RemoveClonePass())
        self.add_pass(SizeAdjustConv2DPass())
        self.add_pass(ConvertExpandCopyToRepeatPass())
        self.add_pass(UnsqueezeBeforeRepeatPass())
        self.add_pass(UnsqueezeScalarPlaceholdersPass(exported_program))
        self.add_pass(CastInt64ToInt32Pass(exported_program))
        self.add_pass(MatchArgRanksPass(exported_program))
        self.add_pass(KeepDimsFalseToSqueezePass())
        self.add_pass(Conv1dUnsqueezePass(exported_program))
        self.add_pass(DecomposeSelectPass())
        self.add_pass(ConvertSqueezesToViewPass())

        self.add_pass(FuseViewCopyTransform())
        self.add_pass(FuseConstantOpsPass(exported_program))
        self.add_pass(InsertTableOpsPass(exported_program))
        self.add_pass(AnnotateChannelsLastDimOrder())
        self.add_pass(InsertRescalePass())

        return self._transform(exported_program.graph_module)

    def transform_to_backend_pipeline(self, exported_program: ExportedProgram):
        """Apply passes before transforming program to backend"""
        if self.tosa_spec == TosaSpecification.create_from_string("TOSA-0.80.0+BI"):
            return self._tosa_080_BI_pipeline(exported_program)
        elif self.tosa_spec == TosaSpecification.create_from_string("TOSA-0.80.0+MI"):
            return self._tosa_080_MI_pipeline(exported_program)
        else:
            raise NotImplementedError(
                f"No pass pipeline implemented for {self.tosa_spec=}"
            )

    def transform_for_annotation_pipeline(self, graph_module: GraphModule):
        self.add_pass(ScalarsToAttributePass())
        self.add_pass(ReplaceScalarWithTensorArgPass())
        self.add_pass(DecomposeLayerNormPass())
        self.add_pass(DecomposeVarPass())
        self.add_pass(DecomposeMeanDimPass())
        self.add_pass(DecomposeDivPass())
        self.add_pass(DecomposeSoftmaxesPass())
        self.add_pass(ConvertMinMaxPass())
        return self._transform(graph_module)
