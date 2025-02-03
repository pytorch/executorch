# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from executorch.backends.arm._passes.annotate_channels_last_dim_order_pass import (
    AnnotateChannelsLastDimOrder,
)
from executorch.backends.arm._passes.annotate_decomposed_matmul import (
    AnnotateDecomposedMatmulPass,
)
from executorch.backends.arm._passes.cast_int64_pass import CastInt64ToInt32Pass
from executorch.backends.arm._passes.conv1d_unsqueeze_pass import Conv1dUnsqueezePass
from executorch.backends.arm._passes.convert_expand_copy_to_repeat import (
    ConvertExpandCopyToRepeatPass,
)
from executorch.backends.arm._passes.convert_split_to_slice import (
    ConvertSplitToSlicePass,
)
from executorch.backends.arm._passes.convert_squeezes_to_view import (  # type: ignore[import-not-found]
    ConvertSqueezesToViewPass,
)
from executorch.backends.arm._passes.decompose_batchnorm_pass import (
    DecomposeBatchNormPass,
)
from executorch.backends.arm._passes.decompose_div_pass import DecomposeDivPass
from executorch.backends.arm._passes.decompose_layernorm_pass import (
    DecomposeLayerNormPass,
)
from executorch.backends.arm._passes.decompose_linear_pass import DecomposeLinearPass
from executorch.backends.arm._passes.decompose_meandim_pass import DecomposeMeanDimPass
from executorch.backends.arm._passes.decompose_select import (  # type: ignore[import-not-found]
    DecomposeSelectPass,
)
from executorch.backends.arm._passes.decompose_softmaxes_pass import (
    DecomposeSoftmaxesPass,
)
from executorch.backends.arm._passes.decompose_var_pass import DecomposeVarPass
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    FoldAndAnnotateQParamsPass,
    QuantizeOperatorArguments,
    RetraceFoldedDtypesPass,
)
from executorch.backends.arm._passes.fuse_batchnorm2d_pass import FuseBatchnorm2DPass
from executorch.backends.arm._passes.fuse_quantized_activation_pass import (  # type: ignore[import-not-found]
    FuseQuantizedActivationPass,
)
from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass
from executorch.backends.arm._passes.keep_dims_false_to_squeeze_pass import (
    KeepDimsFalseToSqueezePass,
)
from executorch.backends.arm._passes.match_arg_ranks_pass import MatchArgRanksPass
from executorch.backends.arm._passes.meandim_to_averagepool_pass import (  # type: ignore[attr-defined]
    ConvertMeanDimToAveragePoolPass,
)
from executorch.backends.arm._passes.mm_to_bmm_pass import (  # type: ignore[import-not-found]
    ConvertMmToBmmPass,
)
from executorch.backends.arm._passes.remove_clone_pass import RemoveClonePass
from executorch.backends.arm._passes.scalars_to_attribute_pass import (
    ScalarsToAttributePass,
)
from executorch.backends.arm._passes.size_adjust_conv2d_pass import SizeAdjustConv2DPass
from executorch.backends.arm._passes.unsqueeze_before_repeat_pass import (
    UnsqueezeBeforeRepeatPass,
)
from executorch.backends.arm._passes.unsqueeze_scalar_placeholders_pass import (
    UnsqueezeScalarPlaceholdersPass,
)
from executorch.backends.arm.tosa_specification import TosaSpecification
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

        self.add_pass(AnnotateDecomposedMatmulPass())
        self.add_pass(QuantizeOperatorArguments())
        self.add_pass(FoldAndAnnotateQParamsPass())  # type: ignore[call-arg]
        self.add_pass(RetraceFoldedDtypesPass())
        self.add_pass(InsertTableOpsPass(exported_program))

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

        self.add_pass(AnnotateChannelsLastDimOrder())

        return self._transform(exported_program.graph_module)

    def _tosa_080_MI_pipeline(self, exported_program: ExportedProgram) -> GraphModule:

        self.add_pass(FuseQuantizedActivationPass())
        self.add_pass(RemoveGetItemPass())
        self.add_pass(ConvertSplitToSlicePass())
        self.add_pass(ConvertMmToBmmPass())
        self.add_pass(DecomposeLinearPass())
        self.add_pass(DecomposeBatchNormPass())
        self.add_pass(DecomposeLayerNormPass())
        self.add_pass(DecomposeVarPass())
        self.add_pass(DecomposeMeanDimPass())
        self.add_pass(ConvertMeanDimToAveragePoolPass())
        self.add_pass(DecomposeDivPass())
        self.add_pass(DecomposeSoftmaxesPass())
        self.add_pass(FuseBatchnorm2DPass(exported_program))

        self.add_pass(AnnotateDecomposedMatmulPass())
        self.add_pass(QuantizeOperatorArguments())
        self.add_pass(FoldAndAnnotateQParamsPass())  # type: ignore[call-arg]
        self.add_pass(RetraceFoldedDtypesPass())
        self.add_pass(InsertTableOpsPass(exported_program))

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

        self.add_pass(AnnotateChannelsLastDimOrder())

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
        self.add_pass(DecomposeLayerNormPass())
        self.add_pass(DecomposeVarPass())
        self.add_pass(DecomposeMeanDimPass())
        self.add_pass(DecomposeDivPass())
        self.add_pass(DecomposeSoftmaxesPass())
        return self._transform(graph_module)
