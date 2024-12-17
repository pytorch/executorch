# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from executorch.backends.arm._passes.annotate_channels_last_dim_order_pass import (
    AnnotateChannelsLastDimOrder,
)
from executorch.backends.arm._passes.cast_int64_pass import CastInt64ToInt32Pass
from executorch.backends.arm._passes.conv1d_unsqueeze_pass import Conv1dUnsqueezePass
from executorch.backends.arm._passes.convert_expand_copy_to_repeat import (
    ConvertExpandCopyToRepeatPass,
)
from executorch.backends.arm._passes.convert_split_to_slice import (
    ConvertSplitToSlicePass,
)
from executorch.backends.arm._passes.decompose_div_pass import DecomposeDivPass
from executorch.backends.arm._passes.decompose_layernorm_pass import (
    DecomposeLayerNormPass,
)
from executorch.backends.arm._passes.decompose_linear_pass import DecomposeLinearPass
from executorch.backends.arm._passes.decompose_meandim_pass import DecomposeMeanDimPass
from executorch.backends.arm._passes.decompose_softmaxes_pass import (
    DecomposeSoftmaxesPass,
)
from executorch.backends.arm._passes.decompose_var_pass import DecomposeVarPass
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    FoldAndAnnotateQParamsPass,
    QuantizeFullArgument,
)
from executorch.backends.arm._passes.keep_dims_false_to_squeeze_pass import (
    KeepDimsFalseToSqueezePass,
)
from executorch.backends.arm._passes.match_arg_ranks_pass import MatchArgRanksPass
from executorch.backends.arm._passes.meandim_to_averagepool_pass import (
    ConvertMeanDimToAveragePool,
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
from executorch.backends.xnnpack._passes.remove_getitem_op import RemoveGetItemPass
from executorch.exir import ExportedProgram
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_manager import PassManager


class ArmPassManager(PassManager):

    def _transform(self, graph_module: torch.fx.GraphModule):
        return self(graph_module).graph_module

    def transform_to_backend_pipeline(
        self, exported_program: ExportedProgram, compile_spec: list[CompileSpec]
    ):
        """Apply passes before transforming program to backend"""
        self.add_pass(CastInt64ToInt32Pass(exported_program))
        self.add_pass(RemoveGetItemPass())
        self.add_pass(UnsqueezeScalarPlaceholdersPass(exported_program))
        self.add_pass(SizeAdjustConv2DPass())
        self.add_pass(RemoveClonePass())
        self.add_pass(ConvertExpandCopyToRepeatPass())
        self.add_pass(DecomposeLayerNormPass())
        self.add_pass(UnsqueezeBeforeRepeatPass())
        self.add_pass(DecomposeVarPass())
        self.add_pass(ConvertMeanDimToAveragePool())
        self.add_pass(DecomposeMeanDimPass())
        self.add_pass(MatchArgRanksPass(exported_program))
        self.add_pass(DecomposeDivPass())
        self.add_pass(KeepDimsFalseToSqueezePass())
        self.add_pass(ConvertSplitToSlicePass())
        self.add_pass(Conv1dUnsqueezePass(exported_program))
        self.add_pass(DecomposeSoftmaxesPass())
        self.add_pass(DecomposeLinearPass())
        self.add_pass(QuantizeFullArgument())
        self.add_pass(
            FoldAndAnnotateQParamsPass(
                [
                    exir_ops.edge.aten.minimum.default,
                    exir_ops.edge.aten.maximum.default,
                    exir_ops.edge.aten.add.Tensor,
                    exir_ops.edge.aten.avg_pool2d.default,
                    exir_ops.edge.aten.convolution.default,
                    exir_ops.edge.aten.full.default,
                ]
            )
        )
        for spec in compile_spec:
            if spec.key == "permute_memory_format":
                memory_format = spec.value.decode()
                if memory_format == "nhwc":
                    self.add_pass(AnnotateChannelsLastDimOrder())

        return self._transform(exported_program.graph_module)

    def transform_for_annotation_pipeline(self, graph_module: torch.fx.GraphModule):
        self.add_pass(ScalarsToAttributePass())
        self.add_pass(DecomposeLayerNormPass())
        self.add_pass(DecomposeVarPass())
        self.add_pass(DecomposeMeanDimPass())
        self.add_pass(DecomposeDivPass())
        self.add_pass(DecomposeSoftmaxesPass())
        return self._transform(graph_module)
