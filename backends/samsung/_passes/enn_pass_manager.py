# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.samsung._passes import (
    AnnotateQparamsPass,
    AnnotateScalarParametersPass,
    ComputeConstAttrs,
    ConstantPropPass,
    Conv1dToConv2d,
    DecomposeEinsum,
    DecomposeGlu,
    DecomposeLinalgVectorNorm,
    DecomposeRoll,
    FoldQDQPass,
    FuseActivationPass,
    InsertQDQPass,
    RecomposeRmsNorm,
    RemoveUselessOpPass,
    ReplaceInfValues,
    ReplaceOpsWithScalar,
)
from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform

from executorch.backends.transforms.decompose_sdpa import (
    DecomposeScaledDotProductAttention,
)
from executorch.backends.transforms.fuse_batch_norm_with_conv import (
    FuseBatchNormWithConvPass,
)
from executorch.backends.transforms.remove_clone_ops import RemoveCloneOpsTransform
from executorch.backends.transforms.remove_getitem_op import RemoveGetItemPass

from executorch.exir import ExportedProgram
from executorch.exir.pass_manager import PassManager
from torch.fx import GraphModule


class EnnPassManager(PassManager):
    def __init__(self) -> None:
        super().__init__()

    def _transform(self, graph_module: GraphModule):
        return self(graph_module).graph_module

    # before annotator, for quant models
    def transform_for_annotation_pass(self, graph_module: GraphModule):
        self.add_pass(DecomposeScaledDotProductAttention())
        self.add_pass(DecomposeGlu())
        self.add_pass(DecomposeEinsum())
        self.add_pass(DecomposeRoll())
        self.add_pass(DecomposeLinalgVectorNorm())
        self.add_pass(ReplaceInfValues())
        return self._transform(graph_module)

    # before partition, for float models
    def transform_for_export_pass(self, exported_program: ExportedProgram):
        self.add_pass(ComputeConstAttrs())
        self.add_pass(DecomposeRoll())
        self._transform(exported_program.graph_module)
        return exported_program

    # in preprocess
    def transform_for_preprocess_pass(self, exported_program: ExportedProgram):
        self.add_pass(RemoveUselessOpPass())
        self.add_pass(RemoveCloneOpsTransform())
        self.add_pass(AnnotateQparamsPass(exported_program))
        self.add_pass(ConstantPropPass(exported_program))
        self.add_pass(FuseActivationPass())
        self.add_pass(FoldQDQPass())
        self.add_pass(Conv1dToConv2d(exported_program))
        self.add_pass(FuseBatchNormWithConvPass(exported_program))
        self.add_pass(AddmmToLinearTransform())
        self.add_pass(ReplaceOpsWithScalar())
        self.add_pass(RemoveGetItemPass())
        self.add_pass(InsertQDQPass(exported_program))
        self.add_pass(AnnotateScalarParametersPass(exported_program))
        self.add_pass(RecomposeRmsNorm())
        return self._transform(exported_program.graph_module)
