# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from executorch.backends.arm._passes import (
    FoldAndAnnotateQParamsPass,
    ScalarsToAttributePass,
)
from executorch.backends.cortex_m.passes import (
    QuantizedLinearFusionPass,
    QuantizedOpFusionPass,
    ReplaceQuantNodesPass,
)
from executorch.backends.transforms.replace_scalar_with_tensor import (
    ReplaceScalarWithTensorArgPass,
)
from executorch.backends.xnnpack._passes import XNNPACKPassManager
from executorch.exir.pass_base import ExportPass


class CortexMPassManager(XNNPACKPassManager):

    pass_list: list[ExportPass] = [
        FoldAndAnnotateQParamsPass,
        ReplaceScalarWithTensorArgPass,
        ReplaceQuantNodesPass,
        QuantizedOpFusionPass,
        QuantizedLinearFusionPass,
    ]

    pass_list_transform_for_annotation: list[ExportPass] = [
        ScalarsToAttributePass,
        ReplaceScalarWithTensorArgPass,
    ]

    def __init__(self, exported_program, passes=None):
        super().__init__(exported_program, passes or self.pass_list)

    def transform_for_annotation(self, model):
        passes = self.pass_list_transform_for_annotation
        for p in passes:
            model = p().call(model).graph_module
        return model
