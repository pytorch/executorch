# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import inspect

from executorch.backends.arm._passes import (
    FoldAndAnnotateQParamsPass,
    ScalarsToAttributePass,
)
from executorch.backends.cortex_m.passes import (
    ActivationFusionPass,
    ClampHardswishPass,
    ConvertToCortexMPass,
    DecomposeHardswishPass,
    DecomposeMeanPass,
    QuantizedOpFusionPass,
    ReplaceQuantNodesPass,
)
from executorch.backends.transforms.replace_scalar_with_tensor import (
    ReplaceScalarWithTensorArgPass,
)
from executorch.exir.pass_base import ExportPass
from executorch.exir.pass_manager import PassManager
from executorch.exir.program._program import _transform
from torch.export import ExportedProgram


class CortexMPassManager(PassManager):

    pass_list: list[ExportPass] = [
        FoldAndAnnotateQParamsPass,
        ReplaceScalarWithTensorArgPass,
        ReplaceQuantNodesPass,
        ActivationFusionPass,
        DecomposeHardswishPass,
        QuantizedOpFusionPass,
        ConvertToCortexMPass,
    ]

    pass_list_transform_for_annotation: list[ExportPass] = [
        ScalarsToAttributePass,
        ReplaceScalarWithTensorArgPass,
        ClampHardswishPass,
        DecomposeMeanPass,
    ]

    def __init__(self, exported_program, passes=None):
        self.exported_program = exported_program
        if passes is not None:
            self.passes = passes
        else:
            self.passes = self.pass_list

    def transform_for_annotation(self, model):
        passes = self.pass_list_transform_for_annotation
        for p in passes:
            model = p().call(model).graph_module
        return model

    def transform(self) -> ExportedProgram:
        ep = self.exported_program
        for pass_ in self.passes:
            signature = inspect.signature(pass_.__init__)
            if "exported_program" in signature.parameters:
                transform_pass = pass_(ep)
            elif issubclass(pass_, ExportPass):
                transform_pass = pass_()
            else:
                raise RuntimeError(
                    f"Expecting ExportPass or ExportPass(), but got pass: {pass_} with type: {type(pass_)}"
                )
            ep = _transform(ep, transform_pass)
        return ep
