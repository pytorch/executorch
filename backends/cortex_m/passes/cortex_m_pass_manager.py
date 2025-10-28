# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


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
        ReplaceScalarWithTensorArgPass,
        ReplaceQuantNodesPass,
        QuantizedOpFusionPass,
        QuantizedLinearFusionPass,
    ]

    def __init__(self, exported_program, passes=None):
        super().__init__(exported_program, passes or self.pass_list)
