# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import inspect
from typing import Callable, cast, Optional, Type

from executorch.backends.arm._passes import (
    FoldAndAnnotateQParamsPass,
    ScalarsToAttributePass,
)
from executorch.backends.transforms.remove_getitem_op import RemoveGetItemPass
from executorch.backends.transforms.replace_scalar_with_tensor import (
    ReplaceScalarWithTensorArgPass,
)
from executorch.exir.pass_base import ExportPass
from executorch.exir.pass_manager import PassManager
from executorch.exir.program._program import _transform, lift_constant_tensor_pass
from torch.export import ExportedProgram
from torch.fx.passes.infra.pass_base import PassResult

from torch.nn import Module

from .activation_fusion_pass import ActivationFusionPass
from .clamp_hardswish_pass import ClampHardswishPass
from .convert_to_cortex_m_pass import ConvertToCortexMPass
from .decompose_hardswish_pass import DecomposeHardswishPass
from .decompose_mean_pass import DecomposeMeanPass
from .quantized_op_fusion_pass import QuantizedOpFusionPass
from .replace_quant_nodes_pass import ReplaceQuantNodesPass

PassClass = Type[ExportPass]


class CortexMPassManager(PassManager):
    pass_list: list[PassClass] = [
        # Run before folding so qparams attach to max_pool2d values, not tuple + getitem.
        RemoveGetItemPass,
        FoldAndAnnotateQParamsPass,
        ReplaceScalarWithTensorArgPass,
        ReplaceQuantNodesPass,
        ActivationFusionPass,
        DecomposeHardswishPass,
        QuantizedOpFusionPass,
        ConvertToCortexMPass,
    ]

    pass_list_transform_for_annotation: list[PassClass] = [
        ScalarsToAttributePass,
        ReplaceScalarWithTensorArgPass,
        ClampHardswishPass,
        DecomposeMeanPass,
    ]

    def __init__(
        self, exported_program, passes: Optional[list[PassClass]] = None
    ) -> None:
        super().__init__(passes=[])
        self.exported_program = exported_program
        # PassManager.passes is typed as callables; this manager stores pass classes which are initialized at transform time with the exported_program.
        self.passes: list[PassClass] = (  # type: ignore[assignment]
            passes if passes is not None else self.pass_list  # type: ignore[assignment]
        )

    def transform_for_annotation(self, model):
        passes = self.pass_list_transform_for_annotation
        for p in passes:
            model = p().call(model).graph_module
        return model

    def transform(self) -> ExportedProgram:
        ep = self.exported_program
        for pass_cls in self.passes:
            signature = inspect.signature(pass_cls)
            if "exported_program" in signature.parameters:
                ep_pass_ctor = cast(Callable[[ExportedProgram], ExportPass], pass_cls)
                transform_pass = ep_pass_ctor(ep)
            else:
                transform_pass = pass_cls()
            pass_callable = cast(Callable[[Module], PassResult], transform_pass)
            ep = _transform(ep, pass_callable)

        # All constant tensors should be lifted to buffers at this point, re-run
        # lift_constant_tensor_pass in case new ones have been introduced by the passes above.
        ep = lift_constant_tensor_pass(ep)
        return ep
