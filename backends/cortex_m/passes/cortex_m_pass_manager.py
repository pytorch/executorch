# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import inspect
from typing import Any, Optional, Type

from executorch.backends.arm._passes import (
    FoldAndAnnotateQParamsPass,
    ScalarsToAttributePass,
)
from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
from executorch.backends.transforms.remove_getitem_op import RemoveGetItemPass
from executorch.backends.transforms.replace_scalar_with_tensor import (
    ReplaceScalarWithTensorArgPass,
)
from executorch.exir.pass_base import ExportPass
from executorch.exir.pass_manager import PassManager
from executorch.exir.program._program import _transform, lift_constant_tensor_pass
from torch.export import ExportedProgram

from .activation_fusion_pass import ActivationFusionPass
from .clamp_hardswish_pass import ClampHardswishPass
from .convert_to_cortex_m_pass import ConvertToCortexMPass
from .decompose_hardswish_pass import DecomposeHardswishPass
from .decompose_mean_pass import DecomposeMeanPass
from .quantized_clamp_activation_pass import QuantizedClampActivationPass
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
        QuantizedClampActivationPass,
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
        self,
        exported_program: ExportedProgram | None,
        passes: Optional[list[PassClass]] = None,
        target_config: Optional[CortexMTargetConfig] = None,
    ) -> None:
        """Initialize the Cortex-M pass manager.

        Args:
            exported_program: The exported program to transform. Required
                before calling ``transform()``; may be ``None`` for callers
                that only use ``transform_for_annotation()``.
            passes: Optional override of the pass list. Defaults to
                ``CortexMPassManager.pass_list``.
            target_config: Compilation target for passes that need it.
                Defaults to ``CortexMTargetConfig(cpu=CortexM.M55)``, which
                resolves through cmsis_nn to the MVE backend — matching the
                pre-config historical behaviour.
        """
        super().__init__(passes=[])
        self.exported_program = exported_program
        # PassManager.passes is typed as callables; this manager stores pass classes which are initialized at transform time with the exported_program.
        self.passes: list[PassClass] = (  # type: ignore[assignment]
            passes if passes is not None else self.pass_list  # type: ignore[assignment]
        )
        self.target_config: CortexMTargetConfig = target_config or CortexMTargetConfig(
            cpu=CortexM.M55
        )

    def transform_for_annotation(self, model):
        passes = self.pass_list_transform_for_annotation
        for p in passes:
            model = p().call(model).graph_module
        return model

    def transform(self) -> ExportedProgram:
        exported_program = self.exported_program
        if not isinstance(exported_program, ExportedProgram):
            raise ValueError(
                f"{type(self).__name__}.transform() needs a real ExportedProgram, "
                f"got {exported_program!r}"
            )

        for pass_cls in self.passes:
            if not isinstance(pass_cls, type):
                raise ValueError(
                    f"{type(self).__name__} expects pass classes, not instances; "
                    f"got {pass_cls!r}"
                )

            signature = inspect.signature(pass_cls)
            kwargs: dict[str, Any] = {}
            if "exported_program" in signature.parameters:
                kwargs["exported_program"] = exported_program
            if "target_config" in signature.parameters:
                kwargs["target_config"] = self.target_config

            transform_pass = pass_cls(**kwargs)
            exported_program = _transform(exported_program, transform_pass)

        # All constant tensors should be lifted to buffers at this point, re-run
        # lift_constant_tensor_pass in case new ones have been introduced.
        exported_program = lift_constant_tensor_pass(exported_program)
        return exported_program
