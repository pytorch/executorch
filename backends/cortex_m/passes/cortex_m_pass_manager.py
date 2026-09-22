# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
import inspect
from typing import Any, Optional, Type

from executorch.backends.arm._passes import (
    DeduplicateGetAttrPass,
    FoldAndAnnotateQParamsPass,
    ScalarsToAttributePass,
)
from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
from executorch.backends.transforms.convert_conv1d_to_conv2d_pass import (
    ConvertConv1dToConv2dPass,
)
from executorch.backends.transforms.remove_getitem_op import RemoveGetItemPass
from executorch.backends.transforms.remove_permutes_around_elementwise_ops import (
    RemovePermutesAroundElementwiseOps,
)
from executorch.backends.transforms.remove_unused_constants_pass import (
    RemoveUnusedConstantsPass,
)
from executorch.backends.transforms.replace_scalar_with_tensor import (
    ReplaceScalarWithTensorArgPass,
)
from executorch.backends.transforms.replace_squeeze_unsqueeze_with_view import (
    ReplaceSqueezeAndUnsqueezeWithViewPass,
)
from executorch.exir.pass_base import (
    ExportedProgramPassBase,
    ExportedProgramPassResult,
    ExportPass,
)
from executorch.exir.pass_manager import ExportedProgramPassManager, PassType
from executorch.exir.program._program import _transform, lift_constant_tensor_pass
from torch.export import ExportedProgram
from torch.fx import GraphModule

from .activation_fusion_pass import ActivationFusionPass
from .aten_to_cortex_m_pass import AtenToCortexMPass
from .clamp_hardswish_pass import ClampHardswishPass
from .decompose_hardswish_pass import DecomposeHardswishPass
from .decompose_mean_pass import DecomposeMeanPass
from .explicit_layout_pass import (
    CortexMCanonicalizeViewCopyPermutePass,
    CortexMReplaceOpsWithChannelsLastVariants,
    ValidateCortexMExplicitLayoutPass,
)
from .fuse_conv_padding_pass import FuseConvPaddingPass
from .initialize_scratch_buffers_pass import InitializeScratchBuffersPass
from .matmul_to_bmm_pass import MatmulToBmmPass
from .quantized_clamp_activation_pass import QuantizedClampActivationPass
from .replace_quant_nodes_pass import ReplaceQuantNodesPass

PassClass = Type[ExportPass | ExportedProgramPassBase]


class LiftConstantTensorsPass(ExportedProgramPassBase):
    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        # The pass manager shallow-copies programs; lifting mutates shared structures.
        graph = copy.deepcopy(exported_program.graph)
        for original, cloned in zip(exported_program.graph.nodes, graph.nodes):
            cloned.name = original.name
        graph_module = GraphModule(exported_program.graph_module, graph)
        graph_module.meta = exported_program.graph_module.meta.copy()
        exported_program._graph_module = graph_module
        exported_program._graph_signature = copy.deepcopy(
            exported_program.graph_signature
        )
        exported_program._state_dict = exported_program.state_dict.copy()

        buffer_count = len(exported_program.graph_signature.buffers)
        exported_program = lift_constant_tensor_pass(exported_program)
        return ExportedProgramPassResult(
            exported_program,
            len(exported_program.graph_signature.buffers) != buffer_count,
        )


class _CortexMLoweringPass(ExportedProgramPassBase):
    def __init__(
        self, pass_classes: list[PassClass], target_config: CortexMTargetConfig
    ) -> None:
        self.pass_classes = pass_classes
        self.target_config = target_config

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        modified = False
        for pass_cls in self.pass_classes:
            signature = inspect.signature(pass_cls)
            kwargs: dict[str, Any] = {}
            if "exported_program" in signature.parameters:
                kwargs["exported_program"] = exported_program
            if "target_config" in signature.parameters:
                kwargs["target_config"] = self.target_config

            transform_pass = pass_cls(**kwargs)
            transformed = _transform(exported_program, transform_pass)
            modified |= transformed is not exported_program
            exported_program = transformed

        return ExportedProgramPassResult(exported_program, modified)


class CortexMPassManager(ExportedProgramPassManager):
    legacy_pass_list: list[PassClass] = [
        # Run before folding so qparams attach to max_pool2d values, not tuple + getitem.
        RemoveGetItemPass,
        FoldAndAnnotateQParamsPass,
        ReplaceScalarWithTensorArgPass,
        ReplaceQuantNodesPass,
        ActivationFusionPass,
        QuantizedClampActivationPass,
        DecomposeHardswishPass,
        AtenToCortexMPass,
        FuseConvPaddingPass,
        InitializeScratchBuffersPass,
        LiftConstantTensorsPass,
        RemoveUnusedConstantsPass,
    ]

    explicit_layout_pass_list: list[PassClass] = [
        RemoveGetItemPass,
        FoldAndAnnotateQParamsPass,
        ReplaceScalarWithTensorArgPass,
        ActivationFusionPass,
        QuantizedClampActivationPass,
        DecomposeHardswishPass,
        ConvertConv1dToConv2dPass,
        CortexMReplaceOpsWithChannelsLastVariants,
        ReplaceSqueezeAndUnsqueezeWithViewPass,
        # Move layout copies across pads before singleton permutations become views.
        RemovePermutesAroundElementwiseOps,
        CortexMCanonicalizeViewCopyPermutePass,
        ValidateCortexMExplicitLayoutPass,
        ReplaceQuantNodesPass,
        AtenToCortexMPass,
        FuseConvPaddingPass,
        InitializeScratchBuffersPass,
        LiftConstantTensorsPass,
        RemoveUnusedConstantsPass,
    ]

    pass_list = legacy_pass_list

    pass_list_transform_for_annotation: list[Type[ExportPass]] = [
        ScalarsToAttributePass,
        ReplaceScalarWithTensorArgPass,
        ClampHardswishPass,
        DecomposeMeanPass,
        MatmulToBmmPass,
        DeduplicateGetAttrPass,
    ]

    def __init__(
        self,
        exported_program: ExportedProgram | None = None,
        passes: Optional[list[PassClass]] = None,
        target_config: Optional[CortexMTargetConfig] = None,
        use_explicit_layout: bool = False,
    ) -> None:
        """Initialize the Cortex-M pass manager.

        Args:
            exported_program: Optional program for the legacy ``transform()``
                entry point. Omit when using ``edge.transform(pass_manager)``.
            passes: Optional override of the pass list. Defaults to
                the legacy or explicit-layout pass list selected by
                ``use_explicit_layout``.
            target_config: Compilation target for passes that need it.
                Defaults to ``CortexMTargetConfig(cpu=CortexM.M55)``, which
                resolves through cmsis_nn to the MVE backend — matching the
                pre-config historical behaviour.
            use_explicit_layout: Select the experimental explicit-layout pass
                sequence. Legacy lowering remains the default.
        """
        self.exported_program = exported_program
        default_passes = (
            self.explicit_layout_pass_list
            if use_explicit_layout
            else self.legacy_pass_list
        )
        pass_classes = passes if passes is not None else default_passes
        for pass_cls in pass_classes:
            if not isinstance(pass_cls, type):
                raise ValueError(
                    f"{type(self).__name__} expects pass classes, not instances; "
                    f"got {pass_cls!r}"
                )
        self.target_config: CortexMTargetConfig = target_config or CortexMTargetConfig(
            cpu=CortexM.M55
        )
        lowering_passes: list[PassType] = [
            _CortexMLoweringPass(pass_classes, self.target_config)
        ]
        super().__init__(lowering_passes)

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

        result = self(exported_program)
        return result.exported_program if result.modified else exported_program
