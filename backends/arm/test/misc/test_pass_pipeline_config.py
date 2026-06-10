# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.arm._passes import (
    DecomposeMaskedFillPass,
    DecomposeSoftmaxPass,
    FuseDuplicateUsersPass,
)
from executorch.backends.arm._passes.arm_pass_manager import ArmPassManager
from executorch.backends.arm.common.pipeline_config import (
    ArmPassPipelineConfig,
    QuantizeInfConfig,
    SoftmaxDecompositionConfig,
)
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.specification import TosaSpecification
from torch.export import export


class ModuleWithInf(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "mask", torch.tensor([float("inf"), float("-inf")], dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mask  # type: ignore[operator]
        x = torch.ops.aten.add.Tensor(x, float("-inf"))
        x = torch.ops.aten.add.Tensor(x, float("inf"))
        return x


def test_pipeline_config_override_outside_compile_spec():
    compile_spec = TosaCompileSpec(
        TosaSpecification.create_from_string("TOSA-1.00+INT")
    )
    default_manager = ArmPassManager(compile_spec)
    default_skip_passes = default_manager._skip_pass_types
    assert FuseDuplicateUsersPass not in default_skip_passes
    assert DecomposeSoftmaxPass not in default_skip_passes

    override_compile_spec = TosaCompileSpec(
        TosaSpecification.create_from_string("TOSA-1.00+INT")
    )
    override_config = ArmPassPipelineConfig(softmax=SoftmaxDecompositionConfig.STABLE)
    override_compile_spec.set_pass_pipeline_config(override_config)
    override_manager = ArmPassManager(override_compile_spec)
    skip_passes = override_manager._skip_pass_types

    assert FuseDuplicateUsersPass not in skip_passes
    assert DecomposeMaskedFillPass in skip_passes


def test_softmax_config_masked_no_target():
    """Test MASKED config: stable softmax, masked fill decomposition enabled."""
    compile_spec = TosaCompileSpec(
        TosaSpecification.create_from_string("TOSA-1.00+INT")
    )
    config = ArmPassPipelineConfig(softmax=SoftmaxDecompositionConfig.MASKED)
    compile_spec.set_pass_pipeline_config(config)
    manager = ArmPassManager(compile_spec)
    skip_passes = manager._skip_pass_types

    # MASKED: use stable softmax
    assert DecomposeSoftmaxPass not in skip_passes
    # MASKED: masked fill decomposition is enabled (not skipped)
    assert DecomposeMaskedFillPass not in skip_passes


def test_softmax_config_stable_no_target():
    """Test STABLE config: stable softmax, no masked fill decomposition."""
    compile_spec = TosaCompileSpec(
        TosaSpecification.create_from_string("TOSA-1.00+INT")
    )
    config = ArmPassPipelineConfig(softmax=SoftmaxDecompositionConfig.STABLE)
    compile_spec.set_pass_pipeline_config(config)
    manager = ArmPassManager(compile_spec)
    skip_passes = manager._skip_pass_types

    # STABLE: use stable softmax
    assert DecomposeSoftmaxPass not in skip_passes
    # STABLE: masked fill decomposition is disabled (skipped)
    assert DecomposeMaskedFillPass in skip_passes


def test_quant_inf_config_reaches_annotation_pipeline():
    QUANT_NEG_INF = -321.0
    QUANT_POS_INF = 123.0

    config = ArmPassPipelineConfig(
        quantize_inf=QuantizeInfConfig(neg_inf=QUANT_NEG_INF, pos_inf=QUANT_POS_INF),
    )
    compile_spec = TosaCompileSpec(
        TosaSpecification.create_from_string("TOSA-1.00+INT")
    )
    compile_spec.set_pass_pipeline_config(config)
    manager = ArmPassManager(compile_spec)
    exported = export(ModuleWithInf(), (torch.zeros(2),), strict=True)

    transformed = manager.transform_for_annotation_pipeline(exported.graph_module)
    tensor_constant_values = sorted(
        constant.item()
        for name, constant in transformed.named_buffers()
        if name.startswith("_tensor_constant")
    )

    assert tensor_constant_values == [QUANT_NEG_INF, QUANT_POS_INF]
