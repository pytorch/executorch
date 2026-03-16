# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.arm._passes import (
    DecomposeMaskedFillPass,
    DecomposeSoftmaxPass,
    DecomposeSoftmaxUnstablePass,
    FuseDuplicateUsersPass,
)
from executorch.backends.arm._passes.arm_pass_manager import ArmPassManager
from executorch.backends.arm.common.pipeline_config import (
    ArmPassPipelineConfig,
    SoftmaxDecompositionConfig,
)
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.specification import TosaSpecification


def test_pipeline_config_override_outside_compile_spec_no_target():
    compile_spec = TosaCompileSpec(
        TosaSpecification.create_from_string("TOSA-1.00+INT")
    )
    default_manager = ArmPassManager(compile_spec)
    default_skip_passes = default_manager._skip_pass_types
    assert FuseDuplicateUsersPass not in default_skip_passes
    assert DecomposeSoftmaxUnstablePass in default_skip_passes

    override_compile_spec = TosaCompileSpec(
        TosaSpecification.create_from_string("TOSA-1.00+INT")
    )
    override_config = ArmPassPipelineConfig()
    override_config.disable_fuse_duplicate_users()
    override_compile_spec.set_pass_pipeline_config(override_config)
    override_manager = ArmPassManager(override_compile_spec)
    skip_passes = override_manager._skip_pass_types

    assert FuseDuplicateUsersPass in skip_passes
    assert DecomposeSoftmaxUnstablePass in skip_passes


def test_softmax_config_masked():
    """Test MASKED config: stable softmax, masked fill decomposition enabled."""
    compile_spec = TosaCompileSpec(
        TosaSpecification.create_from_string("TOSA-1.00+INT")
    )
    config = ArmPassPipelineConfig(softmax=SoftmaxDecompositionConfig.MASKED)
    compile_spec.set_pass_pipeline_config(config)
    manager = ArmPassManager(compile_spec)
    skip_passes = manager._skip_pass_types

    # MASKED: skip unstable softmax, use stable softmax
    assert DecomposeSoftmaxUnstablePass in skip_passes
    assert DecomposeSoftmaxPass not in skip_passes
    # MASKED: masked fill decomposition is enabled (not skipped)
    assert DecomposeMaskedFillPass not in skip_passes


def test_softmax_config_unstable():
    """Test UNSTABLE config: unstable softmax, no masked fill decomposition."""
    compile_spec = TosaCompileSpec(
        TosaSpecification.create_from_string("TOSA-1.00+INT")
    )
    config = ArmPassPipelineConfig(softmax=SoftmaxDecompositionConfig.UNSTABLE)
    compile_spec.set_pass_pipeline_config(config)
    manager = ArmPassManager(compile_spec)
    skip_passes = manager._skip_pass_types

    # UNSTABLE: skip stable softmax, use unstable softmax
    assert DecomposeSoftmaxPass in skip_passes
    assert DecomposeSoftmaxUnstablePass not in skip_passes
    # UNSTABLE: masked fill decomposition is disabled (skipped)
    assert DecomposeMaskedFillPass in skip_passes


def test_softmax_config_stable():
    """Test STABLE config: stable softmax, no masked fill decomposition."""
    compile_spec = TosaCompileSpec(
        TosaSpecification.create_from_string("TOSA-1.00+INT")
    )
    config = ArmPassPipelineConfig(softmax=SoftmaxDecompositionConfig.STABLE)
    compile_spec.set_pass_pipeline_config(config)
    manager = ArmPassManager(compile_spec)
    skip_passes = manager._skip_pass_types

    # STABLE: skip unstable softmax, use stable softmax
    assert DecomposeSoftmaxUnstablePass in skip_passes
    assert DecomposeSoftmaxPass not in skip_passes
    # STABLE: masked fill decomposition is disabled (skipped)
    assert DecomposeMaskedFillPass in skip_passes
