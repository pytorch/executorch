# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.arm._passes import (
    DecomposeSoftmaxUnstablePass,
    FuseDuplicateUsersPass,
)
from executorch.backends.arm._passes.arm_pass_manager import ArmPassManager
from executorch.backends.arm.common.pipeline_config import ArmPassPipelineConfig
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.specification import TosaSpecification


def test_pipeline_config_override_outside_compile_spec():
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
