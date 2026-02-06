# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.arm._passes.rewrite_conv_pass import RewriteConvPass
from executorch.backends.arm.test.misc.test_dw_convs_with_shared_weights import (
    DWConvsModule,
)
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline


def test_rewrite_conv_tosa_FP():
    module = DWConvsModule()
    pipeline = PassPipeline(
        module, module.get_inputs(), passes_with_exported_program=[RewriteConvPass]
    )
    # We can't run TOSA backend dialect operators in eager mode
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()
