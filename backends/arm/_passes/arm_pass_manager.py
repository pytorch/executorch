# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from executorch.backends.arm._passes.annotate_channels_last_dim_order_pass import (
    AnnotateChannelsLastDimOrder,
)
from executorch.backends.arm._passes.convert_expand_copy_to_repeat import (
    ConvertExpandCopyToRepeatPass,
)
from executorch.backends.arm._passes.convert_split_to_slice import (
    ConvertSplitToSlicePass,
)
from executorch.backends.arm._passes.meandim_to_averagepool_pass import (
    ConvertMeanDimToAveragePool,
)
from executorch.backends.arm._passes.remove_clone_pass import RemoveClonePass
from executorch.backends.arm._passes.size_adjust_conv2d_pass import SizeAdjustConv2DPass
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.pass_manager import PassManager


class ArmPassManager(PassManager):

    def _transform(self, graph_module: torch.fx.GraphModule):
        return self(graph_module).graph_module

    def transform_to_backend_pipeline(
        self, graph_module: torch.fx.GraphModule, compile_spec: list[CompileSpec]
    ):
        """Apply passes before transforming program to backend"""
        self.add_pass(SizeAdjustConv2DPass())
        self.add_pass(RemoveClonePass())
        self.add_pass(ConvertExpandCopyToRepeatPass())
        self.add_pass(ConvertMeanDimToAveragePool())
        self.add_pass(ConvertSplitToSlicePass())
        for spec in compile_spec:
            if spec.key == "permute_memory_format":
                memory_format = spec.value.decode()
                if memory_format == "nhwc":
                    self.add_pass(AnnotateChannelsLastDimOrder())

        return self._transform(graph_module)
