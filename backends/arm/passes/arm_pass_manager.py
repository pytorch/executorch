# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.passes.remove_clone_pass import RemoveClone
from executorch.backends.arm.passes.tag_io_quant_pass import TagIOQuant
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.pass_manager import PassManager


class ArmPassManager(PassManager):

    def _transform(self, graph_module: torch.fx.Graph):
        return self(graph_module).graph_module

    def transform_partition_pipeline(
        self, graph_module: torch.fx.Graph, compile_spec: CompileSpec
    ):
        """Apply passes before partitioning"""
        for spec in compile_spec:
            if spec.key == "quantize_io":
                output_format = spec.value.decode()
                if output_format == "True":
                    self.add_pass(TagIOQuant())

        return self._transform(graph_module)

    def transform_to_backend_pipeline(
        self, graph_module: torch.fx.Graph, compile_spec: CompileSpec
    ):
        """Apply passes before transforming program to backend"""
        self.add_pass(RemoveClone())
        for spec in compile_spec:
            if spec.key == "permute_memory_format":
                memory_format = spec.value.decode()
                if memory_format == "nhwc":
                    # self.add_pass(PermuteMemoryFormatPass)
                    pass

        return self._transform(graph_module)
