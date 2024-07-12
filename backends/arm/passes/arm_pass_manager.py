# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.passes.remove_clone_pass import RemoveClonePass
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.pass_manager import PassManager


class ArmPassManager(PassManager):

    def _transform(self, graph_module: torch.fx.Graph):
        return self(graph_module).graph_module

    def transform_to_backend_pipeline(
        self, graph_module: torch.fx.Graph, compile_spec: CompileSpec
    ):
        """Apply passes before transforming program to backend"""
        self.add_pass(RemoveClonePass())

        return self._transform(graph_module)
