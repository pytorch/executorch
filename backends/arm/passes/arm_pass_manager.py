# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.arm.passes.remove_clone_pass import RemoveClone
from executorch.backends.arm.passes.tag_io_quant_pass import TagIOQuant
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.pass_manager import PassManager
from torch.export import ExportedProgram


class ArmPassManager(PassManager):

    def __init__(self, exported_program: ExportedProgram):
        self.exported_program = exported_program
        super().__init__()

    def _transform(self):
        self.__call__(self.exported_program.graph_module)
        return self.exported_program

    def transform_partition_pipeline(self, compile_spec: CompileSpec):
        """Apply passes before partitioning"""
        for spec in compile_spec:
            if spec.key == "quantize_io":
                output_format = spec.value.decode()
                if output_format == "True":
                    self.add_pass(TagIOQuant(self.exported_program))

        return self._transform()

    def transform_to_backend_pipeline(self, compile_spec: CompileSpec):
        """Apply passes before transforming program to backend"""
        self.add_pass(RemoveClone(self.exported_program))
        for spec in compile_spec:
            if spec.key == "permute_memory_format":
                memory_format = spec.value.decode()
                if memory_format == "nhwc":
                    # self.add_pass(PermuteMemoryFormatPass)
                    pass

        return self._transform()
