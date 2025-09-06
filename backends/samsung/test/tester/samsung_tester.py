# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.
from typing import List, Optional, Tuple

import executorch.backends.test.harness.stages as BaseStages
import torch
from executorch.backends.samsung.partition.enn_partitioner import EnnPartitioner

from executorch.backends.test.harness import Tester as TesterBase
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.backend.backend_details import CompileSpec

from torch.export import ExportedProgram


class Export(BaseStages.Export):
    pass


class Quantize(BaseStages.Quantize):
    pass


class ToEdgeTransformAndLower(BaseStages.ToEdgeTransformAndLower):
    def __init__(
        self,
        compile_specs: Optional[List[CompileSpec]] = None,
        edge_compile_config: Optional[EdgeCompileConfig] = None,
    ):
        compile_specs = compile_specs or []
        self.partitioners = [EnnPartitioner(compile_specs=compile_specs)]
        self.edge_compile_config = edge_compile_config or EdgeCompileConfig(
            _skip_dim_order=True, _check_ir_validity=False
        )
        self.edge_dialect_program = None

    def run(
        self, artifact: ExportedProgram, inputs=None, generate_etrecord: bool = False
    ) -> None:
        self.edge_dialect_program = to_edge_transform_and_lower(
            artifact,
            partitioner=self.partitioners,
            compile_config=self.edge_compile_config,
        )


class ToExecutorch(BaseStages.ToExecutorch):
    pass


class SamsungTester(TesterBase):
    def __init__(
        self,
        module: torch.nn.Module,
        example_inputs: Tuple[torch.Tensor],
        compile_specs: Optional[List[CompileSpec]] = None,
    ):
        module.eval()

        super().__init__(
            module=module,
            example_inputs=example_inputs,
            dynamic_shapes=None,
        )

        self.original_module = module
        self.exported_module = module
        self.example_inputs = example_inputs
        self.compile_specs = compile_specs

    def to_edge_transform_and_lower(
        self,
        edge_compile_config: Optional[EdgeCompileConfig] = None,
    ):
        to_edge_transform_and_lower_stage = ToEdgeTransformAndLower(
            self.compile_specs, edge_compile_config
        )

        return super().to_edge_transform_and_lower(to_edge_transform_and_lower_stage)
