# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import copy
import os
import shutil
import tempfile
import unittest

import torch
from executorch.devtools import BundledProgram
from executorch.devtools.etrecord import generate_etrecord, parse_etrecord
from executorch.devtools.inspector import Inspector
from executorch.exir import EdgeCompileConfig, EdgeProgramManager, to_edge
from executorch.exir.capture._config import CaptureConfig
from executorch.exir.program import ExecutorchProgram
from torch.export import export, ExportedProgram
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)

# 定义一个简单的模型用于测试
class SimpleAddModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y


class TestDevtoolsEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.etrecord_path = os.path.join(self.tmp_dir, "etrecord.bin")
        self.etdump_path = os.path.join(self.tmp_dir, "etdump.bin")

        self.model = SimpleAddModel()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def generate_etrecord(self):
        aten_model: ExportedProgram = export(
            self.model,
            (torch.randn(1, 1, 32, 32),),
        )
        edge_program_manager = to_edge(
            aten_model,
            compile_config=EdgeCompileConfig(
                _use_edge_ops=False,
                _check_ir_validity=False,
            ),
        )
        edge_program_manager_copy = copy.deepcopy(edge_program_manager)
        et_program_manager = edge_program_manager.to_executorch()

        generate_etrecord(self.etrecord_path, edge_program_manager_copy, et_program_manager)

    def generate_bundled_program(self):
        method_name = "forward"
        method_graphs = {method_name: export(self.model, (torch.randn(1, 1, 32, 32),))}

        inputs = [torch.randn(1, 1, 32, 32)]
        method_test_suites = [
            MethodTestSuite(
                method_name=method_name,
                test_cases=[MethodTestCase(inputs=inp, expected_outputs=self.model(inp)) for inp in inputs],
            )
        ]
        
        executorch_program = to_edge(method_graphs).to_executorch()
        bundled_program = BundledProgram(
            executorch_program=executorch_program,
            method_test_suites=method_test_suites,
        )

        return bundled_program