# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch import exir
from executorch.exir.backend.runtime_info_schema import RuntimeInfo
from executorch.exir.backend.test.demos.rpc.executor_backend_preprocess import (
    ExecutorBackend,
)
from executorch.exir.tests.models import MLP


class TestBackends(unittest.TestCase):
    def test_compatibility(self):
        mlp = MLP()
        example_inputs = mlp.get_random_inputs()
        exported_program = torch.export.export(mlp, example_inputs)
        edge_program = exir.to_edge(
            exported_program,
            compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
        )

        # Preprocess the subgraph and get the processed bytes
        preprocess_result = ExecutorBackend.preprocess(
            edge_program.exported_program(), []
        )

        def runtime_version(compatible):
            return b"ET_12" if compatible else b"ET_13"

        def binary_version(compatible):
            return b"0" if compatible else b"1"

        for compatible_runtime_version in (True, False):
            for compatible_binary_version in (True, False):
                runtime_info = [
                    RuntimeInfo(
                        "runtime_version", runtime_version(compatible_runtime_version)
                    ),
                    RuntimeInfo(
                        "supported_binary_version",
                        binary_version(compatible_binary_version),
                    ),
                ]
                result = ExecutorBackend.is_compatible(
                    preprocess_result.processed_bytes, [], runtime_info
                )
                self.assertEqual(
                    result, compatible_runtime_version and compatible_binary_version
                )
