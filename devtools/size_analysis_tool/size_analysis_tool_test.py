# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackFloatingPointPartitioner,
)
from executorch.backends.xnnpack.utils.configs import (
    get_xnnpack_executorch_backend_config,
)
from executorch.backends.xnnpack.utils.utils import capture_graph_for_xnnpack

from executorch.devtools.size_analysis_tool.size_analysis_tool import (
    generate_model_size_information,
)
from executorch.exir.backend.backend_api import to_backend, validation_disabled
from executorch.exir.passes.spec_prop_pass import SpecPropPass


class SizeAnalysisToolTest(unittest.TestCase):
    def test_generate_model_size_analysis(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()
                self.conv3d = torch.nn.Conv3d(
                    in_channels=4, out_channels=2, kernel_size=3
                )
                self.conv2d = torch.nn.Conv2d(
                    in_channels=5,
                    out_channels=2,
                    kernel_size=3,
                )
                self.conv_transpose2d = torch.nn.ConvTranspose2d(
                    in_channels=2, out_channels=4, kernel_size=2
                )

            def forward(self, x):
                x = self.sigmoid(x)
                x = self.conv3d(x)
                x = self.conv2d(x)
                x = self.conv_transpose2d(x)
                return x

        mm = MyModel()
        mm.eval()

        test_input = torch.ones(size=(4, 7, 5, 6), dtype=torch.float)

        edge_program = capture_graph_for_xnnpack(mm, (test_input,))
        partitioner = XnnpackFloatingPointPartitioner()

        with validation_disabled():
            delegated_program = edge_program
            delegated_program.exported_program = to_backend(
                edge_program.exported_program, partitioner
            )

        program = delegated_program.to_executorch(
            get_xnnpack_executorch_backend_config([SpecPropPass()]),
        )

        size_information = generate_model_size_information(
            model=program,
            delegate_deserializers=None,
            flatbuffer=program.buffer,
        )

        # Number of Elements -> Other tensor data
        exepected_tensor_data = {
            # Conv3d Weight
            216: {
                "dtype": "float32",
                "element_size": 4,
                "shape": [2, 4, 3, 3, 3],
                "num_bytes": 864,
            },
            # ConvTranspose2d Weight
            32: {
                "dtype": "float32",
                "element_size": 4,
                "shape": [2, 4, 2, 2],
                "num_bytes": 128,
            },
            # ConvTranspose2d Bias
            4: {
                "dtype": "float32",
                "element_size": 4,
                "shape": [4],
                "num_bytes": 16,
            },
            # Conv3d Bias
            2: {
                "dtype": "float32",
                "element_size": 4,
                "shape": [2],
                "num_bytes": 8,
            },
        }

        self.assertEqual(
            len(size_information["tensor_data"]), len(exepected_tensor_data)
        )

        for tensor in size_information["tensor_data"]:
            for k, v in exepected_tensor_data[tensor["numel"]].items():
                self.assertEqual(tensor[k], v)

        # Two delegate blobs: sigmoid and conv2d
        self.assertEqual(len(size_information["delegate_blob_data"]), 2)
