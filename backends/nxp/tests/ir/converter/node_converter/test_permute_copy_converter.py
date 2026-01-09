# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import kgb
import numpy as np
import torch

from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    graph_contains_any_of_ops,
)
from executorch.backends.nxp.tests.models import Conv2dModule
from executorch.exir.dialects._ops import ops as exir_ops
from parameterized import parameterized
from torch.export import ExportedProgram
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


class Conv2dTransposeModule(torch.nn.Module):
    def __init__(self, in_channels: int, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        self.conv = Conv2dModule(
            in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        return torch.transpose(x, self.dim0, self.dim1)


class Conv2dPermuteModule(torch.nn.Module):
    def __init__(self, in_channels: int, perm: tuple[int, ...]):
        super().__init__()
        self.perm = perm
        self.conv = Conv2dModule(
            in_channels=in_channels,
            out_channels=in_channels,
            stride=1,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        x = self.conv(x)
        return torch.permute(x, self.perm)


class PermuteConv2dModule(torch.nn.Module):
    def __init__(self, in_channels: int, perm: tuple[int, ...]):
        super().__init__()
        self.perm = perm
        self.conv = Conv2dModule(
            in_channels=in_channels,
            out_channels=in_channels,
            stride=1,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        x = torch.permute(x, self.perm)
        return self.conv(x)


class PermuteConv2dPermuteModule(torch.nn.Module):
    def __init__(
        self, in_channels: int, perm1: tuple[int, ...], perm2: tuple[int, ...]
    ):
        super().__init__()
        self.perm1 = perm1
        self.perm2 = perm2
        self.conv = Conv2dModule(
            in_channels=in_channels,
            out_channels=in_channels,
            stride=1,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        x = torch.permute(x, self.perm1)
        x = self.conv(x)
        x = torch.permute(x, self.perm2)
        return x


class LinearPermuteModule(torch.nn.Module):
    def __init__(self, in_features: int, perm: tuple[int, ...]):
        super().__init__()
        self.perm = perm
        self.fc = torch.nn.Linear(in_features, in_features)

    def forward(self, x):
        x = self.fc(x)
        return torch.permute(x, self.perm)


class TestPermuteCopyConversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(23)
        np.random.seed(42)

    @parameterized.expand(
        [
            ["QAT; To channel first permutation", (1, 16, 8, 8), (0, 3, 1, 2), True],
            ["PTQ; To channel first permutation", (1, 16, 8, 8), (0, 3, 1, 2), False],
            ["QAT; To channel last permutation", (1, 16, 8, 8), (0, 2, 3, 1), True],
            ["PTQ; To channel last permutation", (1, 16, 8, 8), (0, 2, 3, 1), False],
        ]
    )
    def test_permute_copy_conversion__from_permute_4D__quantized__channels_first_input(
        self, _: str, input_shape, perm, use_qat
    ):
        with kgb.spy_on(
            EdgeProgramToIRConverter.convert_program, call_original=True
        ) as converter_spy:
            model = Conv2dPermuteModule(input_shape[1], perm)

            # Run conversion
            edge_program = to_quantized_edge_program(
                model, input_shape, use_qat=use_qat
            ).exported_program()

            # Make sure the `Permute_copy` was delegated.
            assert not graph_contains_any_of_ops(
                graph=edge_program.graph, ops=[exir_ops.edge.aten.permute_copy.default]
            )
            assert any(
                "lowered_module" in node.name for node in edge_program.graph.nodes
            )

            # Capture generated model
            tflite_flatbuffers_model, io_formats = converter_spy.calls[-1].return_value

            # Capture converted program
            exported_program: ExportedProgram = converter_spy.calls[-1].args[0]

            input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(
                np.int8
            )

            convert_run_compare(
                exported_program,
                input_data,
                tfl_model=tflite_flatbuffers_model,
                atol=1.0,
            )

    @parameterized.expand(
        [
            ["QAT; To channel first permutation", (1, 8, 8, 8), (0, 3, 1, 2), True],
            ["PTQ; To channel first permutation", (1, 8, 8, 8), (0, 3, 1, 2), False],
            ["QAT; To channel last permutation", (1, 8, 8, 8), (0, 2, 3, 1), True],
            ["PTQ; To channel last permutation", (1, 8, 8, 8), (0, 2, 3, 1), False],
        ]
    )
    def test_permute_copy_conversion__from_permute_4D__quantized__channels_first_output(
        self, _: str, input_shape, perm, use_qat
    ):
        with kgb.spy_on(
            EdgeProgramToIRConverter.convert_program, call_original=True
        ) as converter_spy:
            model = PermuteConv2dModule(input_shape[1], perm)

            # Run conversion
            edge_program = to_quantized_edge_program(
                model, input_shape, use_qat=use_qat
            ).exported_program()

            # Make sure the `Permute_copy` was delegated.
            assert not graph_contains_any_of_ops(
                graph=edge_program.graph, ops=[exir_ops.edge.aten.permute_copy.default]
            )
            assert any(
                "lowered_module" in node.name for node in edge_program.graph.nodes
            )

            # Capture generated model
            tflite_flatbuffers_model, io_formats = converter_spy.calls[-1].return_value

            # Capture converted program
            exported_program: ExportedProgram = converter_spy.calls[-1].args[0]

            input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(
                np.int8
            )

            convert_run_compare(
                exported_program,
                input_data,
                tfl_model=tflite_flatbuffers_model,
                atol=1.0,
            )

    @parameterized.expand(
        [
            [
                "QAT; nchw->nhwc ... nchw->nhwc",
                (1, 8, 8, 8),
                (0, 2, 3, 1),
                (0, 2, 3, 1),
                True,
            ],
            [
                "PTQ; nchw->nhwc ... nchw->nhwc",
                (1, 8, 8, 8),
                (0, 2, 3, 1),
                (0, 2, 3, 1),
                False,
            ],
            [
                "QAT; nchw->nhwc ... nhwc->nchw",
                (1, 8, 8, 8),
                (0, 2, 3, 1),
                (0, 3, 1, 2),
                True,
            ],
            [
                "PTQ; nchw->nhwc ... nhwc->nchw",
                (1, 8, 8, 8),
                (0, 2, 3, 1),
                (0, 3, 1, 2),
                False,
            ],
            [
                "QAT; nhwc->nchw ... nhwc->nchw",
                (1, 8, 8, 8),
                (0, 3, 1, 2),
                (0, 3, 1, 2),
                True,
            ],
            [
                "PTQ; nhwc->nchw ... nhwc->nchw",
                (1, 8, 8, 8),
                (0, 3, 1, 2),
                (0, 3, 1, 2),
                False,
            ],
            [
                "QAT; nhwc->nchw ... nchw->nhwc",
                (1, 8, 8, 8),
                (0, 3, 1, 2),
                (0, 2, 3, 1),
                True,
            ],
            [
                "PTQ; nhwc->nchw ... nchw->nhwc",
                (1, 8, 8, 8),
                (0, 3, 1, 2),
                (0, 2, 3, 1),
                False,
            ],
        ]
    )
    def test_permute_copy_conversion__from_permute_4D__quantized__channels_first_io(
        self, _: str, input_shape, perm1, perm2, use_qat
    ):
        with kgb.spy_on(
            EdgeProgramToIRConverter.convert_program, call_original=True
        ) as converter_spy:
            model = PermuteConv2dPermuteModule(input_shape[1], perm1, perm2)

            # Run conversion
            edge_program = to_quantized_edge_program(
                model, input_shape, use_qat=use_qat
            ).exported_program()

            # Make sure the `Permute_copy` was delegated.
            assert not graph_contains_any_of_ops(
                graph=edge_program.graph, ops=[exir_ops.edge.aten.permute_copy.default]
            )
            assert any(
                "lowered_module" in node.name for node in edge_program.graph.nodes
            )

            # Capture generated model
            tflite_flatbuffers_model, io_formats = converter_spy.calls[-1].return_value

            # Capture converted program
            exported_program: ExportedProgram = converter_spy.calls[-1].args[0]

            input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(
                np.int8
            )

            convert_run_compare(
                exported_program,
                input_data,
                tfl_model=tflite_flatbuffers_model,
                atol=1.0,
            )

    @parameterized.expand(
        [
            [
                "QAT; Permutation can be replaced by reshapes",
                (10, 1, 8),
                (0, 2, 1),
                True,
            ],
            [
                "PTQ; Permutation can be replaced by reshapes",
                (10, 1, 8),
                (0, 2, 1),
                False,
            ],
            [
                "QAT; Permutation can be replaced by reshapes",
                (10, 1, 1),
                (2, 1, 0),
                True,
            ],
            [
                "PTQ; Permutation can be replaced by reshapes",
                (10, 1, 1),
                (2, 1, 0),
                False,
            ],
            [
                "QAT; Permutation is identical and can be removed",
                (10, 1, 8),
                (0, 1, 2),
                True,
            ],
            [
                "PTQ; Permutation is identical and can be removed",
                (10, 1, 8),
                (0, 1, 2),
                False,
            ],
        ]
    )
    def test_permute_copy_conversion__from_permute_3D__quantized(
        self, _: str, input_shape, perm, use_qat
    ):
        with kgb.spy_on(
            EdgeProgramToIRConverter.convert_program, call_original=True
        ) as converter_spy:
            # Run conversion
            edge_program = to_quantized_edge_program(
                LinearPermuteModule(input_shape[2], perm), input_shape, use_qat=use_qat
            ).exported_program()

            # Make sure the `Permute_copy` was delegated.
            assert not graph_contains_any_of_ops(
                graph=edge_program.graph, ops=[exir_ops.edge.aten.permute_copy.default]
            )
            assert any(
                "lowered_module" in node.name for node in edge_program.graph.nodes
            )

            # Capture generated model
            tflite_flatbuffers_model, io_formats = converter_spy.calls[-1].return_value

            # Capture converted program
            exported_program: ExportedProgram = converter_spy.calls[-1].args[0]

            input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(
                np.int8
            )

            convert_run_compare(
                exported_program,
                input_data,
                tfl_model=tflite_flatbuffers_model,
                atol=1.0,
            )

    @parameterized.expand(
        [
            ["QAT; Transpose dims 1 and 2", (1, 16, 8, 8), (0, 2, 1, 3), True],
            ["PTQ; Transpose dims 1 and 2", (1, 16, 8, 8), (0, 2, 1, 3), False],
            ["QAT; To (2, 0, 1, 3) permutation", (1, 16, 8, 8), (2, 0, 1, 3), True],
            ["PTQ; To (2, 0, 1, 3) permutation", (1, 16, 8, 8), (2, 0, 1, 3), False],
            ["QAT; To  (3, 1, 2, 0) permutation", (1, 16, 8, 8), (3, 1, 2, 0), True],
            ["PTQ; To  (3, 1, 2, 0) permutation", (1, 16, 8, 8), (3, 1, 2, 0), False],
            ["QAT; To  (3, 1, 0, 2) permutation", (1, 16, 8, 8), (3, 1, 0, 2), True],
            ["PTQ; To  (3, 1, 0, 2) permutation", (1, 16, 8, 8), (3, 1, 0, 2), False],
        ]
    )
    def test_permute_copy_non_delegated_conversion__from_permute_4D__quantized(
        self, _: str, input_shape, perm, use_qat
    ):
        model = Conv2dPermuteModule(input_shape[1], perm)
        edge_program = to_quantized_edge_program(
            model, input_shape, use_qat=use_qat
        ).exported_program()

        nodes = list(edge_program.graph.nodes)
        assert len(nodes) == 8
        assert (
            nodes[5].target == exir_ops.edge.aten.permute_copy.default
        )  # PermuteCopy not delegated.

    @parameterized.expand(
        [
            ["QAT; Transpose dims 1 and 2", (1, 16, 8, 8), 1, 2, True],
            ["PTQ; Transpose dims 1 and 2", (1, 16, 8, 8), 1, 2, False],
            ["QAT; Transpose dims 2 and 3", (1, 16, 8, 8), 2, 3, True],
            ["PTQ; Transpose dims 2 and 3", (1, 16, 8, 8), 2, 3, False],
        ]
    )
    def test_permute_copy_non_delegated_conversion__from_transpose_4D__quantized(
        self, _: str, input_shape, dim0, dim1, use_qat
    ):
        model = Conv2dTransposeModule(input_shape[1], dim0, dim1)
        edge_program = to_quantized_edge_program(
            model, input_shape, use_qat=use_qat
        ).exported_program()

        nodes = list(edge_program.graph.nodes)
        assert len(nodes) == 8
        assert (
            nodes[5].target == exir_ops.edge.aten.permute_copy.default
        )  # PermuteCopy not delegated.
