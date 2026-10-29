# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import struct
import types
import unittest
from unittest import mock

from executorch.backends.mediatek import preprocess
from executorch.exir import to_edge
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.backend.backend_api import to_backend


class _FakeValue:
    dtype = preprocess.torch.float32


class _FakeNode:
    def __init__(self, name: str, op: str, value: object) -> None:
        self.name = name
        self.op = op
        self.meta = {"val": value}


class _FakeEdgeProgram:
    def __init__(self) -> None:
        graph = types.SimpleNamespace(
            nodes=[
                _FakeNode("input", "placeholder", _FakeValue()),
                _FakeNode("result", "call_function", _FakeValue()),
                _FakeNode("output", "output", (_FakeValue(),)),
            ]
        )
        self.graph = graph
        self.graph_module = types.SimpleNamespace(graph=graph)
        self.graph_signature = types.SimpleNamespace(
            user_inputs=["input"], user_outputs=["result"]
        )


class NeuropilotPreprocessTest(unittest.TestCase):
    def test_neuron_adapter_major_rejects_invalid_values(self) -> None:
        invalid_values = [b"", b"0", b"256", b"+9", b"9.0", b" 9", b"\xff"]

        for value in invalid_values:
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "ASCII decimal integer between 1 and 255"
            ):
                preprocess.NeuropilotBackend.preprocess(
                    _FakeEdgeProgram(),
                    [
                        preprocess.CompileSpec("platform-config", b"mt6989"),
                        preprocess.CompileSpec("neuron-adapter-major", value),
                    ],
                )

    def test_neuron_adapter_major_rejects_duplicate_values(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "duplicated keys"):
            preprocess.NeuropilotBackend.preprocess(
                _FakeEdgeProgram(),
                [
                    preprocess.CompileSpec("platform-config", b"mt6989"),
                    preprocess.CompileSpec("neuron-adapter-major", b"8"),
                    preprocess.CompileSpec("neuron-adapter-major", b"9"),
                ],
            )

    def test_existing_converter_call_is_unchanged_without_mlir_export_spec(
        self,
    ) -> None:
        converter = mock.create_autospec(
            preprocess.mtk_converter.PyTorchV2Converter, instance=True
        )
        converter.convert_to_mlir.return_value = "module @main"
        converter_type = mock.Mock()
        converter_type.from_exported_program.return_value = converter

        with mock.patch.object(
            preprocess.mtk_converter, "PyTorchV2Converter", converter_type
        ), mock.patch.object(
            preprocess.mtk_neuron, "compile", return_value=b"compiled"
        ):
            preprocess.NeuropilotBackend.preprocess(
                _FakeEdgeProgram(),
                [preprocess.CompileSpec("platform-config", b"mt6989")],
            )

        converter.convert_to_mlir.assert_called_once_with()

    def test_mlir_export_spec_is_only_forwarded_to_converter(self) -> None:
        converter = mock.create_autospec(
            preprocess.mtk_converter.PyTorchV2Converter, instance=True
        )
        converter.convert_to_mlir.return_value = "module @main"
        converter_type = mock.Mock()
        converter_type.from_exported_program.return_value = converter
        compile_specs = [
            preprocess.CompileSpec("platform-config", b"mt6989"),
            preprocess.CompileSpec("mlir-export-spec", b"example-export-spec"),
            preprocess.CompileSpec("neuron-adapter-major", b"9"),
            preprocess.CompileSpec("ImportForever", b""),
            preprocess.CompileSpec("enable-feature", b""),
            preprocess.CompileSpec("custom-opt", b"value"),
        ]

        with mock.patch.object(
            preprocess.mtk_converter, "PyTorchV2Converter", converter_type
        ), mock.patch.object(
            preprocess.mtk_neuron, "compile", return_value=b"compiled"
        ) as compile_model:
            result = preprocess.NeuropilotBackend.preprocess(
                _FakeEdgeProgram(), compile_specs
            )

        converter.convert_to_mlir.assert_called_once_with(
            mlir_export_spec="example-export-spec"
        )
        compile_model.assert_called_once_with(
            "module @main",
            "--relax-fp32 --opt=3 --platform-config=mt6989 "
            "--enable-feature --custom-opt=value",
        )
        expected = (
            struct.pack("<BIII", preprocess.HEADER_VERSION, 1, 1, len(b"compiled"))
            + b"compiled"
        )
        self.assertEqual(result.processed_bytes, expected)

    def test_neuron_adapter_major_survives_pte_serialization(self) -> None:
        class AddModule(preprocess.torch.nn.Module):
            def forward(self, value):
                return value + value

        converter = mock.create_autospec(
            preprocess.mtk_converter.PyTorchV2Converter, instance=True
        )
        converter.convert_to_mlir.return_value = "module @main"
        converter_type = mock.Mock()
        converter_type.from_exported_program.return_value = converter
        compile_specs = [
            preprocess.CompileSpec("platform-config", b"mt6989"),
            preprocess.CompileSpec("neuron-adapter-major", b"9"),
        ]
        edge_program = to_edge(
            preprocess.torch.export.export(
                AddModule(), (preprocess.torch.ones(1),), strict=True
            )
        )

        with mock.patch.object(
            preprocess.mtk_converter, "PyTorchV2Converter", converter_type
        ), mock.patch.object(
            preprocess.mtk_neuron, "compile", return_value=b"compiled"
        ):
            lowered_module = to_backend(
                preprocess.NeuropilotBackend.__name__,
                edge_program.exported_program(),
                compile_specs,
            )

        program = deserialize_pte_binary(lowered_module.buffer()).program
        serialized_specs = {
            spec.key: spec.value
            for spec in program.execution_plan[0].delegates[0].compile_specs
        }
        self.assertEqual(serialized_specs["neuron-adapter-major"], b"9")
