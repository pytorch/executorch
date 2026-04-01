# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import unittest

import kgb
import numpy as np
import torch

from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters import (
    PermuteCopyConverter,
)
from executorch.backends.nxp.backend.node_format_inference import NodeFormatInference
from executorch.backends.nxp.edge_passes.move_auxiliary_operator_into_separate_qdq_cluster_pass import (
    MoveLeadingAuxiliaryOperatorIntoSeparateQDQClusterPass,
)
from executorch.backends.nxp.edge_passes.neutron_edge_pass_manager import (
    NeutronEdgePassManager,
)
from executorch.backends.nxp.edge_passes.remove_io_quant_ops_pass import (
    RemoveIOQuantOpsPass,
)
from executorch.backends.nxp.neutron_partitioner import QDQClusterRecognizer
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from executorch.backends.nxp.quantizer.utils import calibrate_and_quantize
from executorch.backends.nxp.tests import executors
from executorch.backends.nxp.tests.executorch_pipeline import (
    get_random_calibration_inputs,
    neutron_target_spec,
    to_edge_program,
    to_model_input_spec,
    to_quantized_edge_program,
)
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    graph_contains_any,
    graph_contains_any_of_ops,
    OverrideTargetSupportCheck,
    ToChannelFirstPreprocess,
    ToChannelLastPreprocess,
)
from executorch.exir import EdgeCompileConfig
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.extension.export_util.utils import export_to_edge
from parameterized import parameterized
from torch import nn
from torch.export import ExportedProgram


class SingleConvBlockWithDropout(torch.nn.Module):
    def __init__(
        self, conv_in_channels: int = 3, perform_inplace_dropout: bool = False
    ):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=conv_in_channels, out_channels=64, kernel_size=(4, 4)
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(inplace=perform_inplace_dropout),
        )

    def forward(self, x):
        return self.block(x)


class KWSFinalBlock(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        pool_size = (25, 5)
        self.block = torch.nn.Sequential(
            self.conv_sep_dw(inp=input_shape[1], oup=64),
            nn.Dropout(p=0.4),
            nn.AvgPool2d(kernel_size=pool_size, stride=pool_size),
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=10),
        )

    def conv_sep_dw(self, inp, oup):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=inp, out_channels=inp, kernel_size=3, padding=1, groups=inp
            ),
            nn.BatchNorm2d(num_features=inp, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=oup, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class TransposeReshapeModel(nn.Module):

    def __init__(self, new_shape: list[int]):
        super().__init__()
        self.new_shape = new_shape

    def forward(self, x):
        # `x` should be 4D.

        x = torch.add(x, x)
        x = torch.permute(x, [0, 3, 1, 2])
        # A `clone(memory_format=contiguous)` will be added here during the lowering to edge dialect.
        x = torch.reshape(x, self.new_shape)

        return x


class PermuteCopyReshapeModel(nn.Module):

    def __init__(self, new_shape: list[int], permutation: list[int]):
        super().__init__()
        self.new_shape = new_shape
        self.permutation = permutation

    def forward(self, x):
        # `x` should be 4D.

        x = torch.add(x, x)
        x = torch.permute(x, self.permutation)
        # A `clone(memory_format=contiguous)` will be added here during the lowering to edge dialect.
        x = torch.reshape(x, self.new_shape)
        x = torch.add(x, x)

        return x


class TestCloneConverter(unittest.TestCase):
    __test__ = False  # Prevent interfering with PyTest tests

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(23)
        np.random.seed(23)

    @staticmethod
    def _node_is_clone(node) -> bool:
        clone_ops = [
            exir_ops.edge.aten.clone.default,
            exir_ops.edge.dim_order_ops._clone_dim_order.default,
        ]

        def target_can_be_clone(node):
            if hasattr(node, "op") and node.op == "call_function":
                return "clone" in node.target.__name__

            return False

        return node in clone_ops or target_can_be_clone(node)

    @parameterized.expand(
        list(
            itertools.product(
                [True, False], [(1, 3, 128, 128), (1, 3, 256, 256)], [True, False]
            )
        )
    )
    def test_conv_dropout_quant(
        self, inplace_dropout: bool, input_shape: tuple[int], use_qat: bool
    ):
        model = SingleConvBlockWithDropout(
            conv_in_channels=input_shape[1], perform_inplace_dropout=inplace_dropout
        ).eval()

        with kgb.spy_on(
            EdgeProgramToIRConverter.convert_program,
            call_original=True,
            owner=EdgeProgramToIRConverter,
        ) as converter_spy:
            quantized_program = to_quantized_edge_program(
                model,
                input_shape,
                use_qat=use_qat,
                use_neutron_for_format_conversion=False,
            ).exported_program()

            tflite_flatbuffers_model, _ = converter_spy.calls[-1].return_value
            exported_program: ExportedProgram = converter_spy.calls[-1].args[0]

            assert not graph_contains_any(
                graph=quantized_program.graph,
                condition=TestCloneConverter._node_is_clone,
            )

            input_data = (np.random.random(input_shape) * 50).astype(np.int8)
            convert_run_compare(
                exported_program,
                tfl_model=tflite_flatbuffers_model,
                tflite_input_preprocess=ToChannelLastPreprocess(),
                tflite_output_preprocess=ToChannelFirstPreprocess(),
                input_data=input_data,
                atol=1.0,
            )

    @parameterized.expand(
        list(itertools.product([True, False], [(1, 3, 128, 128), (1, 3, 256, 256)]))
    )
    def test_conv_dropout_no_quant(
        self, inplace_dropout: bool, input_shape: tuple[int]
    ):
        model = SingleConvBlockWithDropout(
            conv_in_channels=input_shape[1], perform_inplace_dropout=inplace_dropout
        ).eval()

        edge_program = to_edge_program(model, input_shape).exported_program()

        has_clone = graph_contains_any_of_ops(
            graph=edge_program.graph,
            ops=[
                exir_ops.edge.aten.clone.default,
                exir_ops.edge.dim_order_ops._clone_dim_order.default,
            ],
        )

        # Clone with inplace=True should not produce clone edge op and vice versa
        assert inplace_dropout ^ has_clone

    @parameterized.expand([("QAT", True), ("PTQ", False)])
    def test_clone_pool_view_copy_quant(
        self, _, use_qat: bool, input_shape: tuple[int] = (1, 64, 25, 5)
    ):
        model = KWSFinalBlock(input_shape).eval()

        with kgb.spy_on(
            EdgeProgramToIRConverter.convert_program,
            call_original=True,
            owner=EdgeProgramToIRConverter,
        ) as converter_spy:
            quantized_program = to_quantized_edge_program(
                model, input_shape, use_qat=use_qat
            ).exported_program()

            tflite_flatbuffers_model, _ = converter_spy.calls[-1].return_value
            exported_program: ExportedProgram = converter_spy.calls[-1].args[0]

            assert not graph_contains_any(
                graph=quantized_program.graph,
                condition=TestCloneConverter._node_is_clone,
            )

            input_data = (np.random.random(input_shape) * 50).astype(np.int8)
            convert_run_compare(
                exported_program,
                tfl_model=tflite_flatbuffers_model,
                tflite_input_preprocess=ToChannelLastPreprocess(),
                input_data=input_data,
                atol=1.0,
            )

    def test_clone__to_contiguous_format(self):
        input_shape = (1, 8, 8, 8)
        new_shape = [1, 32, 2, 8]

        model = TransposeReshapeModel(new_shape).eval()

        calibration_inputs = get_random_calibration_inputs(
            to_model_input_spec(input_shape)
        )

        example_input = calibration_inputs[0]

        exir_program_aten = torch.export.export(model, example_input, strict=True)

        exir_program_aten__module_quant = calibrate_and_quantize(
            exir_program_aten,
            calibration_inputs,
            NeutronQuantizer(neutron_target_spec),
        )

        edge_compile_config = EdgeCompileConfig(_check_ir_validity=False)
        edge_program_manager = export_to_edge(
            exir_program_aten__module_quant,
            example_input,
            edge_compile_config=edge_compile_config,
        )
        # Make sure the `aten.clone` was inserted as expected.
        nodes = list(edge_program_manager.exported_program().graph.nodes)
        assert nodes[9].target == exir_ops.edge.dim_order_ops._clone_dim_order.default
        assert nodes[9].kwargs["dim_order"] == [0, 1, 2, 3]

        # Move the `clone` out of the cluster with the `view_copy`.
        edge_program_manager = edge_program_manager.transform(
            NeutronEdgePassManager(
                [MoveLeadingAuxiliaryOperatorIntoSeparateQDQClusterPass()]
            )
        )

        # Tag QDQ clusters, so the conversion works correctly.
        QDQClusterRecognizer().tag_qdq_clusters(
            list(edge_program_manager.exported_program().graph.nodes)
        )
        edge_program_manager.exported_program().graph_module.recompile()
        edge_program_manager = edge_program_manager.transform(
            [RemoveIOQuantOpsPass(edge_program_manager=edge_program_manager)]
        )

        # Identify the node formats.
        NodeFormatInference(
            edge_program_manager.exported_program()
        ).identify_node_formats()

        # Convert to the IR.
        converted_model, _ = EdgeProgramToIRConverter().convert_program(
            edge_program_manager.exported_program()
        )

        # Make sure the IR version produces the same outputs.
        executors.convert_run_compare(
            edge_program_manager.exported_program(),
            np.random.random_integers(0, 255, input_shape).astype("int8"),
            tfl_model=converted_model,
        )

    def test_clone__to_contiguous_format__non_delegated_permute_copy(self):
        input_shape = (2, 4, 6, 8)
        new_shape = [3, 4, 16, 2]
        permutation = [3, 2, 1, 0]  # Unsupported by default.

        model = PermuteCopyReshapeModel(new_shape, permutation).eval()

        # Prohibit `permute_copy` delegation in case support for the permutation is added in the future.
        def _unsupported_target(*_):
            return False

        with OverrideTargetSupportCheck(
            PermuteCopyConverter, new_target_support_check=_unsupported_target
        ):
            ep = to_quantized_edge_program(model, input_shape).exported_program()

        nodes = list(ep.graph.nodes)
        assert not graph_contains_any_of_ops(
            ep.graph, [exir_ops.edge.aten.clone.default]
        )
        assert nodes[3].name == "executorch_call_delegate"
        assert nodes[5].target == exir_ops.edge.aten.permute_copy.default
        assert nodes[7].name == "executorch_call_delegate_1"
