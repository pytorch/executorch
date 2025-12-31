# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import math
import unittest

import kgb
import numpy as np
import torch
from executorch.backends.nxp.aten_passes.move_activation_before_concat import (
    MoveActivationBeforeConcat,
)
from executorch.backends.nxp.aten_passes.neutron_aten_pass_manager import (
    NeutronAtenPassManager,
)
from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from executorch.backends.nxp.quantizer.utils import calibrate_and_quantize
from executorch.backends.nxp.tests.executorch_pipeline import (
    get_random_calibration_inputs,
    neutron_target_spec,
    to_model_input_spec,
    to_quantized_edge_program,
)
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    graph_contains_any_of_ops,
    ToChannelFirstPreprocess,
    ToChannelLastPreprocess,
)
from executorch.backends.nxp.tests.models import get_activation
from executorch.exir.dialects._ops import ops as exir_ops
from parameterized import parameterized
from torch import nn
from torch.export import ExportedProgram
from torch.fx import GraphModule

concat_cluster_ops = [
    exir_ops.edge.aten.addmm.default,
    exir_ops.edge.aten.convolution.default,
    exir_ops.edge.aten.hardtanh.default,
    exir_ops.edge.aten.relu.default,
    exir_ops.edge.aten.sigmoid.default,
    exir_ops.edge.aten.tanh.default,
    exir_ops.edge.aten.cat.default,
]


# Permutation of all supported combinations of:
# <activation>, <is_inplace>, <use_qat>
all_activation_cases = list(
    itertools.product(
        ["relu", "relu6", "tanh"],
        [True, False],
        [True, False],
    )
) + [
    ("sigmoid", False, True),
    ("sigmoid", False, False),
]


# <activation1>, <activation2>, <act1_inplace>, <act2_inplace>, <use_qat>
all_concat_cluster_cases = [
    ("relu", "relu", True, False, True),
    ("relu", "relu", True, False, False),
    ("relu6", "relu6", False, True, True),
    ("relu6", "relu6", False, True, False),
    ("tanh", "tanh", True, False, True),
    ("tanh", "tanh", True, False, False),
    ("sigmoid", "sigmoid", False, True, True),
    ("sigmoid", "sigmoid", False, True, False),
    ("relu", "relu_hardtanh", True, True, True),
    ("relu", "relu_hardtanh", True, True, False),
]


class ConvConcatActivationModule(torch.nn.Module):
    def __init__(self, activation: str, inplace: bool, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            (3, 3),
            padding=1,
        )

        self.activation = get_activation(activation, inplace)
        self.eval()

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv(x)
        x = torch.cat((x1, x2), dim=1)
        return self.activation(x)


class LinearConcatActivationModule(nn.Module):
    def __init__(
        self, activation: str, inplace: bool, in_channels: int, mode: str = "linear"
    ):
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in [
            "linear",
            "addmm",
            "mm",
        ], "Mode must be 'linear', 'addmm', or 'mm'"

        if self.mode == "linear":
            self.linear = nn.Linear(in_channels, in_channels)
        else:
            # Manual weight and bias for addmm/mm.
            self.weight = nn.Parameter(torch.empty(in_channels, in_channels))
            self.bias = nn.Parameter(torch.empty(in_channels))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        self.activation = get_activation(activation, inplace)
        self.eval()

    def forward(self, x):
        x1, x2 = None, None

        if self.mode == "linear":
            x1 = self.linear(x)
            x2 = self.linear(x)
        if self.mode == "addmm":
            x1 = torch.addmm(self.bias, x, self.weight)
            x2 = torch.addmm(self.bias, x, self.weight)
        elif self.mode == "mm":
            x1 = torch.mm(x, self.weight)
            x2 = torch.mm(x, self.weight)

        x = torch.cat((x1, x2), dim=1)
        return self.activation(x)


class ConvActivationConcatModule(torch.nn.Module):
    def __init__(
        self,
        activation1: str,
        activation2: str,
        act1_inplace: bool,
        act2_inplace: bool,
        in_channels: int,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            (3, 3),
            padding=1,
        )

        self.activation1 = get_activation(activation1, act1_inplace)
        self.activation2 = get_activation(activation2, act2_inplace)
        self.eval()

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.activation1(x1)
        x2 = self.conv(x)
        x2 = self.activation2(x2)
        return torch.cat((x1, x2), dim=1)


class LinearActivationConcatModule(torch.nn.Module):
    def __init__(
        self,
        activation1: str,
        activation2: str,
        act1_inplace: bool,
        act2_inplace: bool,
        in_channels: int,
    ):
        super().__init__()
        self.linear = nn.Linear(in_channels, in_channels)

        self.activation1 = get_activation(activation1, act1_inplace)
        self.activation2 = get_activation(activation2, act2_inplace)
        self.eval()

    def forward(self, x):
        x1 = self.linear(x)
        x1 = self.activation1(x1)
        x2 = self.linear(x)
        x2 = self.activation2(x2)
        return torch.cat((x1, x2), dim=1)


class TestMoveActivationBeforeConcat(unittest.TestCase):
    __test__ = False  # Prevent interfering with PyTest tests.

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(23)
        np.random.seed(42)

    @parameterized.expand(all_activation_cases)
    def test_move_activation_before_concat__conv(self, activation, inplace, is_qat):
        input_shape = (1, 3, 8, 8)
        model = ConvConcatActivationModule(
            activation=activation, inplace=inplace, in_channels=3
        )

        calibration_inputs = get_random_calibration_inputs(
            to_model_input_spec(input_shape)
        )
        example_input = calibration_inputs[0]

        exir_program_aten = torch.export.export(
            model, example_input, strict=True
        ).module()

        outputs_before = [o.detach().numpy() for o in exir_program_aten(*example_input)]
        nodes = list(exir_program_aten.graph.nodes)
        assert len(nodes) == 8
        cat_node = nodes[5]
        assert cat_node.target == torch.ops.aten.cat.default
        assert all(
            neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__aten(
                input_node
            )
            and len(input_node.users) == 1
            for input_node in cat_node.all_input_nodes
        )
        assert (
            neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
                nodes[6]
            )
        )

        # Apply the optimization.
        NeutronAtenPassManager(
            neutron_target_spec,
            [MoveActivationBeforeConcat(neutron_target_spec)],
        )(exir_program_aten)

        nodes = list(exir_program_aten.graph.nodes)

        # Make sure the optimization was applied.
        assert len(nodes) == 9
        cat_node = nodes[7]
        assert cat_node.target == torch.ops.aten.cat.default
        assert all(
            neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
                input_node
            )
            and len(input_node.users) == 1
            for input_node in cat_node.all_input_nodes
        )
        assert nodes[8].target == "output"

        outputs_after = [o.detach().numpy() for o in exir_program_aten(*example_input)]

        # Make sure the model still produces the exact same output.
        assert np.allclose(outputs_before[0], outputs_after[0])

        # Run pre-processing passes of the float32 aten dialect program.
        neutron_aten_pass_manager = NeutronAtenPassManager(neutron_target_spec)
        neutron_aten_pass_manager(exir_program_aten)  # All passes by default.

        exir_program_aten_quant = calibrate_and_quantize(
            exir_program_aten,
            calibration_inputs,
            NeutronQuantizer(neutron_target_spec),
            is_qat=is_qat,
        )

        # Check convolution and activation are in same QDQ cluster.
        nodes = list(exir_program_aten_quant.graph.nodes)
        assert len(nodes) == 26
        assert neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__aten(
            nodes[14]
        )
        assert (
            neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
                nodes[15]
            )
        )
        assert (
            nodes[16].target
            == torch.ops.quantized_decomposed.quantize_per_tensor.default
        )
        assert neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__aten(
            nodes[18]
        )
        assert (
            neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
                nodes[19]
            )
        )
        assert (
            nodes[20].target
            == torch.ops.quantized_decomposed.quantize_per_tensor.default
        )

    @parameterized.expand(all_activation_cases)
    def test_move_activation_before_concat__linear(self, activation, inplace, is_qat):
        input_shape = (1, 8)
        model = LinearConcatActivationModule(
            activation=activation, inplace=inplace, in_channels=8, mode="linear"
        )

        calibration_inputs = get_random_calibration_inputs(
            to_model_input_spec(input_shape)
        )
        example_input = calibration_inputs[0]

        exir_program_aten = torch.export.export(
            model, example_input, strict=True
        ).module()

        outputs_before = [o.detach().numpy() for o in exir_program_aten(*example_input)]
        nodes = list(exir_program_aten.graph.nodes)
        assert len(nodes) == 8
        cat_node = nodes[5]
        assert cat_node.target == torch.ops.aten.cat.default
        assert all(
            neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__aten(
                input_node
            )
            and len(input_node.users) == 1
            for input_node in cat_node.all_input_nodes
        )
        assert (
            neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
                nodes[6]
            )
        )

        # Apply the optimization.
        NeutronAtenPassManager(
            neutron_target_spec,
            [MoveActivationBeforeConcat(neutron_target_spec)],
        )(exir_program_aten)

        nodes = list(exir_program_aten.graph.nodes)

        # Make sure the optimization was applied.
        assert len(nodes) == 9
        cat_node = nodes[7]
        assert cat_node.target == torch.ops.aten.cat.default
        assert all(
            neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
                input_node
            )
            and len(input_node.users) == 1
            for input_node in cat_node.all_input_nodes
        )
        assert nodes[8].target == "output"

        outputs_after = [o.detach().numpy() for o in exir_program_aten(*example_input)]

        # Make sure the model still produces the exact same output.
        assert np.allclose(outputs_before[0], outputs_after[0])

        # Run pre-processing passes of the float32 aten dialect program.
        neutron_aten_pass_manager = NeutronAtenPassManager(neutron_target_spec)
        neutron_aten_pass_manager(exir_program_aten)  # All passes by default.

        exir_program_aten_quant = calibrate_and_quantize(
            exir_program_aten,
            calibration_inputs,
            NeutronQuantizer(neutron_target_spec),
            is_qat=is_qat,
        )

        # Check linear and activation are in same QDQ cluster.
        nodes = list(exir_program_aten_quant.graph.nodes)
        assert len(nodes) == 22
        assert neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__aten(
            nodes[10]
        )
        assert (
            neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
                nodes[11]
            )
        )
        assert (
            nodes[12].target
            == torch.ops.quantized_decomposed.quantize_per_tensor.default
        )
        assert neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__aten(
            nodes[14]
        )
        assert (
            neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
                nodes[15]
            )
        )
        assert (
            nodes[16].target
            == torch.ops.quantized_decomposed.quantize_per_tensor.default
        )

    @parameterized.expand(all_activation_cases)
    def test_move_activation_before_concat__addmm(self, activation, inplace, is_qat):
        input_shape = (1, 8)
        model = LinearConcatActivationModule(
            activation=activation, inplace=inplace, in_channels=8, mode="addmm"
        )

        calibration_inputs = get_random_calibration_inputs(
            to_model_input_spec(input_shape)
        )
        example_input = calibration_inputs[0]

        exir_program_aten = torch.export.export(
            model, example_input, strict=True
        ).module()

        outputs_before = [o.detach().numpy() for o in exir_program_aten(*example_input)]
        nodes = list(exir_program_aten.graph.nodes)
        assert len(nodes) == 8
        cat_node = nodes[5]
        assert cat_node.target == torch.ops.aten.cat.default
        assert all(
            neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__aten(
                input_node
            )
            and len(input_node.users) == 1
            for input_node in cat_node.all_input_nodes
        )
        assert (
            neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
                nodes[6]
            )
        )

        # Apply the optimization.
        NeutronAtenPassManager(
            neutron_target_spec,
            [MoveActivationBeforeConcat(neutron_target_spec)],
        )(exir_program_aten)

        nodes = list(exir_program_aten.graph.nodes)

        # Make sure the optimization was applied.
        assert len(nodes) == 9
        cat_node = nodes[7]
        assert cat_node.target == torch.ops.aten.cat.default
        assert all(
            neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
                input_node
            )
            and len(input_node.users) == 1
            for input_node in cat_node.all_input_nodes
        )
        assert nodes[8].target == "output"

        outputs_after = [o.detach().numpy() for o in exir_program_aten(*example_input)]

        # Make sure the model still produces the exact same output.
        assert np.allclose(outputs_before[0], outputs_after[0])

        # Run pre-processing passes of the float32 aten dialect program.
        neutron_aten_pass_manager = NeutronAtenPassManager(neutron_target_spec)
        neutron_aten_pass_manager(exir_program_aten)  # All passes by default.

        exir_program_aten_quant = calibrate_and_quantize(
            exir_program_aten,
            calibration_inputs,
            NeutronQuantizer(neutron_target_spec),
            is_qat=is_qat,
        )

        # Check addmm and activation are in same QDQ cluster.
        nodes = list(exir_program_aten_quant.graph.nodes)
        assert len(nodes) == 22
        assert neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__aten(
            nodes[10]
        )
        assert (
            neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
                nodes[11]
            )
        )
        assert (
            nodes[12].target
            == torch.ops.quantized_decomposed.quantize_per_tensor.default
        )
        assert neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__aten(
            nodes[14]
        )
        assert (
            neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
                nodes[15]
            )
        )
        assert (
            nodes[16].target
            == torch.ops.quantized_decomposed.quantize_per_tensor.default
        )

    @parameterized.expand(all_activation_cases)
    def test_move_activation_before_concat__mm(self, activation, inplace, is_qat):
        input_shape = (1, 8)
        model = LinearConcatActivationModule(
            activation=activation, inplace=inplace, in_channels=8, mode="mm"
        )

        calibration_inputs = get_random_calibration_inputs(
            to_model_input_spec(input_shape)
        )
        example_input = calibration_inputs[0]

        exir_program_aten = torch.export.export(
            model, example_input, strict=True
        ).module()

        outputs_before = [o.detach().numpy() for o in exir_program_aten(*example_input)]
        nodes = list(exir_program_aten.graph.nodes)
        assert len(nodes) == 7
        cat_node = nodes[4]
        assert cat_node.target == torch.ops.aten.cat.default
        assert all(
            neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__aten(
                input_node
            )
            and len(input_node.users) == 1
            for input_node in cat_node.all_input_nodes
        )
        assert (
            neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
                nodes[5]
            )
        )

        # Apply the optimization.
        NeutronAtenPassManager(
            neutron_target_spec,
            [MoveActivationBeforeConcat(neutron_target_spec)],
        )(exir_program_aten)

        nodes = list(exir_program_aten.graph.nodes)

        # Make sure the optimization was applied.
        assert len(nodes) == 8
        cat_node = nodes[6]
        assert cat_node.target == torch.ops.aten.cat.default
        assert all(
            neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
                input_node
            )
            and len(input_node.users) == 1
            for input_node in cat_node.all_input_nodes
        )
        assert nodes[7].target == "output"

        outputs_after = [o.detach().numpy() for o in exir_program_aten(*example_input)]

        # Make sure the model still produces the exact same output.
        assert np.allclose(outputs_before[0], outputs_after[0])

        # Run pre-processing passes of the float32 aten dialect program.
        neutron_aten_pass_manager = NeutronAtenPassManager(neutron_target_spec)
        neutron_aten_pass_manager(exir_program_aten)  # All passes by default.

        exir_program_aten_quant = calibrate_and_quantize(
            exir_program_aten,
            calibration_inputs,
            NeutronQuantizer(neutron_target_spec),
            is_qat=is_qat,
        )

        # Check mm and activation are in same QDQ cluster.
        nodes = list(exir_program_aten_quant.graph.nodes)
        assert len(nodes) == 19
        assert neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__aten(
            nodes[7]
        )
        assert (
            neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
                nodes[8]
            )
        )
        assert (
            nodes[9].target
            == torch.ops.quantized_decomposed.quantize_per_tensor.default
        )
        assert neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__aten(
            nodes[11]
        )
        assert (
            neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
                nodes[12]
            )
        )
        assert (
            nodes[13].target
            == torch.ops.quantized_decomposed.quantize_per_tensor.default
        )

    @parameterized.expand(all_activation_cases)
    def test_move_activation_before_concat_quantization__conv(
        self, activation, inplace, use_qat
    ):
        with kgb.spy_on(
            EdgeProgramToIRConverter.convert_program,
            call_original=True,
            owner=EdgeProgramToIRConverter,
        ) as converter_spy:
            input_shape = (1, 8, 8, 8)
            model = ConvConcatActivationModule(
                activation=activation, inplace=inplace, in_channels=8
            )

            edge_program = to_quantized_edge_program(
                model,
                input_shape,
                use_qat=use_qat,
                use_neutron_for_format_conversion=False,
            ).exported_program()

            # Make sure that all nodes were delegated.
            assert not graph_contains_any_of_ops(
                graph=edge_program.graph, ops=concat_cluster_ops
            )
            assert any(
                "lowered_module" in node.name for node in edge_program.graph.nodes
            )

            tflite_flatbuffers_model, io_formats = converter_spy.calls[-1].return_value
            exported_program: ExportedProgram = converter_spy.calls[-1].args[0]
            input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(
                np.int8
            )
            convert_run_compare(
                exported_program,
                input_data,
                tfl_model=tflite_flatbuffers_model,
                tflite_input_preprocess=ToChannelLastPreprocess(),
                tflite_output_preprocess=ToChannelFirstPreprocess(),
            )

    @parameterized.expand(all_activation_cases)
    def test_move_activation_before_concat_quantization__linear(
        self, activation, inplace, use_qat
    ):
        with kgb.spy_on(
            EdgeProgramToIRConverter.convert_program,
            call_original=True,
            owner=EdgeProgramToIRConverter,
        ) as converter_spy:
            input_shape = (1, 8)
            model = LinearConcatActivationModule(
                activation=activation, inplace=inplace, in_channels=8, mode="linear"
            )

            edge_program = to_quantized_edge_program(
                model, input_shape, use_qat=use_qat
            ).exported_program()

            # Make sure that all nodes were delegated.
            assert not graph_contains_any_of_ops(
                graph=edge_program.graph, ops=concat_cluster_ops
            )
            assert any(
                "lowered_module" in node.name for node in edge_program.graph.nodes
            )

            tflite_flatbuffers_model, io_formats = converter_spy.calls[-1].return_value
            exported_program: ExportedProgram = converter_spy.calls[-1].args[0]
            input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(
                np.int8
            )
            convert_run_compare(
                exported_program,
                input_data,
                tfl_model=tflite_flatbuffers_model,
            )

    @parameterized.expand(all_activation_cases)
    def test_move_activation_before_concat_quantization__addmm(
        self, activation, inplace, use_qat
    ):
        torch.manual_seed(23)
        with kgb.spy_on(
            EdgeProgramToIRConverter.convert_program,
            call_original=True,
            owner=EdgeProgramToIRConverter,
        ) as converter_spy:
            input_shape = (1, 8)
            model = LinearConcatActivationModule(
                activation=activation, inplace=inplace, in_channels=8, mode="addmm"
            )

            edge_program = to_quantized_edge_program(
                model, input_shape, use_qat=use_qat
            ).exported_program()

            # Make sure that all nodes were delegated.
            assert not graph_contains_any_of_ops(
                graph=edge_program.graph, ops=concat_cluster_ops
            )
            assert any(
                "lowered_module" in node.name for node in edge_program.graph.nodes
            )

            tflite_flatbuffers_model, io_formats = converter_spy.calls[-1].return_value
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

    @parameterized.expand(all_activation_cases)
    def test_move_activation_before_concat_quantization__mm(
        self, activation, inplace, use_qat
    ):
        with kgb.spy_on(
            EdgeProgramToIRConverter.convert_program,
            call_original=True,
            owner=EdgeProgramToIRConverter,
        ) as converter_spy:
            input_shape = (1, 8)
            model = LinearConcatActivationModule(
                activation=activation, inplace=inplace, in_channels=8, mode="mm"
            )

            edge_program = to_quantized_edge_program(
                model, input_shape, use_qat=use_qat
            ).exported_program()

            # Make sure that all nodes were delegated.
            assert not graph_contains_any_of_ops(
                graph=edge_program.graph, ops=concat_cluster_ops
            )
            assert any(
                "lowered_module" in node.name for node in edge_program.graph.nodes
            )

            tflite_flatbuffers_model, io_formats = converter_spy.calls[-1].return_value
            exported_program: ExportedProgram = converter_spy.calls[-1].args[0]
            input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(
                np.int8
            )
            convert_run_compare(
                exported_program,
                input_data,
                tfl_model=tflite_flatbuffers_model,
            )

    @parameterized.expand(all_concat_cluster_cases)
    def test_concat_cluster_quantization__conv(
        self, activation1, activation2, act1_inplace, act2_inplace, use_qat
    ):
        with kgb.spy_on(
            EdgeProgramToIRConverter.convert_program,
            call_original=True,
            owner=EdgeProgramToIRConverter,
        ) as converter_spy:
            with kgb.spy_on(
                calibrate_and_quantize, call_original=True
            ) as quantizer_spy:
                input_shape = (1, 8, 8, 8)
                model = ConvActivationConcatModule(
                    activation1, activation2, act1_inplace, act2_inplace, in_channels=8
                )

                edge_program = to_quantized_edge_program(
                    model,
                    input_shape,
                    use_qat=use_qat,
                    use_neutron_for_format_conversion=False,
                ).exported_program()

                # Make sure that all nodes were delegated.
                assert not graph_contains_any_of_ops(
                    graph=edge_program.graph,
                    ops=concat_cluster_ops,
                )
                assert any(
                    "lowered_module" in node.name for node in edge_program.graph.nodes
                )

                tflite_flatbuffers_model, io_formats = converter_spy.calls[
                    -1
                ].return_value
                exported_program: ExportedProgram = converter_spy.calls[-1].args[0]
                exir_program_aten_quant: GraphModule = quantizer_spy.calls[
                    -1
                ].return_value

                # Check convolution and activation are in same QDQ cluster.
                nodes = list(exir_program_aten_quant.graph.nodes)
                assert len(nodes) == 26
                assert neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__aten(
                    nodes[14]
                )
                assert neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
                    nodes[15]
                )
                assert (
                    nodes[16].target
                    == torch.ops.quantized_decomposed.quantize_per_tensor.default
                )
                assert neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__aten(
                    nodes[18]
                )
                assert neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
                    nodes[19]
                )
                assert (
                    nodes[20].target
                    == torch.ops.quantized_decomposed.quantize_per_tensor.default
                )

                input_data = (
                    np.random.random(input_shape).astype(np.float32) * 50
                ).astype(np.int8)
                convert_run_compare(
                    exported_program,
                    input_data,
                    tfl_model=tflite_flatbuffers_model,
                    tflite_input_preprocess=ToChannelLastPreprocess(),
                    tflite_output_preprocess=ToChannelFirstPreprocess(),
                )

    @parameterized.expand(all_concat_cluster_cases)
    def test_concat_cluster_quantization__linear(
        self, activation1, activation2, act1_inplace, act2_inplace, use_qat
    ):
        with kgb.spy_on(
            EdgeProgramToIRConverter.convert_program,
            call_original=True,
            owner=EdgeProgramToIRConverter,
        ) as converter_spy:
            with kgb.spy_on(
                calibrate_and_quantize, call_original=True
            ) as quantizer_spy:
                input_shape = (1, 8)
                model = LinearActivationConcatModule(
                    activation1, activation2, act1_inplace, act2_inplace, in_channels=8
                )

                edge_program = to_quantized_edge_program(
                    model, input_shape, use_qat=use_qat
                ).exported_program()

                # Make sure that all nodes were delegated.
                assert not graph_contains_any_of_ops(
                    graph=edge_program.graph,
                    ops=concat_cluster_ops,
                )
                assert any(
                    "lowered_module" in node.name for node in edge_program.graph.nodes
                )

                tflite_flatbuffers_model, io_formats = converter_spy.calls[
                    -1
                ].return_value
                exported_program: ExportedProgram = converter_spy.calls[-1].args[0]
                exir_program_aten_quant: GraphModule = quantizer_spy.calls[
                    -1
                ].return_value

                # Check linear and activation are in same QDQ cluster.
                nodes = list(exir_program_aten_quant.graph.nodes)
                assert len(nodes) == 22
                assert neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__aten(
                    nodes[10]
                )
                assert neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
                    nodes[11]
                )
                assert (
                    nodes[12].target
                    == torch.ops.quantized_decomposed.quantize_per_tensor.default
                )
                assert neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__aten(
                    nodes[14]
                )
                assert neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
                    nodes[15]
                )
                assert (
                    nodes[16].target
                    == torch.ops.quantized_decomposed.quantize_per_tensor.default
                )

                input_data = (
                    np.random.random(input_shape).astype(np.float32) * 50
                ).astype(np.int8)
                convert_run_compare(
                    exported_program,
                    input_data,
                    tfl_model=tflite_flatbuffers_model,
                    tflite_input_preprocess=ToChannelLastPreprocess(),
                    tflite_output_preprocess=ToChannelFirstPreprocess(),
                )
