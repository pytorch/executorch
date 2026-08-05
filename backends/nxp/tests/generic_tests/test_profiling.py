# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import ast
import logging
import os
import re
from typing import Any, Union

import numpy as np
import pytest
import torch
from executorch.backends.nxp.tests.graph_verifier import BaseGraphVerifier
from executorch.backends.nxp.tests.model_output_comparator import (
    NumericalStatsOutputComparator,
)
from executorch.backends.nxp.tests.models import AvgPool2dModule, SoftmaxModule
from executorch.backends.nxp.tests.nsys_testing import (
    get_test_name,
    lower_run_compare,
    OUTPUTS_DIR,
)

from executorch.devtools.inspector._inspector import Inspector
from executorch.examples.models.mlperf_tiny import (
    DeepAutoEncoder,
    DSCNNKWS,
    MobileNetV1025,
    ResNet8,
)
from executorch.examples.nxp.experimental.cifar_net.cifar_net import CifarNetModel


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


PATTERN_NEUTRON_MAP = r"Neutron to Edge map was created: (\{.*\})"


def extract_map_from_logs(caplog):
    for record in caplog.records:
        msg = record.getMessage()
        neutron_map_match = re.search(PATTERN_NEUTRON_MAP, msg)
        if neutron_map_match:
            dict_str = neutron_map_match.group(1)
            return ast.literal_eval(dict_str)
    return None


def inspector_check(test_name: str) -> None:
    """
    Validate ExecuTorch Inspector profiling output.

    Checks:
      1. Required profiling artifacts (etrecord.bin, trace.etdump) exist.
      2. Inspector can be created and profiling data can be parsed.
      3. All numeric delegate events except the last contain
         "Neutron kernel" metadata.
      4. The last numeric delegate event contains
         "Profiling dump" metadata.
      5. The profiling dump event does not have associated op types.
    """

    def parse_delegate_metadata(
        delegate_metadatas: list[bytes],
    ) -> Union[list[str], dict[str, Any]]:
        """Metadata parser for Neutron Backend metadata.

        The parser is a callable that deserializes the data and returns neutron kernel number.
        The deserialized data is then added back to the corresponding event in the event block for user consumption.
        """

        metadata_list = []
        for metadata_bytes in delegate_metadatas:
            if len(metadata_bytes) == 1:
                function_code = metadata_bytes[0]
                if function_code == 0:
                    metadata_list.append("Profiling dump")
                else:
                    metadata_list.append("Neutron kernel " + str(function_code))
            else:
                metadata_list.append("Invalid metadata size")
        return metadata_list

    npu_results_path = os.path.join(OUTPUTS_DIR, test_name, "results_npu")
    etrecord_path = os.path.join(npu_results_path, "etrecord.bin")
    etdump_path = os.path.join(npu_results_path, "trace.etdump")

    # Verify profiling artifacts were generated.
    for file_path in (etrecord_path, etdump_path):
        assert os.path.isfile(
            file_path
        ), f"Required profiling file does not exist: {file_path}"

    # Create Inspector and parse profiling data.
    try:
        inspector = Inspector(
            etdump_path=etdump_path,
            etrecord=etrecord_path,
            delegate_metadata_parser=parse_delegate_metadata,
        )
        inspector.print_data_tabular(include_delegate_debug_data=True)

    except Exception as e:
        raise RuntimeError(
            "Failed to create or run Inspector for "
            f"etdump='{etdump_path}', "
            f"etrecord='{etrecord_path}'"
        ) from e

    # Collect delegated profiling events whose names are numeric
    # (0, 1, 2, ..., N). These events are emitted by the Neutron backend.
    numeric_events = [
        event
        for event_block in inspector.event_blocks
        for event in event_block.events
        if str(event.name).isdigit()
    ]

    assert numeric_events, "No numeric delegate profiling events found"

    # All delegate events except the last one should describe
    # individual Neutron kernels.
    for event in numeric_events[:-1]:
        metadata = str(event.delegate_debug_metadatas)

        assert "Neutron kernel" in metadata, (
            f"Event {event.name}: expected 'Neutron kernel', " f"got {metadata}"
        )

    # The final numeric event should represent the profiling dump.
    profiling_dump_event = numeric_events[-1]
    profiling_metadata = str(profiling_dump_event.delegate_debug_metadatas)

    assert "Profiling dump" in profiling_metadata, (
        f"Event {profiling_dump_event.name}: "
        f"expected 'Profiling dump', got {profiling_metadata}"
    )

    # Profiling dump event is expected to have no associated operators.
    assert profiling_dump_event.op_types == [], (
        f"Event {profiling_dump_event.name}: expected empty op_types, "
        f"got {profiling_dump_event.op_types}"
    )


class SimpleParallelPoolModel(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_in = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool2d = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv_out = torch.nn.Conv2d(2 * channels, channels, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = torch.cat((self.max_pool2d(x), self.avg_pool2d(x)), dim=1)
        x = self.conv_out(x)
        return x


class ParallelPoolModel(torch.nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(ch)
        self.conv2 = torch.nn.Conv2d(ch, ch, 3, padding=1)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.avgpool = torch.nn.AvgPool2d(2)
        self.conv_out = torch.nn.Conv2d(2 * ch, ch, 1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = x + residual  # residual connection
        x = torch.cat((self.maxpool(x), self.avgpool(x)), dim=1)  # parallel merge
        x = self.conv_out(x)
        return torch.relu(x)


class TestProfiling:
    @pytest.mark.xfail(
        reason="Profiling support for cmodel and SoftMax fix will be available in Neutron SW 3.2.",
        strict=True,
    )
    def test__softmax(self, caplog, request):
        caplog.set_level(logging.INFO)
        model = SoftmaxModule(-1)
        lower_run_compare(
            model,
            (10,),
            dlg_model_verifier=BaseGraphVerifier(1, []),
            request=request,
            use_profiling=True,
            output_comparator=NumericalStatsOutputComparator(),
        )

        # Neuron map for 1D Softmax with input size 10 should contain 4 nodes:
        # 3 Neuron kernels (pad, softmax, and slice) and 1 unmapped node used for profiling dum
        neutron_map = extract_map_from_logs(caplog)
        assert neutron_map == {
            0: (2,),  # Pad
            1: (2,),  # Softmax
            2: (2,),  # Slice
            3: (),  # Neutron Dump
        }

    def test__simple_parallel_pool(self, caplog, request):
        caplog.set_level(logging.INFO)
        input_shape = (1, 3, 32, 32)
        model = SimpleParallelPoolModel(input_shape[1])
        lower_run_compare(
            model,
            input_shape,
            dlg_model_verifier=BaseGraphVerifier(1, []),
            request=request,
            output_comparator=NumericalStatsOutputComparator(),
            use_neutron_for_format_conversion=False,
            use_profiling=True,
        )
        neutron_map = extract_map_from_logs(caplog)
        assert neutron_map == {
            0: (6,),  # Conv2DStandardV2
            1: (),  # Conv2DDepthwiseV2 (AvgPool)
            2: (7,),  # MaxPool
            3: (),  # TransposeCHW
            4: (),  # TransposeCHW
            5: (),  # TransposeCHW
            6: (),  # Slice
            7: (),  # Pad
            8: (),  # Conv2DPointwise
            9: (),  # Slice
            10: (),  # Neutron Dump
        }

    @pytest.mark.xfail(reason="SoftMax support PR is not merged so far.", strict=True)
    def test__cifar(self, caplog, request):
        caplog.set_level(logging.INFO)
        input_shape = (1, 3, 32, 32)
        model = CifarNetModel()
        lower_run_compare(
            model,
            input_shape,
            dlg_model_verifier=BaseGraphVerifier(1, []),
            request=request,
            output_comparator=NumericalStatsOutputComparator(),
            use_neutron_for_format_conversion=False,
            use_profiling=True,
        )
        neutron_map = extract_map_from_logs(caplog)
        assert neutron_map == {
            0: (10,),  # Pad
            1: (10, 11),  # Conv2DStandardV1 (Pad + Conv2d)
            2: (12,),  # MaxPool
            3: (13, 14),  # Conv2DStandardV1 (Pad + Conv2d)
            4: (15,),  # MaxPool
            5: (16, 17),  # Conv2DStandardV1 (Pad + Conv2d)
            6: (18,),  # MaxPool
            7: (20,),  # FullyConnected
            8: (21,),  # Pad
            9: (21,),  # Softmax
            10: (21,),  # Slice
            11: (),  # Neutron Dump
        }
        inspector_check(get_test_name(request))

    def test__avg_pool(self, caplog, request):
        caplog.set_level(logging.INFO)
        input_shape = (2, 9, 6, 15)
        model = AvgPool2dModule(False, 0)
        lower_run_compare(
            model,
            input_shape,
            dlg_model_verifier=BaseGraphVerifier(1, []),
            request=request,
            output_comparator=NumericalStatsOutputComparator(),
            use_neutron_for_format_conversion=False,
            use_profiling=True,
        )
        neutron_map = extract_map_from_logs(caplog)
        assert neutron_map == {
            0: (2,),  # Pad
            1: (2,),  # Conv2DDepthwiseDense
            2: (2,),  # Slice
            3: (),  # Neutron Dump
        }

    def test__parallel_pool(self, caplog, request):
        caplog.set_level(logging.INFO)
        input_shape = (1, 16, 32, 32)
        model = ParallelPoolModel(input_shape[1])
        lower_run_compare(
            model,
            input_shape,
            dlg_model_verifier=BaseGraphVerifier(1, []),
            request=request,
            output_comparator=NumericalStatsOutputComparator(),
            use_neutron_for_format_conversion=False,
            use_profiling=True,
        )
        neutron_map = extract_map_from_logs(caplog)
        assert neutron_map == {
            0: (8, 9),  # Conv2DStandardV1 (Pad + Conv2d)
            1: (10,),  # Conv2DStandardV1
            2: (11,),  # Add
            3: (),  # Conv2DDepthwiseV2 (AvgPool)
            4: (12,),  # MaxPool
            5: (14,),  # StridedSliceConcat
            6: (15, 16),  # Conv2DPointwise (Conv2D + Relu)
            7: (),  # Neutron Dump
        }

    def test__resnet8(self, caplog, request):
        # Three-stage residual network for the MLPerf Tiny image-classification.
        caplog.set_level(logging.INFO)
        model = ResNet8()
        input_shape = (1, 3, 32, 32)

        lower_run_compare(
            model,
            input_shape,
            dlg_model_verifier=BaseGraphVerifier(1, []),
            request=request,
            output_comparator=NumericalStatsOutputComparator(),
            use_neutron_for_format_conversion=False,
            use_profiling=True,
        )
        neutron_map = extract_map_from_logs(caplog)
        assert neutron_map == {
            0: (14, 15),  # Conv2DStandardV2 (Pad + Conv)
            1: (17, 18),  # Conv2DStandardV1 (Pad + Conv)
            2: (20,),  # Conv2DStandardV1
            3: (21,),  # Add
            4: (22,),  # GlobalBiasScale (Relu)
            5: (28,),  # Conv2DStandardV1
            6: (24, 25),  # Conv2DStandardV1 (Pad + Conv)
            7: (27,),  # Conv2DStandardV1
            8: (29,),  # Add
            9: (30,),  # GlobalBiasScale (Relu)
            10: (36,),  # Conv2DStandardV1
            11: (32, 33),  # Conv2DStandardV1 (Pad + Conv)
            12: (35,),  # Conv2DStandardV1
            13: (37,),  # Add
            14: (38,),  # GlobalBiasScale (Relu)
            15: (),  # GlobalAvgPool (Mean)
            16: (41,),  # FullyConnected
            17: (),  # Neutron Dump
        }

    def test__ds_cnn(self, caplog, request):
        # Depthwise Separable CNN used for keyword spotting in MLCommons Tiny.
        caplog.set_level(logging.INFO)
        model = DSCNNKWS()
        input_shape = (1, 1, 49, 10)

        lower_run_compare(
            model,
            input_shape,
            dlg_model_verifier=BaseGraphVerifier(1, []),
            request=request,
            output_comparator=NumericalStatsOutputComparator(),
            use_neutron_for_format_conversion=False,
            use_profiling=True,
        )
        neutron_map = extract_map_from_logs(caplog)
        assert neutron_map == {
            0: (14, 15),  # Pad (Conv + Relu)
            1: (14, 15),  # Conv2DStandardV2 (Conv + Relu)
            2: (18, 19),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            3: (21, 22),  # Conv2DPointwise (Conv + Relu)
            4: (24, 25),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            5: (27, 28),  # Conv2DDepthwiseV1 (Conv + Relu)
            6: (30, 31),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            7: (33, 34),  # Conv2DPointwise (Conv + Relu)
            8: (36, 37),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            9: (39, 40),  # Conv2DPointwise  (Conv + Relu)
            10: (42,),  # Conv2DDepthwiseDense (AvgPool)
            11: (44,),  # FullyConnected
            12: (),  # Neutron Dump
        }

    def test__mobilenet_v1_025(self, caplog, request):
        # MobileNetV1 with width multiplier 0.25 for the Visual Wake Words.
        caplog.set_level(logging.INFO)
        model = MobileNetV1025()
        input_shape = (1, 3, 96, 96)

        lower_run_compare(
            model,
            input_shape,
            dlg_model_verifier=BaseGraphVerifier(1, []),
            request=request,
            output_comparator=NumericalStatsOutputComparator(),
            use_neutron_for_format_conversion=False,
            use_profiling=True,
        )
        neutron_map = extract_map_from_logs(caplog)
        assert neutron_map == {
            0: (32, 33),  # Conv2DStandardV2 (Conv + Relu)
            1: (35, 36),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            2: (38, 39),  # Conv2DPointwise (Conv + Relu)
            3: (41, 42),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            4: (44, 45),  # Conv2DPointwise (Conv + Relu)
            5: (47, 48),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            6: (50, 51),  # Conv2DPointwise (Conv + Relu)
            7: (53, 54),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            8: (56, 57),  # Conv2DPointwise (Conv + Relu)
            9: (59, 60),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            10: (62, 63),  # Conv2DPointwise (Conv + Relu)
            11: (65, 66),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            12: (68, 69),  # Conv2DPointwise (Conv + Relu)
            13: (71, 72),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            14: (74, 75),  # Conv2DPointwise (Conv + Relu)
            15: (77, 78),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            16: (80, 81),  # Conv2DPointwise (Conv + Relu)
            17: (83, 84),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            18: (86, 87),  # Conv2DPointwise (Conv + Relu)
            19: (89, 90),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            20: (92, 93),  # Conv2DPointwise (Conv + Relu)
            21: (95, 96),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            22: (98, 99),  # Conv2DPointwise (Conv + Relu)
            23: (101, 102),  # Conv2DDepthwiseDense (DepthwiseConv + Relu)
            24: (104, 105),  # Conv2DPointwise (Conv + Relu)
            25: (107, 108),  # Conv2DDepthwiseDense (DepthwiseConv + Relu)
            26: (110, 111),  # Conv2DPointwise (Conv + Relu)
            27: (),  # Mean (GlobalAvgPool)
            28: (114,),  # FullyConnected
            29: (),  # Neutron Dump
        }

    def test__deep_autoencoder(self, caplog, request):
        # MLPerf Tiny anomaly detection deep autoencoder.
        caplog.set_level(logging.INFO)
        model = DeepAutoEncoder()
        input_shape = (1, 640)

        lower_run_compare(
            model,
            input_shape,
            dlg_model_verifier=BaseGraphVerifier(1, []),
            request=request,
            output_comparator=NumericalStatsOutputComparator(),
            use_neutron_for_format_conversion=False,
            use_profiling=True,
        )
        neutron_map = extract_map_from_logs(caplog)
        assert neutron_map == {
            0: (22, 23),  # FullyConnected (FullyConnected + Relu)
            1: (24, 25),  # FullyConnected (FullyConnected + Relu)
            2: (26, 27),  # FullyConnected (FullyConnected + Relu)
            3: (28, 29),  # FullyConnected (FullyConnected + Relu)
            4: (30, 31),  # FullyConnected (FullyConnected + Relu)
            5: (32, 33),  # FullyConnected (FullyConnected + Relu)
            6: (34, 35),  # FullyConnected (FullyConnected + Relu)
            7: (36, 37),  # FullyConnected (FullyConnected + Relu)
            8: (38, 39),  # FullyConnected (FullyConnected + Relu)
            9: (40,),  # FullyConnected
            10: (),  # Neutron Dump
        }
