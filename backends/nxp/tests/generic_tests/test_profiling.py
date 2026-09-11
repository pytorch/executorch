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

# noinspection PyUnusedImports
import pytest
import torch
from executorch.backends.nxp.tests.graph_verifier import BaseGraphVerifier
from executorch.backends.nxp.tests.model_output_comparator import (
    NumericalStatsOutputComparator,
)
from executorch.backends.nxp.tests.nsys_testing import (
    get_test_name,
    lower_run_compare,
    OUTPUTS_DIR,
)

from executorch.backends.nxp.tests.profiling_utils import (
    get_neutron_compiler_version,
    get_neutron_driver_version,
    get_neutron_kernel_kinds,
)
from executorch.backends.nxp.tests.simple_models import AvgPool2dModule, SoftmaxModule
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

    # Global mapping of Neutron kernel IDs to names used by the delegate metadata parser.
    kernel_kinds = {}

    def parse_delegate_metadata(
        delegate_metadatas: list[bytes],
    ) -> Union[list[str], dict[str, Any]]:
        """Metadata parser for Neutron Backend metadata.

        The parser deserializes delegate metadata and converts kernel IDs into human-readable kernel names when available.
        The deserialized data is then added back to the corresponding event in the event block for user consumption.
        """

        metadata_list = []
        for metadata_bytes in delegate_metadatas:
            if len(metadata_bytes) == 1:
                function_code = metadata_bytes[0]
                if function_code == 0:
                    metadata_list.append("Profiling dump")
                else:
                    metadata_list.append(
                        kernel_kinds.get(
                            function_code, "Neutron kernel " + str(function_code)
                        )
                    )
            elif len(metadata_bytes) == 2:
                metadata_list.append("Profiling dump")
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

    # Validate driver/compiler version compatibility and load kernel names
    # used to decode delegate metadata.
    driver_version = get_neutron_driver_version(etdump_path)
    compiler_version = get_neutron_compiler_version()
    if driver_version:
        assert (
            driver_version == compiler_version
        ), "Driver and compiler versions do not match"
        kernel_kinds = get_neutron_kernel_kinds()

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

    # All numeric delegate events except the last contain either
    # resolved kernel names or fallback "Neutron kernel <id>" metadata.
    for event in numeric_events[:-1]:
        metadata = event.delegate_debug_metadatas
        if kernel_kinds:
            assert "Neutron kernel" not in metadata, (
                f"Event {event.name}: expected kernel kind, " f"got {metadata}"
            )
        else:
            assert "Neutron kernel" in metadata, (
                f"Event {event.name}: expected 'Neutron kernel', " f"got {metadata}"
            )

    # The final numeric event should represent the profiling dump.
    profiling_dump_event = numeric_events[-1]
    profiling_metadata = profiling_dump_event.delegate_debug_metadatas

    assert "Profiling dump" in profiling_metadata, (
        f"Event {profiling_dump_event.name}: "
        f"expected 'Profiling dump', got {profiling_metadata}"
    )

    # Profiling dump event is expected to have no associated operators.
    assert not profiling_dump_event.op_types, (
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
            0: (22, 23),  # Conv2DStandardV2 (Pad + Conv)
            1: (24, 25),  # Conv2DStandardV1 (Pad + Conv)
            2: (26,),  # Conv2DStandardV1
            3: (27,),  # Add
            4: (28,),  # GlobalBiasScale (Relu)
            5: (33,),  # Conv2DStandardV1
            6: (29, 30, 31),  # Conv2DStandardV1 (Pad + Conv)
            7: (32,),  # Conv2DStandardV1
            8: (34,),  # Add
            9: (35,),  # GlobalBiasScale (Relu)
            10: (40,),  # Conv2DStandardV1
            11: (36, 37, 38),  # Conv2DStandardV1 (Pad + Conv)
            12: (39,),  # Conv2DStandardV1
            13: (41,),  # Add
            14: (42,),  # GlobalBiasScale (Relu)
            15: (),  # GlobalAvgPool (Mean)
            16: (45,),  # FullyConnected
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
            0: (22, 23, 24),  # Pad (Pad + Conv + Relu)
            1: (22, 23, 24),  # Conv2DStandardV2 (Pad + Conv + Relu)
            2: (26, 27),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            3: (28, 29),  # Conv2DPointwise (Conv + Relu)
            4: (30, 31),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            5: (32, 33),  # Conv2DDepthwiseV1 (Conv + Relu)
            6: (34, 35),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            7: (36, 37),  # Conv2DPointwise (Conv + Relu)
            8: (38, 39),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            9: (40, 41),  # Conv2DPointwise  (Conv + Relu)
            10: (43,),  # Conv2DDepthwiseDense (AvgPool)
            11: (45,),  # FullyConnected
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
            0: (58, 59, 60),  # Conv2DStandardV2 (Pad + Conv + Relu)
            1: (61, 62),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            2: (63, 64),  # Conv2DPointwise (Conv + Relu)
            3: (65, 66, 67),  # Conv2DDepthwiseV1 (Pad + DepthwiseConv + Relu)
            4: (68, 69),  # Conv2DPointwise (Conv + Relu)
            5: (70, 71),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            6: (72, 73),  # Conv2DPointwise (Conv + Relu)
            7: (74, 75, 76),  # Conv2DDepthwiseV1 (Pad + DepthwiseConv + Relu)
            8: (77, 78),  # Conv2DPointwise (Conv + Relu)
            9: (79, 80),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            10: (81, 82),  # Conv2DPointwise (Conv + Relu)
            11: (83, 84, 85),  # Conv2DDepthwiseV1 (Pad + DepthwiseConv + Relu)
            12: (86, 87),  # Conv2DPointwise (Conv + Relu)
            13: (88, 89),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            14: (90, 91),  # Conv2DPointwise (Conv + Relu)
            15: (92, 93),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            16: (94, 95),  # Conv2DPointwise (Conv + Relu)
            17: (96, 97),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            18: (98, 99),  # Conv2DPointwise (Conv + Relu)
            19: (100, 101),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            20: (102, 103),  # Conv2DPointwise (Conv + Relu)
            21: (104, 105),  # Conv2DDepthwiseV1 (DepthwiseConv + Relu)
            22: (106, 107),  # Conv2DPointwise (Conv + Relu)
            23: (108, 109, 110),  # Conv2DDepthwiseDense (Pad + DepthwiseConv + Relu)
            24: (111, 112),  # Conv2DPointwise (Conv + Relu)
            25: (113, 114),  # Conv2DDepthwiseDense (DepthwiseConv + Relu)
            26: (115, 116),  # Conv2DPointwise (Conv + Relu)
            27: (),  # Mean (GlobalAvgPool)
            28: (119,),  # FullyConnected
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
