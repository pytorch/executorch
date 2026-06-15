# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import ast
import logging
import re

import numpy as np
import pytest
import torch
from executorch.backends.nxp.tests.graph_verifier import BaseGraphVerifier
from executorch.backends.nxp.tests.model_output_comparator import (
    NumericalStatsOutputComparator,
)

from executorch.backends.nxp.tests.models import AvgPool2dModule, SoftmaxModule
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare

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


class ParallelPoolModel(torch.nn.Module):
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


class TestProfiling:
    @pytest.mark.xfail(reason="SoftMax support PR is not merged so far.", strict=True)
    def test__softmax(self, caplog):
        caplog.set_level(logging.INFO)
        model = SoftmaxModule(-1)
        lower_run_compare(
            model,
            (10,),
            dlg_model_verifier=BaseGraphVerifier(1, []),
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

    def test__parallel_pool(self, caplog):
        caplog.set_level(logging.INFO)
        input_shape = (1, 3, 32, 32)
        model = ParallelPoolModel(input_shape[1])
        lower_run_compare(
            model,
            input_shape,
            dlg_model_verifier=BaseGraphVerifier(1, []),
            output_comparator=NumericalStatsOutputComparator(),
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
    def test__cifar(self, caplog):
        caplog.set_level(logging.INFO)
        input_shape = (1, 3, 32, 32)
        model = CifarNetModel()
        lower_run_compare(
            model,
            input_shape,
            dlg_model_verifier=BaseGraphVerifier(1, []),
            output_comparator=NumericalStatsOutputComparator(),
            use_profiling=True,
        )
        neutron_map = extract_map_from_logs(caplog)
        assert neutron_map == {
            0: (),  # Pad
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

    def test__avg_pool(self, caplog):
        caplog.set_level(logging.INFO)
        input_shape = (2, 9, 6, 15)
        model = AvgPool2dModule(False, 0)
        lower_run_compare(
            model,
            input_shape,
            dlg_model_verifier=BaseGraphVerifier(1, []),
            output_comparator=NumericalStatsOutputComparator(),
            use_profiling=True,
        )
        neutron_map = extract_map_from_logs(caplog)
        assert neutron_map == {
            0: (2,),  # Pad
            1: (2,),  # Conv2DDepthwiseDense
            2: (2,),  # Slice
            3: (),  # Neutron Dump
        }
