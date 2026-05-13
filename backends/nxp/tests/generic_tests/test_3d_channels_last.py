# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.executorch_pipeline import ModelInputSpec
from executorch.backends.nxp.tests.graph_verifier import (
    BaseGraphVerifier,
    NonDelegatedNode,
)
from executorch.backends.nxp.tests.model_output_comparator import (
    NumericalStatsOutputComparator,
)
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare, ReferenceModel


class Conv1DModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(2, 2, 3)

    def forward(self, x):
        return self.conv(x)


def test_3d_channels_last_dim_order__conv(mocker):
    model = Conv1DModel().eval()

    input_spec = [ModelInputSpec((1, 2, 5), dim_order=torch.channels_last)]

    comparator = NumericalStatsOutputComparator(max_mse_error=8e-3)
    lower_run_compare(
        model,
        input_spec,
        dataset_creator=RandomDatasetCreator(),
        output_comparator=comparator,
        dlg_model_verifier=BaseGraphVerifier(
            1, [NonDelegatedNode("aten_view_copy_default", 2)]
        ),
        mocker=mocker,
        reference_model=ReferenceModel.QUANTIZED_EDGE_PYTHON,
    )
