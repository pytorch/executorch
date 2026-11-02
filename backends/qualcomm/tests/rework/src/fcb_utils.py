# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.qualcomm.export_utils import QcomChipset
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    to_edge_transform_and_lower_to_qnn,
)


class FcbWeightSharingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Conv2d(3, 32, 3)
        self.second = torch.nn.Conv2d(32, 32, 3)

    def forward(self, x):
        return self.second(torch.relu(self.first(x)))


def make_fcb_weight_sharing_model():
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = FcbWeightSharingModel().eval()
    inputs = (torch.linspace(-1, 1, 3 * 16 * 16).reshape(1, 3, 16, 16),)
    return model, inputs


def fcb_target_socs(primary_soc: QcomChipset):
    other_soc = (
        QcomChipset.SM8650 if primary_soc != QcomChipset.SM8650 else QcomChipset.SM8750
    )
    return [primary_soc, other_soc]


def make_fcb_weight_sharing_specs(
    soc_models: list[QcomChipset], fcb_reference_weight_sharing: bool
):
    return generate_qnn_executorch_compiler_spec(
        soc_model=soc_models,
        backend_options=[
            generate_htp_compiler_spec(use_fp16=False, use_weight_sharing=False)
            for _ in soc_models
        ],
        fcb_reference_weight_sharing=fcb_reference_weight_sharing,
    )


def lower_fcb_weight_sharing_model(module: torch.nn.Module, inputs, compiler_specs):
    return to_edge_transform_and_lower_to_qnn(
        module=module,
        inputs=inputs,
        compiler_specs=compiler_specs,
    ).to_executorch()
