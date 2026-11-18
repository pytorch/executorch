# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Demo model for backends/arm/scripts/check_model_support.py.

The model intentionally mixes commonly table-matched operators (add and relu)
with ``torch.sort``, which is absent from the generated Arm backend support
tables at the time this example is added. The demo is therefore expected to
report ``INCONCLUSIVE`` in the fast pre-backend operator-list check. This does
not by itself prove that the selected backend rejects the model.

"""

import torch


class FastSupportDemo(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(x + 1.0)
        values, _ = torch.sort(x, dim=-1)
        return values


ModelUnderTest = FastSupportDemo()
ModelInputs = (torch.randn(1, 8),)
