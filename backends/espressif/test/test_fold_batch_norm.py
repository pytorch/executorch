# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
import torch.nn as nn

from executorch.backends.espressif._passes.fold_batch_norm import (
    count_conv_bn,
    fold_batch_norm,
)

models = pytest.importorskip("torchvision.models")


torch.set_grad_enabled(False)


class ConvBnStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU6(),
            nn.Conv2d(8, 8, 3, padding=1, groups=8),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.features(x)


def _randomize_batch_norm(model: nn.Module) -> None:
    generator = torch.Generator().manual_seed(0)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.running_mean.copy_(
                torch.randn(module.num_features, generator=generator) * 0.1
            )
            module.running_var.copy_(
                torch.rand(module.num_features, generator=generator) + 0.5
            )
            module.weight.copy_(
                torch.rand(module.num_features, generator=generator) + 0.5
            )
            module.bias.copy_(
                torch.randn(module.num_features, generator=generator) * 0.1
            )


def test_folding_removes_batch_norm_and_preserves_outputs():
    model = ConvBnStack().eval()
    _randomize_batch_norm(model)
    x = torch.randn(2, 3, 16, 16)
    expected = model(x)
    folded, count = fold_batch_norm(model, inplace=False)
    assert count == 2
    assert count_conv_bn(folded) == (2, 0)
    assert count_conv_bn(model) == (2, 2)
    assert torch.allclose(folded(x), expected, atol=1e-5)


def test_folding_mobilenet_v2_matches_eager():
    model = models.mobilenet_v2(weights=None).eval()
    _randomize_batch_norm(model)
    convs, norms = count_conv_bn(model)
    folded, count = fold_batch_norm(model, inplace=False)
    assert count == norms > 0
    assert count_conv_bn(folded) == (convs, 0)
    x = torch.randn(1, 3, 96, 96)
    assert torch.allclose(folded(x), model(x), atol=1e-4, rtol=1e-4)


def test_training_mode_is_rejected():
    model = ConvBnStack().train()
    try:
        fold_batch_norm(model)
    except ValueError:
        return
    raise AssertionError("fold_batch_norm accepted a model in training mode")
