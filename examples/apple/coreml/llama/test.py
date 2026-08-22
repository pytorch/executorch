# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

sys.path.insert(0, ".")
import copy

import torch
from utils import replace_linear_with_split_linear


def get_split_model(
    model,
    out_target_split_size=1,
    out_max_splits=1,
    in_target_split_size=1,
    in_max_splits=1,
):
    model_copy = copy.deepcopy(model)
    replace_linear_with_split_linear(
        model_copy,
        out_target_split_size,
        out_max_splits,
        in_target_split_size,
        in_max_splits,
    )
    return model_copy


def test_split_model():
    inputs = torch.randn(10, 5, 1, 512)

    model = torch.nn.Sequential(*[torch.nn.Linear(512, 1024, bias=False)])
    model1 = get_split_model(model, 64, 2, 64, 1000)
    model2 = get_split_model(model, 64, 2, 64, 1)
    model3 = get_split_model(model, 64, 1, 64, 1000)

    assert torch.allclose(model(inputs), model1(inputs), atol=1e-5)
    assert torch.allclose(model(inputs), model2(inputs), atol=1e-5)
    assert torch.allclose(model(inputs), model3(inputs), atol=1e-5)


if __name__ == "__main__":
    test_split_model()
