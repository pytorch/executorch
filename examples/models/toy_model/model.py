# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.backend.compile_spec_schema import CompileSpec

from ..model_base import EagerModelBase


class MulModule(torch.nn.Module, EagerModelBase):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input, other):
        return input * other

    def get_eager_model(self) -> torch.nn.Module:
        return self

    def get_example_inputs(self):
        return (torch.randn(3, 2), torch.randn(3, 2))


class LinearModule(torch.nn.Module, EagerModelBase):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, arg):
        return self.linear(arg)

    def get_eager_model(self) -> torch.nn.Module:
        return self

    def get_example_inputs(self):
        return (torch.randn(3, 3),)


class AddModule(torch.nn.Module, EagerModelBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x + y
        return z

    def get_eager_model(self) -> torch.nn.Module:
        return self

    def get_example_inputs(self):
        return (torch.ones(1), torch.ones(1))


class AddMulModule(torch.nn.Module, EagerModelBase):
    def __init__(self):
        super().__init__()

    def forward(self, a, x, b):
        y = torch.mm(a, x)
        z = torch.add(y, b)
        return z

    def get_eager_model(self) -> torch.nn.Module:
        return self

    def get_example_inputs(self):
        return (torch.ones(2, 2), 2 * torch.ones(2, 2), 3 * torch.ones(2, 2))

    def get_compile_spec(self):
        max_value = self.get_example_inputs()[0].shape[0]
        return [CompileSpec("max_value", bytes([max_value]))]


class SoftmaxModule(torch.nn.Module, EagerModelBase):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        z = self.softmax(x)
        return z

    def get_eager_model(self) -> torch.nn.Module:
        return self

    def get_example_inputs(self):
        return (torch.ones(2, 2),)


class Conv1dModule(torch.nn.Module, EagerModelBase):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )

    def forward(self, x):
        return self.conv1d(x)

    def get_eager_model(self) -> torch.nn.Module:
        return self

    def get_example_inputs(self):
        return (torch.randn(1, 3, 10),)


class SdpaModule(torch.nn.Module, EagerModelBase):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        out = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        return out

    def get_eager_model(self) -> torch.nn.Module:
        return self

    def get_example_inputs(self):
        # Input shape: (batch, num_heads, seq_len, head_dim)
        batch_size = 2
        num_heads = 8
        seq_len = 128
        head_dim = 64
        query = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16
        )
        key = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16
        )
        value = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16
        )
        return (query, key, value)
