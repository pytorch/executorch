# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


class OpSequencesAddConv2d(torch.nn.Module):
    """
    Module which include sequences of Memory Format sensitive ops. forward runs
    [num_sequences] sequences of [ops_per_sequences] ops. Each sequence is
    followed by an add to separate the sequences
    """

    def __init__(self, num_sequences, ops_per_sequence):
        super().__init__()
        self.num_ops = num_sequences * ops_per_sequence
        self.num_sequences = num_sequences

        self.op_sequence = torch.nn.ModuleList()
        for _ in range(num_sequences):
            inner = torch.nn.ModuleList()
            for _ in range(ops_per_sequence):
                inner.append(
                    torch.nn.Conv2d(
                        in_channels=1,
                        out_channels=1,
                        kernel_size=(3, 3),
                        padding=1,
                        bias=False,
                    )
                )
            self.op_sequence.append(inner)

    def forward(self, x):
        for seq in self.op_sequence:
            for op in seq:
                x = op(x)
            x = x + x
        return x + x
