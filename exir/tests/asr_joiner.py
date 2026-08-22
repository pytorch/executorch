# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn


class ASRJoiner(nn.Module):
    """
    ASR joiner implementation following the code in https://fburl.com/code/ierfau7c
    Have a local implementation has the benefit that we don't need pull in the
    heavy dependencies and wait for a few minutes to run tests.
    """

    def __init__(self, B=1, H=10, T=1, U=1, D=768) -> None:
        """
        B: source batch size
        H: number of hypotheses for beam search
        T: source sequence length
        U: target sequence length
        D: encoding (some sort of embedding?) dimension
        """
        super().__init__()
        self.B, self.H, self.T, self.U, self.D = B, H, T, U, D
        # The module looks like:
        # SequentialContainer(
        #   (module_list): ModuleList(
        #     (0): ReLULayer(inplace=False)
        #     (1): LinearLayer(input_dim=768, output_dim=4096, bias=True, context_dim=0, pruning_aware_training=False, parameter_noise=0.1, qat_qconfig=None, freeze_rex_pattern=None)
        #   )
        # )
        self.module = nn.Sequential(
            nn.ReLU(),
            nn.Linear(D, 4096),
        )

    def forward(self, src_encodings, src_lengths, tgt_encodings, tgt_lengths):
        """
        One simplification we make here is we assume src_encodings and tgt_encodings
        are not None. In the originally implementation, either can be None.
        """
        H = tgt_encodings.shape[0] // src_encodings.shape[0]
        B = src_encodings.shape[0]
        new_order = (
            (torch.arange(B).view(-1, 1).repeat(1, H).view(-1))
            .long()
            .to(device=src_encodings.device)
        )
        # src_encodings: (B, T, D) -> (B*H, T, D)
        src_encodings = torch.index_select(
            src_encodings, dim=0, index=new_order
        ).contiguous()

        # src_lengths: (B,) -> (B*H,)
        src_lengths = torch.index_select(
            src_lengths, dim=0, index=new_order.to(device=src_lengths.device)
        )

        # src_encodings: (B*H, T, D) -> (B*H, T, 1, D)
        src_encodings = src_encodings.unsqueeze(dim=2).contiguous()

        # tgt_encodings: (B*H, U, D) -> (B*H, 1, U, D)
        tgt_encodings = tgt_encodings.unsqueeze(dim=1).contiguous()

        # joint_encodings: (B*H, T, U, D)
        joint_encodings = src_encodings + tgt_encodings

        output = F.log_softmax(self.module(joint_encodings), dim=-1)

        return output, src_lengths, tgt_lengths

    def get_random_inputs(self):
        return (
            torch.rand(self.B, self.T, self.D),
            torch.randint(0, 10, (self.B,)),
            torch.rand(self.B * self.H, self.U, self.D),
            torch.randint(0, 10, (self.B,)),
        )
