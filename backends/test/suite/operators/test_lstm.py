# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch
from executorch.backends.test.suite.flow import TestFlow

from executorch.backends.test.suite.operators import (
    dtype_test,
    operator_test,
    OperatorTest,
)
from torch.nn.quantizable.modules.rnn import LSTM as QuantizableLSTM


def _get_lstm_cls(use_quantizable_lstm: bool):
    return QuantizableLSTM if use_quantizable_lstm else torch.nn.LSTM


class Model(torch.nn.Module):
    def __init__(
        self,
        input_size=64,
        hidden_size=32,
        num_layers=1,
        bias=True,
        batch_first=True,
        dropout=0.0,
        bidirectional=False,
        use_quantizable_lstm: bool = False,
    ):
        super().__init__()
        lstm_cls = _get_lstm_cls(use_quantizable_lstm)
        self.lstm = lstm_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        return self.lstm(x)[0]  # Return only the output, not the hidden states


@operator_test
class LSTM(OperatorTest):
    @dtype_test
    def test_lstm_dtype(self, flow: TestFlow, dtype) -> None:
        use_quantizable_lstm = flow.quantize
        self._test_op(
            Model(num_layers=2, use_quantizable_lstm=use_quantizable_lstm).to(dtype),
            ((torch.rand(1, 10, 64) * 10).to(dtype),),  # (batch=1, seq_len, input_size)
            flow,
        )

    @dtype_test
    def test_lstm_no_bias_dtype(self, flow: TestFlow, dtype) -> None:
        use_quantizable_lstm = flow.quantize
        self._test_op(
            Model(
                num_layers=2, bias=False, use_quantizable_lstm=use_quantizable_lstm
            ).to(dtype),
            ((torch.rand(1, 10, 64) * 10).to(dtype),),
            flow,
        )

    def test_lstm_feature_sizes(self, flow: TestFlow) -> None:
        use_quantizable_lstm = flow.quantize
        self._test_op(
            Model(
                input_size=32,
                hidden_size=16,
                use_quantizable_lstm=use_quantizable_lstm,
            ),
            (torch.randn(1, 8, 32),),  # (batch=1, seq_len, input_size)
            flow,
        )
        self._test_op(
            Model(
                input_size=128,
                hidden_size=64,
                use_quantizable_lstm=use_quantizable_lstm,
            ),
            (torch.randn(1, 12, 128),),
            flow,
        )
        self._test_op(
            Model(
                input_size=256,
                hidden_size=128,
                use_quantizable_lstm=use_quantizable_lstm,
            ),
            (torch.randn(1, 6, 256),),
            flow,
        )
        self._test_op(
            Model(
                input_size=16,
                hidden_size=32,
                use_quantizable_lstm=use_quantizable_lstm,
            ),
            (torch.randn(1, 5, 16),),
            flow,
        )

    def test_lstm_batch_sizes(self, flow: TestFlow) -> None:
        use_quantizable_lstm = flow.quantize
        self._test_op(
            Model(use_quantizable_lstm=use_quantizable_lstm),
            (torch.randn(8, 10, 64),),
            flow,
        )
        self._test_op(
            Model(use_quantizable_lstm=use_quantizable_lstm),
            (torch.randn(32, 10, 64),),
            flow,
        )
        self._test_op(
            Model(use_quantizable_lstm=use_quantizable_lstm),
            (torch.randn(100, 10, 64),),
            flow,
        )

    def test_lstm_seq_lengths(self, flow: TestFlow) -> None:
        use_quantizable_lstm = flow.quantize
        self._test_op(
            Model(use_quantizable_lstm=use_quantizable_lstm),
            (torch.randn(1, 5, 64),),
            flow,
        )
        self._test_op(
            Model(use_quantizable_lstm=use_quantizable_lstm),
            (torch.randn(1, 20, 64),),
            flow,
        )
        self._test_op(
            Model(use_quantizable_lstm=use_quantizable_lstm),
            (torch.randn(1, 50, 64),),
            flow,
        )

    def test_lstm_batch_first_false(self, flow: TestFlow) -> None:
        use_quantizable_lstm = flow.quantize
        self._test_op(
            Model(batch_first=False, use_quantizable_lstm=use_quantizable_lstm),
            (torch.randn(10, 1, 64),),  # (seq_len, batch=1, input_size)
            flow,
        )

    def test_lstm_num_layers(self, flow: TestFlow) -> None:
        use_quantizable_lstm = flow.quantize
        self._test_op(
            Model(num_layers=2, use_quantizable_lstm=use_quantizable_lstm),
            (torch.randn(1, 10, 64),),
            flow,
        )
        self._test_op(
            Model(num_layers=3, use_quantizable_lstm=use_quantizable_lstm),
            (torch.randn(1, 10, 64),),
            flow,
        )

    def test_lstm_bidirectional(self, flow: TestFlow) -> None:
        use_quantizable_lstm = flow.quantize
        self._test_op(
            Model(bidirectional=True, use_quantizable_lstm=use_quantizable_lstm),
            (torch.randn(1, 10, 64),),
            flow,
        )

    def test_lstm_with_dropout(self, flow: TestFlow) -> None:
        # Note: Dropout is only effective with num_layers > 1
        use_quantizable_lstm = flow.quantize
        self._test_op(
            Model(num_layers=2, dropout=0.2, use_quantizable_lstm=use_quantizable_lstm),
            (torch.randn(1, 10, 64),),
            flow,
        )

    def test_lstm_with_initial_states(self, flow: TestFlow) -> None:
        # Create a model that accepts initial states
        class ModelWithStates(torch.nn.Module):
            def __init__(self, use_quantizable_lstm: bool = False):
                super().__init__()
                lstm_cls = _get_lstm_cls(use_quantizable_lstm)
                self.lstm = lstm_cls(
                    input_size=64,
                    hidden_size=32,
                    num_layers=2,
                    batch_first=True,
                )

            def forward(self, x, h0, c0):
                return self.lstm(x, (h0, c0))[0]  # Return only the output

        batch_size = 1
        num_layers = 2
        hidden_size = 32
        use_quantizable_lstm = flow.quantize

        self._test_op(
            ModelWithStates(use_quantizable_lstm=use_quantizable_lstm),
            (
                torch.randn(batch_size, 10, 64),  # input
                torch.randn(num_layers, batch_size, hidden_size),  # h0
                torch.randn(num_layers, batch_size, hidden_size),  # c0
            ),
            flow,
        )

    def test_lstm_return_hidden_states(self, flow: TestFlow) -> None:
        # Create a model that returns both output and hidden states
        class ModelWithHiddenStates(torch.nn.Module):
            def __init__(self, use_quantizable_lstm: bool = False):
                super().__init__()
                lstm_cls = _get_lstm_cls(use_quantizable_lstm)
                self.lstm = lstm_cls(
                    input_size=64,
                    hidden_size=32,
                    num_layers=2,
                    batch_first=True,
                )

            def forward(self, x):
                # Return the complete output tuple: (output, (h_n, c_n))
                output, (h_n, c_n) = self.lstm(x)
                return output, h_n, c_n

        batch_size = 1
        seq_len = 10
        input_size = 64
        use_quantizable_lstm = flow.quantize

        self._test_op(
            ModelWithHiddenStates(use_quantizable_lstm=use_quantizable_lstm),
            (torch.randn(batch_size, seq_len, input_size),),
            flow,
        )
