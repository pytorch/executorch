# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


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
    ):
        super().__init__()
        self.lstm = torch.nn.LSTM(
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


@parameterize_by_dtype
def test_lstm_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(
        Model(num_layers=2).to(dtype),
        ((torch.rand(1, 10, 64) * 10).to(dtype),),  # (batch=1, seq_len, input_size)
    )


@parameterize_by_dtype
def test_lstm_no_bias_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(
        Model(num_layers=2, bias=False).to(dtype),
        ((torch.rand(1, 10, 64) * 10).to(dtype),),
    )


def test_lstm_feature_sizes(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(input_size=32, hidden_size=16),
        (torch.randn(1, 8, 32),),  # (batch=1, seq_len, input_size)
    )
    test_runner.lower_and_run_model(
        Model(input_size=128, hidden_size=64),
        (torch.randn(1, 12, 128),),
    )
    test_runner.lower_and_run_model(
        Model(input_size=256, hidden_size=128),
        (torch.randn(1, 6, 256),),
    )
    test_runner.lower_and_run_model(
        Model(input_size=16, hidden_size=32),
        (torch.randn(1, 5, 16),),
    )


def test_lstm_batch_sizes(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(8, 10, 64),),
    )
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(32, 10, 64),),
    )
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(100, 10, 64),),
    )


def test_lstm_seq_lengths(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(1, 5, 64),),
    )
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(1, 20, 64),),
    )
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(1, 50, 64),),
    )


def test_lstm_batch_first_false(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(batch_first=False),
        (torch.randn(10, 1, 64),),  # (seq_len, batch=1, input_size)
    )


def test_lstm_num_layers(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(num_layers=2),
        (torch.randn(1, 10, 64),),
    )
    test_runner.lower_and_run_model(
        Model(num_layers=3),
        (torch.randn(1, 10, 64),),
    )


def test_lstm_bidirectional(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(bidirectional=True),
        (torch.randn(1, 10, 64),),
    )


def test_lstm_with_dropout(test_runner) -> None:
    # Note: Dropout is only effective with num_layers > 1
    test_runner.lower_and_run_model(
        Model(num_layers=2, dropout=0.2),
        (torch.randn(1, 10, 64),),
    )


def test_lstm_with_initial_states(test_runner) -> None:
    # Create a model that accepts initial states
    class ModelWithStates(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(
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

    test_runner.lower_and_run_model(
        ModelWithStates(),
        (
            torch.randn(batch_size, 10, 64),  # input
            torch.randn(num_layers, batch_size, hidden_size),  # h0
            torch.randn(num_layers, batch_size, hidden_size),  # c0
        ),
    )


def test_lstm_return_hidden_states(test_runner) -> None:
    # Create a model that returns both output and hidden states
    class ModelWithHiddenStates(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(
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

    test_runner.lower_and_run_model(
        ModelWithHiddenStates(),
        (torch.randn(batch_size, seq_len, input_size),),
    )
