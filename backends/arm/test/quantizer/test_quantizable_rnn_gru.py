# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for quantizable RNN and GRU modules.

Verifies that quantizable versions produce outputs matching standard nn.RNN/nn.GRU.
"""

import torch
from executorch.backends.arm.quantizable import GRU, RNN


class TestQuantizableRNN:
    """Tests for the quantizable RNN module."""

    def test_rnn_basic_forward(self):
        """Test basic RNN forward pass matches nn.RNN."""
        input_size, hidden_size, seq_len, batch = 10, 20, 5, 3
        x = torch.randn(seq_len, batch, input_size)

        # Create standard RNN
        std_rnn = torch.nn.RNN(input_size, hidden_size, batch_first=False)

        # Create quantizable RNN and copy weights
        q_rnn = RNN(input_size, hidden_size, batch_first=False)
        q_rnn.layers[0].layer_fw.cell.input_linear.weight.data = std_rnn.weight_ih_l0.data.clone()
        q_rnn.layers[0].layer_fw.cell.input_linear.bias.data = std_rnn.bias_ih_l0.data.clone()
        q_rnn.layers[0].layer_fw.cell.hidden_linear.weight.data = std_rnn.weight_hh_l0.data.clone()
        q_rnn.layers[0].layer_fw.cell.hidden_linear.bias.data = std_rnn.bias_hh_l0.data.clone()

        std_out, std_h = std_rnn(x)
        q_out, q_h = q_rnn(x)

        torch.testing.assert_close(q_out, std_out, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(q_h, std_h, rtol=1e-4, atol=1e-5)

    def test_rnn_bidirectional(self):
        """Test bidirectional RNN matches nn.RNN."""
        input_size, hidden_size, seq_len, batch = 10, 20, 5, 3
        x = torch.randn(seq_len, batch, input_size)

        std_rnn = torch.nn.RNN(input_size, hidden_size, bidirectional=True)

        q_rnn = RNN(input_size, hidden_size, bidirectional=True)
        # Copy forward weights
        q_rnn.layers[0].layer_fw.cell.input_linear.weight.data = std_rnn.weight_ih_l0.data.clone()
        q_rnn.layers[0].layer_fw.cell.input_linear.bias.data = std_rnn.bias_ih_l0.data.clone()
        q_rnn.layers[0].layer_fw.cell.hidden_linear.weight.data = std_rnn.weight_hh_l0.data.clone()
        q_rnn.layers[0].layer_fw.cell.hidden_linear.bias.data = std_rnn.bias_hh_l0.data.clone()
        # Copy backward weights
        q_rnn.layers[0].layer_bw.cell.input_linear.weight.data = (
            std_rnn.weight_ih_l0_reverse.data.clone()
        )
        q_rnn.layers[0].layer_bw.cell.input_linear.bias.data = std_rnn.bias_ih_l0_reverse.data.clone()
        q_rnn.layers[0].layer_bw.cell.hidden_linear.weight.data = (
            std_rnn.weight_hh_l0_reverse.data.clone()
        )
        q_rnn.layers[0].layer_bw.cell.hidden_linear.bias.data = (
            std_rnn.bias_hh_l0_reverse.data.clone()
        )

        std_out, std_h = std_rnn(x)
        q_out, q_h = q_rnn(x)

        torch.testing.assert_close(q_out, std_out, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(q_h, std_h, rtol=1e-4, atol=1e-5)

    def test_rnn_batch_first(self):
        """Test batch_first RNN matches nn.RNN."""
        input_size, hidden_size, seq_len, batch = 10, 20, 5, 3
        x = torch.randn(batch, seq_len, input_size)

        std_rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True)

        q_rnn = RNN(input_size, hidden_size, batch_first=True)
        q_rnn.layers[0].layer_fw.cell.input_linear.weight.data = std_rnn.weight_ih_l0.data.clone()
        q_rnn.layers[0].layer_fw.cell.input_linear.bias.data = std_rnn.bias_ih_l0.data.clone()
        q_rnn.layers[0].layer_fw.cell.hidden_linear.weight.data = std_rnn.weight_hh_l0.data.clone()
        q_rnn.layers[0].layer_fw.cell.hidden_linear.bias.data = std_rnn.bias_hh_l0.data.clone()

        std_out, std_h = std_rnn(x)
        q_out, q_h = q_rnn(x)

        torch.testing.assert_close(q_out, std_out, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(q_h, std_h, rtol=1e-4, atol=1e-5)

    def test_rnn_relu_nonlinearity(self):
        """Test RNN with ReLU nonlinearity."""
        input_size, hidden_size, seq_len, batch = 10, 20, 5, 3
        x = torch.randn(seq_len, batch, input_size)

        std_rnn = torch.nn.RNN(input_size, hidden_size, nonlinearity="relu")

        q_rnn = RNN(input_size, hidden_size, nonlinearity="relu")
        q_rnn.layers[0].layer_fw.cell.input_linear.weight.data = std_rnn.weight_ih_l0.data.clone()
        q_rnn.layers[0].layer_fw.cell.input_linear.bias.data = std_rnn.bias_ih_l0.data.clone()
        q_rnn.layers[0].layer_fw.cell.hidden_linear.weight.data = std_rnn.weight_hh_l0.data.clone()
        q_rnn.layers[0].layer_fw.cell.hidden_linear.bias.data = std_rnn.bias_hh_l0.data.clone()

        std_out, std_h = std_rnn(x)
        q_out, q_h = q_rnn(x)

        torch.testing.assert_close(q_out, std_out, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(q_h, std_h, rtol=1e-4, atol=1e-5)


class TestQuantizableGRU:
    """Tests for the quantizable GRU module."""

    def test_gru_basic_forward(self):
        """Test basic GRU forward pass matches nn.GRU."""
        input_size, hidden_size, seq_len, batch = 10, 20, 5, 3
        x = torch.randn(seq_len, batch, input_size)

        # Create standard GRU
        std_gru = torch.nn.GRU(input_size, hidden_size, batch_first=False)

        # Create quantizable GRU and copy weights
        # GRU uses combined input_linear and hidden_linear (3*hidden_size)
        q_gru = GRU(input_size, hidden_size, batch_first=False)
        q_gru.layers[0].layer_fw.cell.input_linear.weight.data = std_gru.weight_ih_l0.data.clone()
        q_gru.layers[0].layer_fw.cell.input_linear.bias.data = std_gru.bias_ih_l0.data.clone()
        q_gru.layers[0].layer_fw.cell.hidden_linear.weight.data = std_gru.weight_hh_l0.data.clone()
        q_gru.layers[0].layer_fw.cell.hidden_linear.bias.data = std_gru.bias_hh_l0.data.clone()

        std_out, std_h = std_gru(x)
        q_out, q_h = q_gru(x)

        torch.testing.assert_close(q_out, std_out, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(q_h, std_h, rtol=1e-4, atol=1e-5)

    def test_gru_bidirectional(self):
        """Test bidirectional GRU matches nn.GRU."""
        input_size, hidden_size, seq_len, batch = 10, 20, 5, 3
        x = torch.randn(seq_len, batch, input_size)

        std_gru = torch.nn.GRU(input_size, hidden_size, bidirectional=True)

        q_gru = GRU(input_size, hidden_size, bidirectional=True)

        # Copy forward weights
        q_gru.layers[0].layer_fw.cell.input_linear.weight.data = std_gru.weight_ih_l0.data.clone()
        q_gru.layers[0].layer_fw.cell.input_linear.bias.data = std_gru.bias_ih_l0.data.clone()
        q_gru.layers[0].layer_fw.cell.hidden_linear.weight.data = std_gru.weight_hh_l0.data.clone()
        q_gru.layers[0].layer_fw.cell.hidden_linear.bias.data = std_gru.bias_hh_l0.data.clone()

        # Copy backward weights
        q_gru.layers[0].layer_bw.cell.input_linear.weight.data = (
            std_gru.weight_ih_l0_reverse.data.clone()
        )
        q_gru.layers[0].layer_bw.cell.input_linear.bias.data = std_gru.bias_ih_l0_reverse.data.clone()
        q_gru.layers[0].layer_bw.cell.hidden_linear.weight.data = (
            std_gru.weight_hh_l0_reverse.data.clone()
        )
        q_gru.layers[0].layer_bw.cell.hidden_linear.bias.data = (
            std_gru.bias_hh_l0_reverse.data.clone()
        )

        std_out, std_h = std_gru(x)
        q_out, q_h = q_gru(x)

        torch.testing.assert_close(q_out, std_out, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(q_h, std_h, rtol=1e-4, atol=1e-5)

    def test_gru_batch_first(self):
        """Test batch_first GRU matches nn.GRU."""
        input_size, hidden_size, seq_len, batch = 10, 20, 5, 3
        x = torch.randn(batch, seq_len, input_size)

        std_gru = torch.nn.GRU(input_size, hidden_size, batch_first=True)

        q_gru = GRU(input_size, hidden_size, batch_first=True)
        q_gru.layers[0].layer_fw.cell.input_linear.weight.data = std_gru.weight_ih_l0.data.clone()
        q_gru.layers[0].layer_fw.cell.input_linear.bias.data = std_gru.bias_ih_l0.data.clone()
        q_gru.layers[0].layer_fw.cell.hidden_linear.weight.data = std_gru.weight_hh_l0.data.clone()
        q_gru.layers[0].layer_fw.cell.hidden_linear.bias.data = std_gru.bias_hh_l0.data.clone()

        std_out, std_h = std_gru(x)
        q_out, q_h = q_gru(x)

        torch.testing.assert_close(q_out, std_out, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(q_h, std_h, rtol=1e-4, atol=1e-5)

    def test_gru_no_bias(self):
        """Test GRU without bias matches nn.GRU."""
        input_size, hidden_size, seq_len, batch = 10, 20, 5, 3
        x = torch.randn(seq_len, batch, input_size)

        std_gru = torch.nn.GRU(input_size, hidden_size, bias=False)

        q_gru = GRU(input_size, hidden_size, bias=False)
        q_gru.layers[0].layer_fw.cell.input_linear.weight.data = std_gru.weight_ih_l0.data.clone()
        q_gru.layers[0].layer_fw.cell.hidden_linear.weight.data = std_gru.weight_hh_l0.data.clone()

        std_out, std_h = std_gru(x)
        q_out, q_h = q_gru(x)

        torch.testing.assert_close(q_out, std_out, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(q_h, std_h, rtol=1e-4, atol=1e-5)
