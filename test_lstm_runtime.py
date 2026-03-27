#!/usr/bin/env python3
"""Test LSTM + Embedding with different input shapes at runtime."""

import torch
import torch.nn as nn
from torch.export import Dim, export
from torch.export._patches import register_lstm_while_loop_decomposition

from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig, ExecutorchBackendConfig
from executorch.runtime import Runtime


class LSTMWithEmbedding(nn.Module):
    def __init__(self, vocab_size=100, embed_dim=32, hidden_size=64, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x, h0, c0):
        embedded = self.embedding(x)
        output, (hn, cn) = self.lstm(embedded, (h0, c0))
        return output, hn, cn


def main():
    model = LSTMWithEmbedding()
    model.eval()

    batch_size = 1
    seq_len = 16
    hidden_size = 64
    num_layers = 1

    x = torch.randint(0, 100, (batch_size, seq_len))
    h0 = torch.zeros(num_layers, batch_size, hidden_size)
    c0 = torch.zeros(num_layers, batch_size, hidden_size)

    seq = Dim("seq", min=1, max=128)
    dynamic_shapes = {"x": {1: seq}, "h0": {}, "c0": {}}

    print("Exporting with seq_len=16...")
    with register_lstm_while_loop_decomposition():
        exported = export(model, (x, h0, c0), dynamic_shapes=dynamic_shapes)

    edge_config = EdgeCompileConfig(_check_ir_validity=False)
    et_config = ExecutorchBackendConfig()

    with register_lstm_while_loop_decomposition():
        edge = to_edge_transform_and_lower(exported, compile_config=edge_config)
    et_program = edge.to_executorch(config=et_config)

    pte_path = "/tmp/lstm_embedding_runtime_test.pte"
    with open(pte_path, "wb") as f:
        et_program.write_to_file(f)
    print("  Saved .pte file")

    print("\nLoading runtime...")
    runtime = Runtime.get()
    program = runtime.load_program(pte_path)
    method = program.load_method("forward")

    # Test with SAME shape (seq_len=16)
    print("\nTest 1: Inference with seq_len=16 (same as export)...")
    x_16 = torch.randint(0, 100, (batch_size, 16))
    outputs = method.execute([x_16, h0, c0])
    print(f"  Output shape: {outputs[0].shape}")
    assert outputs[0].shape == (1, 16, 64), f"Expected (1, 16, 64), got {outputs[0].shape}"
    print("  PASSED!")

    # Test with DIFFERENT shape (seq_len=5) - this was the original bug
    print("\nTest 2: Inference with seq_len=5 (different from export)...")
    x_5 = torch.randint(0, 100, (batch_size, 5))
    method2 = program.load_method("forward")
    outputs = method2.execute([x_5, h0, c0])
    print(f"  Output shape: {outputs[0].shape}")
    assert outputs[0].shape == (1, 5, 64), f"Expected (1, 5, 64), got {outputs[0].shape}"
    print("  PASSED!")

    # Test with another different shape (seq_len=32)
    print("\nTest 3: Inference with seq_len=32...")
    x_32 = torch.randint(0, 100, (batch_size, 32))
    method3 = program.load_method("forward")
    outputs = method3.execute([x_32, h0, c0])
    print(f"  Output shape: {outputs[0].shape}")
    assert outputs[0].shape == (1, 32, 64), f"Expected (1, 32, 64), got {outputs[0].shape}"
    print("  PASSED!")

    # Verify numerical correctness
    print("\nTest 4: Numerical correctness check (seq_len=5)...")
    with torch.no_grad():
        expected = model(x_5, h0, c0)
    method4 = program.load_method("forward")
    actual = method4.execute([x_5, h0, c0])
    max_diff = (expected[0] - actual[0]).abs().max().item()
    print(f"  Max absolute difference: {max_diff}")
    assert max_diff < 1e-4, f"Numerical mismatch: max diff = {max_diff}"
    print("  PASSED!")

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
