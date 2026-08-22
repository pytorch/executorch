"""Export Silero VAD (16kHz) to ExecuTorch.

Reimplements the model in eager PyTorch, loads weights from the official
JIT model, and exports as a single-method .pte.

Usage:
    python export_silero_vad.py --jit-model /path/to/silero_vad.jit
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from torch.export import export


SAMPLE_RATE = 16000
WINDOW_SIZE = 512
CONTEXT_SIZE = 64
INPUT_SIZE = WINDOW_SIZE + CONTEXT_SIZE  # 576
HIDDEN_DIM = 128
STFT_FILTERS = 258
FREQ_BINS = STFT_FILTERS // 2  # 129


class SileroVAD16k(nn.Module):
    """Eager PyTorch reimplementation of Silero VAD (16kHz).

    Architecture: learned STFT -> 4-layer CNN encoder -> LSTMCell -> sigmoid.
    Weights are loaded from the official JIT model's _model sub-module.
    """

    def __init__(self):
        super().__init__()
        self.stft_conv = nn.Conv1d(
            1, STFT_FILTERS, kernel_size=256, stride=128, bias=False
        )
        self.encoder = nn.ModuleList(
            [
                nn.Conv1d(FREQ_BINS, HIDDEN_DIM, 3, stride=1, padding=1),
                nn.Conv1d(HIDDEN_DIM, 64, 3, stride=2, padding=1),
                nn.Conv1d(64, 64, 3, stride=2, padding=1),
                nn.Conv1d(64, HIDDEN_DIM, 3, stride=1, padding=1),
            ]
        )
        self.lstm_cell = nn.LSTMCell(HIDDEN_DIM, HIDDEN_DIM)
        self.final_conv = nn.Conv1d(HIDDEN_DIM, 1, 1)

    def forward(
        self, x: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (1, 576) — 64 context + 512 audio samples
        # state: (2, 1, 128) — [h, c] LSTM state

        # Learned STFT
        x = F.pad(x, (0, 64), mode="reflect")
        x = x.unsqueeze(1)
        x = self.stft_conv(x)
        real, imag = x[:, :FREQ_BINS], x[:, FREQ_BINS:]
        x = (real**2 + imag**2).sqrt()

        # CNN encoder
        for conv in self.encoder:
            x = F.relu(conv(x))
        x = x.squeeze(-1)

        # LSTM
        h, c = self.lstm_cell(x, (state[0], state[1]))
        new_state = torch.stack([h, c], dim=0)

        # Decoder
        x = F.relu(h).unsqueeze(-1)
        x = torch.sigmoid(self.final_conv(x))
        x = x.squeeze(1).mean(dim=1, keepdim=True)

        return x, new_state


def load_model(jit_path: str) -> SileroVAD16k:
    """Load weights from JIT model into eager PyTorch reimplementation."""
    jit_model = torch.jit.load(jit_path, map_location="cpu")
    jit_sd = jit_model._model.state_dict()

    model = SileroVAD16k()

    key_map = {
        "stft.forward_basis_buffer": "stft_conv.weight",
        "encoder.0.reparam_conv.weight": "encoder.0.weight",
        "encoder.0.reparam_conv.bias": "encoder.0.bias",
        "encoder.1.reparam_conv.weight": "encoder.1.weight",
        "encoder.1.reparam_conv.bias": "encoder.1.bias",
        "encoder.2.reparam_conv.weight": "encoder.2.weight",
        "encoder.2.reparam_conv.bias": "encoder.2.bias",
        "encoder.3.reparam_conv.weight": "encoder.3.weight",
        "encoder.3.reparam_conv.bias": "encoder.3.bias",
        "decoder.rnn.weight_ih": "lstm_cell.weight_ih",
        "decoder.rnn.weight_hh": "lstm_cell.weight_hh",
        "decoder.rnn.bias_ih": "lstm_cell.bias_ih",
        "decoder.rnn.bias_hh": "lstm_cell.bias_hh",
        "decoder.decoder.2.weight": "final_conv.weight",
        "decoder.decoder.2.bias": "final_conv.bias",
    }

    mapped_sd = {}
    for jit_key, eager_key in key_map.items():
        mapped_sd[eager_key] = jit_sd[jit_key]

    model.load_state_dict(mapped_sd)
    model.eval()

    # Verify against JIT model
    with torch.no_grad():
        sample_x = torch.randn(1, INPUT_SIZE)
        sample_state = torch.zeros(2, 1, HIDDEN_DIM)

        jit_out, jit_state = jit_model._model(sample_x, sample_state)
        eager_out, eager_state = model(sample_x, sample_state)

        max_diff = (jit_out - eager_out).abs().max().item()
        print(f"  Verification: max output diff = {max_diff:.2e}")
        assert max_diff < 1e-5, f"Output mismatch: {max_diff}"

    return model


def export_model(model: SileroVAD16k):
    """Export the model with explicit state passing."""
    sample_x = torch.randn(1, INPUT_SIZE)
    sample_state = torch.zeros(2, 1, HIDDEN_DIM)

    print("  Exporting forward...")
    programs = {
        "forward": export(
            model,
            (sample_x, sample_state),
            strict=False,
        )
    }

    metadata = {
        "sample_rate": SAMPLE_RATE,
        "window_size": WINDOW_SIZE,
        "context_size": CONTEXT_SIZE,
    }

    return programs, metadata


def lower_to_executorch(programs, metadata=None, backend="portable"):
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )

        print("\nLowering to ExecuTorch with XNNPACK...")
        partitioner = {"forward": [XnnpackPartitioner()]}
    else:
        print("\nLowering to ExecuTorch...")
        partitioner = []

    constant_methods = {}
    if metadata:
        for key, value in metadata.items():
            constant_methods[key] = value

    et_prog = to_edge_transform_and_lower(
        programs,
        partitioner=partitioner,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=constant_methods if constant_methods else None,
    )
    return et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export Silero VAD (16kHz) to ExecuTorch"
    )
    parser.add_argument("--output-dir", default="./silero_vad_exports")
    parser.add_argument(
        "--jit-model",
        type=str,
        required=True,
        help="Path to silero_vad.jit file",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="xnnpack",
        choices=["portable", "xnnpack"],
        help="Backend for acceleration (default: xnnpack)",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...")
    model = load_model(args.jit_model)

    print("\nExporting...")
    programs, metadata = export_model(model)

    et = lower_to_executorch(programs, metadata=metadata, backend=args.backend)

    pte_path = os.path.join(args.output_dir, "silero_vad.pte")
    print(f"\nSaving ExecuTorch program to: {pte_path}")
    with open(pte_path, "wb") as f:
        et.write_to_file(f)
    print(f"Saved {os.path.getsize(pte_path) / 1024:.1f} KB")

    print("\nDone!")


if __name__ == "__main__":
    main()
