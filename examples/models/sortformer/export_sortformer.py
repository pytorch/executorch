"""Export streaming Sortformer diarizer components to ExecuTorch.

Exports nvidia/diar_streaming_sortformer_4spk-v2 (117M param speaker diarization)
as a multi-method .pte with three methods: preprocessor, pre_encode, encode.

Usage:
    python export_sortformer.py --nemo-path /path/to/model.nemo
    python export_sortformer.py --hf-model nvidia/diar_streaming_sortformer_4spk-v2
"""

import argparse
import os
from typing import Optional

import torch

from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from torch import nn
from torch.export import Dim, export


class PreprocessorWrapper(torch.nn.Module):
    """Wraps NeMo's AudioToMelSpectrogramPreprocessor for single-sample export.

    Input: 1D audio waveform (N,) and length (1,) int64.
    Output: mel spectrogram (1, 128, T_mel) and mel_len (1,) int64.
    """

    def __init__(self, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor

    def forward(
        self, audio: torch.Tensor, length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio_signal = audio.unsqueeze(0)
        mel, mel_len = self.preprocessor(input_signal=audio_signal, length=length)
        return mel, mel_len


class PreEncodeWrapper(torch.nn.Module):
    """Wraps ConvSubsampling (8x downsampling) for per-chunk export.

    Input: chunk (1, T_mel_chunk, 128) float, chunk_len (1,) int64.
    Output: embs (1, T_sub, 512) float, emb_len (1,) int64.

    Re-implements ConvSubsampling.forward without MaskedConvSequential's
    masking logic, which creates data-dependent guards that block torch.export.
    Masking is unnecessary for single-sample inference with valid-length chunks.
    """

    def __init__(self, pre_encode):
        super().__init__()
        self.conv_layers = nn.ModuleList(list(pre_encode.conv))
        self.out = pre_encode.out
        self._left_padding = pre_encode._left_padding
        self._right_padding = pre_encode._right_padding
        self._kernel_size = pre_encode._kernel_size
        self._stride = pre_encode._stride
        self._ceil_mode = pre_encode._ceil_mode
        self._sampling_num = pre_encode._sampling_num

    def forward(
        self, chunk: torch.Tensor, chunk_len: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from nemo.collections.asr.parts.submodules.subsampling import calc_length

        out_lengths = calc_length(
            chunk_len,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )
        x = chunk.unsqueeze(1)  # (B, 1, T, feat_in)
        for layer in self.conv_layers:
            x = layer(x)
        # flatten static dims C and F (not dynamic T) to avoid symbolic guards
        x = x.permute(0, 2, 1, 3).flatten(2)  # (B, T_sub, C*F)
        x = self.out(x)
        return x, out_lengths.to(torch.int64)


class EncodeWrapper(torch.nn.Module):
    """Wraps conformer + encoder_proj + transformer + speaker head.

    Input: embs (1, T_total, 512) float, emb_len (1,) int64.
    Output: preds (1, T_total, 4) float — per-frame speaker probabilities.

    The C++ runner concatenates [spkcache, fifo, chunk_embs] into a single
    contiguous tensor and passes it here. Calls encoder with bypass_pre_encode=True
    to skip ConvSubsampling (already run separately).
    """

    def __init__(self, encoder, encoder_proj, transformer_encoder, sortformer_modules):
        super().__init__()
        self.encoder = encoder
        self.encoder_proj = encoder_proj if encoder_proj is not None else nn.Identity()
        self.transformer_encoder = transformer_encoder
        self.sortformer_modules = sortformer_modules

    def forward(self, embs: torch.Tensor, emb_len: torch.Tensor) -> torch.Tensor:
        # Conformer layers (skip pre_encode since input is already subsampled)
        encoded, enc_len = self.encoder(
            audio_signal=embs, length=emb_len, bypass_pre_encode=True
        )
        # encoded shape: (B, d_model, T) — transpose to (B, T, d_model)
        encoded = encoded.transpose(1, 2)

        # Project from conformer dim to transformer dim: (B, T, 512) -> (B, T, 192)
        encoded = self.encoder_proj(encoded)

        # Transformer encoder
        encoder_mask = self.sortformer_modules.length_to_mask(enc_len, encoded.shape[1])
        trans_out = self.transformer_encoder(
            encoder_states=encoded, encoder_mask=encoder_mask
        )

        # Speaker head: Linear(192->192) -> ReLU -> Linear(192->4) -> Sigmoid
        preds = self.sortformer_modules.forward_speaker_sigmoids(trans_out)
        preds = preds * encoder_mask.unsqueeze(-1)
        return preds


def load_model(nemo_path: Optional[str] = None, hf_model: Optional[str] = None):
    """Load SortformerEncLabelModel from .nemo file or HuggingFace."""
    from nemo.collections.asr.models import SortformerEncLabelModel

    if nemo_path:
        model = SortformerEncLabelModel.restore_from(
            nemo_path, map_location="cpu", strict=False
        )
    elif hf_model:
        model = SortformerEncLabelModel.from_pretrained(hf_model, map_location="cpu")
    else:
        model = SortformerEncLabelModel.from_pretrained(
            "nvidia/diar_streaming_sortformer_4spk-v2", map_location="cpu"
        )

    model.eval()
    model.cpu()
    return model


def _rel_shift_export(self, x):
    """Export-safe rel_shift using gather instead of pad+view.

    The original rel_shift does pad → view(b,h,-1,T) → slice → view(b,h,T,2T-1).
    The view ops mix two T-dependent dimensions, creating a (2*T*T)//T guard that
    the symbolic solver can't simplify. This replaces the entire operation with a
    single gather that directly produces the shifted+trimmed output (b,h,T,T).
    """
    b, h, qlen, pos_len = x.size()
    arange = torch.arange(qlen, device=x.device)
    # output[i,j] = input[i, j - i + qlen - 1]
    idx = arange.unsqueeze(0) - arange.unsqueeze(1) + (qlen - 1)
    idx = idx.unsqueeze(0).unsqueeze(0).expand(b, h, -1, -1)
    return torch.gather(x, 3, idx)


def prepare_for_export(model):
    """Patch model to remove torch.export blockers.

    1. Preprocessor pad_to creates a data-dependent branch on frame count % 16.
    2. ConformerEncoder.update_max_seq_length dynamically extends positional
       encoding buffers based on input size.
    3. RelPositionMultiHeadAttention.rel_shift uses pad+view that creates
       unsolvable (2*T*T)//T guards — replaced with gather-based version.
    """
    model.preprocessor.featurizer.pad_to = 0
    model.encoder.set_max_audio_length(5000)
    model.encoder.update_max_seq_length = lambda seq_length, device: None

    import types

    from nemo.collections.asr.parts.submodules.multi_head_attention import (
        RelPositionMultiHeadAttention,
    )

    for layer in model.encoder.layers:
        if isinstance(layer.self_attn, RelPositionMultiHeadAttention):
            layer.self_attn.rel_shift = types.MethodType(
                _rel_shift_export, layer.self_attn
            )


def export_all(model, backend: Optional[str] = None):
    """Export preprocessor, pre_encode, and encode methods.

    Args:
        model: The SortformerEncLabelModel to export.
        backend: Target backend ("xnnpack" or "portable").
    """
    programs = {}

    sample_rate = model.preprocessor._cfg.sample_rate
    window_stride = float(model.preprocessor._cfg.window_stride)
    subsampling_factor = int(model.encoder.subsampling_factor)

    prepare_for_export(model)

    # --- Method 1: preprocessor ---
    preprocessor_wrapper = PreprocessorWrapper(model.preprocessor)
    preprocessor_wrapper.eval()

    max_audio_samples = int(sample_rate * 120)  # 120 seconds max
    sample_audio = torch.randn(max_audio_samples, dtype=torch.float)
    sample_length = torch.tensor([sample_audio.shape[0]], dtype=torch.int64)

    # Force CPU path to avoid data-dependent CUDA conditionals in preprocessor
    old_cuda_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False

    print("  Exporting preprocessor...")
    programs["preprocessor"] = export(
        preprocessor_wrapper,
        (sample_audio, sample_length),
        dynamic_shapes={
            "audio": {0: Dim("audio_len", min=1600, max=max_audio_samples)},
            "length": {},
        },
        strict=False,
    )

    torch.cuda.is_available = old_cuda_is_available

    # --- Method 2: pre_encode ---
    pre_encode_wrapper = PreEncodeWrapper(model.encoder.pre_encode)
    pre_encode_wrapper.eval()

    feat_in = getattr(model.encoder, "_feat_in", 128)
    max_chunk_mel = 4000
    sample_chunk = torch.randn(1, max_chunk_mel, feat_in, dtype=torch.float)
    sample_chunk_len = torch.tensor([max_chunk_mel], dtype=torch.int64)

    print("  Exporting pre_encode...")
    programs["pre_encode"] = export(
        pre_encode_wrapper,
        (sample_chunk, sample_chunk_len),
        # Static shapes: conv-derived symbolic expression 1+((L-1)//8) creates
        # an unsolvable guard when hitting nn.Linear. Static shapes are practical
        # since streaming chunk sizes are fixed per config.
        strict=False,
    )

    # --- Method 3: encode ---

    encoder_proj = model.sortformer_modules.encoder_proj
    encode_wrapper = EncodeWrapper(
        encoder=model.encoder,
        encoder_proj=encoder_proj,
        transformer_encoder=model.transformer_encoder,
        sortformer_modules=model.sortformer_modules,
    )
    encode_wrapper.eval()

    d_model = model.encoder.d_model
    max_total_frames = 1000
    sample_embs = torch.randn(1, max_total_frames, d_model, dtype=torch.float)
    sample_emb_len = torch.tensor([max_total_frames], dtype=torch.int64)

    print("  Exporting encode...")
    programs["encode"] = export(
        encode_wrapper,
        (sample_embs, sample_emb_len),
        dynamic_shapes={
            # min=2: attention mask logic requires T>1
            "embs": {1: Dim("total_frames", min=2, max=max_total_frames)},
            "emb_len": {},
        },
        strict=False,
    )

    # --- Metadata ---
    fc_d_model = int(model._cfg.encoder.d_model)
    tf_d_model = int(model._cfg.model_defaults.tf_d_model)
    max_num_of_spks = int(model._cfg.max_num_of_spks)
    spkcache_len = int(model.sortformer_modules.spkcache_len)

    metadata = {
        "sample_rate": int(sample_rate),
        "window_stride": window_stride,
        "subsampling_factor": subsampling_factor,
        "fc_d_model": fc_d_model,
        "tf_d_model": tf_d_model,
        "max_num_of_spks": max_num_of_spks,
        "spkcache_len": spkcache_len,
    }

    return programs, metadata


def lower_to_executorch(programs, metadata=None, backend="portable"):
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )

        print("\nLowering to ExecuTorch with XNNPACK...")
        partitioner = {}
        for key in programs.keys():
            if key == "preprocessor":
                partitioner[key] = []
            else:
                partitioner[key] = [XnnpackPartitioner()]
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
        description="Export streaming Sortformer diarizer to ExecuTorch"
    )
    parser.add_argument("--output-dir", default="./sortformer_exports")
    parser.add_argument(
        "--nemo-path",
        type=str,
        help="Path to .nemo model file",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        help="HuggingFace model ID (default: nvidia/diar_streaming_sortformer_4spk-v2)",
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
    model = load_model(nemo_path=args.nemo_path, hf_model=args.hf_model)

    print("\nExporting components...")
    programs, metadata = export_all(model, backend=args.backend)

    et = lower_to_executorch(programs, metadata=metadata, backend=args.backend)

    pte_path = os.path.join(args.output_dir, "sortformer.pte")
    print(f"\nSaving ExecuTorch program to: {pte_path}")
    with open(pte_path, "wb") as f:
        et.write_to_file(f)
    print(f"Saved {os.path.getsize(pte_path) / (1024 * 1024):.1f} MB")

    print("\nDone!")


if __name__ == "__main__":
    main()
