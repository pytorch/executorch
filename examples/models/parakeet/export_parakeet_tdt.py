"""Export nvidia/parakeet-tdt-0.6b-v3 components to ExecuTorch."""

import argparse
import os
import shutil
import tarfile
import tempfile
from typing import Optional

import torch
import torchaudio

from executorch.examples.models.parakeet.quantize import quantize_model_
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from torch.export import Dim, export


def load_audio(audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
    """Load audio file and resample to target sample rate."""

    waveform, sr = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    return waveform


def greedy_decode_eager(
    encoder_output: torch.Tensor, encoder_len: torch.Tensor, model
) -> list[int]:
    hypotheses = model.decoding.rnnt_decoder_predictions_tensor(
        encoder_output=encoder_output,
        encoded_lengths=encoder_len,
        return_hypotheses=True,
    )
    return hypotheses[0].y_sequence


class EncoderWithProjection(torch.nn.Module):
    """Encoder that outputs projected features ready for the joint network."""

    def __init__(self, encoder, joint):
        super().__init__()
        self.encoder = encoder
        self.project_encoder = joint.project_encoder

    def forward(
        self, audio_signal: torch.Tensor, length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Run encoder: [B, feat_in, T_mel] -> [B, enc_dim, T_enc]
        encoded, enc_len = self.encoder(audio_signal=audio_signal, length=length)
        # Transpose: [B, enc_dim, T_enc] -> [B, T_enc, enc_dim]
        encoded_t = encoded.transpose(1, 2)
        # Project: [B, T_enc, enc_dim] -> [B, T_enc, joint_hidden]
        f_proj = self.project_encoder(encoded_t)
        return f_proj, enc_len


class DecoderStep(torch.nn.Module):
    """Single decoder RNN step that outputs projected features for the joint network."""

    def __init__(self, decoder, joint):
        super().__init__()
        self.decoder = decoder
        self.project_prednet = joint.project_prednet
        self.pred_hidden = decoder.pred_hidden
        self.pred_rnn_layers = getattr(decoder, "pred_rnn_layers", 2)

    def forward(
        self, token: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Run decoder RNN step
        g, new_state = self.decoder.predict(y=token, state=[h, c], add_sos=False)
        # Project decoder output: [B, 1, pred_hidden] -> [B, 1, joint_hidden]
        g_proj = self.project_prednet(g)
        return g_proj, new_state[0], new_state[1]


def greedy_decode_executorch(
    f_proj: torch.Tensor,
    encoder_len: int,
    program,
    blank_id: int,
    num_rnn_layers: int = 2,
    pred_hidden: int = 640,
    max_symbols_per_step: int = 10,
    durations: list[int] | None = None,
) -> list[int]:
    """Greedy TDT decoding using ExecuTorch methods.

    Args:
        f_proj: Projected encoder output [B, T, joint_hidden] (already transposed and projected)
        encoder_len: Number of valid encoder frames
        program: ExecuTorch program with loaded methods
        blank_id: Token ID for blank
        num_rnn_layers: Number of RNN layers in decoder
        pred_hidden: Hidden size of decoder RNN
        max_symbols_per_step: Maximum symbols per frame
        durations: Duration values for TDT

    Returns:
        List of decoded token IDs
    """
    if durations is None:
        durations = [0, 1, 2, 3, 4]

    hypothesis = []

    decoder_step_method = program.load_method("decoder_step")
    joint_method = program.load_method("joint")

    # Initialize decoder state
    h = torch.zeros(num_rnn_layers, 1, pred_hidden)
    c = torch.zeros(num_rnn_layers, 1, pred_hidden)

    # Prime decoder with SOS (blank_id) to match NeMo TDT behavior
    sos_token = torch.tensor([[blank_id]], dtype=torch.long)
    sos_result = decoder_step_method.execute([sos_token, h, c])
    g_proj = sos_result[0]
    h = sos_result[1]
    c = sos_result[2]

    t = 0
    symbols_on_frame = 0

    while t < encoder_len:
        f_t = f_proj[:, t : t + 1, :].contiguous()

        joint_out = joint_method.execute([f_t, g_proj])
        k = joint_out[0].item()
        dur_idx = joint_out[1].item()
        dur = durations[dur_idx]

        if k == blank_id:
            t += max(dur, 1)
            symbols_on_frame = 0
        else:
            hypothesis.append(k)

            token = torch.tensor([[k]], dtype=torch.long)
            result = decoder_step_method.execute([token, h, c])
            g_proj = result[0]
            h = result[1]
            c = result[2]

            t += dur

            if dur == 0:
                symbols_on_frame += 1
                if symbols_on_frame >= max_symbols_per_step:
                    t += 1
                    symbols_on_frame = 0
            else:
                symbols_on_frame = 0

    return hypothesis


def transcribe_executorch(audio_path: str, model, et_buffer) -> str:
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(et_buffer)

    # Get sample rate from model
    sample_rate = model.preprocessor._cfg.sample_rate

    with torch.no_grad():
        audio = load_audio(audio_path, sample_rate=sample_rate)
        preprocessor_method = program.load_method("preprocessor")
        audio_1d = audio.squeeze(0)
        audio_len = torch.tensor([audio_1d.shape[0]], dtype=torch.int64)
        proc_result = preprocessor_method.execute([audio_1d, audio_len])
        mel = proc_result[0]
        mel_len = proc_result[1].item()

        encoder_method = program.load_method("encoder")
        mel_len_tensor = torch.tensor([mel_len], dtype=torch.int64)
        enc_result = encoder_method.execute([mel, mel_len_tensor])
        f_proj = enc_result[0]
        encoded_len = enc_result[1].item()

        vocab_size = model.tokenizer.vocab_size
        tokens = greedy_decode_executorch(
            f_proj,
            encoded_len,
            program,
            blank_id=vocab_size,
            num_rnn_layers=model.decoder.pred_rnn_layers,
            pred_hidden=model.decoder.pred_hidden,
        )

        return model.tokenizer.ids_to_text(tokens)


def transcribe_eager(audio_path: str, model) -> str:
    with torch.no_grad():
        audio = load_audio(audio_path)
        mel, mel_len = model.preprocessor(
            input_signal=audio, length=torch.tensor([audio.shape[1]])
        )
        encoded, encoded_len = model.encoder(audio_signal=mel, length=mel_len)
        tokens = greedy_decode_eager(encoded, encoded_len, model)
        return model.tokenizer.ids_to_text(tokens)


def load_model():
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/parakeet-tdt-0.6b-v3", map_location="cpu"
    )
    model.eval()
    model.cpu()
    return model


def extract_tokenizer(output_dir: str) -> str | None:
    """Extract tokenizer.model from the cached .nemo file.

    Args:
        output_dir: Directory to save the tokenizer.model file.

    Returns:
        Path to the extracted tokenizer.model, or None if extraction failed.
    """
    from huggingface_hub import hf_hub_download

    # Download/get cached .nemo file path
    nemo_path = hf_hub_download(
        repo_id="nvidia/parakeet-tdt-0.6b-v3",
        filename="parakeet-tdt-0.6b-v3.nemo",
    )

    # .nemo files are tar archives - extract tokenizer.model
    tokenizer_filename = "tokenizer.model"
    output_path = os.path.join(output_dir, tokenizer_filename)

    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(nemo_path, "r") as tar:
            # Find tokenizer.model in the archive (may be in root or subdirectory)
            tokenizer_member = None
            for member in tar.getmembers():
                if member.name.endswith(tokenizer_filename):
                    tokenizer_member = member
                    break

            if tokenizer_member is None:
                print(f"Warning: {tokenizer_filename} not found in .nemo archive")
                return None

            # Extract to temp directory
            tar.extract(tokenizer_member, tmpdir)
            extracted_path = os.path.join(tmpdir, tokenizer_member.name)

            # Copy to output directory
            shutil.copy2(extracted_path, output_path)

    print(f"Extracted tokenizer to: {output_path}")
    return output_path


class JointWithArgmax(torch.nn.Module):
    """Joint network that returns token and duration indices directly."""

    def __init__(self, joint, num_token_classes):
        super().__init__()
        self.joint = joint
        self.num_token_classes = num_token_classes

    def forward(self, f, g):
        logits = self.joint.joint_after_projection(f, g).squeeze()
        token_id = logits[: self.num_token_classes].argmax()
        duration_idx = logits[self.num_token_classes :].argmax()
        return token_id, duration_idx


class PreprocessorWrapper(torch.nn.Module):
    def __init__(self, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor

    def forward(
        self, audio: torch.Tensor, length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio_signal = audio.unsqueeze(0)
        mel, mel_len = self.preprocessor(input_signal=audio_signal, length=length)
        return mel, mel_len


def export_all(
    model,
    dtype=torch.float,
    backend: Optional[str] = None,
    # Encoder quantization args
    qlinear_encoder: Optional[str] = None,
    qlinear_encoder_group_size: int = 32,
    qlinear_encoder_packing_format: Optional[str] = None,
    # Decoder quantization args
    qlinear: Optional[str] = None,
    qlinear_group_size: int = 32,
    qlinear_packing_format: Optional[str] = None,
    # Embedding quantization args (decoder has the embedding layer)
    qembedding: Optional[str] = None,
    qembedding_group_size: int = 0,
):
    """Export all model components.

    The maximum audio duration is determined by the model's internal
    max_audio_length (~50 seconds for Parakeet with max_audio_length=5000).

    Args:
        model: The NeMo ASR model to export.
        dtype: Data type for floating-point tensors (default: torch.float).
        backend: Target backend ("cuda", "xnnpack", etc.).
        qlinear_encoder: Quantization config for encoder linear layers.
        qlinear_encoder_group_size: Group size for encoder linear quantization.
        qlinear_encoder_packing_format: Packing format for encoder linear layers.
        qlinear: Quantization config for decoder linear layers.
        qlinear_group_size: Group size for decoder linear quantization.
        qlinear_packing_format: Packing format for decoder linear layers.
        qembedding: Quantization config for embedding layers ("4w", "8w").
        qembedding_group_size: Group size for embedding quantization (default: 0 = per-axis).
    """
    programs = {}

    # Determine device based on backend (preprocessor always stays on CPU)
    device = torch.device("cuda" if backend == "cuda" else "cpu")

    # Get audio parameters from model config
    sample_rate = model.preprocessor._cfg.sample_rate
    window_stride = float(model.preprocessor._cfg.window_stride)

    # Get encoder's actual limit from NeMo model
    encoder_max_frames = model.encoder.max_audio_length  # typically 5000
    max_audio_sec = int(encoder_max_frames * window_stride)

    max_audio_samples = int(sample_rate * max_audio_sec)
    max_mel_frames = int(max_audio_sec / window_stride)

    preprocessor_wrapper = PreprocessorWrapper(model.preprocessor)
    preprocessor_wrapper.eval()

    sample_audio = torch.randn(max_audio_samples, dtype=torch.float)
    sample_length = torch.tensor([sample_audio.shape[0]], dtype=torch.int64)
    # The preprocessor uses different code paths when CUDA is available, which include
    # data-dependent conditionals that torch.export cannot handle. Force CPU path.
    old_cuda_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    programs["preprocessor"] = export(
        preprocessor_wrapper,
        (sample_audio, sample_length),
        dynamic_shapes={
            # min=1600 samples = 0.1 sec @ 16kHz, max aligned with encoder limit
            "audio": {0: Dim("audio_len", min=1600, max=max_audio_samples)},
            "length": {},
        },
        strict=False,
    )
    torch.cuda.is_available = old_cuda_is_available

    # Move model to CUDA after preprocessor export (preprocessor must stay on CPU)
    if backend == "cuda":
        model.cuda()

    feat_in = getattr(model.encoder, "_feat_in", 128)
    # Use max_mel_frames as example to ensure Dim.AUTO infers the full range.
    # Smaller examples cause Dim.AUTO to infer narrow bounds.
    audio_signal = torch.randn(1, feat_in, max_mel_frames, dtype=dtype, device=device)
    length = torch.tensor([max_mel_frames], dtype=torch.int64, device=device)
    encoder_with_proj = EncoderWithProjection(model.encoder, model.joint)
    encoder_with_proj.eval()

    if qlinear_encoder:
        print("Quantizing encoder...")
        quantize_model_(
            encoder_with_proj,
            qlinear_config=qlinear_encoder,
            qlinear_group_size=qlinear_encoder_group_size,
            qlinear_packing_format=qlinear_encoder_packing_format,
        )

    programs["encoder"] = export(
        encoder_with_proj,
        (),
        kwargs={"audio_signal": audio_signal, "length": length},
        dynamic_shapes={
            # Use Dim.AUTO - explicit bounds fail due to different size guards on different devices
            "audio_signal": {2: Dim.AUTO},
            "length": {},
        },
        strict=False,
    )

    num_layers = model.decoder.pred_rnn_layers
    pred_hidden = model.decoder.pred_hidden
    decoder_step = DecoderStep(model.decoder, model.joint)
    decoder_step.eval()

    if qlinear or qembedding:
        print("Quantizing decoder...")
        quantize_model_(
            decoder_step,
            qlinear_config=qlinear,
            qlinear_group_size=qlinear_group_size,
            qlinear_packing_format=qlinear_packing_format,
            qembedding_config=qembedding,
            qembedding_group_size=qembedding_group_size,
        )

    token = torch.tensor([[0]], dtype=torch.long, device=device)
    h = torch.zeros(num_layers, 1, pred_hidden, dtype=dtype, device=device)
    c = torch.zeros(num_layers, 1, pred_hidden, dtype=dtype, device=device)
    programs["decoder_step"] = export(
        decoder_step,
        (token, h, c),
        dynamic_shapes={"token": {}, "h": {}, "c": {}},
        strict=False,
    )

    joint_hidden = model.joint.joint_hidden
    num_token_classes = model.tokenizer.vocab_size + 1  # +1 for blank

    f_proj = torch.randn(1, 1, joint_hidden, dtype=dtype, device=device)
    g_proj = torch.randn(1, 1, joint_hidden, dtype=dtype, device=device)
    programs["joint"] = export(
        JointWithArgmax(model.joint, num_token_classes),
        (f_proj, g_proj),
        dynamic_shapes={"f": {}, "g": {}},
        strict=False,
    )

    sample_rate = model.preprocessor._cfg.sample_rate
    window_stride = float(model.preprocessor._cfg.window_stride)
    encoder_subsampling_factor = int(getattr(model.encoder, "subsampling_factor", 8))

    metadata = {
        "num_rnn_layers": num_layers,
        "pred_hidden": pred_hidden,
        "joint_hidden": joint_hidden,
        "vocab_size": model.tokenizer.vocab_size,
        "blank_id": model.tokenizer.vocab_size,
        "sample_rate": sample_rate,
        "window_stride": window_stride,
        "encoder_subsampling_factor": encoder_subsampling_factor,
    }

    return programs, metadata


def _create_xnnpack_partitioners(programs):
    """Create XNNPACK partitioners for all programs except preprocessor."""
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackDynamicallyQuantizedPartitioner,
        XnnpackPartitioner,
    )

    print("\nLowering to ExecuTorch with XNNPACK...")
    partitioner = {}
    for key in programs.keys():
        if key == "preprocessor":
            partitioner[key] = []
        else:
            # Use both partitioners:
            # 1. XnnpackDynamicallyQuantizedPartitioner for dynamic quantization (8da4w)
            # 2. XnnpackPartitioner for remaining ops
            partitioner[key] = [
                XnnpackDynamicallyQuantizedPartitioner(),
                XnnpackPartitioner(),
            ]
    return partitioner, programs


# This custom decomposition is the key to making Parakeet run on the Metal backend.
# Without this, linear gets decomposed in a way that doesn't work for us.
# When input/weight tensors are 2D and bias is present, this gets decomposed into addmm and
# reinterpret_tensor_wrapper gets called on the bias, to make it look like a 2D tensor.
# On one hand, this requires us to implement addmm in the Metal backend. But more importantly,
# the reinterpret_tensor_wrapper call makes its way to ExecuTorch, causing a call to executorch::extension::from_blob
# with a 0 stride. ExecuTorch doesn't support that, and raises an error.
# This decomposition avoids that problem, and also avoids having to implement addmm.
def _linear_bias_decomposition(input, weight, bias=None):
    """Decompose linear with bias into matmul + add."""
    # linear(input, weight) = input @ weight.T
    # Use matmul instead of mm to handle batched inputs (3D+)
    weight_t = torch.ops.aten.t.default(weight)
    out = torch.ops.aten.matmul.default(input, weight_t)
    if bias is not None:
        return torch.ops.aten.add.Tensor(out, bias)
    return out


def _create_metal_partitioners(programs):
    """Create Metal partitioners for all programs except preprocessor."""
    from executorch.backends.apple.metal.metal_backend import MetalBackend
    from executorch.backends.apple.metal.metal_partitioner import MetalPartitioner

    print("\nLowering to ExecuTorch with Metal...")

    # Run decompositions for non-preprocessor programs
    updated_programs = {}
    for key, ep in programs.items():
        # print(f"Running decompositions for {key}")
        # print(ep.graph_module)
        if key != "preprocessor":
            updated_programs[key] = ep.run_decompositions(
                {torch.ops.aten.linear.default: _linear_bias_decomposition}
            )
        else:
            updated_programs[key] = ep

    partitioner = {}
    for key in updated_programs.keys():
        if key == "preprocessor":
            partitioner[key] = []
        else:
            compile_specs = [MetalBackend.generate_method_name_compile_spec(key)]
            partitioner[key] = [MetalPartitioner(compile_specs)]
    return partitioner, updated_programs


def _create_cuda_partitioners(programs, is_windows=False):
    """Create CUDA partitioners for all programs except preprocessor."""
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.exir.backend.compile_spec_schema import CompileSpec
    from torch._inductor.decomposition import conv1d_to_conv2d

    print(f"\nLowering to ExecuTorch with CUDA{' (Windows)' if is_windows else ''}...")

    # Run decompositions for non-preprocessor programs
    updated_programs = {}
    for key, ep in programs.items():
        if key != "preprocessor":
            updated_programs[key] = ep.run_decompositions(
                {torch.ops.aten.conv1d.default: conv1d_to_conv2d}
            )
        else:
            updated_programs[key] = ep

    partitioner = {}
    for key in updated_programs.keys():
        if key == "preprocessor":
            partitioner[key] = []
        else:
            compile_specs = [CudaBackend.generate_method_name_compile_spec(key)]
            if is_windows:
                compile_specs.append(CompileSpec("platform", "windows".encode("utf-8")))
            partitioner[key] = [CudaPartitioner(compile_specs)]
    return partitioner, updated_programs


def lower_to_executorch(programs, metadata=None, backend="portable"):
    if backend == "xnnpack":
        partitioner, programs = _create_xnnpack_partitioners(programs)
    elif backend == "metal":
        partitioner, programs = _create_metal_partitioners(programs)
    elif backend in ("cuda", "cuda-windows"):
        partitioner, programs = _create_cuda_partitioners(
            programs, is_windows=(backend == "cuda-windows")
        )
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
            do_quant_fusion_and_const_prop=True,
        ),
    )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./parakeet_tdt_exports")
    parser.add_argument(
        "--audio", type=str, help="Path to audio file for transcription test"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="xnnpack",
        choices=["portable", "xnnpack", "metal", "cuda", "cuda-windows"],
        help="Backend for acceleration (default: xnnpack)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Model dtype for Metal/CUDA backends (default: fp32)",
    )

    # Decoder quantization arguments
    parser.add_argument(
        "--qlinear",
        type=str,
        choices=["4w", "8w", "8da4w", "8da8w"],
        help="Quantization config for decoder linear layers",
    )
    parser.add_argument(
        "--qlinear_group_size",
        type=int,
        default=32,
        help="Group size for decoder linear quantization (default: 32)",
    )
    parser.add_argument(
        "--qlinear_packing_format",
        type=str,
        choices=["tile_packed_to_4d"],
        help="Packing format for decoder linear layers",
    )

    # Encoder quantization arguments
    parser.add_argument(
        "--qlinear_encoder",
        type=str,
        choices=["4w", "8w", "8da4w", "8da8w"],
        help="Quantization config for encoder linear layers",
    )
    parser.add_argument(
        "--qlinear_encoder_group_size",
        type=int,
        default=32,
        help="Group size for encoder linear quantization (default: 32)",
    )
    parser.add_argument(
        "--qlinear_encoder_packing_format",
        type=str,
        choices=["tile_packed_to_4d"],
        help="Packing format for encoder linear layers",
    )

    # Embedding quantization arguments (decoder has the embedding layer)
    parser.add_argument(
        "--qembedding",
        type=str,
        choices=["4w", "8w"],
        help="Quantization config for decoder embedding layer",
    )
    parser.add_argument(
        "--qembedding_group_size",
        type=int,
        default=0,
        help="Group size for embedding quantization (default: 0 = per-axis)",
    )

    args = parser.parse_args()

    # Validate dtype
    if args.dtype == "fp16":
        parser.error("fp16 is not yet supported")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Extracting tokenizer...")
    extract_tokenizer(args.output_dir)

    print("Loading model...")
    model = load_model()

    # Convert model to specified dtype for Metal/CUDA backends
    if args.dtype == "bf16":
        print("Converting model to bfloat16...")
        model = model.to(torch.bfloat16)
    elif args.dtype == "fp16":
        print("Converting model to float16...")
        model = model.to(torch.float16)

    print("\nExporting components...")
    export_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float
    programs, metadata = export_all(
        model,
        dtype=export_dtype,
        backend=args.backend,
        # Encoder quantization
        qlinear_encoder=args.qlinear_encoder,
        qlinear_encoder_group_size=args.qlinear_encoder_group_size,
        qlinear_encoder_packing_format=args.qlinear_encoder_packing_format,
        # Decoder quantization
        qlinear=args.qlinear,
        qlinear_group_size=args.qlinear_group_size,
        qlinear_packing_format=args.qlinear_packing_format,
        # Embedding quantization
        qembedding=args.qembedding,
        qembedding_group_size=args.qembedding_group_size,
    )

    et = lower_to_executorch(programs, metadata=metadata, backend=args.backend)

    pte_path = os.path.join(args.output_dir, "model.pte")
    print(f"\nSaving ExecuTorch program to: {pte_path}")
    with open(pte_path, "wb") as f:
        et.write_to_file(f)
    print(f"Saved {os.path.getsize(pte_path) / (1024 * 1024):.1f} MB")

    # Save .ptd data files (e.g., CUDA delegate data)
    if et._tensor_data:
        print(f"\nSaving {len(et._tensor_data)} data file(s)...")
        et.write_tensor_data_to_file(args.output_dir)

    if args.audio:
        print("\n" + "=" * 60)
        print("Testing transcription...")
        print("=" * 60)

        print("\n[Eager PyTorch]")
        eager_text = transcribe_eager(args.audio, model)
        print(f"  Result: {eager_text}")

        print("\n[ExecuTorch Runtime]")
        et_text = transcribe_executorch(args.audio, model, et.buffer)
        print(f"  Result: {et_text}")

        if eager_text == et_text:
            print("\n✓ Transcriptions match!")
        else:
            print("\n✗ Transcriptions differ!")
            print(f"  Eager: {eager_text}")
            print(f"  ET:    {et_text}")

    print("\nDone!")


if __name__ == "__main__":
    main()
