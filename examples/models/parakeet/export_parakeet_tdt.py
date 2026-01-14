"""Export nvidia/parakeet-tdt-0.6b-v3 components to ExecuTorch."""

import argparse
import os
import shutil
import tarfile
import tempfile

import torch
import torchaudio
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


class DecoderPredict(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.pred_hidden = decoder.pred_hidden
        self.pred_rnn_layers = getattr(decoder, "pred_rnn_layers", 2)

    def forward(
        self, token: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g, new_state = self.decoder.predict(y=token, state=[h, c], add_sos=False)
        return g, new_state[0], new_state[1]


def greedy_decode_executorch(
    encoder_output: torch.Tensor,
    encoder_len: int,
    program,
    blank_id: int,
    vocab_size: int,
    num_rnn_layers: int = 2,
    pred_hidden: int = 640,
    max_symbols_per_step: int = 10,
    durations: list[int] | None = None,
) -> list[int]:
    if durations is None:
        durations = [0, 1, 2, 3, 4]

    hypothesis = []
    num_token_classes = vocab_size + 1

    encoder_output = encoder_output.transpose(1, 2)

    proj_enc_method = program.load_method("joint_project_encoder")
    f_proj = proj_enc_method.execute([encoder_output.contiguous()])[0]

    decoder_predict_method = program.load_method("decoder_predict")
    proj_dec_method = program.load_method("joint_project_decoder")
    joint_method = program.load_method("joint")

    h = torch.zeros(num_rnn_layers, 1, pred_hidden)
    c = torch.zeros(num_rnn_layers, 1, pred_hidden)

    sos_g = torch.zeros(1, 1, pred_hidden)
    g_proj = proj_dec_method.execute([sos_g])[0]

    t = 0
    symbols_on_frame = 0

    while t < encoder_len:
        f_t = f_proj[:, t : t + 1, :].contiguous()

        joint_out = joint_method.execute([f_t, g_proj])

        full_logits = joint_out[0].squeeze()
        token_logits = full_logits[:num_token_classes]
        duration_logits = full_logits[num_token_classes:]

        k = token_logits.argmax().item()
        dur_idx = duration_logits.argmax().item()
        dur = durations[dur_idx]

        if k == blank_id:
            t += max(dur, 1)
            symbols_on_frame = 0
        else:
            hypothesis.append(k)

            token = torch.tensor([[k]], dtype=torch.long)
            result = decoder_predict_method.execute([token, h, c])
            g = result[0]
            h = result[1]
            c = result[2]

            g_proj = proj_dec_method.execute([g])[0]
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
        encoded = enc_result[0]
        encoded_len = enc_result[1].item()

        vocab_size = model.tokenizer.vocab_size
        tokens = greedy_decode_executorch(
            encoded,
            encoded_len,
            program,
            blank_id=vocab_size,
            vocab_size=vocab_size,
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


class JointAfterProjection(torch.nn.Module):
    def __init__(self, joint):
        super().__init__()
        self.joint = joint

    def forward(self, f, g):
        return self.joint.joint_after_projection(f, g)


class JointProjectEncoder(torch.nn.Module):
    def __init__(self, joint):
        super().__init__()
        self.joint = joint

    def forward(self, f):
        return self.joint.project_encoder(f)


class JointProjectDecoder(torch.nn.Module):
    def __init__(self, joint):
        super().__init__()
        self.joint = joint

    def forward(self, g):
        return self.joint.project_prednet(g)


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


def export_all(model):
    programs = {}

    preprocessor_wrapper = PreprocessorWrapper(model.preprocessor)
    preprocessor_wrapper.eval()
    sample_audio = torch.randn(16000 * 10)
    sample_length = torch.tensor([sample_audio.shape[0]], dtype=torch.int64)
    # The preprocessor definition changes if cuda is available (likely due to making it cuda graphable).
    # Unfortunately that new definition is not supported by export, so we need to stop that from happening.
    old_cuda_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    programs["preprocessor"] = export(
        preprocessor_wrapper,
        (sample_audio, sample_length),
        dynamic_shapes={
            "audio": {0: Dim("audio_len", min=1600, max=16000 * 600)},
            "length": {},
        },
        strict=False,
    )
    torch.cuda.is_available = old_cuda_is_available

    feat_in = getattr(model.encoder, "_feat_in", 128)
    audio_signal = torch.randn(1, feat_in, 100)
    length = torch.tensor([100], dtype=torch.int64)
    programs["encoder"] = export(
        model.encoder,
        (),
        kwargs={"audio_signal": audio_signal, "length": length},
        dynamic_shapes={"audio_signal": {2: Dim.AUTO}, "length": {}},
        strict=False,
    )

    decoder_predict = DecoderPredict(model.decoder)
    decoder_predict.eval()
    token = torch.tensor([[0]], dtype=torch.long)
    num_layers = model.decoder.pred_rnn_layers
    pred_hidden = model.decoder.pred_hidden
    h = torch.zeros(num_layers, 1, pred_hidden)
    c = torch.zeros(num_layers, 1, pred_hidden)
    programs["decoder_predict"] = export(
        decoder_predict,
        (token, h, c),
        dynamic_shapes={"token": {}, "h": {}, "c": {}},
        strict=False,
    )

    joint_hidden = model.joint.joint_hidden

    f_proj = torch.randn(1, 1, joint_hidden)
    g_proj = torch.randn(1, 1, joint_hidden)
    programs["joint"] = export(
        JointAfterProjection(model.joint),
        (f_proj, g_proj),
        dynamic_shapes={"f": {}, "g": {}},
        strict=False,
    )

    enc_output_dim = getattr(model.encoder, "_feat_out", 1024)

    programs["joint_project_encoder"] = export(
        JointProjectEncoder(model.joint),
        (torch.randn(1, 25, enc_output_dim),),
        dynamic_shapes={"f": {1: Dim("enc_time", min=1, max=60000)}},
        strict=False,
    )

    programs["joint_project_decoder"] = export(
        JointProjectDecoder(model.joint),
        (torch.randn(1, 1, pred_hidden),),
        dynamic_shapes={"g": {}},
        strict=False,
    )

    sample_rate = model.preprocessor._cfg.sample_rate
    metadata = {
        "num_rnn_layers": num_layers,
        "pred_hidden": pred_hidden,
        "joint_hidden": joint_hidden,
        "vocab_size": model.tokenizer.vocab_size,
        "blank_id": model.tokenizer.vocab_size,
        "sample_rate": sample_rate,
    }

    return programs, metadata


def lower_to_executorch(programs, metadata=None, backend="portable"):
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )

    partitioner = {}
    if "preprocessor" in programs:
        partitioner["preprocessor"] = [XnnpackPartitioner()]

    if backend == "xnnpack":

        print("\nLowering to ExecuTorch with XNNPACK...")
        for key in programs.keys():
            if key != "preprocessor":
                partitioner[key] = [XnnpackPartitioner()]

    elif backend in ("cuda", "cuda-windows"):
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from executorch.exir.backend.compile_spec_schema import CompileSpec
        from torch._inductor.decomposition import conv1d_to_conv2d

        print(
            f"\nLowering to ExecuTorch with CUDA{' (Windows)' if backend == 'cuda-windows' else ''}..."
        )

        for key, ep in programs.items():
            if key != "preprocessor":
                programs[key] = ep.run_decompositions(
                    {torch.ops.aten.conv1d.default: conv1d_to_conv2d}
                )

        for key in programs.keys():
            if key != "preprocessor":
                compile_specs = [CudaBackend.generate_method_name_compile_spec(key)]
                if backend == "cuda-windows":
                    compile_specs.append(
                        CompileSpec("platform", "windows".encode("utf-8"))
                    )
                partitioner[key] = [CudaPartitioner(compile_specs)]

    # else:
    #     print("\nLowering to ExecuTorch...")
    #     partitioner = []

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
    print(et_prog.exported_program("preprocessor").graph_module)
    return et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
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
        default="portable",
        choices=["portable", "xnnpack", "cuda", "cuda-windows"],
        help="Backend for acceleration (default: portable)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Extracting tokenizer...")
    extract_tokenizer(args.output_dir)

    print("Loading model...")
    model = load_model()

    print("\nExporting components...")
    programs, metadata = export_all(model)

    et = lower_to_executorch(programs, metadata=metadata, backend=args.backend)

    pte_path = os.path.join(args.output_dir, "parakeet_tdt.pte")
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
