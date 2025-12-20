#!/usr/bin/env python3
"""Export nvidia/parakeet-tdt-0.6b-v3 components to ExecuTorch."""

import os

import torch
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.passes import MemoryPlanningPass
from torch.export import Dim, export


def load_audio(audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
    """Load audio file and resample to target sample rate."""
    try:
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)
    except (ImportError, Exception):
        from scipy.io import wavfile

        sr, data = wavfile.read(audio_path)
        if data.dtype == "int16":
            data = data.astype("float32") / 32768.0
        elif data.dtype == "int32":
            data = data.astype("float32") / 2147483648.0
        waveform = torch.from_numpy(data).unsqueeze(0)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        try:
            import torchaudio

            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        except ImportError:
            from scipy import signal

            num_samples = int(len(waveform[0]) * sample_rate / sr)
            resampled = signal.resample(waveform[0].numpy(), num_samples)
            waveform = torch.from_numpy(resampled).unsqueeze(0).float()

    return waveform


def greedy_decode_eager(encoder_output: torch.Tensor, encoder_len: torch.Tensor, model) -> list[int]:
    """Greedy decode using NeMo's built-in decoding."""
    hypotheses = model.decoding.rnnt_decoder_predictions_tensor(
        encoder_output=encoder_output,
        encoded_lengths=encoder_len,
        return_hypotheses=True,
    )
    return hypotheses[0].y_sequence


class DecoderPredict(torch.nn.Module):
    """Wrapper for decoder.predict() with LSTM state."""

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
    """TDT duration-aware greedy decode using ExecuTorch runtime."""
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

    # Debug: print first few tokens
    debug_count = 0

    # Scan over the encoder output
    while t < encoder_len:
        f_t = f_proj[:, t : t + 1, :].contiguous()

        joint_out = joint_method.execute([f_t, g_proj])

        full_logits = joint_out[0].squeeze()
        token_logits = full_logits[:num_token_classes]
        duration_logits = full_logits[num_token_classes:]

        k = token_logits.argmax().item()
        dur_idx = duration_logits.argmax().item()
        dur = durations[dur_idx]

        # TDT decoding: joint network outputs both token logits and duration logits.
        # - If blank: skip forward by predicted duration (min 1 frame)
        # - If token: emit it, update decoder state, advance by duration.
        #   Duration=0 means "emit another token on this frame" (up to max_symbols_per_step).
        if k == blank_id:
            t += max(dur, 1)
            symbols_on_frame = 0
        else:
            if debug_count < 20:
                print(f"Token[{debug_count}]: t={t} k={k} dur={dur}")
                debug_count += 1
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
    """Transcribe audio file using ExecuTorch runtime."""
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(et_buffer)

    with torch.no_grad():
        audio = load_audio(audio_path)

        mel, mel_len = model.preprocessor(input_signal=audio, length=torch.tensor([audio.shape[1]]))
        print(mel.shape)

        encoder_method = program.load_method("encoder")
        enc_result = encoder_method.execute([mel, mel_len])
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
    """Transcribe audio file using eager PyTorch model."""
    with torch.no_grad():
        audio = load_audio(audio_path)
        mel, mel_len = model.preprocessor(input_signal=audio, length=torch.tensor([audio.shape[1]]))
        encoded, encoded_len = model.encoder(audio_signal=mel, length=mel_len)
        tokens = greedy_decode_eager(encoded, encoded_len, model)
        return model.tokenizer.ids_to_text(tokens)


def load_model():
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    model.eval()
    return model


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


def export_all(model):
    """Export all components, return dict of ExportedPrograms."""
    programs = {}

    feat_in = getattr(model.encoder, "_feat_in", 128)
    print(f"Encoder feat_in: {feat_in}")
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

    f_proj = torch.randn(1, 1, 640)
    g_proj = torch.randn(1, 1, 640)
    programs["joint"] = export(
        JointAfterProjection(model.joint),
        (f_proj, g_proj),
        dynamic_shapes={"f": {}, "g": {}},
        strict=False,
    )

    programs["joint_project_encoder"] = export(
        JointProjectEncoder(model.joint),
        (torch.randn(1, 25, 1024),),
        dynamic_shapes={"f": {1: Dim("enc_time", min=1, max=60000)}},
        strict=False,
    )

    pred_hidden = getattr(model.decoder, "pred_hidden", 640)
    programs["joint_project_decoder"] = export(
        JointProjectDecoder(model.joint),
        (torch.randn(1, 1, pred_hidden),),
        dynamic_shapes={"g": {}},
        strict=False,
    )

    return programs


def lower_to_executorch(programs, backend="portable"):
    """Lower all ExportedPrograms to ExecuTorch."""
    partitioner = None

    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

        print("\nLowering to ExecuTorch with XNNPACK...")
        partitioner = [XnnpackPartitioner()]

    elif backend in ("cuda", "cuda-windows"):
        from torch._inductor.decomposition import conv1d_to_conv2d

        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from executorch.exir.backend.compile_spec_schema import CompileSpec

        print(f"\nLowering to ExecuTorch with CUDA{' (Windows)' if backend == 'cuda-windows' else ''}...")

        # Decompose conv1d to conv2d for Triton kernel generation
        for key, ep in programs.items():
            programs[key] = ep.run_decompositions({torch.ops.aten.conv1d.default: conv1d_to_conv2d})

        partitioner = {}
        for key in programs.keys():
            compile_specs = [CudaBackend.generate_method_name_compile_spec(key)]
            if backend == "cuda-windows":
                compile_specs.append(CompileSpec("platform", "windows".encode("utf-8")))
            partitioner[key] = [CudaPartitioner(compile_specs)]

    else:
        print("\nLowering to ExecuTorch")
        partitioner = []

    et_prog = to_edge_transform_and_lower(
        programs,
        partitioner=partitioner,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
    )
    return et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )


def export_preprocessor(model, output_dir: str, backend: str = "portable"):
    """Export NeMo's preprocessor to ExecuTorch."""
    
    class PreprocessorWrapper(torch.nn.Module):
        def __init__(self, preprocessor):
            super().__init__()
            self.preprocessor = preprocessor
        
        def forward(self, audio: torch.Tensor) -> torch.Tensor:
            # audio is 1D: [num_samples]
            # Add batch dimension and compute length
            audio_signal = audio.unsqueeze(0)  # [1, num_samples]
            length = torch.tensor([audio.shape[0]], dtype=torch.int64)
            
            mel, mel_len = self.preprocessor(input_signal=audio_signal, length=length)
            return mel
    
    wrapper = PreprocessorWrapper(model.preprocessor)
    wrapper.eval()
    
    # Export with dynamic audio length
    sample_audio = torch.randn(16000 * 10)  # 10 seconds
    
    preprocessor_ep = export(
        wrapper,
        (sample_audio,),
        dynamic_shapes={"audio": {0: Dim("audio_len", min=1600, max=16000 * 600)}},
        strict=False,
    )
    
    partitioner = []
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        partitioner = [XnnpackPartitioner()]
    
    et_prog = to_edge_transform_and_lower(
        {"forward": preprocessor_ep},
        partitioner=partitioner,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
    )
    
    et_preprocessor = et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )
    
    pte_path = os.path.join(output_dir, "parakeet_preprocessor.pte")
    print(f"Saving preprocessor to: {pte_path}")
    with open(pte_path, "wb") as f:
        et_preprocessor.write_to_file(f)
    
    return pte_path


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./parakeet_tdt_exports")
    parser.add_argument("--audio", type=str, help="Path to audio file for transcription test")
    parser.add_argument(
        "--backend",
        choices=["portable", "xnnpack", "cuda", "cuda-windows"],
        default="portable",
        help="Backend for acceleration",
    )
    parser.add_argument(
        "--export-preprocessor",
        action="store_true",
        help="Export NeMo's preprocessor to ExecuTorch",
    )
    parser.add_argument(
        "--test-preprocessor",
        type=str,
        help="Test exported preprocessor against NeMo's native preprocessor",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...")
    model = load_model()

    if args.test_preprocessor:
        print("\nTesting preprocessor...")
        from executorch.runtime import Runtime
        
        audio = load_audio(args.test_preprocessor)
        audio_1d = audio.squeeze(0)  # [num_samples]
        
        print(f"Python audio shape: {audio.shape}, first 5 samples: {audio[0, :5].tolist()}")
        
        # NeMo's native preprocessor
        mel_native, mel_len_native = model.preprocessor(
            input_signal=audio, 
            length=torch.tensor([audio.shape[1]])
        )
        print(f"NeMo mel shape: {mel_native.shape}, mel_len: {mel_len_native.item()}")
        
        # Exported preprocessor
        pte_path = os.path.join(args.output_dir, "parakeet_preprocessor.pte")
        with open(pte_path, "rb") as f:
            runtime = Runtime.get()
            program = runtime.load_program(f.read())
            method = program.load_method("forward")
            mel_exported = method.execute([audio_1d])[0]
        print(f"Exported mel shape: {mel_exported.shape}")
        
        # Compare
        mel_native_np = mel_native.numpy()
        mel_exported_np = mel_exported.numpy()
        
        max_diff = abs(mel_native_np - mel_exported_np).max()
        mean_diff = abs(mel_native_np - mel_exported_np).mean()
        print(f"Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        
        if max_diff < 1e-4:
            print("✓ Preprocessors match!")
        else:
            print("✗ Preprocessors differ!")
            # Print first few values
            print(f"Native [0,0,:5]: {mel_native_np[0,0,:5]}")
            print(f"Exported [0,0,:5]: {mel_exported_np[0,0,:5]}")
        return

    if args.export_preprocessor:
        print("\nExporting preprocessor...")
        export_preprocessor(model, args.output_dir, args.backend)
        print("Preprocessor exported!")
        if not args.audio:
            return

    print("\nExporting components...")
    programs = export_all(model)

    et = lower_to_executorch(programs, backend=args.backend)

    # Save the .pte file
    pte_path = os.path.join(args.output_dir, "parakeet_tdt.pte")
    print(f"\nSaving ExecuTorch program to: {pte_path}")
    with open(pte_path, "wb") as f:
        et.write_to_file(f)
    print(f"Saved {os.path.getsize(pte_path) / (1024 * 1024):.1f} MB")

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
