#!/usr/bin/env python3
"""
Static export for Parakeet TDT with TensorRT.
Uses dynamic shapes for preprocessor (XNNPACK) but STATIC shapes for encoder (TRT fix).
"""
import os
import sys
import torch
from torch.export import export, Dim
import argparse

from executorch.backends.nvidia.tensorrt.partitioner import TensorRTPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig

# Constants 
MAX_AUDIO_SEC = 50  # Max audio duration in seconds (model limit)
SAMPLE_RATE = 16000
WINDOW_STRIDE = 0.01  # 10ms window stride

MAX_AUDIO_SAMPLES = int(SAMPLE_RATE * MAX_AUDIO_SEC)
MAX_MEL_FRAMES = int(MAX_AUDIO_SEC / WINDOW_STRIDE)  # 5000 frames for 50s


class PreprocessorWrapper(torch.nn.Module):
    def __init__(self, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor

    def forward(self, audio: torch.Tensor, length: torch.Tensor):
        mel, mel_len = self.preprocessor(
            input_signal=audio.unsqueeze(0),
            length=length,
        )
        return mel, mel_len


class EncoderWithProjection(torch.nn.Module):
    def __init__(self, encoder, joint):
        super().__init__()
        self.encoder = encoder
        self.enc_proj = joint.enc

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor):
        encoded, enc_len = self.encoder(audio_signal=audio_signal, length=length)
        f_proj = self.enc_proj(encoded)
        return f_proj, enc_len


class DecoderStep(torch.nn.Module):
    def __init__(self, decoder, joint):
        super().__init__()
        self.embedding = decoder.prediction.embed
        self.lstm = decoder.prediction.dec_rnn
        self.pred_proj = joint.pred

    def forward(self, token, h, c):
        embed = self.embedding(token)
        embed = embed.transpose(0, 1)
        out, (h_new, c_new) = self.lstm(embed, (h, c))
        out = out.transpose(0, 1)
        g_proj = self.pred_proj(out)
        return g_proj, h_new, c_new


class JointWithArgmax(torch.nn.Module):
    def __init__(self, joint, num_classes):
        super().__init__()
        self.joint_net = joint.joint_net
        self.num_classes = num_classes

    def forward(self, f: torch.Tensor, g: torch.Tensor):
        joint_in = f + g
        logits = self.joint_net(joint_in)
        token_logits = logits[:, :, :self.num_classes]
        duration_logits = logits[:, :, self.num_classes:]
        token_id = torch.argmax(token_logits, dim=-1, keepdim=True)
        duration_idx = torch.argmax(duration_logits, dim=-1, keepdim=True)
        return torch.cat([token_id, duration_idx], dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./parakeet_trt_static")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = "cuda"

    print("Loading Parakeet model...")
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    model = EncDecRNNTBPEModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    model.eval()
    model.cuda()

    programs = {}

    # ========== Preprocessor (DYNAMIC - uses XNNPACK, not TRT) ==========
    print("Exporting preprocessor (dynamic shapes, XNNPACK backend)...")
    preprocessor = PreprocessorWrapper(model.preprocessor)
    preprocessor.float()
    preprocessor.eval()

    sample_audio = torch.randn(MAX_AUDIO_SAMPLES, dtype=torch.float)
    sample_length = torch.tensor([MAX_AUDIO_SAMPLES], dtype=torch.int64)

    old_cuda = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    programs["preprocessor"] = export(
        preprocessor,
        (sample_audio, sample_length),
        dynamic_shapes={
            "audio": {0: Dim("audio_len", min=1600, max=MAX_AUDIO_SAMPLES)},
            "length": {},
        },
        strict=False,
    )
    torch.cuda.is_available = old_cuda

    # ========== Encoder (STATIC - key fix for TRT!) ==========
    print("Exporting encoder with STATIC shapes (TRT fix)...")
    feat_in = getattr(model.encoder, "_feat_in", 128)
    audio_signal = torch.randn(1, feat_in, MAX_MEL_FRAMES, dtype=dtype, device=device)
    length = torch.tensor([MAX_MEL_FRAMES], dtype=torch.int64, device=device)

    encoder = EncoderWithProjection(model.encoder, model.joint)
    encoder.eval()
    if dtype == torch.float16:
        encoder.half()

    programs["encoder"] = export(
        encoder,
        (),
        kwargs={"audio_signal": audio_signal, "length": length},
        dynamic_shapes=None,  # STATIC! This is the key fix.
        strict=False,
    )

    # ========== Decoder step (static) ==========
    print("Exporting decoder_step...")
    num_layers = model.decoder.pred_rnn_layers
    pred_hidden = model.decoder.pred_hidden
    
    decoder = DecoderStep(model.decoder, model.joint)
    decoder.eval()
    if dtype == torch.float16:
        decoder.half()

    token = torch.tensor([[0]], dtype=torch.long, device=device)
    h = torch.zeros(num_layers, 1, pred_hidden, dtype=dtype, device=device)
    c = torch.zeros(num_layers, 1, pred_hidden, dtype=dtype, device=device)

    programs["decoder_step"] = export(
        decoder,
        (token, h, c),
        dynamic_shapes=None,
        strict=False,
    )

    # ========== Joint (static) ==========
    print("Exporting joint...")
    joint_hidden = model.joint.joint_hidden
    num_token_classes = model.tokenizer.vocab_size + 1

    joint = JointWithArgmax(model.joint, num_token_classes)
    joint.eval()
    if dtype == torch.float16:
        joint.half()

    f = torch.randn(1, 1, joint_hidden, dtype=dtype, device=device)
    g = torch.randn(1, 1, joint_hidden, dtype=dtype, device=device)

    programs["joint"] = export(
        joint,
        (f, g),
        dynamic_shapes=None,
        strict=False,
    )

    # ========== Metadata ==========
    sample_rate = model.preprocessor._cfg.sample_rate
    window_stride = float(model.preprocessor._cfg.window_stride)
    encoder_subsampling = int(getattr(model.encoder, "subsampling_factor", 8))

    metadata = {
        "num_rnn_layers": num_layers,
        "pred_hidden": pred_hidden,
        "joint_hidden": joint_hidden,
        "vocab_size": model.tokenizer.vocab_size,
        "blank_id": model.tokenizer.vocab_size,
        "sample_rate": sample_rate,
        "window_stride": window_stride,
        "encoder_subsampling_factor": encoder_subsampling,
    }

    # ========== Lower to ExecuTorch ==========
    print("Lowering to ExecuTorch with TensorRT backend...")
    trt_partitioner = TensorRTPartitioner()
    xnnpack_partitioner = XnnpackPartitioner()

    edge_config = EdgeCompileConfig(_check_ir_validity=False)
    
    method_progs = {}
    for name, prog in programs.items():
        print(f"  Lowering {name}...")
        if name == "preprocessor":
            edge = to_edge_transform_and_lower(prog, partitioner=[xnnpack_partitioner], compile_config=edge_config)
        else:
            edge = to_edge_transform_and_lower(prog, partitioner=[trt_partitioner], compile_config=edge_config)
        method_progs[name] = edge

    # Add metadata methods
    for key, value in metadata.items():
        print(f"  Adding metadata: {key}={value}")
        class MetaModule(torch.nn.Module):
            def __init__(self, v):
                super().__init__()
                if isinstance(v, float):
                    self.register_buffer("val", torch.tensor(v))
                else:
                    self.register_buffer("val", torch.tensor(v, dtype=torch.int64))
            def forward(self):
                return self.val

        meta = MetaModule(value)
        meta_prog = export(meta, ())
        edge = to_edge_transform_and_lower(meta_prog, compile_config=edge_config)
        method_progs[key] = edge

    # Merge into single program
    print("Merging methods...")
    merged = None
    for name, edge in method_progs.items():
        if merged is None:
            merged = edge.transform(lambda x: x)
            if name != "forward":
                old_progs = merged._edge_programs
                if "forward" in old_progs:
                    old_progs[name] = old_progs.pop("forward")
        else:
            prog_to_merge = edge._edge_programs.get("forward", list(edge._edge_programs.values())[0])
            merged._edge_programs[name] = prog_to_merge

    print("Converting to ExecuTorch...")
    et_prog = merged.to_executorch()

    output_path = os.path.join(args.output_dir, "model.pte")
    print(f"Saving to {output_path}...")
    with open(output_path, "wb") as f:
        et_prog.write_to_file(f)

    # Extract tokenizer
    print("Extracting tokenizer...")
    import shutil
    tokenizer_path = None
    for path in [
        os.path.join(os.path.expanduser("~"), ".cache/huggingface/hub/models--nvidia--parakeet-tdt-0.6b-v3"),
    ]:
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith(".model"):
                    tokenizer_path = os.path.join(root, f)
                    break
    if tokenizer_path:
        shutil.copy(tokenizer_path, os.path.join(args.output_dir, "tokenizer.model"))
        print(f"  Copied tokenizer from {tokenizer_path}")

    print(f"\n{'='*60}")
    print("Static encoder export complete!")
    print(f"{'='*60}")
    print(f"  Model: {output_path}")
    print(f"  Encoder static mel frames: {MAX_MEL_FRAMES} (~{MAX_AUDIO_SEC}s audio)")
    print(f"\n  Key difference: Encoder uses STATIC shapes for TRT")
    print(f"                  (Preprocessor still uses dynamic shapes with XNNPACK)")

if __name__ == "__main__":
    with torch.no_grad():
        main()
