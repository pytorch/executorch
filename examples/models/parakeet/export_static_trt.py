#!/usr/bin/env python3
"""
Static shape TensorRT export for Parakeet TDT model.

This is a workaround for the dynamic shape bug in the TRT encoder export.
By using static shapes, the attention masking and position encoding should
work correctly.

Usage:
    python export_static_trt.py --output-dir ./parakeet_trt_static
"""

import argparse
import os
import sys
import torch
from torch.export import export

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from executorch.backends.nvidia.tensorrt.partitioner import TensorRTPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program

# Import wrappers from the main export script
from export_parakeet_tdt import (
    EncoderWithProjection,
    DecoderStep,
    JointWithArgmax,
    PreprocessorWrapper,
    extract_tokenizer,
)

# Fixed audio length for static export (10 seconds at 16kHz)
STATIC_AUDIO_SAMPLES = 160000  # 10 seconds
STATIC_MEL_FRAMES = 1001  # Corresponding mel frames for 10s audio


def export_static(model, dtype=torch.float16, output_dir="./parakeet_trt_static"):
    """Export Parakeet model with STATIC shapes for TensorRT."""
    
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    programs = {}
    
    # ========== Preprocessor (static shape) ==========
    print("\nExporting preprocessor with static shapes...")
    preprocessor_wrapper = PreprocessorWrapper(model.preprocessor)
    preprocessor_wrapper.float()
    preprocessor_wrapper.eval()

    sample_audio = torch.randn(STATIC_AUDIO_SAMPLES, dtype=torch.float)
    sample_length = torch.tensor([STATIC_AUDIO_SAMPLES], dtype=torch.int64)
    
    # Force CPU path for preprocessor
    old_cuda_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    programs["preprocessor"] = export(
        preprocessor_wrapper,
        (sample_audio, sample_length),
        dynamic_shapes=None,  # STATIC - no dynamic shapes!
        strict=False,
    )
    torch.cuda.is_available = old_cuda_is_available
    
    # ========== Encoder (STATIC shape - key fix) ==========
    print("\nExporting encoder with STATIC shapes (workaround for dynamic shape bug)...")
    if device == "cuda":
        model.cuda()
    
    feat_in = getattr(model.encoder, "_feat_in", 128)
    # Use STATIC mel frame count
    audio_signal = torch.randn(1, feat_in, STATIC_MEL_FRAMES, dtype=dtype, device=device)
    length = torch.tensor([STATIC_MEL_FRAMES], dtype=torch.int64, device=device)
    
    encoder_with_proj = EncoderWithProjection(model.encoder, model.joint)
    encoder_with_proj.eval()
    if dtype == torch.float16:
        encoder_with_proj.half()
    
    programs["encoder"] = export(
        encoder_with_proj,
        (),
        kwargs={"audio_signal": audio_signal, "length": length},
        dynamic_shapes=None,  # STATIC - no dynamic shapes!
        strict=False,
    )
    
    # ========== Decoder (already static) ==========
    print("\nExporting decoder_step...")
    num_layers = model.decoder.pred_rnn_layers
    pred_hidden = model.decoder.pred_hidden
    decoder_step = DecoderStep(model.decoder, model.joint)
    decoder_step.eval()
    if dtype == torch.float16:
        decoder_step.half()
    
    token = torch.tensor([[0]], dtype=torch.long, device=device)
    h = torch.zeros(num_layers, 1, pred_hidden, dtype=dtype, device=device)
    c = torch.zeros(num_layers, 1, pred_hidden, dtype=dtype, device=device)
    programs["decoder_step"] = export(
        decoder_step,
        (token, h, c),
        dynamic_shapes=None,
        strict=False,
    )
    
    # ========== Joint (already static) ==========
    print("\nExporting joint...")
    joint_hidden = model.joint.joint_hidden
    num_token_classes = model.tokenizer.vocab_size + 1
    
    f_proj = torch.randn(1, 1, joint_hidden, dtype=dtype, device=device)
    g_proj = torch.randn(1, 1, joint_hidden, dtype=dtype, device=device)
    joint_module = JointWithArgmax(model.joint, num_token_classes)
    if dtype == torch.float16:
        joint_module.half()
    
    programs["joint"] = export(
        joint_module,
        (f_proj, g_proj),
        dynamic_shapes=None,
        strict=False,
    )
    
    # ========== Metadata ==========
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


def main():
    parser = argparse.ArgumentParser(
        description="Export Parakeet TDT with STATIC shapes for TensorRT (workaround)"
    )
    parser.add_argument(
        "--output-dir",
        default="./parakeet_trt_static",
        help="Output directory for exported model",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp32", "fp16"],
        default="fp16",
        help="Data type for export",
    )
    args = parser.parse_args()
    
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    
    # Load Parakeet model
    print("Loading Parakeet TDT model...")
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    model = EncDecRNNTBPEModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    model.eval()
    
    # Export with static shapes
    programs, metadata = export_static(model, dtype=dtype, output_dir=args.output_dir)
    
    # Lower to ExecuTorch with TensorRT
    print("\nLowering to ExecuTorch with TensorRT backend...")
    
    trt_partitioner = TensorRTPartitioner()
    xnnpack_partitioner = XnnpackPartitioner()
    
    # Build method programs
    method_programs = {}
    
    for name, prog in programs.items():
        print(f"  Lowering {name}...")
        if name == "preprocessor":
            edge = to_edge_transform_and_lower(prog, partitioner=[xnnpack_partitioner])
        else:
            edge = to_edge_transform_and_lower(prog, partitioner=[trt_partitioner])
        method_programs[name] = edge
    
    # Add metadata methods
    for key, value in metadata.items():
        print(f"  Adding metadata: {key}={value}")
        
        class MetaModule(torch.nn.Module):
            def __init__(self, v):
                super().__init__()
                if isinstance(v, float):
                    self.register_buffer("v", torch.tensor(v))
                else:
                    self.register_buffer("v", torch.tensor(v, dtype=torch.int64))
            def forward(self):
                return self.v
        
        meta_mod = MetaModule(value)
        meta_prog = export(meta_mod, ())
        edge = to_edge_transform_and_lower(meta_prog)
        method_programs[key] = edge
    
    # Merge all into single program
    print("\nMerging methods into single program...")
    merged = None
    method_names = list(method_programs.keys())
    
    for i, name in enumerate(method_names):
        edge_prog = method_programs[name]
        print(f"  [{i+1}/{len(method_names)}] Adding: {name}")
        
        if merged is None:
            # First program - rename 'forward' to the method name
            if name != "forward":
                merged = edge_prog.transform(lambda x: x)
                # Get the underlying program and rename
                old_progs = merged._edge_programs
                if "forward" in old_progs:
                    old_progs[name] = old_progs.pop("forward")
            else:
                merged = edge_prog
        else:
            # Merge subsequent programs
            try:
                # Get the program to merge
                prog_to_merge = edge_prog._edge_programs.get(
                    "forward", 
                    list(edge_prog._edge_programs.values())[0]
                )
                merged._edge_programs[name] = prog_to_merge
            except Exception as e:
                print(f"    Warning: merge issue for {name}: {e}")
    
    # Export to .pte
    print("\nExporting to .pte file...")
    et_prog = merged.to_executorch()
    
    output_path = os.path.join(args.output_dir, "model.pte")
    with open(output_path, "wb") as f:
        et_prog.write_to_file(f)
    print(f"Saved model to: {output_path}")
    
    # Copy tokenizer
    print("\nExtracting tokenizer...")
    extract_tokenizer(args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"Static shape export complete!")
    print(f"{'='*60}")
    print(f"  Model: {output_path}")
    print(f"  Static audio length: {STATIC_AUDIO_SAMPLES} samples ({STATIC_AUDIO_SAMPLES/16000:.1f}s)")
    print(f"  Static mel frames: {STATIC_MEL_FRAMES}")
    print(f"\nNOTE: Input audio must be exactly {STATIC_AUDIO_SAMPLES/16000:.1f}s long.")
    print(f"      Pad shorter audio or truncate longer audio before inference.")


if __name__ == "__main__":
    with torch.no_grad():
        main()
