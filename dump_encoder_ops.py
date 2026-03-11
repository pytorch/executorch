#!/usr/bin/env python3
"""Dump the FX graph ops that would be sent to TRT converters for the encoder."""
import sys
import torch
from collections import Counter

sys.path.insert(0, "/home/gasoonjia/trt/executorch/examples/models/parakeet")
from export_parakeet_tdt import EncoderWithProjection

print("Loading model...")
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/parakeet-tdt-0.6b-v3", map_location="cpu"
)
model.eval()

encoder_with_proj = EncoderWithProjection(model.encoder, model.joint)
encoder_with_proj.eval()

# Trace with export
print("\nExporting FX graph...")
T_mel = 5000
mel = torch.randn(1, 128, T_mel)
mel_len = torch.tensor([T_mel], dtype=torch.int64)

with torch.no_grad():
    ep = torch.export.export(
        encoder_with_proj,
        (mel,),
        kwargs={"length": mel_len},
    )

print(f"\nFX graph has {len(list(ep.graph.nodes))} nodes")
print(f"\nOperation counts:")

op_counts = Counter()
op_details = []
for node in ep.graph.nodes:
    if node.op == "call_function":
        op_name = str(node.target)
        # Get short name
        if hasattr(node.target, "__name__"):
            short = node.target.__name__
        else:
            short = op_name.split(".")[-1] if "." in op_name else op_name
        op_counts[op_name] += 1

        # Get output shape
        val = node.meta.get("val", None)
        if val is not None and isinstance(val, torch.Tensor):
            shape_str = str(list(val.shape))
        elif val is not None and isinstance(val, (tuple, list)):
            shapes = []
            for v in val:
                if isinstance(v, torch.Tensor):
                    shapes.append(str(list(v.shape)))
                else:
                    shapes.append(str(v))
            shape_str = ", ".join(shapes)
        else:
            shape_str = str(val)
        op_details.append((node.name, op_name, shape_str))

# Print summary
for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
    print(f"  {count:4d}x {op}")

# Print first 50 ops to show the pipeline
print(f"\nFirst 50 call_function nodes:")
for name, op, shape in op_details[:50]:
    print(f"  {name:40s} | {op:50s} | {shape}")

print(f"\nLast 20 call_function nodes:")
for name, op, shape in op_details[-20:]:
    print(f"  {name:40s} | {op:50s} | {shape}")
