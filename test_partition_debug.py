#!/usr/bin/env python3
"""Debug: check how the graph gets partitioned with real mask logic.
Show what ops go to TRT vs stay on CPU."""
import sys
import torch
import torch.nn as nn
from collections import Counter

sys.path.insert(0, "/home/gasoonjia/trt/executorch/examples/models/parakeet")

from executorch.exir import to_edge
from executorch.backends.nvidia.tensorrt.partitioner import TensorRTPartitioner
from export_parakeet_tdt import PreprocessorWrapper

print("Loading model...")
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/parakeet-tdt-0.6b-v3", map_location="cpu"
)
model.eval()
encoder = model.encoder

# Prepare mel
import soundfile as sf
data, sr = sf.read("/home/gasoonjia/trt/executorch/examples/models/parakeet/real_speech.wav", dtype="float32")
if data.ndim > 1:
    data = data.mean(axis=1)
audio_1d = torch.from_numpy(data)
audio_len = torch.tensor([audio_1d.shape[0]], dtype=torch.int64)
preproc = PreprocessorWrapper(model.preprocessor)
preproc.float().eval()
with torch.no_grad():
    mel, mel_len = preproc(audio_1d, audio_len)
mel_len_val = mel_len.item()
mel_len_t = torch.tensor([mel_len_val], dtype=torch.int64)


class Encoder1BlockRealMask(nn.Module):
    """1 Conformer block with real mask logic."""
    def __init__(self, enc):
        super().__init__()
        self.pre_encode = enc.pre_encode
        self.pos_enc = enc.pos_enc
        self.layers = nn.ModuleList([enc.layers[0]])

    def forward(self, audio_signal, length):
        audio_signal = torch.transpose(audio_signal, 1, 2)
        audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
        audio_signal, pos_emb = self.pos_enc(x=audio_signal)
        max_len = audio_signal.shape[1]

        # Real mask creation
        att_mask = torch.ones(1, max_len, max_len, dtype=torch.bool, device=audio_signal.device)
        pad_mask = torch.arange(0, max_len, device=audio_signal.device).expand(
            length.size(0), -1
        ) < length.unsqueeze(-1)
        pad_mask_for_att = pad_mask.unsqueeze(1).repeat([1, max_len, 1])
        pad_mask_for_att = torch.logical_and(pad_mask_for_att, pad_mask_for_att.transpose(1, 2))
        att_mask = torch.logical_and(pad_mask_for_att, att_mask)
        att_mask = ~att_mask
        pad_mask = ~pad_mask

        for layer in self.layers:
            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
            )
        return audio_signal, length


class Encoder1BlockSimpleMask(nn.Module):
    """1 Conformer block with simple mask logic."""
    def __init__(self, enc):
        super().__init__()
        self.pre_encode = enc.pre_encode
        self.pos_enc = enc.pos_enc
        self.layers = nn.ModuleList([enc.layers[0]])

    def forward(self, audio_signal, length):
        audio_signal = torch.transpose(audio_signal, 1, 2)
        audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
        audio_signal, pos_emb = self.pos_enc(x=audio_signal)
        max_len = audio_signal.shape[1]

        pad_mask = torch.arange(max_len, device=audio_signal.device).unsqueeze(0) < length.unsqueeze(1)
        att_mask = pad_mask.unsqueeze(1).unsqueeze(1)

        for layer in self.layers:
            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
            )
        return audio_signal, length


def analyze_partitions(name, submodel, mel, mel_len_t):
    print(f"\n{'='*70}")
    print(f"Analyzing: {name}")
    print(f"{'='*70}")

    with torch.no_grad():
        ep = torch.export.export(submodel, (mel,), kwargs={"length": mel_len_t})

    # Count ops in exported graph
    export_ops = Counter()
    for n in ep.graph.nodes:
        if n.op == "call_function":
            export_ops[str(n.target)] += 1
    print(f"\nExported graph: {len(list(ep.graph.nodes))} nodes")
    print(f"  call_function ops: {sum(export_ops.values())}")

    # Check for masked_fill
    for op, cnt in sorted(export_ops.items()):
        if "mask" in op.lower() or "where" in op.lower() or "fill" in op.lower():
            print(f"  MASK-RELATED: {cnt:3d}x {op}")

    # Convert to edge
    edge = to_edge(ep)

    # Count ops in edge IR (after decomposition)
    edge_ops = Counter()
    edge_graph = edge.exported_program().graph
    for n in edge_graph.nodes:
        if n.op == "call_function":
            edge_ops[str(n.target)] += 1
    print(f"\nEdge IR: {len(list(edge_graph.nodes))} nodes")
    print(f"  call_function ops: {sum(edge_ops.values())}")
    for op, cnt in sorted(edge_ops.items()):
        if "mask" in op.lower() or "where" in op.lower() or "fill" in op.lower():
            print(f"  MASK-RELATED: {cnt:3d}x {op}")

    # Partition
    edge = edge.to_backend(TensorRTPartitioner())
    final_graph = edge.exported_program().graph

    # Analyze partitioned graph
    trt_delegates = 0
    cpu_ops = Counter()
    delegate_inputs = {}
    for n in final_graph.nodes:
        if n.op == "call_function":
            op_name = str(n.target)
            if "executorch_call_delegate" in op_name:
                trt_delegates += 1
                # Count inputs to this delegate
                num_inputs = len(n.args) - 1  # first arg is the delegate itself
                delegate_inputs[f"delegate_{trt_delegates}"] = num_inputs
                print(f"\n  TRT Delegate #{trt_delegates}: {num_inputs} inputs")
                for i, arg in enumerate(n.args[1:]):
                    if isinstance(arg, torch.fx.Node):
                        val = arg.meta.get("val", None)
                        if isinstance(val, torch.Tensor):
                            print(f"    input[{i}]: {arg.name} shape={list(val.shape)} dtype={val.dtype}")
                        elif isinstance(val, (list, tuple)):
                            shapes = [list(v.shape) if isinstance(v, torch.Tensor) else str(v) for v in val]
                            print(f"    input[{i}]: {arg.name} shapes={shapes}")
                        else:
                            print(f"    input[{i}]: {arg.name} val_type={type(val).__name__}")
            else:
                cpu_ops[op_name] += 1

    print(f"\n  Total TRT delegates: {trt_delegates}")
    if cpu_ops:
        print(f"  CPU ops ({sum(cpu_ops.values())} total):")
        for op, cnt in sorted(cpu_ops.items(), key=lambda x: -x[1]):
            print(f"    {cnt:3d}x {op}")
    else:
        print(f"  CPU ops: NONE (fully delegated)")

    # Show new vs removed ops (what got decomposed)
    new_in_edge = set(edge_ops) - set(export_ops)
    removed_in_edge = set(export_ops) - set(edge_ops)
    if removed_in_edge:
        print(f"\n  Ops decomposed (in export, not in edge):")
        for op in sorted(removed_in_edge):
            print(f"    {export_ops[op]:3d}x {op}")
    if new_in_edge:
        print(f"\n  Ops added by decomposition (in edge, not in export):")
        for op in sorted(new_in_edge):
            print(f"    {edge_ops[op]:3d}x {op}")


# Test both
simple = Encoder1BlockSimpleMask(encoder)
simple.eval()
analyze_partitions("1-block SIMPLE mask", simple, mel, mel_len_t)

real = Encoder1BlockRealMask(encoder)
real.eval()
analyze_partitions("1-block REAL mask", real, mel, mel_len_t)
