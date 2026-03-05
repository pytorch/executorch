"""Isolate which op in a single conformer layer causes TRT divergence.

Usage:
  python debug_conformer_ops.py <test_name>

Where test_name is one of:
  ln, ff1, ff2, selfattn, conv, matmul, bn, all_noproj, single_layer
"""

import logging
import os
import sys
import torch
import torch.nn as nn
from torch.export import Dim, export

os.environ.setdefault("TRT_LOG_LEVEL", "2")
logging.getLogger("executorch.backends.nvidia.tensorrt").setLevel(logging.WARNING)
logging.getLogger("executorch.backends.nvidia.tensorrt.backend").setLevel(logging.WARNING)

from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from executorch.runtime import Runtime

from examples.models.parakeet.export_parakeet_tdt import load_model, load_audio

AUDIO_PATH = "/home/dev/models/parakeet_trt_fp32/output30.wav"


def get_trt_partitioner():
    from executorch.backends.nvidia.tensorrt.compile_spec import (
        TensorRTCompileSpec, TensorRTPrecision,
    )
    from executorch.backends.nvidia.tensorrt.partitioner import TensorRTPartitioner
    compile_specs = TensorRTCompileSpec(precision=TensorRTPrecision.FP32).to_compile_specs()
    return [TensorRTPartitioner(compile_specs=compile_specs)]


def export_and_compare(name, module, example_kwargs, dynamic_shapes, real_inputs, eager_out):
    """Export module with TRT, run it, and compare to eager output."""
    print(f"  Exporting {name}...", flush=True)
    ep = export(module, (), kwargs=example_kwargs, dynamic_shapes=dynamic_shapes, strict=False)

    # Print ops in the graph
    from collections import Counter
    op_counts = Counter()
    for node in ep.graph_module.graph.nodes:
        if node.op == "call_function":
            if hasattr(node.target, "_schema"):
                schema = node.target._schema
                op_name = schema.name.replace("::", ".")
                overload = getattr(schema, "overload_name", "")
                target = f"{op_name}.{overload}" if overload else f"{op_name}.default"
            else:
                target = str(node.target)
            op_counts[target] += 1
    print(f"  Ops: {dict(op_counts)}")

    print(f"  Lowering {name} to TRT...", flush=True)
    et_prog = to_edge_transform_and_lower(
        {"forward": ep},
        partitioner={"forward": get_trt_partitioner()},
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
    )
    et = et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False, alloc_graph_output=True),
        ),
    )

    print(f"  Running {name} via ET runtime...", flush=True)
    runtime = Runtime.get()
    program = runtime.load_program(et.buffer)
    method = program.load_method("forward")
    result = method.execute(real_inputs)

    et_out = result[0]
    if hasattr(et_out, 'numpy'):
        et_out = torch.tensor(et_out.numpy())
    diff = (eager_out.float() - et_out.float()).abs()
    print(f"  RESULT {name}: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}", flush=True)
    return diff.max().item()


class LayerNormTest(nn.Module):
    def __init__(self, ln):
        super().__init__()
        self.ln = ln
    def forward(self, x: torch.Tensor):
        return self.ln(x)


class FeedForwardTest(nn.Module):
    def __init__(self, norm, ff):
        super().__init__()
        self.norm = norm
        self.ff = ff
    def forward(self, x: torch.Tensor):
        return self.ff(self.norm(x))


class ConvModuleTest(nn.Module):
    def __init__(self, norm, conv):
        super().__init__()
        self.norm = norm
        self.conv = conv
    def forward(self, x: torch.Tensor):
        return self.conv(self.norm(x))


class SelfAttnTest(nn.Module):
    def __init__(self, norm, self_attn, pos_emb):
        super().__init__()
        self.norm = norm
        self.self_attn = self_attn
        self.register_buffer("pos_emb", pos_emb)
    def forward(self, x: torch.Tensor):
        y = self.norm(x)
        out = self.self_attn(query=y, key=y, value=y, mask=None, pos_emb=self.pos_emb)
        if isinstance(out, tuple):
            out = out[0]
        return out


class MatMulTest(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight)
    def forward(self, x: torch.Tensor):
        return torch.matmul(x, self.weight.t())


class BatchNormTest(nn.Module):
    def __init__(self, bn):
        super().__init__()
        self.bn = bn
    def forward(self, x: torch.Tensor):
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class FullLayerWithPosEmb(nn.Module):
    """Full single conformer layer with positional embedding baked in."""
    def __init__(self, layer, pos_emb):
        super().__init__()
        self.layer = layer
        self.register_buffer("pos_emb", pos_emb)
    def forward(self, x: torch.Tensor):
        out = self.layer(x=x, att_mask=None, pos_emb=self.pos_emb)
        if isinstance(out, tuple):
            out = out[0]
        return out


def get_pre_encoded_input(model):
    """Get the pre-encoded input that conformer layers receive."""
    with torch.no_grad():
        audio = load_audio(AUDIO_PATH, sample_rate=16000)
        mel, mel_len = model.preprocessor(
            input_signal=audio, length=torch.tensor([audio.shape[1]])
        )
        x = mel.transpose(1, 2)
        x, lengths = model.encoder.pre_encode(x=x, lengths=mel_len)
        if isinstance(x, tuple):
            x = x[0]
        x, pos_emb = model.encoder.pos_enc(x=x)
    return x, pos_emb


def main():
    test_name = sys.argv[1] if len(sys.argv) > 1 else "all"

    print("Loading NeMo model...")
    model = load_model()
    model.eval()

    print("Getting pre-encoded input...")
    x, pos_emb = get_pre_encoded_input(model)
    print(f"Input: shape={tuple(x.shape)}")

    layer0 = model.encoder.layers[0]
    example = {"x": torch.randn_like(x)}
    # Use static shapes to avoid dynamic shape pybind issue
    dyn = {"x": {}}

    tests = {}

    if test_name in ("ln", "all"):
        print("\n=== Layer Norms ===")
        for ln_name in ["norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"]:
            ln = getattr(layer0, ln_name, None)
            if ln is None:
                continue
            wrapper = LayerNormTest(ln)
            wrapper.eval()
            with torch.no_grad():
                eager_out = wrapper(x)
            tests[f"LN_{ln_name}"] = export_and_compare(
                f"LN_{ln_name}", wrapper, example, dyn, [x], eager_out
            )

    if test_name in ("ff1", "all"):
        print("\n=== Feed-Forward 1 ===")
        wrapper = FeedForwardTest(layer0.norm_feed_forward1, layer0.feed_forward1)
        wrapper.eval()
        with torch.no_grad():
            eager_out = wrapper(x)
        tests["FF1"] = export_and_compare("FF1", wrapper, example, dyn, [x], eager_out)

    if test_name in ("ff2", "all"):
        print("\n=== Feed-Forward 2 ===")
        wrapper = FeedForwardTest(layer0.norm_feed_forward2, layer0.feed_forward2)
        wrapper.eval()
        with torch.no_grad():
            eager_out = wrapper(x)
        tests["FF2"] = export_and_compare("FF2", wrapper, example, dyn, [x], eager_out)

    if test_name in ("selfattn", "all"):
        print("\n=== Self-Attention ===")
        wrapper = SelfAttnTest(layer0.norm_self_att, layer0.self_attn, pos_emb)
        wrapper.eval()
        with torch.no_grad():
            eager_out = wrapper(x)
        tests["SelfAttn"] = export_and_compare("SelfAttn", wrapper, example, dyn, [x], eager_out)

    if test_name in ("conv", "all"):
        print("\n=== Conv Module ===")
        wrapper = ConvModuleTest(layer0.norm_conv, layer0.conv)
        wrapper.eval()
        with torch.no_grad():
            eager_out = wrapper(x)
        tests["ConvModule"] = export_and_compare("ConvModule", wrapper, example, dyn, [x], eager_out)

    if test_name in ("matmul", "all"):
        print("\n=== MatMul (Q projection) ===")
        w = layer0.self_attn.linear_q.weight.data.clone()
        wrapper = MatMulTest(w)
        wrapper.eval()
        with torch.no_grad():
            eager_out = wrapper(x)
        tests["MatMul_Q"] = export_and_compare("MatMul_Q", wrapper, example, dyn, [x], eager_out)

    if test_name in ("bn", "all"):
        print("\n=== BatchNorm ===")
        if hasattr(layer0.conv, "batch_norm"):
            bn = layer0.conv.batch_norm
            wrapper = BatchNormTest(bn)
            wrapper.eval()
            with torch.no_grad():
                eager_out = wrapper(x)
            tests["BatchNorm"] = export_and_compare("BatchNorm", wrapper, example, dyn, [x], eager_out)

    if test_name in ("single_layer", "all"):
        print("\n=== Full Single Conformer Layer (with pos_emb) ===")
        wrapper = FullLayerWithPosEmb(layer0, pos_emb)
        wrapper.eval()
        with torch.no_grad():
            eager_out = wrapper(x)
        tests["SingleLayer"] = export_and_compare("SingleLayer", wrapper, example, dyn, [x], eager_out)

    if tests:
        print("\n=== Summary ===")
        for name, diff in tests.items():
            status = "OK" if diff < 0.01 else "WARN" if diff < 0.1 else "BAD"
            print(f"  {name}: max_diff={diff:.6f} [{status}]")


if __name__ == "__main__":
    main()
