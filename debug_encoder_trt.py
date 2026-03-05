"""Debug encoder TRT vs eager divergence by layer-by-layer bisection."""

import logging
import os
import sys
import torch
from collections import Counter
from torch.export import Dim, export

# Suppress TRT verbose logging before importing anything TRT-related
os.environ.setdefault("TRT_LOG_LEVEL", "2")
try:
    import tensorrt as trt
    trt_logger = trt.Logger(trt.Logger.WARNING)
except ImportError:
    pass

logging.getLogger("executorch.backends.nvidia.tensorrt").setLevel(logging.WARNING)

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
        TensorRTCompileSpec,
        TensorRTPrecision,
    )
    from executorch.backends.nvidia.tensorrt.partitioner import TensorRTPartitioner
    compile_specs = TensorRTCompileSpec(precision=TensorRTPrecision.FP32).to_compile_specs()
    return [TensorRTPartitioner(compile_specs=compile_specs)]


def export_and_run_trt(module, example_kwargs, dynamic_shapes, real_inputs):
    ep = export(module, (), kwargs=example_kwargs, dynamic_shapes=dynamic_shapes, strict=False)
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
    runtime = Runtime.get()
    program = runtime.load_program(et.buffer)
    method = program.load_method("forward")
    return method.execute(real_inputs)


def et_to_torch(val):
    if hasattr(val, 'numpy'):
        return torch.tensor(val.numpy())
    return val if isinstance(val, torch.Tensor) else torch.tensor(val)


def compare_tensors(name, eager, et):
    et = et_to_torch(et)
    diff = (eager.float() - et.float()).abs()
    print(f"  {name}: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")
    return diff.max().item()


class EncoderFirstNLayers(torch.nn.Module):
    def __init__(self, encoder, num_layers):
        super().__init__()
        self.encoder = encoder
        self.num_layers = num_layers
        self._orig_layers = list(encoder.layers)

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor):
        import torch.nn as nn
        self.encoder.layers = nn.ModuleList(self._orig_layers[:self.num_layers])
        try:
            encoded, enc_len = self.encoder(audio_signal=audio_signal, length=length)
        finally:
            self.encoder.layers = nn.ModuleList(self._orig_layers)
        return encoded, enc_len


def redirect_stderr_to_devnull():
    """Context manager to silence C++ stderr output (TRT verbose logs)."""
    import contextlib
    @contextlib.contextmanager
    def suppressed():
        old_fd = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        try:
            yield
        finally:
            os.dup2(old_fd, 2)
            os.close(devnull)
            os.close(old_fd)
    return suppressed()


def step2_layer_bisection(model, mel_eager, mel_len_eager):
    """Compare encoder output with different numbers of conformer layers."""
    print("\n" + "=" * 70)
    print("STEP 2: Layer-by-layer bisection")
    print("=" * 70)
    sys.stdout.flush()

    total_layers = len(model.encoder.layers)
    print(f"Encoder has {total_layers} conformer layers")

    feat_in = getattr(model.encoder, "_feat_in", 128)
    max_mel_frames = mel_eager.shape[2]

    test_counts = [1, 2, 4, 8, 12, 18, 24]
    test_counts = [n for n in test_counts if n <= total_layers]
    if total_layers not in test_counts:
        test_counts.append(total_layers)

    results = {}

    for n_layers in test_counts:
        print(f"\n--- {n_layers} layer(s) ---", flush=True)

        wrapper = EncoderFirstNLayers(model.encoder, n_layers)
        wrapper.eval()

        with torch.no_grad():
            out_eager, len_eager = wrapper(audio_signal=mel_eager, length=mel_len_eager)

        audio_signal = torch.randn(1, feat_in, max_mel_frames)
        length = torch.tensor([max_mel_frames], dtype=torch.int64)

        try:
            with redirect_stderr_to_devnull():
                et_result = export_and_run_trt(
                    wrapper,
                    {"audio_signal": audio_signal, "length": length},
                    {"audio_signal": {2: Dim.AUTO}, "length": {}},
                    [mel_eager, mel_len_eager],
                )
            out_et = et_result[0]
            max_diff = compare_tensors(f"encoder_{n_layers}L", out_eager, out_et)
            results[n_layers] = max_diff
        except Exception as e:
            print(f"  ERROR: {e}")
            results[n_layers] = -1.0

        sys.stdout.flush()

    print("\n--- Summary ---")
    for n, d in sorted(results.items()):
        status = "OK" if d < 0.01 else "WARN" if d < 0.1 else "BAD" if d > 0 else "ERR"
        print(f"  {n:3d} layers: max_diff={d:.6f} [{status}]")

    return results


def main():
    print("Loading NeMo model...")
    model = load_model()
    model.eval()

    print("Loading audio and running preprocessor...")
    with torch.no_grad():
        audio = load_audio(AUDIO_PATH, sample_rate=16000)
        mel_eager, mel_len_eager = model.preprocessor(
            input_signal=audio, length=torch.tensor([audio.shape[1]])
        )
    print(f"Mel: shape={tuple(mel_eager.shape)}, len={mel_len_eager.item()}")

    results = step2_layer_bisection(model, mel_eager, mel_len_eager)

    print("\nDone!")


if __name__ == "__main__":
    main()
