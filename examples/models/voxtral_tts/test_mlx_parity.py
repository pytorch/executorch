"""MLX parity tests for Voxtral TTS — LM decoder + semantic head.

Parity is measured against the XNNPACK fp32 baseline. Cosine ≥ 0.998 on
the last prefill hidden state and semantic argmax identical on the first
predicted frame — these gate M-2 (export) of the MLX bring-up per
MLX_DEV.md. For Metal parity tests see test_metal_parity.py (independent
file, different backend).

Skips when:
  - Host is not macOS.
  - `executorch.backends.mlx` is not importable (MLX delegate not built).
  - `torch.ops.mlx.custom_sdpa` / `torch.ops.mlx.rope` are not registered
    (the MLX Python bindings are missing from the env).
  - Voxtral-4B-TTS-2603 checkpoint is not available
    (set ``$VOXTRAL_TTS_MODEL_DIR`` or drop it at
    ``~/models/mistralai/Voxtral-4B-TTS-2603``).

Run:
    pytest -xvs examples/models/voxtral_tts/test_mlx_parity.py
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

VOXTRAL_DIR_ENV = "VOXTRAL_TTS_MODEL_DIR"
DEFAULT_VOXTRAL_DIR = Path.home() / "models/mistralai/Voxtral-4B-TTS-2603"


def _voxtral_dir() -> Path | None:
    p = Path(os.environ.get(VOXTRAL_DIR_ENV, DEFAULT_VOXTRAL_DIR))
    return p if (p / "params.json").exists() else None


def _mlx_available() -> tuple[bool, str]:
    """Check whether the MLX backend is usable on this host.

    Returns ``(ok, reason)``. When ``ok`` is False, ``reason`` describes why.
    """
    if sys.platform != "darwin":
        return False, f"MLX requires macOS (got {sys.platform})"
    try:
        import executorch.backends.mlx  # noqa: F401
    except (AttributeError, ImportError, OSError) as e:
        return False, f"executorch.backends.mlx not importable: {e}"
    try:
        import executorch.backends.mlx.custom_ops  # noqa: F401
    except (AttributeError, ImportError, OSError) as e:
        return False, f"executorch.backends.mlx.custom_ops not importable: {e}"
    if not hasattr(torch.ops, "mlx"):
        return False, "torch.ops.mlx namespace not registered"
    for op_name in ("custom_sdpa", "rope"):
        if not hasattr(torch.ops.mlx, op_name):
            return False, f"torch.ops.mlx.{op_name} not registered"
    return True, ""


_MLX_OK, _MLX_REASON = _mlx_available()


# ---------------------------------------------------------------------------
# Sanity check — import the MLX symbols that model.py added for the bring-up
# ---------------------------------------------------------------------------


def test_mlx_symbols_present():
    """MLXStaticKVCache and MLXSDPA must exist in model.py after Phase M-1."""
    import model  # noqa: E402

    missing = [
        sym for sym in ("MLXStaticKVCache", "MLXSDPA") if not hasattr(model, sym)
    ]
    assert not missing, (
        f"MLX helpers missing from model.py: {missing}. " f"See MLX_DEV.md Phase M-1."
    )


def test_export_script_uses_local_model_module():
    """The export script must use this source tree's model.py, not site-packages."""
    import export_voxtral_tts  # noqa: E402

    assert (
        Path(inspect.getfile(export_voxtral_tts.load_model)).resolve()
        == Path(__file__).with_name("model.py").resolve()
    )


def test_mlx_qembedding_default_group_size_is_supported():
    """MLX 8w embeddings need grouped quantization, not TorchAO's per-axis default."""
    import export_voxtral_tts  # noqa: E402

    args = SimpleNamespace(qembedding="8w", qembedding_group_size=None)
    export_voxtral_tts._apply_mlx_arg_defaults(args, backend_for_export="mlx")

    assert args.qembedding_group_size == 128


def test_mlx_rejects_quantized_codec():
    """Native MLX codec lowering is only validated for the unquantized codec."""
    import export_voxtral_tts  # noqa: E402

    parser = argparse.ArgumentParser()
    args = SimpleNamespace(qlinear_codec="4w")

    with pytest.raises(SystemExit):
        export_voxtral_tts._validate_mlx_args(parser, args, backend_for_export="mlx")


def test_mlx_codec_export_uses_mlx_lowering(monkeypatch, tmp_path):
    """Codec decoder should lower through MLX for native codec execution."""
    import export_voxtral_tts  # noqa: E402

    captured = {}

    def fake_export_codec_decoder(*args, **kwargs):
        return {"forward": object()}, {"n": "meta"}

    class FakeLoweredCodec:
        _tensor_data = {}

        def write_to_file(self, f):
            f.write(b"pte")

    def fake_lower_to_executorch(programs, metadata, backend, triton_kernel_mode):
        captured["backend"] = backend
        captured["triton_kernel_mode"] = triton_kernel_mode
        return FakeLoweredCodec()

    monkeypatch.setattr(
        export_voxtral_tts, "export_codec_decoder", fake_export_codec_decoder
    )
    monkeypatch.setattr(
        export_voxtral_tts, "lower_to_executorch", fake_lower_to_executorch
    )

    args = SimpleNamespace(
        backend="mlx",
        max_codec_frames=256,
        output_dir=str(tmp_path),
        qlinear_codec=None,
        qlinear_codec_group_size=None,
    )
    export_voxtral_tts._export_codec_pte(model=object(), args=args, device="cpu")

    assert captured == {"backend": "mlx", "triton_kernel_mode": "ON"}


@pytest.mark.skipif(not _MLX_OK, reason=_MLX_REASON or "MLX not available")
def test_mlx_rope_traditional_matches_apply_rotary_emb():
    """torch.ops.mlx.rope(traditional=True) matches TTS's apply_rotary_emb on
    a tiny tensor.

    Catches any Mistral-interleaved RoPE convention mismatch before the
    full-model tests (which are expensive to set up).
    """
    from model import apply_rotary_emb, precompute_freqs_cis

    torch.manual_seed(0)
    head_dim = 128
    n_heads = 4
    n_kv_heads = 2
    max_seq_len = 32
    rope_theta = 1_000_000.0

    freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, max_seq_len, rope_theta)

    # Single-token decode step at position 5.
    start_pos = 5
    input_pos = torch.tensor([start_pos], dtype=torch.long)
    q = torch.randn(1, 1, n_heads, head_dim, dtype=torch.float32)
    k = torch.randn(1, 1, n_kv_heads, head_dim, dtype=torch.float32)

    # Reference: pair-interleaved apply_rotary_emb via precomputed freqs.
    q_ref, k_ref = apply_rotary_emb(q, k, freqs_cos[input_pos], freqs_sin[input_pos])

    # MLX: on-the-fly via torch.ops.mlx.rope on BHSD tensors.
    q_mlx = torch.ops.mlx.rope(
        q.transpose(1, 2), head_dim, start_pos, traditional=True, base=rope_theta
    ).transpose(1, 2)
    k_mlx = torch.ops.mlx.rope(
        k.transpose(1, 2), head_dim, start_pos, traditional=True, base=rope_theta
    ).transpose(1, 2)

    torch.testing.assert_close(q_mlx, q_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(k_mlx, k_ref, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Full-model parity tests (skip if no checkpoint)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def models():
    if not _MLX_OK:
        pytest.skip(_MLX_REASON or "MLX not available")
    vdir = _voxtral_dir()
    if vdir is None:
        pytest.skip(
            f"Voxtral-4B-TTS-2603 checkpoint not found "
            f"(set ${VOXTRAL_DIR_ENV} or place at {DEFAULT_VOXTRAL_DIR})"
        )
    from model import load_model  # noqa: E402

    cpu = load_model(
        str(vdir), max_seq_len=4096, dtype=torch.float32, backend="xnnpack"
    )
    cpu.eval()
    try:
        mlx_model = load_model(
            str(vdir), max_seq_len=4096, dtype=torch.float32, backend="mlx"
        )
    except (ValueError, AttributeError, ImportError) as e:
        pytest.skip(f"MLX backend not yet wired in load_model: {e}")
    mlx_model.eval()
    return cpu, mlx_model


def test_prefill_hidden_parity(models):
    """MLX decoder prefill matches XNNPACK baseline.

    Cosine ≥ 0.998 on the final hidden state. Threshold set conservatively
    for fp32-vs-fp32 execution — with bf16 KV cache (the likely production
    config) the threshold may need to relax to ≥ 0.995.
    """
    import torch.nn.functional as F

    cpu, mlx_model = models
    torch.manual_seed(42)
    embeds = torch.randn(1, 230, 3072, dtype=torch.float32)
    pos = torch.arange(230, dtype=torch.long)

    with torch.no_grad():
        h_cpu = cpu.decoder(embeds, pos)
        h_mlx = mlx_model.decoder(embeds, pos)

    cos = F.cosine_similarity(h_cpu[0, -1], h_mlx[0, -1], dim=0).item()
    assert cos > 0.998, f"prefill hidden cosine = {cos:.6f} (expected > 0.998)"


def test_first_frame_semantic_argmax_match(models):
    """Semantic head argmax on the first predicted frame agrees with XNNPACK."""
    cpu, mlx_model = models
    torch.manual_seed(42)
    embeds = torch.randn(1, 230, 3072, dtype=torch.float32)
    pos = torch.arange(230, dtype=torch.long)

    with torch.no_grad():
        h_cpu = cpu.decoder(embeds, pos)[0, -1].unsqueeze(0)
        h_mlx = mlx_model.decoder(embeds, pos)[0, -1].unsqueeze(0)
        sem_cpu = cpu.flow_head.semantic_logits(h_cpu)
        sem_mlx = mlx_model.flow_head.semantic_logits(h_mlx)

    argmax_cpu = sem_cpu[0].argmax().item()
    argmax_mlx = sem_mlx[0].argmax().item()
    assert (
        argmax_cpu == argmax_mlx
    ), f"semantic argmax mismatch: cpu={argmax_cpu} mlx={argmax_mlx}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-xvs"]))
