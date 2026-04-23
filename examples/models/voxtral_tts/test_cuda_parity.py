"""CUDA parity tests for Voxtral TTS.

Guards the new CUDA code paths added in 2026-04 (StaticKVCache, StandardSDPA,
_build_causal_mask_bool, _conv1d_as_matmul, _conv_transpose1d_as_matmul) against
silent regressions. All tests run in eager mode — they don't require a CUDA
build of ExecuTorch, only PyTorch + CUDA + the Voxtral checkpoint.

Skips cleanly if CUDA isn't available or the checkpoint isn't on disk, so this
is safe to keep in the default test suite.

Run:
    pytest -xvs examples/models/voxtral_tts/test_cuda_parity.py
or:
    python examples/models/voxtral_tts/test_cuda_parity.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import (  # noqa: E402
    _conv1d_as_matmul,
    _conv_transpose1d_as_matmul,
    load_model,
)


VOXTRAL_DIR_ENV = "VOXTRAL_TTS_MODEL_DIR"
DEFAULT_VOXTRAL_DIR = Path.home() / "models/mistralai/Voxtral-4B-TTS-2603"


def _voxtral_dir() -> Path | None:
    p = Path(os.environ.get(VOXTRAL_DIR_ENV, DEFAULT_VOXTRAL_DIR))
    return p if (p / "params.json").exists() else None


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


# ---------------------------------------------------------------------------
# Conv-as-matmul math parity (no checkpoint needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "in_ch,out_ch,k,stride,dilation",
    [
        (1024, 1024, 3, 1, 1),  # codec mid conv
        (1024, 1024, 4, 1, 1),  # ConvTranspose decomp shape
        (1024, 240, 3, 1, 1),  # codec output_proj
        (1024, 1024, 7, 1, 1),  # first conv
    ],
)
def test_conv1d_as_matmul_matches_f_conv1d(in_ch, out_ch, k, stride, dilation):
    # Disable TF32 — A100 uses it for matmul by default, which gives ~1e-2
    # vs cuDNN conv. Strict fp32 keeps the rewrite within 1e-4.
    prev_tf32_mm = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.manual_seed(0)
        weight = torch.randn(out_ch, in_ch, k, device="cuda", dtype=torch.float32)
        bias = torch.randn(out_ch, device="cuda", dtype=torch.float32)
        x = torch.randn(1, in_ch, 256, device="cuda", dtype=torch.float32)

        y_ref = F.conv1d(x, weight, bias, stride=stride, padding=0, dilation=dilation)
        y_alt = _conv1d_as_matmul(x, weight, bias, stride=stride, dilation=dilation)
        assert y_ref.shape == y_alt.shape
        diff = (y_ref - y_alt).abs().max().item()
        rms = y_ref.float().pow(2).mean().sqrt().item()
        rel = diff / (rms + 1e-9)
        # fp32 matmul reduction order vs cuDNN: very small numerical drift.
        assert rel < 1e-3, f"max abs diff = {diff}, rel = {rel}"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_mm
        torch.backends.cudnn.allow_tf32 = prev_tf32_cudnn


@pytest.mark.parametrize(
    "in_ch,out_ch,k,stride",
    [
        (1024, 1024, 4, 2),  # upsample 2x
        (1024, 512, 4, 2),  # upsample with channel reduction
        (1024, 512, 3, 1),  # stride-1 ConvTranspose
        (1024, 240, 8, 4),  # extreme stride
    ],
)
def test_conv_transpose1d_as_matmul_matches_f_conv_transpose1d(
    in_ch, out_ch, k, stride
):
    prev_tf32_mm = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.manual_seed(0)
        weight = torch.randn(in_ch, out_ch, k, device="cuda", dtype=torch.float32)
        bias = torch.randn(out_ch, device="cuda", dtype=torch.float32)
        x = torch.randn(1, in_ch, 64, device="cuda", dtype=torch.float32)

        y_ref = F.conv_transpose1d(x, weight, bias, stride=stride, padding=0)
        y_alt = _conv_transpose1d_as_matmul(x, weight, bias, stride=stride)
        assert y_ref.shape == y_alt.shape
        diff = (y_ref - y_alt).abs().max().item()
        rms = y_ref.float().pow(2).mean().sqrt().item()
        rel = diff / (rms + 1e-9)
        assert rel < 1e-3, f"max abs diff = {diff}, rel = {rel}"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_mm
        torch.backends.cudnn.allow_tf32 = prev_tf32_cudnn


# ---------------------------------------------------------------------------
# Full-model parity tests — need the Voxtral checkpoint
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def models():
    vdir = _voxtral_dir()
    if vdir is None:
        pytest.skip(
            f"Voxtral-4B-TTS-2603 checkpoint not found "
            f"(set ${VOXTRAL_DIR_ENV} or place at {DEFAULT_VOXTRAL_DIR})"
        )
    print(f"\nLoading models from {vdir}...", flush=True)
    cpu = load_model(
        str(vdir), max_seq_len=4096, dtype=torch.float32, backend="xnnpack"
    )
    cpu.eval()
    cuda_model = load_model(
        str(vdir), max_seq_len=4096, dtype=torch.float32, backend="cuda"
    )
    cuda_model.cuda().eval()
    return cpu, cuda_model


def test_prefill_hidden_parity(models):
    """CUDA decoder prefill matches XNNPACK baseline on random embeddings.

    Cosine threshold 0.998 — set by the bf16 SDPA cast inside StandardSDPA.
    Set tighter (0.9999) when full fp32 eager comparisons. See PROGRESS.md
    Phase 7+8 for context on _build_causal_mask_bool and the bf16 isolation.
    """
    cpu, cuda_model = models
    torch.manual_seed(42)
    embeds = torch.randn(1, 230, 3072, dtype=torch.float32)
    pos = torch.arange(230, dtype=torch.long)

    with torch.no_grad():
        h_cpu = cpu.decoder(embeds, pos)
        h_cuda = cuda_model.decoder(embeds.cuda(), pos.cuda()).cpu()

    cos = F.cosine_similarity(h_cpu[0, -1], h_cuda[0, -1], dim=0).item()
    assert cos > 0.998, f"prefill hidden cosine = {cos:.6f} (expected > 0.998)"


def test_first_frame_semantic_argmax_match(models):
    """First-frame semantic argmax must be identical to baseline.

    Captures the regression Codex caught: missing causal mask in CUDA path
    sent semantic_head down the wrong logit branch starting at frame 0.
    """
    cpu, cuda_model = models
    torch.manual_seed(42)
    embeds = torch.randn(1, 230, 3072, dtype=torch.float32)
    pos = torch.arange(230, dtype=torch.long)

    with torch.no_grad():
        h_cpu = cpu.decoder(embeds, pos)[0, -1].unsqueeze(0)
        h_cuda = cuda_model.decoder(embeds.cuda(), pos.cuda())[0, -1].unsqueeze(0)
        sem_cpu = cpu.flow_head.semantic_logits(h_cpu)
        sem_cuda = cuda_model.flow_head.semantic_logits(h_cuda).cpu()

    argmax_cpu = sem_cpu[0].argmax().item()
    argmax_cuda = sem_cuda[0].argmax().item()
    top5_cpu = set(torch.topk(sem_cpu[0], 5).indices.tolist())
    top5_cuda = set(torch.topk(sem_cuda[0], 5).indices.tolist())
    assert (
        argmax_cpu == argmax_cuda
    ), f"semantic argmax mismatch: cpu={argmax_cpu} cuda={argmax_cuda}"
    overlap = len(top5_cpu & top5_cuda)
    assert overlap >= 4, f"top-5 overlap = {overlap}/5 (expected >= 4)"


def test_codec_matmul_rewrite_parity(models):
    """Full codec_decoder forward with the conv-as-matmul rewrite produces
    fp32 output bit-equivalent to the F.conv1d / F.conv_transpose1d baseline.
    """
    import model as tts_model

    cpu, _ = models
    cpu.codec_decoder.eval()

    codes = torch.zeros(1, cpu.config.n_codebooks, 256, dtype=torch.long)
    codes[0, 0, :] = 100
    codes[0, 1:, :] = 12

    # Current path uses _conv1d_as_matmul / _conv_transpose1d_as_matmul.
    with torch.no_grad():
        y_alt = cpu.codec_decoder(codes)

    # Monkey-patch back to F.conv1d / F.conv_transpose1d for the reference.
    orig_c1 = tts_model._conv1d_as_matmul
    orig_ct = tts_model._conv_transpose1d_as_matmul
    try:
        tts_model._conv1d_as_matmul = lambda x, w, b, stride, dilation: F.conv1d(
            x, w, b, stride=stride, padding=0, dilation=dilation
        )
        tts_model._conv_transpose1d_as_matmul = (
            lambda x, w, b, stride: F.conv_transpose1d(
                x, w, b, stride=stride, padding=0
            )
        )
        with torch.no_grad():
            y_ref = cpu.codec_decoder(codes)
    finally:
        tts_model._conv1d_as_matmul = orig_c1
        tts_model._conv_transpose1d_as_matmul = orig_ct

    diff = (y_ref - y_alt).abs().max().item()
    # Codec accumulates many fp32 ops; allow 1e-3 numerical drift.
    assert diff < 1e-3, f"codec output max abs diff = {diff}"


# ---------------------------------------------------------------------------
# Allow `python test_cuda_parity.py` direct invocation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-xvs"]))
