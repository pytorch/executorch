# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Synthetic dry-run validation for the 4-method export contract.

Builds a TINY Gemma4_31B + Gemma4_31BVisionTower in memory (random init,
~few-MB total) and runs the same 4-method export structure as
``export._export_cuda`` against it. The goal is to validate the **structure**
of the refactor — that the produced .pte exposes exactly the four contract
methods — WITHOUT paying the wall-clock cost of a real 31B prequant + export
(~50 minutes).

Note on backend lowering
========================
The real ``_export_cuda`` runs each ExportedProgram through
``CudaPartitioner`` + AOTInductor + the executorch CUDA SDPA / int4 triton
kernels. Those backend kernels assume the weights are ``Int4Tensor`` packed
(produced by ``quantize_and_save.py``), so a synthetic random bf16 model
can't reasonably go through that path. Instead, this test wires the same
4-method ``to_edge_transform_and_lower`` call with an EMPTY per-method
partitioner map. That exercises:

  * the 4 ``torch.export`` calls (with the same wrappers / dynamic shapes
    that ``_export_cuda`` uses),
  * the ``to_edge_transform_and_lower`` combination step,
  * ``to_executorch()``,
  * the final method-name set on the produced .pte.

The CUDA-backend path itself is exercised by the real run captured in
``examples/models/gemma4_31b/tests/test_export_methods.py``
(``EXECUTORCH_TEST_RUN_EXPORT=1``).

Run from the executorch repo root with the ``et`` conda env::

    conda run -n et python -m pytest \\
      examples/models/gemma4_31b/tests/test_export_synthetic.py -v -s
"""

from __future__ import annotations

import gc
import os
import tempfile

import pytest
import torch
from executorch.examples.models.gemma4_31b.vision_tower import Gemma4VisionConfig

from executorch.examples.models.gemma4_31b.model import (
    Gemma4_31B,
    Gemma4_31BConfig,
    materialize_runtime_buffers,
)


_EXPECTED_METHODS = {"embed_text", "vision_encoder", "prefill", "decode"}


def _tiny_vision_config() -> Gemma4VisionConfig:
    """A vision config small enough to export in seconds.

    Numbers chosen so:
      - hidden_size (64) is divisible by num_attention_heads (2) → head_dim 32.
      - patch_size (16) → patch_dim = 3 * 16 * 16 = 768 (matches the contract).
      - pooling_kernel_size = 3 → P = 9 * N (matches the contract).
      - position_embedding_size cut to 256 (vs 10240 prod) to keep the
        position table tiny.
    """
    return Gemma4VisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        patch_size=16,
        pooling_kernel_size=3,
        position_embedding_size=256,
        max_position_embeddings=256,
        rope_theta=100.0,
        standardize=True,
        use_clipped_linears=False,
        default_output_length=4,
        in_channels=3,
    )


def _tiny_text_config(vision_config: Gemma4VisionConfig) -> Gemma4_31BConfig:
    """A text config that exercises BOTH sliding and full attention layers."""
    return Gemma4_31BConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        num_global_key_value_heads=2,
        global_head_dim=64,
        attention_k_eq_v=True,
        sliding_rope_theta=10_000.0,
        full_rope_theta=1_000_000.0,
        full_partial_rotary_factor=0.25,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        final_logit_softcapping=30.0,
        tie_word_embeddings=False,
        sliding_window=32,
        layer_types=["sliding_attention", "full_attention"],
        vision_config=vision_config,
        max_seq_len=64,
    )


def _build_tiny_model() -> tuple[Gemma4_31B, Gemma4_31BConfig]:
    """Construct a tiny Gemma4_31B with vision attached (random init).

    Mirrors the dtype layout the real ``_export_cuda`` sees after
    ``load_and_pack_for_cuda`` + ``materialize_runtime_buffers(bf16)``:
      - All learnable params: bf16
      - layer_scalar buffer: bf16 (so residual stream stays bf16)
      - std_bias / std_scale buffers: fp32
      - KV cache / inv_freq / embed_normalizer / logit_softcap / cache_positions:
        filled in by ``materialize_runtime_buffers`` from the meta build below.
    """
    torch.manual_seed(0)
    vision_config = _tiny_vision_config()
    text_config = _tiny_text_config(vision_config)

    # Build on meta so materialize_runtime_buffers can fill in runtime buffers.
    with torch.device("meta"):
        model = Gemma4_31B(text_config)

    state_dict: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        t = torch.empty(param.shape, dtype=torch.bfloat16)
        t.normal_(mean=0.0, std=0.02)
        state_dict[name] = t
    for name, buf in model.named_buffers():
        if name.endswith("layer_scalar"):
            state_dict[name] = torch.ones(buf.shape, dtype=torch.bfloat16)
        elif name.endswith("std_bias"):
            state_dict[name] = torch.zeros(buf.shape, dtype=torch.float32)
        elif name.endswith("std_scale"):
            state_dict[name] = torch.ones(buf.shape, dtype=torch.float32)

    _missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    assert not unexpected, f"Unexpected keys: {unexpected[:5]}"

    model.eval()
    return model, text_config


def _export_4_methods_no_backend(
    model: Gemma4_31B,
    config: Gemma4_31BConfig,
    output_dir: str,
    *,
    max_vision_soft_tokens: int,
) -> tuple[str, dict[str, "torch.export.ExportedProgram"]]:
    """Mirror of ``export._export_cuda``'s structural steps, but with NO
    backend partitioner (so the CUDA triton kernels' Int4-only assumptions
    aren't exercised). The 4 ``torch.export`` calls, the wrapper / dynamic
    shape choices, the ``to_edge_transform_and_lower`` combine step, and
    the ``to_executorch`` step are intentionally identical to production.

    Returns: (pte_path, programs_dict). The programs dict is returned BEFORE
    lowering so the test can introspect mutable-buffer FQN identity across
    methods (the bug runner_dev hit was that the `__class__` swap fragmented
    these FQNs and KV caches stopped being shared).
    """
    from executorch.examples.models.gemma4_31b.export import (
        _BoundMethodForward,
        _build_vision_encoder_wrapper,
    )
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from executorch.exir.passes import MemoryPlanningPass
    from torch.export import Dim, export

    materialize_runtime_buffers(model, dtype=torch.bfloat16)

    max_prefill = min(config.max_seq_len - 1, config.sliding_window * 2)
    hidden_size = config.hidden_size

    # ---------- prefill (inputs_embeds, dynamic seq_len) ----------
    seq_dim = Dim("seq_len", min=5, max=max_prefill)
    print(f"Exporting prefill (T in [5, {max_prefill}], inputs_embeds)...")
    with torch.no_grad():
        prefill_ep = export(
            model,
            (
                torch.zeros((1, max_prefill, hidden_size), dtype=torch.bfloat16),
                torch.arange(max_prefill, dtype=torch.long),
                torch.tensor([1.0], dtype=torch.float32),
            ),
            dynamic_shapes=({1: seq_dim}, {0: seq_dim}, None),
            strict=True,
        )

    # ---------- decode (tokens, T=1, static) ----------
    print("Exporting decode (T=1, tokens input)...")
    with _BoundMethodForward(model, model.decode_forward), torch.no_grad():
        decode_ep = export(
            model,
            (
                torch.tensor([[0]], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
                torch.tensor([1.0], dtype=torch.float32),
            ),
            strict=True,
        )

    # ---------- embed_text ----------
    print(f"Exporting embed_text (T in [1, {max_prefill}])...")
    embed_text_seq_dim = Dim("embed_text_seq_len", min=1, max=max_prefill)
    with _BoundMethodForward(model, model.embed_text), torch.no_grad():
        embed_text_ep = export(
            model,
            (torch.zeros((1, max_prefill), dtype=torch.long),),
            dynamic_shapes=({1: embed_text_seq_dim},),
            strict=True,
        )

    # ---------- vision_encoder ----------
    pks = config.vision_config.pooling_kernel_size
    max_patches = (pks * pks) * max_vision_soft_tokens
    patch_dim = config.vision_config.patch_dim
    print(
        f"Exporting vision_encoder "
        f"(P in [{pks*pks}, {max_patches}], soft tokens up to "
        f"{max_vision_soft_tokens})..."
    )
    ve_wrapper = _build_vision_encoder_wrapper(model, config)
    num_groups_dim = Dim("vision_num_groups", min=1, max=max_vision_soft_tokens)
    num_patches_dim = (pks * pks) * num_groups_dim
    with torch.no_grad():
        vision_encoder_ep = export(
            ve_wrapper,
            (
                torch.zeros((1, max_patches, patch_dim), dtype=torch.float32),
                torch.zeros((1, max_patches, 2), dtype=torch.long),
            ),
            dynamic_shapes=(
                {1: num_patches_dim},
                {1: num_patches_dim},
            ),
            strict=True,
        )

    programs: dict[str, "torch.export.ExportedProgram"] = {
        "embed_text": embed_text_ep,
        "vision_encoder": vision_encoder_ep,
        "prefill": prefill_ep,
        "decode": decode_ep,
    }

    # No partitioner — we want to validate the structural refactor only.
    partitioner_map: dict[str, list] = {name: [] for name in programs}
    transform_passes: dict[str, list] = {name: [] for name in programs}

    constant_methods: dict[str, object] = {
        "get_max_seq_len": config.max_seq_len,
        "get_vocab_size": config.vocab_size,
        "get_n_layers": config.num_hidden_layers,
        "get_max_prefill_chunk": max_prefill,
        "get_max_vision_soft_tokens": int(max_vision_soft_tokens),
        "use_kv_cache": True,
        "use_sdpa_with_kv_cache": False,
        "enable_dynamic_shape": True,
    }

    print(
        f"Combining {len(programs)} methods to ExecuTorch (no backend partitioner): "
        f"{', '.join(programs.keys())}..."
    )
    et_prog = to_edge_transform_and_lower(
        programs,
        partitioner=partitioner_map,
        transform_passes=transform_passes,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=constant_methods,
    )

    et_program = et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                share_mutable_buffers=True,
            ),
            emit_mutable_buffer_names=True,
        ),
    )

    os.makedirs(output_dir, exist_ok=True)
    pte_path = os.path.join(output_dir, "model.pte")
    with open(pte_path, "wb") as f:
        et_program.write_to_file(f)
    # Keep `ve_wrapper` alive only until lowering completes; we no longer
    # need it. (Don't `del model` either — caller is the test which doesn't
    # touch it again.)
    del ve_wrapper

    return pte_path, programs


def _kv_cache_buffer_targets(ep) -> set[str]:
    """Return the FQN targets of all KV-cache mutable buffers in an EP."""
    fqns = _mutable_buffer_fqns(ep)
    return {f for f in fqns if "kv_cache" in f}


def _mutable_buffer_fqns(ep) -> set[str]:
    """All mutable-buffer FQNs visible in this ExportedProgram's signature.

    Per executorch/exir/passes/memory_planning_pass.py:
      - ``inputs_to_buffers`` maps placeholder name → mutable buffer FQN.
      - ``run_multimethod`` groups specs by FQN string and assigns a single
        shared ``mem_id=2 + mem_offset`` per FQN. Combined with
        ``share_memory_arenas=true`` in the Module ctor, **identical FQN
        strings across two ExportedPrograms are the necessary and
        sufficient condition** for runtime KV-cache sharing across methods.
    """
    return set(ep.graph_signature.inputs_to_buffers.values())


def test_synthetic_4method_export_produces_correct_method_set():
    """Build a tiny multimodal model, run the structural mirror of
    ``_export_cuda``, and check:
      1. the exported .pte exposes the 4 contract methods,
      2. prefill and decode share IDENTICAL mutable-buffer FQNs (the
         necessary-and-sufficient condition for KV cache sharing across
         methods at runtime — see executorch/exir/passes/memory_planning_pass.py).
    """
    model, config = _build_tiny_model()

    with tempfile.TemporaryDirectory(prefix="gemma4_31b_synth_") as tmp_out:
        pte_path, programs = _export_4_methods_no_backend(
            model,
            config,
            tmp_out,
            max_vision_soft_tokens=4,
        )

        # ---- Gate: 4 contract methods present in the .pte ----
        assert os.path.exists(pte_path), f"export did not produce {pte_path}"

        size_mb = os.path.getsize(pte_path) / 1024**2
        assert size_mb < 200, f"Synthetic .pte is {size_mb:.1f} MB; expected <200 MB."

        from executorch.runtime import Runtime

        runtime = Runtime.get()
        program = runtime.load_program(pte_path)
        methods = set(program.method_names)
        missing = _EXPECTED_METHODS - methods
        assert not missing, (
            f"Synthetic .pte is missing contract methods: {sorted(missing)}\n"
            f"  present: {sorted(methods)}"
        )
        assert "get_max_vision_soft_tokens" in methods, (
            "constant_method 'get_max_vision_soft_tokens' is missing from "
            f"the synthetic .pte: {sorted(methods)}"
        )

        # ---- Gate: prefill and decode share IDENTICAL mutable-buffer FQNs ----
        # This is the regression test for runner_dev's smoking-gun bug. The
        # previous `model.__class__ = _Gemma4_31BFor*` swap pattern fragmented
        # the FQNs across the 4 ExportedPrograms, so share_mutable_buffers
        # couldn't unify the KV cache buffers — prefill wrote them, decode
        # read fresh zeros, garbage came out. The bound-method swap fixes it
        # by keeping the same model instance & class across all 4 exports.
        prefill_fqns = _mutable_buffer_fqns(programs["prefill"])
        decode_fqns = _mutable_buffer_fqns(programs["decode"])

        # KV cache must be present in both.
        kv_in_prefill = {f for f in prefill_fqns if "kv_cache" in f}
        kv_in_decode = {f for f in decode_fqns if "kv_cache" in f}
        assert (
            kv_in_prefill
        ), f"prefill has no kv_cache mutable-buffer FQNs; got: {sorted(prefill_fqns)}"
        assert (
            kv_in_decode
        ), f"decode has no kv_cache mutable-buffer FQNs; got: {sorted(decode_fqns)}"

        # Strong assertion: every kv_cache FQN in prefill must also appear in
        # decode (and vice versa). String equality is the necessary+sufficient
        # condition for memory_planning_pass to assign the same mem_id, which
        # — combined with share_memory_arenas=true in the runner — physically
        # backs them with one tensor.
        prefill_only = kv_in_prefill - kv_in_decode
        decode_only = kv_in_decode - kv_in_prefill
        assert kv_in_prefill == kv_in_decode, (
            "prefill / decode KV-cache FQNs differ — share_mutable_buffers "
            "WILL NOT unify them at runtime, so prefill writes and decode "
            "reads from a fresh-zero cache (the bug runner_dev hit).\n"
            f"  prefill only: {sorted(prefill_only)}\n"
            f"  decode only:  {sorted(decode_only)}"
        )

        # Also assert the canonical layer-0 sliding-attention KV cache FQN
        # is present so we'd catch any future renaming of the KV cache module.
        canonical = "layers.0.self_attn.kv_cache.k_cache"
        assert canonical in kv_in_prefill, (
            f"Expected canonical KV cache FQN {canonical!r} not found.\n"
            f"  prefill kv FQNs: {sorted(kv_in_prefill)}"
        )
