# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license in the root LICENSE.

"""Export dense Gemma 4 from the canonical GGUF for CUDA batching."""

import argparse
import gc
import json
from pathlib import Path

import torch
from executorch.examples.models.gemma4_31b.batching_model import PackedGemma4
from executorch.examples.models.gemma4_31b.model import Gemma4_31B, Gemma4_31BConfig, materialize_runtime_buffers
from executorch.examples.models.muse_glimmer.export import common
from executorch.examples.models.muse_glimmer.export.export_solo import (
    batched_sequence_manifest, device_residency, sampler_programs, step_width_spec,
)
from executorch.extension.llm.export.model_metadata import (
    write_activation_dtype, write_logits_to_keep_mode, write_max_context_len,
    write_max_seq_len, write_vocab_size,
)
from torch.export import Dim, export


def gemma_sampler_programs(vocab_size, suppressed_token_ids):
    """Apply the checkpoint's input-only token filter in GPU device samplers."""
    from executorch.extension.llm.batching.sampler import BatchArgmax, BatchSampler, NUM_PARAMS
    from executorch.examples.models.muse_glimmer.export.export_solo import MAX_SAMPLE_ROWS
    blocked = tuple(sorted(set(int(i) for i in suppressed_token_ids)))
    if any(i < 0 or i >= vocab_size for i in blocked):
        raise ValueError("Invalid checkpoint suppressed token id")

    def mask_logits(logits):
        ids = torch.arange(logits.shape[-1], device=logits.device)
        allowed = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
        for token in blocked:
            allowed = allowed & (ids != token)
        return torch.where(allowed, logits, -float("inf"))

    class MaskedArgmax(BatchArgmax):
        def forward(self, logits):
            return super().forward(mask_logits(logits))

    class MaskedSampler(BatchSampler):
        def forward(self, logits, params):
            return super().forward(mask_logits(logits), params)

    # A deterministic regression oracle must choose the best allowed token even
    # when the two input-only modality markers dominate the unfiltered logits.
    probe = torch.zeros(2, vocab_size)
    probe[:, list(blocked)] = 1e6
    probe[0, 7] = 10
    probe[1, 9] = 10
    assert torch.equal(MaskedArgmax()(probe), torch.tensor([7, 9]))
    rows = Dim("sample_rows", min=1, max=MAX_SAMPLE_ROWS)
    logits = torch.zeros(MAX_SAMPLE_ROWS, vocab_size)
    params = torch.zeros(MAX_SAMPLE_ROWS, NUM_PARAMS)
    return {
        "sample": export(MaskedSampler(), (logits, params),
                         dynamic_shapes=({0: rows}, {0: rows}), strict=True),
        "sample_argmax": export(MaskedArgmax(), (logits,),
                               dynamic_shapes=({0: rows},), strict=True),
    }


def export_forward_programs(model, max_step_tokens, forward_sizes):
    sizes = tuple(sorted(set(forward_sizes)))
    if not sizes or sizes[0] != 1 or sizes[-1] < 4 or sizes[-1] >= max_step_tokens:
        raise ValueError("Static forward sizes must include 1 and 4, below max step")
    if max_step_tokens >= model.config.max_seq_len:
        raise ValueError("Max step must be below context")
    programs = {}
    with torch.no_grad():
        for size in sizes:
            print(f"Exporting forward_{size} (T={size})...", flush=True)
            programs[f"forward_{size}"] = export(model, (
                torch.zeros((1, size), dtype=torch.long),
                torch.arange(size, dtype=torch.long), torch.zeros(size, dtype=torch.long),
            ), strict=True)
        print("Exporting forward_others...", flush=True)
        width = Dim("width", min=max(sizes[-1], 5), max=max_step_tokens)
        rows = Dim("rows", min=5, max=64)
        programs["forward_others"] = export(model, (
            torch.zeros((1, max_step_tokens), dtype=torch.long),
            torch.arange(max_step_tokens, dtype=torch.long), torch.zeros(5, dtype=torch.long),
        ), dynamic_shapes=({1: width}, {0: width}, {0: rows}), strict=True)
    return programs


def export_cuda_batching(model, output_dir, max_step_tokens=512, max_kv_seqs=33,
                         forward_sizes=(1, 2, 4, 8, 16, 32, 64), suppressed_token_ids=()):
    import executorch.backends.cuda.quantize_op_dispatch  # noqa: F401
    import torch._inductor.config as inductor_config
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.backends.cuda.passes.lower_offgraph_kv import (
        OFFGRAPH_KV_COMPILE_SPEC, OFFGRAPH_KV_STEP_WIDTH_COMPILE_SPEC,
    )
    from executorch.exir import (
        EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower,
    )
    from executorch.exir.backend.compile_spec_schema import CompileSpec
    from executorch.exir.passes import MemoryPlanningPass

    if max_kv_seqs <= 0:
        raise ValueError("Sequence slots must be positive")
    inductor_config.coordinate_descent_tuning = False
    inductor_config.aot_inductor.compile_wrapper_opt_level = "O2"
    inductor_config.aot_inductor.precompile_headers = False
    inductor_config.cpp_cache_precompile_headers = False
    manifest = batched_sequence_manifest(model.sequence_manifest(max_step_tokens), max_kv_seqs)
    metadata = model.cache_geometry()
    metadata.update(write_max_context_len(model.config.max_seq_len))
    metadata.update(write_max_seq_len(max_step_tokens))
    metadata.update(write_vocab_size(model.config.vocab_size))
    metadata.update(write_activation_dtype("bf16"))
    metadata.update(write_logits_to_keep_mode("selected"))
    metadata.update({"get_offgraph_kv_max_seqs": max_kv_seqs,
                     "get_min_prefill_chunk": 5, "get_max_prefill_chunk": max_step_tokens,
                     "use_sampling": False, "has_vision_encoder": False})
    programs = export_forward_programs(model, max_step_tokens, forward_sizes)
    forwards = list(programs)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    programs.update(gemma_sampler_programs(int(metadata["get_vocab_size"]), suppressed_token_ids))
    metadata["get_suppressed_token_count"] = len(suppressed_token_ids)
    for index, token in enumerate(suppressed_token_ids):
        metadata[f"get_suppressed_token_{index}"] = int(token)

    def partitioner(name):
        specs = [CudaBackend.generate_method_name_compile_spec(name)]
        if name in forwards:
            specs.extend([
                CompileSpec("low_memory_mode", b"ON"),
                CompileSpec(OFFGRAPH_KV_COMPILE_SPEC, manifest.encode()),
                CompileSpec(OFFGRAPH_KV_STEP_WIDTH_COMPILE_SPEC, step_width_spec(name)),
            ])
        return CudaPartitioner(specs)

    print(f"Lowering {', '.join(programs)} to ExecuTorch (CUDA)...", flush=True)
    edge = to_edge_transform_and_lower(programs,
        partitioner={name: [partitioner(name)] for name in programs},
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
        constant_methods=metadata)
    del programs
    gc.collect()
    torch.cuda.empty_cache()
    program = edge.to_executorch(config=ExecutorchBackendConfig(
        extract_delegate_segments=True, do_quant_fusion_and_const_prop=True,
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        propagate_device_config=device_residency(forwards, True),
        emit_mutable_buffer_names=True))
    del edge
    gc.collect()
    common.save_pte(program, output_dir, None)
    return program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-len", type=int, default=12288)
    parser.add_argument("--max-step-tokens", type=int, default=512)
    parser.add_argument("--max-kv-seqs", type=int, default=33)
    args = parser.parse_args()
    config = Gemma4_31BConfig.from_hf_config(args.config)
    config.max_seq_len = args.max_seq_len
    from executorch.examples.models.gemma4_31b.batching_gguf_loader import iter_gemma_gguf
    from executorch.examples.models.gemma4_31b.gguf_loader import gguf_to_model_key
    from executorch.examples.models.gemma4_31b.cuda_packers import convert_quantized_tensors_for_cuda
    from executorch.extension.llm.export.load import load_checkpoint
    with torch.device("meta"):
        model = Gemma4_31B(config)
    print("Loading canonical Gemma GGUF...", flush=True)
    load_checkpoint(args.gguf, model, key_map=gguf_to_model_key,
        tie_map={"embed_tokens.weight": ("lm_head.weight", True)},
        dtype=torch.bfloat16, raw_iter=iter_gemma_gguf)
    materialize_runtime_buffers(model, dtype=torch.bfloat16)
    convert_quantized_tensors_for_cuda(model)
    from executorch.examples.models.gemma4_31b.cuda_source_transformations import (
        cuda_source_transformations,
    )
    cuda_source_transformations(model, use_turboquant=False)
    model = PackedGemma4(model).eval()
    generation_path = Path(args.config).with_name("generation_config.json")
    generation = json.loads(generation_path.read_text())
    suppressed = tuple(sorted(set(generation.get("suppress_tokens", []))))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "sampler-policy.json").write_text(json.dumps(dict(
        source=str(generation_path), stop_tokens=generation["eos_token_id"],
        suppress_tokens=suppressed, temperature=0, gpu_argmax=True,
    ), indent=2) + "\n")
    print("GEMMA_CHECKPOINT_SAMPLER_POLICY", suppressed, flush=True)
    export_cuda_batching(model, str(output), args.max_step_tokens, args.max_kv_seqs,
                        suppressed_token_ids=suppressed)


if __name__ == "__main__":
    main()
