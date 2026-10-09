# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export the toy decoder the CudaExecutor GPU test runs.

Writes two artifacts, each ``model.pte`` and ``aoti_cuda_blob.ptd`` -- a
two-layer decoder (one full-history layer, one sliding-window layer) exported
as the static ``forward_{N}`` methods and the dynamic ``forward_others``
CudaExecutor drives, lowered in the cell layout:

- ``dense/``: forward_{1,2,4,8}, and forward_others over [8, 32] tokens;
- ``sparse/``: forward_{1,4}, and forward_others over [4, 32] tokens selecting
  at least five rows, as a model whose kernels switch at a small width must,
  publishing that bound. Steps of 2..3 tokens pad across the gap to 4.
- ``device_sampling/``: dense's methods with their logits kept on the device,
  plus the ``sample`` and ``sample_argmax`` device samplers
  (extension/llm/batching/sampler.py).

Each has ``expected.txt``: one ``prompt;continuation`` line per prompt, the
greedy continuation computed eagerly with the neutral reference cache.

The residual stream carries each token's embedding at a large scale and the
LM head maps it to a fixed successor, while attention adds a smaller term. The
greedy argmax therefore keeps a wide margin over bf16 noise, so the exported
program must reproduce the eager tokens exactly, yet attention still feeds
every logit.

    python -m executorch.backends.cuda.batching.test.export_toy_decoder --output-dir DIR
"""

import argparse
import json
import os

import torch
import torch.nn as nn

VOCAB = 32
DIM = 64
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = 16
WINDOWS = (0, 4)  # layer 0 keeps its whole history, layer 1 a window of 4
MAX_CONTEXT = 64
MAX_STEP = 32  # forward_others' widest step: longer batches slice
MAX_CELLS = 256
EMBED_SCALE = 4.0

PROMPTS = ([3], [5, 9, 1], [7, 2, 2, 8, 4, 6, 1], list(range(1, 21)))
NEW_TOKENS = 6


class ToyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.embed = nn.Embedding(VOCAB, DIM)
        self.q = nn.ModuleList()
        self.k = nn.ModuleList()
        self.v = nn.ModuleList()
        self.o = nn.ModuleList()
        for _ in WINDOWS:
            self.q.append(nn.Linear(DIM, N_HEADS * HEAD_DIM, bias=False))
            self.k.append(nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False))
            self.v.append(nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False))
            self.o.append(nn.Linear(N_HEADS * HEAD_DIM, DIM, bias=False))
        self.head = nn.Linear(DIM, VOCAB, bias=False)
        with torch.no_grad():
            self.embed.weight.zero_()
            self.embed.weight[:, :VOCAB] = EMBED_SCALE * torch.eye(VOCAB)
            for linear in (*self.q, *self.k, *self.v):
                linear.weight.normal_(0.0, 0.3, generator=generator)
            for linear in self.o:
                linear.weight.normal_(0.0, 0.05, generator=generator)
            successor = torch.randperm(VOCAB, generator=generator)
            self.head.weight.zero_()
            self.head.weight[successor, torch.arange(VOCAB)] = 1.0

    def forward(self, tokens, input_pos, logits_to_keep):
        x = self.embed(tokens)
        length = tokens.shape[1]
        position = input_pos.reshape(-1, 1)
        for layer in range(len(WINDOWS)):
            q = self.q[layer](x).view(1, length, N_HEADS, HEAD_DIM).transpose(1, 2)
            k = self.k[layer](x).view(1, length, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
            v = self.v[layer](x).view(1, length, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
            y = torch.ops.kvcache.update_and_attend(
                q, k, v, position, layer, HEAD_DIM**-0.5, q.dtype
            )
            x = x + self.o[layer](y.transpose(1, 2).reshape(1, length, -1))
        return self.head(x[:, logits_to_keep, :])[0].float()


def _manifest() -> str:
    return json.dumps(
        {
            "version": 1,
            "layout": "cell",
            "maximum_capacity": MAX_CELLS,
            "max_cells": MAX_CELLS,
            "max_write": MAX_STEP,
            "layers": [
                {"layer_id": i, "policy": "ring" if w else "flat", "window": w}
                for i, w in enumerate(WINDOWS)
            ],
        }
    )


def _greedy(model: ToyDecoder, prompt) -> list:
    from executorch.extension.llm.cache.reference_cache import (
        CacheConfig,
        LayerPolicy,
        SequenceReferenceCache,
    )
    from executorch.extension.llm.cache.update_and_attend import REGISTRY

    cache = SequenceReferenceCache(
        CacheConfig(
            n_layers=len(WINDOWS),
            n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            capacity=MAX_CONTEXT,
            layers=tuple(
                LayerPolicy.ring(w) if w else LayerPolicy.flat() for w in WINDOWS
            ),
        )
    )
    key = "toy-decoder-greedy"
    REGISTRY.install(key, cache)
    try:
        with REGISTRY.active(key), torch.no_grad():
            generated = []
            tokens = list(prompt)
            start = 0
            for _ in range(NEW_TOKENS):
                step = torch.tensor([tokens[start:]])
                logits = model(
                    step,
                    torch.arange(start, len(tokens)),
                    torch.tensor([step.shape[1] - 1]),
                )
                start = len(tokens)
                token = int(logits[0].argmax())
                generated.append(token)
                tokens.append(token)
            return generated
    finally:
        REGISTRY.uninstall(key)


ARTIFACTS = {
    "dense": {"static": (1, 2, 4, 8), "min_rows": 1},
    "sparse": {"static": (1, 4), "min_rows": 5},
    "device_sampling": {"static": (1, 2, 4, 8), "min_rows": 1, "device_sampling": True},
}


def export(
    output_dir: str, static_widths, min_rows: int, device_sampling: bool = False
) -> None:
    import torch._inductor.config as inductor_config
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.backends.cuda.passes.lower_offgraph_kv import (
        OFFGRAPH_KV_COMPILE_SPEC,
        OFFGRAPH_KV_STEP_WIDTH_COMPILE_SPEC,
    )
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from executorch.exir.backend.compile_spec_schema import CompileSpec
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.exir.passes.propagate_device_config import PropagateDeviceConfig
    from executorch.extension.llm.batching.sampler import (
        BatchArgmax,
        BatchSampler,
        NUM_PARAMS,
    )
    from executorch.extension.llm.export.model_metadata import (
        write_cache_geometry,
        write_logits_to_keep_mode,
        write_max_context_len,
        write_max_seq_len,
        write_vocab_size,
    )
    from torch.export import Dim

    inductor_config.aot_inductor.precompile_headers = False
    torch.manual_seed(0)
    model = ToyDecoder().eval()
    expected = [_greedy(model, prompt) for prompt in PROMPTS]

    model = model.to(dtype=torch.bfloat16)
    long = {"dtype": torch.long}

    def inputs(width: int):
        return (
            torch.zeros(1, width, **long),
            torch.arange(width, **long),
            torch.zeros(width, **long),
        )

    widest_static = max(static_widths)
    width = Dim("width", min=widest_static, max=MAX_STEP)
    rows = Dim("rows", min=min_rows, max=MAX_STEP)
    programs = {}
    with torch.no_grad():
        for n in static_widths:
            programs[f"forward_{n}"] = torch.export.export(
                model, inputs(n), strict=True
            )
        programs["forward_others"] = torch.export.export(
            model,
            inputs(MAX_STEP),
            dynamic_shapes=({1: width}, {0: width}, {0: rows}),
            strict=True,
        )
    forward_methods = list(programs)
    if device_sampling:
        sample_rows = Dim("sample_rows", min=1, max=8)
        logits = torch.zeros(8, VOCAB)
        programs["sample"] = torch.export.export(
            BatchSampler(),
            (logits, torch.zeros(8, NUM_PARAMS)),
            dynamic_shapes=({0: sample_rows}, {0: sample_rows}),
            strict=True,
        )
        programs["sample_argmax"] = torch.export.export(
            BatchArgmax(), (logits,), dynamic_shapes=({0: sample_rows},), strict=True
        )

    def partitioner(name: str) -> CudaPartitioner:
        if name not in forward_methods:
            return CudaPartitioner(
                [CudaBackend.generate_method_name_compile_spec(name)]
            )
        return CudaPartitioner(
            [
                CudaBackend.generate_method_name_compile_spec(name),
                CompileSpec("low_memory_mode", b"ON"),
                CompileSpec(OFFGRAPH_KV_COMPILE_SPEC, _manifest().encode()),
                CompileSpec(OFFGRAPH_KV_STEP_WIDTH_COMPILE_SPEC, b"1:0"),
            ]
        )

    constant_methods = {
        **write_max_context_len(MAX_CONTEXT),
        **write_max_seq_len(MAX_STEP),
        **write_vocab_size(VOCAB),
        **write_logits_to_keep_mode("selected"),
        **write_cache_geometry(
            [N_KV_HEADS] * len(WINDOWS), [HEAD_DIM] * len(WINDOWS), list(WINDOWS)
        ),
        "get_offgraph_kv_max_cells": MAX_CELLS,
    }
    if min_rows > 1:
        constant_methods["get_min_prefill_chunk"] = min_rows
    # With device sampling the forwards' logits stay on the device, planned in
    # device memory, and the samplers take them there.
    device = {
        name: PropagateDeviceConfig(
            skip_d2h_for_method_outputs=device_sampling and name in forward_methods,
            skip_h2d_for_method_inputs=device_sampling and name not in forward_methods,
        )
        for name in programs
    }
    program = to_edge_transform_and_lower(
        programs,
        partitioner={name: [partitioner(name)] for name in programs},
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
        constant_methods=constant_methods,
    ).to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            propagate_device_config=device,
        )
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "model.pte"), "wb") as f:
        program.write_to_file(f)
    program.write_tensor_data_to_file(output_dir)
    with open(os.path.join(output_dir, "expected.txt"), "w") as f:
        for prompt, continuation in zip(PROMPTS, expected):
            f.write(
                " ".join(map(str, prompt))
                + ";"
                + " ".join(map(str, continuation))
                + "\n"
            )
    print(f"Wrote {output_dir}/model.pte, aoti_cuda_blob.ptd, expected.txt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    output_dir = parser.parse_args().output_dir
    for name, artifact in ARTIFACTS.items():
        export(
            os.path.join(output_dir, name),
            artifact["static"],
            artifact["min_rows"],
            artifact.get("device_sampling", False),
        )


if __name__ == "__main__":
    main()
