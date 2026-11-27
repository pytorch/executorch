# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tiny float-cache decoders and malformed programs for C++ runner tests."""

import argparse
import os

import torch
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge
from executorch.exir.passes import MemoryPlanningPass

CONTEXT = 16
PREFILL = 4
VOCAB = 32
HIDDEN = 16
HEADS = 2
LAYERS = 2
PROMPT = [3, 7, 1, 9, 4, 12]
NEW_TOKENS = 6


def _heads(x, weight):
    return (x @ weight.t()).view(1, x.shape[1], HEADS, -1).transpose(1, 2)


class TinyHfStaticDecoder(torch.nn.Module):
    def __init__(self, return_state=False):
        super().__init__()
        self.return_state = return_state
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(VOCAB, HIDDEN)
        self.qkvo = torch.nn.ModuleList(
            torch.nn.Linear(HIDDEN, 4 * HIDDEN, bias=False) for _ in range(LAYERS)
        )
        self.head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, input_ids, attention_mask, position_ids, *caches):
        steps = input_ids.shape[1]
        x = self.embed(input_ids)
        for layer, keys, values, counter in zip(
            self.qkvo, caches[0::3], caches[1::3], caches[2::3]
        ):
            slots = torch.arange(steps) + counter
            q, k, v, o = layer.weight.split(HIDDEN)
            keys.index_copy_(2, slots, _heads(x, k).to(keys.dtype))
            values.index_copy_(2, slots, _heads(x, v).to(values.dtype))
            counter.add_(steps)
            attn = torch.nn.functional.scaled_dot_product_attention(
                _heads(x, q), keys.float(), values.float(), attn_mask=attention_mask
            )
            x = x + attn.transpose(1, 2).reshape(1, steps, HIDDEN) @ o.t()
        logits = self.head(x)
        return (logits, *caches) if self.return_state else logits


def _inputs(tokens, start, steps):
    positions = torch.arange(start, start + steps)
    mask = (torch.arange(CONTEXT)[None] <= positions[:, None])[None, None]
    ids = torch.tensor([list(tokens) + [tokens[-1]] * (steps - len(tokens))])
    return ids, mask, positions[None]


def _caches(dtype=torch.float32):
    shape = (1, HEADS, CONTEXT, HIDDEN // HEADS)
    return [
        tensor
        for _ in range(LAYERS)
        for tensor in (
            torch.zeros(shape, dtype=dtype),
            torch.zeros(shape, dtype=dtype),
            torch.zeros((), dtype=torch.long),
        )
    ]


def expected_tokens(model, dtype):
    caches, start = _caches(dtype), 0
    generated = []
    with torch.no_grad():
        while start < len(PROMPT):
            chunk = PROMPT[start : start + PREFILL]
            for counter in caches[2::3]:
                counter.fill_(start)
            output = model(*_inputs(chunk, start, PREFILL), *caches)
            logits = (output[0] if model.return_state else output)[0, len(chunk) - 1]
            start += len(chunk)
        for _ in range(NEW_TOKENS):
            token = int(logits.argmax())
            generated.append(token)
            for counter in caches[2::3]:
                counter.fill_(start)
            output = model(*_inputs([token], start, 1), *caches)
            logits = (output[0] if model.return_state else output)[0, 0]
            start += 1
    return generated


def _metadata(dtype=torch.float32):
    return {
        "get_io_contract": "hf_static",
        "get_io_contract_version": 1,
        "get_cache_dtype": "float32" if dtype == torch.float32 else "float16",
        "get_n_layers": LAYERS,
        "get_max_seq_len": CONTEXT,
    }


def _lower(programs, metadata):
    edge = to_edge(
        programs,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        constant_methods=metadata,
    )
    return edge.to_executorch(
        ExecutorchBackendConfig(
            run_reinplace_pass=True,
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False, alloc_graph_output=False
            ),
        )
    )


def export(model, dtype):
    programs = {
        name: torch.export.export(
            model, (*_inputs([0], 0, steps), *_caches(dtype)), strict=False
        )
        for name, steps in (("prefill", PREFILL), ("decode", 1))
    }
    metadata = _metadata(dtype)
    metadata.update(
        get_test_prompt=PROMPT, get_test_expected=expected_tokens(model, dtype)
    )
    return _lower(programs, metadata)


class ContractProbe(torch.nn.Module):
    """Keep the input ABI and mutable outputs without running attention."""

    def __init__(self, first_cache, logits_shape=None, state_case=None):
        super().__init__()
        self.first_cache = first_cache
        self.logits_shape = logits_shape
        self.state_case = state_case

    def forward(self, *inputs):
        for tensor in inputs[self.first_cache :]:
            tensor.add_(1)
        shape = self.logits_shape or (1, inputs[0].shape[1], VOCAB)
        logits = torch.zeros(shape) + inputs[0].sum().float()
        if self.state_case == "state_count":
            return logits, inputs[self.first_cache]
        if self.state_case == "state_shape":
            state = list(inputs[self.first_cache :])
            state[0] = state[0].reshape(1, 4, CONTEXT, 4)
            return logits, *state
        return logits


def export_probe(case):
    metadata = _metadata()
    overrides = {
        "newer_version": {"get_io_contract_version": 2},
        "zero_version": {"get_io_contract_version": 0},
        "version_type": {"get_io_contract_version": "1"},
        "dtype_mismatch": {"get_cache_dtype": "float16"},
        "dtype_unknown": {"get_cache_dtype": "uint16"},
        "dtype_type": {"get_cache_dtype": 1},
        "layers_mismatch": {"get_n_layers": 3},
        "layers_zero": {"get_n_layers": 0},
        "layers_type": {"get_n_layers": "2"},
        "context_zero": {"get_max_seq_len": 0},
        "context_short": {"get_max_seq_len": 8},
        "context_type": {"get_max_seq_len": "16"},
        "context_large": {"get_max_seq_len": 1 << 31},
        "context_budget_large": {"get_max_context_len": 1 << 31},
        "unknown_contract": {"get_io_contract": "hf_static_future"},
        "contract_type": {"get_io_contract": 1},
    }
    metadata.update(overrides.get(case, {}))
    if case.startswith("missing_"):
        del metadata["get_" + case.removeprefix("missing_")]
    programs = {}
    for name, steps in (("prefill", PREFILL), ("decode", 1)):
        inputs, first_cache, logits_shape = _probe_inputs(case, name, steps)
        model = ContractProbe(first_cache, logits_shape, case)
        programs[name] = torch.export.export(model, tuple(inputs), strict=False)
    return _lower(programs, metadata)


def _probe_inputs(case, name, steps):
    inputs = list(_inputs([0], 0, steps)) + _caches()
    first_cache = 3
    if case in ("windows", "all_window"):
        window_mask = torch.ones(1, 1, steps, 6, dtype=torch.bool)
        if case == "windows":
            inputs.insert(2, window_mask)
            first_cache = 4
        else:
            inputs[1] = window_mask
        layers = LAYERS if case == "all_window" else 1
        for layer in range(layers):
            for c in (first_cache + 3 * layer, first_cache + 3 * layer + 1):
                inputs[c] = torch.zeros(1, HEADS, 6, HIDDEN // HEADS)
    shapes = {
        ("prefill", "tokens_batch"): [(0, (2, steps), torch.long)],
        ("prefill", "mask_rows"): [(1, (1, 1, 1, CONTEXT), torch.bool)],
        ("prefill", "mask_batch"): [(1, (2, 1, steps, CONTEXT), torch.bool)],
        ("prefill", "mask_heads"): [(1, (1, 2, steps, CONTEXT), torch.bool)],
        ("prefill", "mask_width"): [(1, (1, 1, steps, CONTEXT + 1), torch.bool)],
        ("prefill", "position_rows"): [(2, (1, 1), torch.long)],
        ("prefill", "position_batch"): [(2, (2, steps), torch.long)],
        ("prefill", "kv_shape"): [
            (4, (1, HEADS, HIDDEN // HEADS, CONTEXT), torch.float32)
        ],
        ("decode", "method_shape"): [
            (c, (1, 4, CONTEXT, 4), torch.float32) for c in (3, 4)
        ],
        ("decode", "method_dtype"): [
            (c, (1, HEADS, CONTEXT, HIDDEN // HEADS * 2), torch.float16) for c in (3, 4)
        ],
    }
    for index, shape, dtype in shapes.get((name, case), []):
        inputs[index] = torch.zeros(shape, dtype=dtype)
    if case == "cache_batch":
        for c in (3, 4):
            inputs[c] = torch.zeros(2, HEADS, CONTEXT, HIDDEN // HEADS)
    logits_shape = {
        ("prefill", "logits_rank"): (VOCAB,),
        ("prefill", "logits_last_only"): (1, 1, VOCAB),
        ("prefill", "logits_rows"): (1, steps + 1, VOCAB),
        ("decode", "method_vocab"): (1, 1, VOCAB + 1),
    }.get((name, case))
    return inputs, first_cache, logits_shape


PROBE_CASES = (
    "windows",
    "all_window",
    "newer_version",
    "zero_version",
    "version_type",
    "missing_io_contract_version",
    "dtype_mismatch",
    "dtype_unknown",
    "dtype_type",
    "missing_cache_dtype",
    "layers_mismatch",
    "layers_zero",
    "layers_type",
    "missing_n_layers",
    "context_zero",
    "context_short",
    "context_type",
    "context_large",
    "context_budget_large",
    "missing_max_seq_len",
    "unknown_contract",
    "contract_type",
    "tokens_batch",
    "mask_rows",
    "mask_batch",
    "mask_heads",
    "mask_width",
    "position_rows",
    "position_batch",
    "cache_batch",
    "kv_shape",
    "method_shape",
    "method_dtype",
    "logits_rank",
    "logits_last_only",
    "logits_rows",
    "method_vocab",
    "state_count",
    "state_shape",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    for name, dtype, return_state in (
        ("HfStaticTiny", torch.float32, False),
        ("HfStaticTinyHalf", torch.float16, False),
        ("HfStaticTinyWithState", torch.float32, True),
    ):
        model = TinyHfStaticDecoder(return_state).eval().requires_grad_(False)
        with open(os.path.join(args.outdir, name + ".pte"), "wb") as f:
            export(model, dtype).write_to_file(f)
    for case in PROBE_CASES:
        with open(os.path.join(args.outdir, "HfStatic_" + case + ".pte"), "wb") as f:
            export_probe(case).write_to_file(f)
    with open(os.path.join(args.outdir, "hf_static_fixtures.stamp"), "w") as f:
        f.write("float-cache v1 fixtures\n")


if __name__ == "__main__":
    main()
