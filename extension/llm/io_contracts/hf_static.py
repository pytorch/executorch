# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Python drivers for the version-1 ``hf_static`` float-cache contract."""

from typing import Callable, Dict, List, Sequence

import torch
from executorch.extension.llm.export.hf_static import (
    CACHE_PREFIX,
    DECODE_METHOD,
    HfStaticLayout,
    IO_CONTRACT,
    IO_CONTRACT_VERSION,
    PREFILL_METHOD,
)
from executorch.extension.llm.io_contracts.kv_cache_io import (
    constant_method,
    tensor_dtype,
)


class HfStaticDriver:
    """Drive flat static-cache steps for calibration and eager validation.

    ``steps`` maps each query length to a callable returning logits followed by
    layer-ordered keys, values and counters. With ``in_place``, keep the caller's
    buffers instead of adopting the returned state.
    """

    def __init__(
        self, steps: Dict[int, Callable], layout: HfStaticLayout, in_place: bool = False
    ):
        self.steps, self.layout, self.in_place = steps, layout, in_place
        self.state = [
            torch.zeros(shape, dtype=dtype)
            for shape, dtype in zip(layout.state_shapes, layout.state_dtypes)
        ]
        self.filled = 0

    def reset(self) -> None:
        for tensor in self.state:
            tensor.zero_()
        self.filled = 0

    def forward(  # noqa: C901
        self, tokens: Sequence[int], start: int, query_len: int
    ) -> torch.Tensor:
        """Run a nonempty chunk at ``start`` and return its last real logits."""
        n, length = len(tokens), self.layout.max_seq_len
        if not 0 < n <= query_len or query_len not in self.steps:
            raise ValueError("The chunk must fit an available query length.")
        if start < 0 or start + query_len > length:
            raise ValueError(
                f"Query at {start} with length {query_len} exceeds context {length}."
            )
        pos = torch.arange(start, start + query_len, dtype=self.layout.position_dtype)
        inputs, state = [], iter(self.state)
        for tensor in self.state:
            if tensor.dim() == 0:
                tensor.fill_(start)
        for name in self.layout.input_names:
            if name == "input_ids":
                ids = list(tokens) + [tokens[-1]] * (query_len - n)
                inputs.append(torch.tensor([ids], dtype=self.layout.token_dtype))
            elif name == "position_ids":
                inputs.append(pos[None])
            elif name.startswith("attention_mask"):
                width = self.layout.mask_widths[name]
                mask = torch.arange(width)[None] <= pos[:, None]
                mask[n:] = False
                inputs.append(mask[None, None])
            else:
                inputs.append(next(state))
        out = self.steps[query_len](*inputs)
        if not self.in_place:
            self.state = [t.clone() for t in out[1:]]
        for tensor in self.state:
            if tensor.dim() == 0:
                tensor.fill_(start + n)
        self.filled = start + n
        return out[0].reshape(-1, out[0].shape[-1])[n - 1]

    def run(self, tokens: Sequence[int]) -> torch.Tensor:
        """Append tokens, using decode for a tail whose padded chunk would overflow."""
        tokens = list(tokens)
        if not tokens or self.filled + len(tokens) > self.layout.max_seq_len:
            raise ValueError("Tokens must be nonempty and fit the remaining context.")
        offset, chunk, logits = 0, max(self.steps), None
        while offset < len(tokens):
            q = chunk
            if len(tokens) - offset == 1 or self.filled + q > self.layout.max_seq_len:
                q = 1
            if q not in self.steps:
                raise ValueError(
                    "A single-token decode method is required for this tail."
                )
            part = tokens[offset : offset + q]
            logits = self.forward(part, self.filled, q)
            offset += len(part)
        return logits

    def generate(self, prompt: Sequence[int], max_new_tokens: int) -> List[int]:
        """Greedy sequence including the prompt, for export-time calibration."""
        self.reset()
        tokens = list(prompt)
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative.")
        logits = self.run(tokens)
        for _ in range(max_new_tokens):
            if len(tokens) >= self.layout.max_seq_len:
                break
            tokens.append(int(logits.argmax()))
            logits = self.run(tokens[-1:])
        return tokens


class HfStaticRunner:
    """Own the shared cache buffers of a loaded ``hf_static`` program."""

    def __init__(self, program):  # noqa: C901
        version = constant_method(program, "get_io_contract_version")
        if (
            not isinstance(version, int)
            or isinstance(version, bool)
            or not 1 <= version <= IO_CONTRACT_VERSION
        ):
            raise ValueError(
                f"Unsupported hf_static contract version {version!r}; supported: 1."
            )
        if constant_method(program, "get_io_contract") != IO_CONTRACT:
            raise ValueError("Expected an hf_static program.")
        length = constant_method(program, "get_max_seq_len")
        layers = constant_method(program, "get_n_layers")
        dtype_name = constant_method(program, "get_cache_dtype")
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise ValueError("get_max_seq_len must be a positive integer.")
        if not isinstance(layers, int) or isinstance(layers, bool) or layers <= 0:
            raise ValueError("get_n_layers must be a positive integer.")
        self.methods = {}
        layouts = {}
        for name in (DECODE_METHOD, PREFILL_METHOD):
            if name not in program.method_names:
                continue
            method = program.load_method(name)
            meta = method.metadata
            count = meta.num_inputs()
            if count < 3 + 3 * layers:
                raise ValueError(f"{name}: not enough inputs for {layers} layers.")
            inputs = [meta.input_tensor_meta(i) for i in range(count)]
            shapes = [torch.Size(t.sizes()) for t in inputs]
            dtypes = [tensor_dtype(t) for t in inputs]
            masks = count - 2 - 3 * layers
            query = shapes[0][-1] if len(shapes[0]) == 2 else 0
            if query <= 0 or shapes[0] != (1, query) or dtypes[0] != torch.int64:
                raise ValueError(f"{name}: invalid input_ids shape or dtype.")
            if name == DECODE_METHOD and query != 1:
                raise ValueError("decode must take exactly one token.")
            if shapes[1 + masks] != (1, query) or dtypes[1 + masks] != torch.int64:
                raise ValueError(f"{name}: invalid position_ids shape or dtype.")
            widths = {}
            for i in range(masks):
                shape = shapes[1 + i]
                if (
                    len(shape) != 4
                    or shape[:3] != (1, 1, query)
                    or shape[-1] <= 0
                    or shape[-1] > length
                    or dtypes[1 + i] != torch.bool
                ):
                    raise ValueError(f"{name}: invalid attention mask shape or dtype.")
                widths[f"attention_mask_{i}"] = shape[-1]
            state_shapes, state_dtypes = shapes[-3 * layers :], dtypes[-3 * layers :]
            cache_shapes = []
            for i in range(layers):
                k, v, counter = state_shapes[3 * i : 3 * i + 3]
                kd, vd, cd = state_dtypes[3 * i : 3 * i + 3]
                if (
                    len(k) != 4
                    or k != v
                    or k[0] != 1
                    or any(d <= 0 for d in k)
                    or k[2] > length
                    or counter != ()
                    or cd != torch.int64
                ):
                    raise ValueError(f"{name}: invalid cache shapes or counter.")
                if (
                    kd != vd
                    or kd not in (torch.float16, torch.float32)
                    or dtype_name != str(kd).removeprefix("torch.")
                ):
                    raise ValueError(
                        f"{name}: get_cache_dtype does not match the float cache inputs."
                    )
                if k[2] not in widths.values() or query > k[2]:
                    raise ValueError(
                        f"{name}: mask/cache width mismatch or query longer than cache."
                    )
                cache_shapes.extend((k, v))
            if query > length or meta.num_outputs() not in (
                1 + 3 * layers,
                1 + 6 * layers,
            ):
                raise ValueError(f"{name}: invalid context or output count.")
            # ExecuTorch places mutated inputs before the user logits.
            expected_state = sorted(
                (tuple(shape), str(dtype))
                for shape, dtype in zip(state_shapes, state_dtypes)
            )
            mutation_state = sorted(
                (
                    tuple(meta.output_tensor_meta(i).sizes()),
                    str(tensor_dtype(meta.output_tensor_meta(i))),
                )
                for i in range(3 * layers)
            )
            if mutation_state != expected_state:
                raise ValueError(
                    f"{name}: mutated outputs differ from the cache inputs."
                )
            if meta.num_outputs() == 1 + 6 * layers:
                for i, (shape, dtype) in enumerate(zip(state_shapes, state_dtypes)):
                    info = meta.output_tensor_meta(3 * layers + 1 + i)
                    if torch.Size(info.sizes()) != shape or tensor_dtype(info) != dtype:
                        raise ValueError(
                            f"{name}: user state output {i} differs from its input."
                        )
            logits = meta.output_tensor_meta(3 * layers)
            if (
                len(logits.sizes()) != 3
                or tuple(logits.sizes())[:2] != (1, query)
                or not tensor_dtype(logits).is_floating_point
            ):
                raise ValueError(f"{name}: invalid logits shape or dtype.")
            names = ["input_ids", *widths, "position_ids"] + [
                f"{CACHE_PREFIX}{i}" for i in range(3 * layers)
            ]
            layouts[name] = HfStaticLayout(
                names,
                query,
                length,
                layers,
                cache_shapes,
                state_shapes,
                state_dtypes[0],
                widths,
                state_dtypes=state_dtypes,
                token_dtype=dtypes[0],
                position_dtype=dtypes[1 + masks],
                window_masks=[n for n, w in widths.items() if w < length],
            )
            self.methods[name] = method
        if DECODE_METHOD not in self.methods:
            raise ValueError("An hf_static program requires decode.")
        layout = layouts[DECODE_METHOD]
        steps = {}
        for name, method in self.methods.items():
            other = layouts[name]
            if (
                other.state_shapes,
                other.state_dtypes,
                other.mask_widths,
                other.token_dtype,
                other.position_dtype,
            ) != (
                layout.state_shapes,
                layout.state_dtypes,
                layout.mask_widths,
                layout.token_dtype,
                layout.position_dtype,
            ):
                raise ValueError(
                    "prefill and decode must have identical cache and mask layouts."
                )
            if other.query_len in steps:
                raise ValueError("prefill and decode must have distinct query lengths.")

            def step(*inputs, method=method):
                outputs = method.execute(inputs)
                state = (
                    outputs[3 * layers + 1 :]
                    if len(outputs) == 1 + 6 * layers
                    else inputs[-3 * layers :]
                )
                return [outputs[3 * layers], *state]

            steps[other.query_len] = step
        self.driver = HfStaticDriver(steps, layout, in_place=True)
        self.eos_ids = constant_method(program, "get_eos_ids", [])
        if not isinstance(self.eos_ids, list):
            self.eos_ids = [self.eos_ids]

    @property
    def filled(self):
        return self.driver.filled

    def reset(self):
        self.driver.reset()

    def forward(self, tokens):
        return self.driver.run(tokens)

    def generate(self, prompt_ids, max_new_tokens):
        """Return new greedy tokens only, stopping at EOS or a full context."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative.")
        self.reset()
        logits = self.forward(prompt_ids)
        generated = []
        while (
            len(generated) < max_new_tokens
            and self.filled < self.driver.layout.max_seq_len
        ):
            token = int(logits.argmax())
            generated.append(token)
            if token in self.eos_ids:
                break
            logits = self.forward([token])
        return generated


try:
    from transformers.exporters import register_io_contract
except ImportError:
    pass
else:
    from executorch.extension.llm.io_contracts._transformers import HfStaticModelRunner

    register_io_contract(IO_CONTRACT, HfStaticModelRunner)
