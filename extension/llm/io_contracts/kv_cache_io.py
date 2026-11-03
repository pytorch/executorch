# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Python driver for programs that speak the ``kvcache_io`` I/O contract.

The contract (written by ``backends/qualcomm/utils/kv_cache_io.py``, driven in
C++ by ``KvCacheIOManager``) is one decoder step per method::

    (input_tokens, atten_mask, pos_ids, past_k_0..N-1, past_v_0..N-1)
        -> (logits, new_k_0..N-1, new_v_0..N-1)

``kv_forward`` takes one token, an optional ``prefill_forward`` a fixed chunk.
Past K is stored transposed ``[1, H, D, P]`` and V as ``[1, H, P, D]``; the
step returns only its own K/V slice and reads the past under an attention mask
the caller writes. This driver owns those buffers, so callers only exchange
tokens and logits.
"""

from typing import Dict, List, Optional, Sequence

import torch

CONTRACT = "kvcache_io"
PREFILL_METHOD = "prefill_forward"
DECODE_METHOD = "kv_forward"


# ExecuTorch ScalarType codes (c10 numbering) of the dtypes the contract uses.
_DTYPES = {
    0: torch.uint8,
    1: torch.int8,
    2: torch.int16,
    3: torch.int32,
    4: torch.int64,
    5: torch.float16,
    6: torch.float32,
    7: torch.float64,
    11: torch.bool,
    15: torch.bfloat16,
    27: torch.uint16,
}


def tensor_dtype(info) -> torch.dtype:
    """A runtime tensor's declared dtype, accepting c10 codes or torch dtypes."""
    dtype = info.dtype()
    if isinstance(dtype, torch.dtype):
        return dtype
    if int(dtype) not in _DTYPES:
        raise ValueError(f"Unsupported tensor dtype code {dtype!r}.")
    return _DTYPES[int(dtype)]


def constant_method(program, name: str, default=None):
    if name not in program.method_names:
        return default
    value = program.load_method(name).execute([])
    return value[0] if len(value) == 1 else list(value)


def io_contract(program) -> Optional[str]:
    """The I/O contract a loaded program names, or None."""
    return constant_method(program, "get_io_contract")


class _Method:
    """One contract method with its own token, mask and position inputs."""

    def __init__(self, program, name: str):
        self.name = name
        self.method = program.load_method(name)
        meta = self.method.metadata
        self.num_layers = (meta.num_inputs() - 3) // 2
        tokens, mask, positions = (meta.input_tensor_meta(i) for i in range(3))
        self.ar_len = tokens.sizes()[-1]
        self.context_len = mask.sizes()[-1]
        self.past_len = self.context_len - self.ar_len
        self.tokens = torch.zeros(list(tokens.sizes()), dtype=tensor_dtype(tokens))
        self.mask = torch.zeros(list(mask.sizes()), dtype=tensor_dtype(mask))
        self.positions = torch.zeros(
            list(positions.sizes()), dtype=tensor_dtype(positions)
        )
        self.cache_shapes = [
            (
                list(meta.input_tensor_meta(i).sizes()),
                tensor_dtype(meta.input_tensor_meta(i)),
            )
            for i in range(3, 3 + 2 * self.num_layers)
        ]

    def fill_mask(self, n_past: int, n_valid: int) -> None:
        # Visible slots are 0 in a float mask and the top code in a quantized
        # one; the program's encoding maps that code to 0.
        dtype = self.mask.dtype
        if dtype.is_floating_point:
            hidden, visible = -65535.0, 0.0
        else:
            hidden, visible = 0, torch.iinfo(dtype).max
        self.mask.fill_(hidden)
        rows = self.mask.view(self.ar_len, self.context_len)
        rows[:, :n_past] = visible
        for r in range(self.ar_len):
            own = min(r, max(n_valid - 1, 0))
            rows[r, self.past_len : self.past_len + own + 1] = visible


class KvCacheIORunner:
    """Greedy-capable driver for ``kvcache_io`` programs.

    ``programs`` holds one program with both methods (a hybrid export) or the
    separate decode and prompt programs. Prompts run through
    ``prefill_forward`` in chunks when there is one, else token by token.
    """

    def __init__(self, programs: Sequence):
        self.methods: Dict[str, _Method] = {}
        for program in programs:
            for name in (DECODE_METHOD, PREFILL_METHOD):
                if name in program.method_names and name not in self.methods:
                    self.methods[name] = _Method(program, name)
        if not self.methods:
            raise ValueError(
                f"A {CONTRACT} program has a {DECODE_METHOD} or {PREFILL_METHOD} method."
            )
        # Single tokens go through the decode step, prompts through the
        # prompt step; a program with only one serves both.
        self.step_method = (
            self.methods.get(DECODE_METHOD) or self.methods[PREFILL_METHOD]
        )
        self.prompt_method = self.methods.get(PREFILL_METHOD) or self.step_method
        first = programs[0]
        self.eos_ids = constant_method(first, "get_eos_ids", [])
        if not isinstance(self.eos_ids, list):
            self.eos_ids = [self.eos_ids]
        self.logits_scale = constant_method(first, "get_logits_scale", 1.0)
        self.logits_zero_point = constant_method(first, "get_logits_zero_point", 0)
        self.num_layers = self.step_method.num_layers
        self.max_seq_len = self.step_method.context_len
        # Past caches by past length: methods with different step sizes lay
        # the same positions out with different strides.
        self.caches: Dict[int, List[torch.Tensor]] = {}
        for method in self.methods.values():
            if method.past_len not in self.caches:
                self.caches[method.past_len] = [
                    torch.zeros(shape, dtype=dtype)
                    for shape, dtype in method.cache_shapes
                ]
        self.filled = 0
        self.active_past_len: Optional[int] = None

    def reset(self) -> None:
        for cache in self.caches.values():
            for tensor in cache:
                tensor.zero_()
        self.filled = 0
        self.active_past_len = None

    def past(self, past_len: Optional[int] = None) -> List[torch.Tensor]:
        """The K caches then V caches in the layout of ``past_len`` (default:
        the one holding the filled positions)."""
        return self.caches[self.active_past_len if past_len is None else past_len]

    def _rearrange(self, past_len: int) -> None:
        src = self.active_past_len
        if src is not None and src != past_len and self.filled:
            self.adopt(self.caches[src], self.filled, past_len)
        self.active_past_len = past_len

    def adopt(
        self,
        caches: Sequence[torch.Tensor],
        filled: int,
        past_len: Optional[int] = None,
    ) -> None:
        """Copy ``filled`` positions of caches in another layout into the
        layout of ``past_len`` (default: the single-token step's)."""
        past_len = self.step_method.past_len if past_len is None else past_len
        dst = self.caches[past_len]
        n = self.num_layers
        for i in range(n):
            dst[i][..., :filled] = caches[i][..., :filled]
            dst[n + i][..., :filled, :] = caches[n + i][..., :filled, :]
        self.filled = filled
        self.active_past_len = past_len

    def _step(self, method: _Method, chunk: List[int]) -> torch.Tensor:
        n, start = len(chunk), self.filled
        if start + n > method.past_len + (1 if method.ar_len == 1 else 0):
            raise ValueError(
                f"Context full: position {start} with {method.past_len} slots of past."
            )
        padded = chunk + [chunk[-1]] * (method.ar_len - n)
        method.tokens.view(-1)[:] = torch.tensor(padded, dtype=method.tokens.dtype)
        method.positions.view(-1)[:] = torch.tensor(
            [min(start + i, method.context_len - 1) for i in range(method.ar_len)],
            dtype=method.positions.dtype,
        )
        self._rearrange(method.past_len)
        method.fill_mask(start, n)
        caches = self.caches[method.past_len]
        outputs = method.method.execute(
            [method.tokens, method.mask, method.positions, *caches]
        )
        layers = self.num_layers
        for i in range(layers):
            caches[i][..., start : start + n] = outputs[1 + i][..., :n]
            caches[layers + i][..., start : start + n, :] = outputs[1 + layers + i][
                ..., :n, :
            ]
        self.filled = start + n
        logits = outputs[0].reshape(-1, outputs[0].shape[-1])[n - 1]
        if not logits.dtype.is_floating_point:
            logits = (
                logits.to(torch.int64) - self.logits_zero_point
            ).float() * self.logits_scale
        return logits

    def forward(self, tokens: Sequence[int]) -> torch.Tensor:
        """Run ``tokens`` after the filled positions; the last token's logits."""
        tokens = list(tokens)
        if not tokens:
            raise ValueError("No tokens to run.")
        method = self.prompt_method if len(tokens) > 1 else self.step_method
        logits = None
        for i in range(0, len(tokens), method.ar_len):
            logits = self._step(method, tokens[i : i + method.ar_len])
        return logits

    def generate(self, prompt_ids: Sequence[int], max_new_tokens: int) -> List[int]:
        """Greedy continuation of ``prompt_ids`` (new tokens only), stopping
        at an end-of-sequence id or a full context."""
        self.reset()
        logits = self.forward(prompt_ids)
        generated: List[int] = []
        while len(generated) < max_new_tokens:
            token = int(torch.argmax(logits))
            generated.append(token)
            if token in self.eos_ids or self.filled >= self.max_seq_len:
                break
            logits = self.forward([token])
        return generated
