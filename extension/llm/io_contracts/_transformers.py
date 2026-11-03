# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``transformers`` exporters runner for ``kvcache_io`` programs."""

import torch
from executorch.extension.llm.io_contracts.hf_static import HfStaticRunner
from executorch.extension.llm.io_contracts.kv_cache_io import KvCacheIORunner
from transformers.exporters import ModelRunner
from transformers.exporters.metadata import ExportMetadata


class KvCacheIOModelRunner(ModelRunner):
    """Runs a ``kvcache_io`` program under ``ExportedGenerator``.

    The program keeps the cache in buffers this runner owns, so it is a
    state-owning runner: the generation loop feeds token ids and reads logits,
    and a prompt program hands its filled cache to the decode program through
    ``state_tensors`` / ``adopt_state``.
    """

    owns_state = True

    def __init__(self, program, export_metadata=None, **kwargs):
        self.driver = KvCacheIORunner([program])
        self.export_metadata = ExportMetadata.from_dict(export_metadata)
        # One program with both steps serves the whole loop.
        self.drives_generation = (
            self.driver.prompt_method is not self.driver.step_method
        )
        layers = self.driver.num_layers
        self.state_paths = frozenset(
            [f"past_k_{i}" for i in range(layers)]
            + [f"past_v_{i}" for i in range(layers)]
        )
        self.input_names = ("input_ids",)
        self.cache_inputs = ("past_key_values",)
        if self.export_metadata.dtype is None:
            self.dtype = torch.float32

    @property
    def state_length(self) -> int:
        return self.driver.filled

    def state_tensors(self, paths=None) -> dict:
        caches = self.driver.past()
        layers = self.driver.num_layers
        named = {f"past_k_{i}": caches[i] for i in range(layers)}
        named.update({f"past_v_{i}": caches[layers + i] for i in range(layers)})
        return {k: v for k, v in named.items() if paths is None or k in paths}

    def adopt_state(self, leaves: dict, length: int) -> None:
        layers = self.driver.num_layers
        self.driver.adopt(
            [leaves[f"past_k_{i}"] for i in range(layers)]
            + [leaves[f"past_v_{i}"] for i in range(layers)],
            length,
        )

    def reset_state(self) -> None:
        self.driver.reset()

    def __call__(self, input_ids=None, **kwargs) -> dict:
        if input_ids is None or input_ids.shape[0] != 1:
            raise ValueError("kvcache_io programs take one sequence of input_ids.")
        logits = self.driver.forward(input_ids[0].tolist())
        return {"logits": logits.view(1, 1, -1)}


class HfStaticModelRunner(ModelRunner):
    """State-owning HF generator adapter for a fixed prefill/decode program."""

    owns_state = True
    drives_generation = True

    def __init__(self, program, export_metadata=None, **kwargs):
        self.driver = HfStaticRunner(program)
        metadata = dict(export_metadata or {})
        metadata["kwargs"] = {
            **metadata.get("kwargs", {}),
            "attention_mask": {"rank": 2, "dtype": "bool"},
        }
        self.export_metadata = ExportMetadata.from_dict(metadata)
        self.input_names = ("input_ids", "attention_mask", "position_ids")
        self.cache_inputs = ("past_key_values",)
        self.state_paths = frozenset()
        self.dtype = self.driver.driver.layout.dtype

    @property
    def state_length(self):
        return self.driver.filled

    def reset_state(self):
        self.driver.reset()

    def _validate_mask(self, mask, query_len):
        message = "hf_static does not support caller-padded or custom attention masks."
        if (
            not isinstance(mask, torch.Tensor)
            or mask.ndim not in (2, 4)
            or mask.shape[0] != 1
        ):
            raise ValueError(message)
        start = self.driver.filled
        if mask.ndim == 2:
            width = min(mask.shape[1], start + query_len)
            if mask.shape[1] < query_len or not bool((mask[:, :width] == 1).all()):
                raise ValueError(message)
            return
        if tuple(mask.shape[:3]) != (1, 1, query_len) or mask.shape[-1] < query_len:
            raise ValueError(message)
        if mask.dtype == torch.bool:
            keep = mask
        elif mask.dtype.is_floating_point and bool(
            ((mask == 0) | (mask <= -10000)).all()
        ):
            keep = mask == 0
        else:
            raise ValueError(message)
        width = mask.shape[-1]
        first = max(0, start + query_len - width)
        slots = torch.arange(first, first + width, device=mask.device)
        positions = torch.arange(start, start + query_len, device=mask.device)
        causal = (slots[None] <= positions[:, None])[None, None]
        if torch.equal(keep, causal):
            return
        windows = set(self.driver.driver.layout.mask_widths.values())
        for window in windows:
            windowed = (
                causal & ((positions[:, None] - slots[None]) < window)[None, None]
            )
            if torch.equal(keep, windowed):
                return
        raise ValueError(message)

    def __call__(
        self, input_ids=None, attention_mask=None, position_ids=None, **kwargs
    ):
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("hf_static programs take one sequence of input_ids.")
        if attention_mask is not None:
            masks = (
                attention_mask.values()
                if isinstance(attention_mask, dict)
                else (attention_mask,)
            )
            for mask in masks:
                if mask is None:
                    continue
                self._validate_mask(mask, input_ids.shape[1])
        if position_ids is not None:
            expected = torch.arange(
                self.driver.filled,
                self.driver.filled + input_ids.shape[1],
                device=position_ids.device,
            )[None]
            if position_ids.shape != expected.shape or not torch.equal(
                position_ids, expected
            ):
                raise ValueError(
                    "hf_static requires consecutive absolute position_ids."
                )
        logits = self.driver.forward(input_ids[0].tolist())
        return {"logits": logits.view(1, 1, -1)}
