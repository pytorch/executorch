# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Any, Dict, Tuple, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree

if TYPE_CHECKING:
    from executorch.examples.models.llama.static_attention import (
        StaticAttentionIOManager,
    )


class SplitLinearModule(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        out_target_split_size=1,
        out_max_splits=1,
        in_target_split_size=1,
        in_max_splits=1,
    ):
        super(SplitLinearModule, self).__init__()
        self.out_split_sizes = self._get_split_sizes(
            out_features, out_target_split_size, out_max_splits
        )
        self.in_split_sizes = self._get_split_sizes(
            in_features, in_target_split_size, in_max_splits
        )
        print(
            f"Splitting out_features={out_features} into {len(self.out_split_sizes)} of size {self.out_split_sizes[0]}."
        )
        print(
            f"Splitting in_features={in_features} into {len(self.in_split_sizes)} of size {self.in_split_sizes[0]}."
        )

        # self.ops contains a list of linear ops for different pieces of the output matrix
        # The index of an op at (in_idx, out_idx) is given by self.op_index(in_idx, out_idx)
        self.ops = torch.nn.ModuleList()
        for idx_out, s_out in enumerate(self.out_split_sizes):
            for idx_in, s_in in enumerate(self.in_split_sizes):
                assert len(self.ops) == self.op_index(idx_in, idx_out)
                self.ops.append(torch.nn.Linear(s_in, s_out, bias=False))

    def op_index(self, in_index, out_index):
        idx = out_index * len(self.in_split_sizes) + in_index
        return idx

    def _get_split_sizes(self, n_features, target_split_size, max_splits):
        num_splits = max(n_features // target_split_size, 1)
        if num_splits > max_splits:
            num_splits = max_splits

        split_size = n_features // num_splits
        split_remainder = n_features % num_splits
        if split_remainder > 0:
            raise ValueError(
                f"Cannot split {n_features} with target_split_size={target_split_size} and max_splits={max_splits} because it leaves a remainder of {split_remainder}."
            )

        ret = [split_size for _ in range(num_splits)]
        return ret

    def set_params(self, weight):
        split_weights = []
        for w_out in weight.split(self.out_split_sizes, dim=0):
            for w in w_out.split(self.in_split_sizes, dim=1):
                split_weights.append(w)

        for i, split in enumerate(self.ops):
            split.weight = torch.nn.Parameter(split_weights[i])

    def forward(self, x):
        if len(self.in_split_sizes) == 1:
            out_chunks = [op(x) for op in self.ops]
        else:
            x_splits = x.split(self.in_split_sizes, dim=-1)
            out_chunks = [
                torch.sum(
                    torch.stack(
                        [
                            self.ops[self.op_index(in_idx, out_idx)].forward(
                                x_splits[in_idx]
                            )
                            for in_idx in range(len(self.in_split_sizes))
                        ],
                    ),
                    dim=0,
                )
                for out_idx in range(len(self.out_split_sizes))
            ]

        return torch.concat(out_chunks, dim=-1)


def replace_linear_with_split_linear(
    model,
    out_target_split_size,
    out_max_splits,
    in_target_split_size,
    in_max_splits=1,
):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            assert module.bias is None, "SplitLinearModule does not support bias"
            new_module = SplitLinearModule(
                module.in_features,
                module.out_features,
                out_target_split_size,
                out_max_splits,
                in_target_split_size,
                in_max_splits,
            )
            new_module.set_params(module.weight)
            setattr(model, name, new_module)
        else:
            replace_linear_with_split_linear(
                module,
                out_target_split_size,
                out_max_splits,
                in_target_split_size,
                in_max_splits,
            )


def setup_multifunction_managers(
    config,
    prefill_input_len: int,
    decode_input_len: int,
    shared_cache_len: int,
    dtype: torch.dtype = torch.float16,
    mask_val: float = float("-inf"),
    style: str = "smart_mask",
):
    """
    Create prefill and decode StaticAttentionIOManager instances with shared cache buffers.

    Both managers use the same cache_len so they share cache memory directly.
    Returns (decode_mgr, prefill_mgr, prefill_mask).
    """
    from executorch.examples.models.llama.static_attention import (
        StaticAttentionIOManager,
    )

    mgr = StaticAttentionIOManager(
        config,
        input_len=decode_input_len,
        cache_lens=shared_cache_len,
        batch_size=1,
        dtype=dtype,
        style=style,
        mask_val=mask_val,
    )

    prefill_mgr = StaticAttentionIOManager(
        config,
        input_len=prefill_input_len,
        cache_lens=shared_cache_len,
        batch_size=1,
        dtype=dtype,
        style=style,
        mask_val=mask_val,
    )

    # Share cache buffers — no copying needed
    prefill_mgr.k_caches = mgr.k_caches
    prefill_mgr.v_caches = mgr.v_caches
    prefill_mask = prefill_mgr.masks

    return mgr, prefill_mgr, prefill_mask


def create_pte_wrapper(
    decode_method,
    prefill_method,
    mgr: "StaticAttentionIOManager",
    prefill_seq_len: int,
    prefill_mask: Dict[str, torch.Tensor],
):
    """
    Create a wrapper function that adapts PTE execution to the interface
    expected by StaticAttentionIOManager.

    This multifunction version selects between prefill and decode methods
    based on the input sequence length. Both methods use the SAME cache_len,
    so the cache buffer is shared directly without any slicing or copying.

    The wrapper:
    - Takes (tokens, options_dict) like the eager model
    - Selects prefill or decode method based on token count
    - Uses the same cache buffer for both methods (no slicing needed)
    - Flattens inputs using pytree
    - Executes the appropriate PTE method
    - Reconstructs outputs to match eager model format: (logits, {"out_cache_state": (k_dict, v_dict)})

    Args:
        decode_method: The PTE method for decode (seqlen=1)
        prefill_method: The PTE method for prefill (seqlen=input_len)
        mgr: StaticAttentionIOManager with caches sized for shared cache_len
        prefill_seq_len: The sequence length for prefill
        prefill_mask: Pre-computed mask tensor for prefill method
    """

    k_cache_keys = list(mgr.k_caches.keys())
    v_cache_keys = list(mgr.v_caches.keys())

    timing_stats = {
        "flatten_time": 0.0,
        "execute_time": 0.0,
        "reconstruct_time": 0.0,
        "detection_time": 0.0,
        "options_build_time": 0.0,
        "call_count": 0,
    }

    def wrapper(
        tokens: torch.Tensor, options: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        timing_stats["call_count"] += 1

        t0 = time.perf_counter()

        # Detect actual sequence length.
        # StaticAttentionIOManager._run_once pads tokens with zeros on the right.
        # For decode (1 actual token), positions 1+ are all zeros.
        padded_seq_len = tokens.shape[1]
        if padded_seq_len > 1 and (tokens[0, 1:] == 0).all():
            actual_seq_len = 1
        else:
            actual_seq_len = padded_seq_len

        is_prefill = actual_seq_len == prefill_seq_len

        t1 = time.perf_counter()
        timing_stats["detection_time"] += t1 - t0

        t0 = time.perf_counter()

        # Get the input cache state from options
        in_k_caches, in_v_caches = options["in_cache_state"]

        # Both prefill and decode use the same cache_len, so no slicing needed!
        # Just select the appropriate method and mask.
        if is_prefill:
            method = prefill_method
            adapted_mask = prefill_mask
        else:
            method = decode_method
            adapted_mask = mgr.masks

        adapted_options = {
            "masks": adapted_mask,
            "freqs_cos_override": options["freqs_cos_override"],
            "freqs_sin_override": options["freqs_sin_override"],
            "in_cache_state": (in_k_caches, in_v_caches),  # Same cache for both!
        }

        if "last_valid_token_pos" in options:
            adapted_options["last_valid_token_pos"] = options["last_valid_token_pos"]

        inputs = (tokens, adapted_options)

        t1 = time.perf_counter()
        timing_stats["options_build_time"] += t1 - t0

        t0 = time.perf_counter()
        flat_inputs, _ = pytree.tree_flatten(inputs)
        t1 = time.perf_counter()
        timing_stats["flatten_time"] += t1 - t0

        t0 = time.perf_counter()
        outputs = method.execute(flat_inputs)
        t1 = time.perf_counter()
        timing_stats["execute_time"] += t1 - t0

        t0 = time.perf_counter()

        logits = outputs[0]

        num_layers = len(k_cache_keys)
        k_updates = outputs[1 : 1 + num_layers]
        v_updates = outputs[1 + num_layers : 1 + 2 * num_layers]

        k_cache_dict = dict(zip(k_cache_keys, k_updates))
        v_cache_dict = dict(zip(v_cache_keys, v_updates))

        attn_updates = {"out_cache_state": (k_cache_dict, v_cache_dict)}

        t1 = time.perf_counter()
        timing_stats["reconstruct_time"] += t1 - t0

        return logits, attn_updates

    def print_timing_stats():
        n = timing_stats["call_count"]
        if n > 0:
            print(f"\n=== Wrapper Timing Stats ({n} calls) ===")
            print(
                f"  Detection time:   {timing_stats['detection_time']*1000:.2f}ms total, {timing_stats['detection_time']/n*1000:.4f}ms avg"
            )
            print(
                f"  Options build:    {timing_stats['options_build_time']*1000:.2f}ms total, {timing_stats['options_build_time']/n*1000:.4f}ms avg"
            )
            print(
                f"  Flatten time:     {timing_stats['flatten_time']*1000:.2f}ms total, {timing_stats['flatten_time']/n*1000:.4f}ms avg"
            )
            print(
                f"  Execute time:     {timing_stats['execute_time']*1000:.2f}ms total, {timing_stats['execute_time']/n*1000:.3f}ms avg"
            )
            print(
                f"  Reconstruct time: {timing_stats['reconstruct_time']*1000:.2f}ms total, {timing_stats['reconstruct_time']/n*1000:.4f}ms avg"
            )
            total = (
                timing_stats["detection_time"]
                + timing_stats["options_build_time"]
                + timing_stats["flatten_time"]
                + timing_stats["execute_time"]
                + timing_stats["reconstruct_time"]
            )
            print(
                f"  Total wrapper:    {total*1000:.2f}ms total, {total/n*1000:.3f}ms avg"
            )
            print(
                f"  Execute is {timing_stats['execute_time']/total*100:.1f}% of wrapper time"
            )
            expected_tps = 1000 / (timing_stats["execute_time"] / n * 1000)
            print(f"  Expected tok/s from execute alone: {expected_tps:.1f}")

    wrapper.print_timing_stats = print_timing_stats
    wrapper.timing_stats = timing_stats

    return wrapper
