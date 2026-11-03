# Hugging Face static-cache I/O contract

**Version:** 1 (`get_io_contract = "hf_static"`)

## TL;DR

An `hf_static` program keeps Hugging Face's per-step static-cache tensor signature. It is not a drop-in replacement for HF's single dynamic `forward` method: ExecuTorch exports fixed-length `prefill` / `decode` methods and requires the padding and counter protocol below. The runner owns the cache buffers; the program writes into them.

## Goals / Non-Goals

This contract defines the inputs, outputs, metadata and runner responsibilities for batch-one float-cache generation. It does not define runtime-owned dynamic caches, cache-state import / fork, quantized caches, or arbitrary query lengths. Sliding-window correctness is CPU-tested; this document does not claim sliding-window device performance past context 512.

## Methods and metadata

`decode` takes one token. An optional `prefill` method takes a fixed `N` tokens per call. Both methods share the same cache shapes, dtypes and buffers. If `prefill` is absent, prompts run through `decode`.

Required metadata methods:

| Method | Value / meaning |
|---|---|
| `get_io_contract` | `"hf_static"` |
| `get_io_contract_version` | `1`; a version-1 runner rejects newer versions |
| `get_max_seq_len` | Maximum total context length; not necessarily the longest physical cache when all layers slide |
| `get_n_layers` | Number of key / value / counter triples |
| `get_cache_dtype` | `"float32"` or `"float16"`, matching every cache input in every method |

The exporter also records model vocabulary / BOS / EOS metadata when available and `use_kv_cache = True`. Runners validate contract metadata against method tensor metadata before allocating or writing buffers. Version 1 has no per-cache quantization scales or zero points.

## Inputs and outputs

Inputs are flattened in HF's exported order:

1. `input_ids`: int64 `[1, N]` (`N = 1` for decode).
2. One or more bool attention masks: `[1, 1, N, W]`, in the exported order.
3. `position_ids`: int64 `[1, N]`.
4. Per layer, in layer order: key cache `[1, H_kv, L_i, D]`, value cache of the same shape and dtype, and a scalar int64 counter.

Full-attention cache length `L_i` is the context length. A sliding-attention layer keeps `min(sliding_window, context_length)` rows. Sliding layers are determined from model configuration / HF cache layer types, not cache-size comparisons or mask names. Every layer's attention must be rewritten; an incomplete match is an export error.

The exported Python module returns logits followed by the cache / counter state. The low-level ExecuTorch method prepends `3 * n_layers` user-input mutation outputs, then exposes logits at index `3 * n_layers`. The mutation prefix is opaque: its internal order need not match input order (for example, counters can precede K/V). Real HF graphs can also return the original `3 * n_layers` state tensors again after logits; runtimes accept either valid layout and do not treat that repeated state as a second owned cache. Mutation-prefix outputs write into their input buffers. The optional ordered state suffix may contain unplanned functional outputs (notably counters); the C++ runner binds each unplanned suffix output to the corresponding existing state buffer. Python native bindings may clone returned tensors, so callers must not infer program aliasing from returned Torch `data_ptr()` values. The C++ IOManager and registered Python host driver consume the fixed logits index and retain their existing state buffers. Python native bindings may clone returned outputs; those binding-owned copies are separate from the driver's persistent cache buffers and from the exported graph. The Python host API is not claimed to be zero-copy. In a padded chunk, Python may mark all padding rows false; the window-mask marking is mandatory, while unused full-attention padding rows can remain causal as in the C++ driver. Logits are floating-point `[1, N, vocabulary]`. For a padded prompt call, sample from row `real_tokens - 1`, not the last padded row.

## Positions, counters and padding

The runner's `start_pos` is the source of truth. Before every call, set every counter to `start_pos` and supply positions `start_pos .. start_pos + N - 1`. Real tokens must fit the total context. A fixed-length padded call must also fit physically; near the context boundary, run the remaining real tokens through `decode` instead of rejecting a prompt that fits.

A prompt chunk with `r < N` real tokens repeats the last real token in its padding. Full-attention masks are causal by absolute position. In a sliding-window mask, each real query row must have at least one true entry and every padding row must be all-false. The program counts nonempty rows to determine `r`; this is a contract extension, not standard HF runner behavior.

The program may advance counters by the static query length `N`, including padding. The runner must restore the logical count to `start_pos + r` after a padded call, or overwrite all counters with the next logical `start_pos` before execution. A reset returns the logical position and counters to zero. Unwritten cache slots need not contain zeros because masks exclude them.

Sliding caches retain the most recent rows in chronological order. When a call crosses the window, the program shifts the retained rows and writes the real new rows on CPU. `prefill_len` must not exceed the smallest sliding cache length. Padding writes must use the last real row's data whenever they share its destination, so correctness does not depend on duplicate-index write order.

HF's eager `StaticSlidingWindowLayer` may stop advancing its tensor `cumulative_length` once full, while its Python bookkeeping tracks the absolute position. This program's counters continue advancing as absolute positions. Cache state cannot be handed between HF eager and this contract by copying counters alone.

## Runners and boundaries

ExecuTorch's C++ `HfStaticIOManager` and the registered Python `hf_static` runner implement this protocol. The Python runner declares `drives_generation = True` for HF's `ExportedGenerator` contract hook. Importing `executorch.extension.llm.io_contracts` registers the adapters when the transformers checkout exposes that hook; a stock HF runner without it is not claimed to load these programs. Generation takes one unpadded sequence with standard absolute positions; caller-defined mask / position semantics are not supported. The runner's own fixed-chunk padding is supported.

Cache allocation is backend-neutral. An optional I/O allocator can place buffers in QNN shared memory. Graph rewriting and CPU writes live in the LLM extension; QNN partitioning and HTP tuning remain in the Qualcomm backend.

Cache inspection, adoption and forking are not exposed by the current IOManager. A future runtime-owned / forking contract needs its own state ownership API and version rather than silently changing version 1.

## Version history

| Version | Definition |
|---|---|
| 1 | Batch-one fixed-length decode / prefill, float caches, in-place state mutation, sliding-mask padding protocol |

Experimental quantized caches, `sha` attention and full-length sliding-cache mode are excluded from this contract and the upstream series.
