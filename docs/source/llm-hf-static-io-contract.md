# Hugging Face static-cache I/O contract

**Version:** 1 (`get_io_contract = "hf_static"`)

## TL;DR

Each step keeps Hugging Face's static-cache interface: token IDs, attention masks, position IDs and full K/V caches in; logits and the same caches updated in place out. The runner owns the buffers; the program writes into them.

The one adaptation is fixed shapes. Instead of HF's single variable-length `forward`, ExecuTorch exports fixed-length `prefill` / `decode` methods and requires the padding / counter protocol below, because Hexagon Tensor Processor (HTP) execution needs static shapes. This adaptation is labeled `hf_static` v1. `decode` is required and `prefill` is optional; the program is not a drop-in replacement for an arbitrary HF runner.

## Goals / Non-Goals

This contract defines the inputs, outputs, metadata and runner responsibilities for batch-one float-cache generation. It does not define runtime-owned dynamic caches, cache-state import / fork, quantized caches, or arbitrary query lengths. CPU tests provide sliding-window correctness evidence only; they do not establish sliding-window device support, longer-context device support or device latency.

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

Cache and counter state must be unplanned CPU inputs in the contiguous layout the runner allocates. The Python loader rejects memory-planned or device-resident state: runtime-owned copies would otherwise lose updates to the caller's buffers. Its public `TensorInfo` metadata does not expose dimension order, so externally produced programs must have their contiguous cache layout checked at export.

The exported Python module returns logits followed by the cache / counter state. The low-level ExecuTorch method prepends `3 * n_layers` user-input mutation outputs, then exposes logits at index `3 * n_layers`. The mutation prefix is opaque: its internal order need not match input order (for example, counters can precede K/V). Mutation-prefix outputs write into their input buffers.

Real HF graphs can also return the original `3 * n_layers` state tensors again after logits. Runtimes accept either valid layout and do not treat that repeated state as a second owned cache. The optional ordered state suffix may contain unplanned functional outputs (notably counters); the C++ runner binds each unplanned suffix output to the corresponding existing state buffer.

The C++ IOManager and registered Python host driver consume the fixed logits index and retain their existing state buffers. Python native bindings may clone returned outputs, so callers must not infer program aliasing from returned Torch `data_ptr()` values. Those binding-owned copies are separate from the driver's persistent cache buffers and from the exported graph. The Python host API is not claimed to be zero-copy.

In a padded chunk, Python may mark all padding rows false; the window-mask marking is mandatory, while unused full-attention padding rows can remain causal as in the C++ driver. Logits are floating-point `[1, N, vocabulary]`. For a padded prompt call, sample from row `real_tokens - 1`, not the last padded row.

## Positions, counters and padding

The runner's `start_pos` is the source of truth. Before every call, set every counter to `start_pos` and supply positions `start_pos .. start_pos + N - 1`. Real tokens must fit the total context. A fixed-length padded call must also fit physically; near the context boundary, run the remaining real tokens through `decode` instead of rejecting a prompt that fits.

A prompt chunk with `r < N` real tokens repeats the last real token in its padding. Full-attention masks are causal by absolute position. In a sliding-window mask, each real query row must have at least one true entry and every padding row must be all-false. The program counts nonempty rows to determine `r`; this is part of the fixed-shape adaptation, not standard HF runner behavior.

The program may advance counters by the static query length `N`, including padding. The runner must restore the logical count to `start_pos + r` after a padded call, or overwrite all counters with the next logical `start_pos` before execution. A reset returns the logical position and counters to zero. Unwritten cache slots need not contain zeros because masks exclude them.

Sliding caches retain the most recent rows in chronological order. When a call crosses the window, the program shifts the retained rows and writes the real token rows on CPU. `prefill_len` must not exceed the smallest sliding cache length. Padding writes must use the last real row's data whenever they share its destination, so correctness does not depend on duplicate-index write order.

HF's eager `StaticSlidingWindowLayer` may stop advancing its tensor `cumulative_length` once full, while its Python bookkeeping tracks the absolute position. This program's counters continue advancing as absolute positions. Cache state cannot be handed between HF eager and this contract by copying counters alone.

## Runners and boundaries

ExecuTorch's C++ `HfStaticIOManager` and the registered Python `hf_static` runner implement this protocol. The C++ IOManager does not require transformers. The Python runner declares `drives_generation = True` for HF's `ExportedGenerator` contract hook.

HF's own Python runner can load these programs only when transformers provides `transformers.exporters.register_io_contract`. For this work, that hook exists only in a local transformers branch, not stock transformers. ExecuTorch guards the hook import with `try` / `except ImportError`. Importing `executorch.extension.llm.io_contracts` registers the adapters only when the hook is available; otherwise registration is skipped, and stock HF loading is not enabled.

Generation takes one unpadded sequence with standard absolute positions; caller-defined mask / position semantics are not supported. The runner's own fixed-chunk padding is supported.

Cache allocation is backend-neutral. An optional I/O allocator can place buffers in QNN shared memory. Graph rewriting and CPU writes live in the LLM extension; QNN partitioning and HTP tuning remain in the Qualcomm backend. This allocation interface does not establish device support or zero-copy behavior for every backend / runner combination.

KV-cache policies, including eviction, long-context handling, reset and fork, belong in the runner's IOManager, not in QNN partitioning or the model exporter. Version 1 implements buffer ownership, the fixed-shape protocol and reset; it does not provide arbitrary context extension or cache-state transfer from HF eager execution. The program's sliding-cache shift / write implements the exported model's window semantics, not a general runner eviction policy.

Cache inspection and adopt / fork / evict hooks are follow-up work and are not exposed by the current IOManager. They need explicit state-ownership semantics and an appropriate contract version rather than silently changing version 1.

## Version history

| Version | Definition |
|---|---|
| 1 | Batch-one fixed-length decode / prefill, float caches, in-place state mutation, sliding-mask padding protocol |

Experimental quantized caches, `sha` attention and full-length sliding-cache mode are excluded from this contract and the upstream series.
