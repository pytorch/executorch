# Explicit session checkpoints

A checkpoint contains committed model state at a particular token position. It
can be copied to CPU memory, saved to a file, loaded, and restored into a new
session. Applications choose the prefix, file, compatibility identity, restore
position, and error recovery. Starting the server does not discover or load files;
finishing a request does not save one.

The CPU object is independent of the source session and worker. It contains token
IDs, logical cache positions, retained layer ranges, dtype and geometry, and owned
bytes. An executor using speculative decoding must also export its draft cache and
any committed target features that the draft has not consumed. Generated tokens
that have not been fed back into the target, sampling policy, and request options
are excluded.

## Python flow

The native worker must implement the checkpoint executor interface. Use the same
`MultiplexedWorkerClient` that owns the session. This example assumes `worker` was
created with `spawn_multiplexed_worker` and `prefix` is the application's chosen
prompt:

```python
import asyncio
from types import SimpleNamespace

from executorch.examples.llm_server.python.checkpoint import (
    checkpoint_identity,
    load_checkpoint,
    save_checkpoint,
)

# Include every artifact and setting that can change cached activations.
# Hashing reads these files; applications may instead supply a trusted digest
# from their model installation manifest.
identity = checkpoint_identity(
    [model_path, tokenizer_path],  # Also include external weights and adapters.
    {"backend": backend, "dflash": use_dflash, "rope": rope_configuration},
)

# The existing generation interface requires a positive output budget. Its
# prediction is excluded by explicitly capturing the prompt's committed end.
stream = worker.generate(
    prefix,
    SimpleNamespace(session_id="prepare", max_new_tokens=1, temperature=0),
)
async for _ in stream:
    pass
stats = await stream.wait()
checkpoint = await worker.export_checkpoint_to_cpu(
    "prepare", position=stats.num_prompt_tokens, identity=identity
)
await asyncio.to_thread(save_checkpoint, checkpoint, "prefix.etcp")
await worker.close_session("prepare")
```

In a later process, after starting a worker with the matching model and settings:

```python
checkpoint = await asyncio.to_thread(
    load_checkpoint, "prefix.etcp", expected_identity=identity
)
await worker.restore_checkpoint_from_cpu(
    checkpoint,
    "conversation",  # Must be an unused session key.
    position=len(checkpoint.tokens),
    expected_identity=identity,
)

# Send a complete next prompt that extends this token history, using
# session_id="conversation". The serving runtime prefills only its suffix.
```

The identity is an application-supplied contract. It must describe the worker
actually being used; a caller-supplied model name is not a weights fingerprint.
The native executor additionally checks its format version, cache geometry, dtype,
draft configuration, and the requested position. A shared API does not imply
cross-backend byte compatibility. Currently backend-specific executor formats
reject restoring each other's checkpoints.

## Partial matches

Compare exactly the token IDs the worker would consume, including BOS and any chat
formatting. Comparing strings or message lists is insufficient.

```python
position = checkpoint.common_prefix_length(next_prompt_token_ids)
# Leave at least one prompt token to process: checkpoints do not store logits.
position = min(position, max(0, len(next_prompt_token_ids) - 1))
checkpoint.validate_restore_position(position)
await worker.restore_checkpoint_from_cpu(
    checkpoint, "conversation", position=position, expected_identity=identity
)
```

A token match does not prove that a restore is possible. A sliding-window layer may
have overwritten history needed at an earlier position. Both Python preflight and
the native cache reject such positions. Ordinary sliding-window attention needs
`window - 1` prior rows; a DFlash draft block needs `window` prior rows. Applications
can choose another saved checkpoint or run a fresh prefill. No fallback is selected
by these operations.

The next prompt must extend the restored history to reuse it. If it equals the
restored history, the current serving planner starts a full prefill because the
checkpoint does not contain logits. For an exact prompt retry, explicitly restore
one token before its end, as above, and send the complete prompt. That restore
position must still pass the retained-history checks.

## Ownership, errors, and cost

- Export requires an idle, healthy session with a token-only history. A capture
  cannot silently omit image embeddings or other opaque inputs.
- Restore creates a new session. An existing key is an error. Invalid state or an
  allocation failure does not replace another session or leave a partial session.
- Export/import execute between model forwards. Device synchronization and CPU
  copies are real work. The serving control thread also waits for these operations,
  which delays admission and prompt preparation across sessions.
- The current JSONL transport exchanges bulk bytes through private temporary local
  files, off its reader/writer threads. These transfers include temporary-file I/O
  and host copies. One checkpoint transfer is admitted at a time. This transport
  detail is separate from the durable file format and retention policy.
- Cancelling the Python waiter waits for the transfer to settle before releasing
  its buffers. It does not undo a completed restore; close the destination key if
  the application abandons it.
- Token IDs are signed 32-bit little-endian values stored first in the binary
  payload, followed by layer K/V buffers and draft features. JSON carries only
  buffer descriptors, so longer histories do not enlarge control frames or the
  file header. The file version is `ETCP0002` and the transport manifest version
  is `2`; earlier draft checkpoints must be regenerated.
- `save_checkpoint` writes a replacement file and atomically publishes it after
  flushing. `load_checkpoint` checks version, bounds, identity, and a checksum over
  metadata and payload. Errors propagate. Files contain no pickle or executable
  objects. CPU loading defaults to a 16 GiB payload limit; `max_bytes` changes it
  explicitly.
- File access is local to the Python application and its child worker. These are
  not HTTP file-path endpoints. Keep application checkpoints private as they hold
  state derived from prompts.

The native primitives are `Session::export_checkpoint_to_cpu_async` and
`Runner::restore_checkpoint_from_cpu_async`. They call the optional executor
checkpoint methods on the engine thread. `cache::CheckpointControl` is the shared
CPU-transfer interface for cache implementations. Other stateful architectures
must implement complete continuation snapshots before enabling export.
