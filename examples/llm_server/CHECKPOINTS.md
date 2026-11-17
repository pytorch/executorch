# Explicit session checkpoints

A checkpoint contains committed model state at a particular token position. It
can be copied to CPU memory, saved to a file, loaded, and restored into a new
session. Applications explicitly choose when to save and resume. The advanced API
also exposes the prefix, file, compatibility identity, and restore position.
Starting the server does not discover or load files;
finishing a request does not save one.

## Save and resume a session

Configure a `SessionStore` once at application startup with the worker's model
artifacts and execution settings. It computes the compatibility fingerprint; chat
and agent code only needs the session ID. Include all external weights and
adapters in `model_files`, and backend, RoPE, and draft settings in
`configuration`. Recreate this configuration when starting a new process.

```python
import asyncio

from executorch.examples.llm_server.python.session_store import SessionStore
from executorch.examples.llm_server.python.multiplexed_worker_client import (
    spawn_multiplexed_worker,
)

store = await SessionStore.create(
    "./saved-sessions",
    model_files=[model_path, tokenizer_path, *external_weight_paths],
    configuration=worker_configuration,
)
worker = await spawn_multiplexed_worker(worker_command, session_store=store)
```

After a conversation turn finishes, explicitly save before closing:

```python
await worker.save_session("chat-42")
await worker.close()  # The saved session remains on disk.
```

After reopening the application and starting a matching worker with its store:

```python
session_id = await worker.resume_session("chat-42")
# Continue with worker.generate(full_next_prompt, options), using this session_id.
```

The native runtime chooses the current committed checkpoint boundary. Resume
loads that saved state into a new session with the same ID and returns the ID.
The application sends its complete next prompt. To reuse the saved state, that
prompt must extend exactly the saved token IDs, including chat formatting and
committed generated tokens. Re-rendering messages can change those IDs and cause
a cold prefill. `save_session` and `resume_session` require no token positions,
checkpoint objects, or per-call fingerprints. The advanced API below supports
explicitly restoring an earlier matching prefix.

`SessionRuntime` exposes the same `save_session(id)` and `resume_session(id)`
methods. Within its request-capacity limit, they retain its per-session lease
through native and file I/O, so saves wait for earlier same-session work.
When using the worker client directly, wait
for generation to finish before saving. Closing a session or worker never saves
implicitly and does not delete its saved checkpoint. Saving again atomically
replaces that session's previous file. A client serializes both API levels'
checkpoint operations through file I/O, bounding loaded checkpoint memory and
preserving save order.

Saving requires a nonempty, idle, healthy, token-only history. A turn cancelled,
failed, or cut by a stop string leaves state that cannot be saved; empty/reset
sessions and image or other opaque histories are also rejected. Save raises and
preserves the previous file. A later successful token-only turn can make the
session saveable again. Keep the application's conversation history independently
so it can rebuild state when saving is unavailable.

`SessionStore.create(..., max_bytes=...)` sets the payload limit for both save and
load; the default is 16 GiB. A larger save fails before replacing the previous
file. On POSIX, the store directory must be owned by the current user and must not
be group- or other-writable. A newly created store uses mode 0700; an existing
owned mode-0755 directory is also accepted.

Persist messages, tool results, and UI state in the application independently of
this model state. Resume requires an unused live session ID. A missing or corrupt
file, incompatible model/configuration, or unsupported checkpoint raises an
error; the application explicitly chooses whether to rebuild from its history.
Queued saves and resumes can be cancelled without starting. With
`MultiplexedWorkerClient`, cancellation of a started transfer waits for settlement
and may leave a saved file or restored session. With `SessionRuntime`, the
cancelled caller returns immediately while the transfer settles under its session
lease and continues to occupy request capacity; later admitted same-session
operations wait behind that settlement. Neither
API undoes a completed save or restore.
Model loading and checkpoint transfer still take time; this avoids recomputing
saved tokens, not starting the model itself. There are no HTTP persistence
endpoints or automatic save-on-exit hooks in this API.

To remove a saved conversation, first finish all pending persistence operations.
For a worker client, await its outstanding save/resume tasks; `client.close()`
alone does not wait for Python file writes. For `SessionRuntime`,
`await runtime.aclose_worker()` closes the worker and waits for its background
operations to settle. Then explicitly delete the saved state:

```python
removed = await asyncio.to_thread(store.delete, "chat-42")
```

This returns whether a file existed. It does not close or reset a live session.
There is no automatic retention or deletion on model changes. A killed writer can
leave hidden temporary files named `.<session-hash>.etcp.*`. After all writers have
stopped, an application or operator may remove these temporary files; startup does
not guess whether another process is still writing.
Export/import staging can also leave `executorch-checkpoint-*` directories in the
system temporary directory after a process is killed. Remove those only after
the application and its workers have exited.

## Advanced checkpoint API

The CPU object is independent of the source session and worker. It contains token
IDs, logical cache positions, retained layer ranges, dtype and geometry, and owned
bytes. An executor using speculative decoding must also export its draft cache and
any committed target features that the draft has not consumed. Generated tokens
that have not been fed back into the target, sampling policy, and request options
are excluded.

### Export and restore a chosen position

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
if position == 0:
    await worker.open_session("conversation")  # Explicitly choose a fresh prefill.
else:
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
  state derived from prompts. The checksum detects corruption, not authenticity;
  load only trusted files. Loads reject nonregular files and, on POSIX, symlinks.

The native primitives are `Session::export_checkpoint_to_cpu_async` and
`Runner::restore_checkpoint_from_cpu_async`. They call the optional executor
checkpoint methods on the engine thread. `cache::CheckpointControl` is the shared
CPU-transfer interface for cache implementations. Other stateful architectures
must implement complete continuation snapshots before enabling export.

The native transfer messages and ordering rules are specified in the
[worker protocol](spec/README.md#explicit-checkpoint-transfers).
