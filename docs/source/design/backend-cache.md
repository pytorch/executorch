# Backend Init Caching

## Problem

Backend initialization in ExecuTorch can be expensive. Backends like XNNPACK
repack weights during `init()`, which involves parsing the processed data
segment, allocating aligned buffers, and transforming tensor layouts. For large
models this takes hundreds of milliseconds on every load — even though the
output is identical across runs on the same device.

Today there is no way to persist these artifacts. Every `Program::load_method()`
call pays the full initialization cost.

## Design

### Goals

- Let backends persist init artifacts (packed weights, compiled graphs) and
  restore them on subsequent loads, skipping the processed data segment entirely.
- Keep the runtime layer (`runtime/`) free of string manipulation, heap
  allocation, and filesystem I/O.
- Provide a concrete filesystem-backed implementation in `extension/` that
  handles aligned allocation, atomic writes, and directory structure.
- Avoid TOCTOU races in the cache API.

### Non-goals

- Cache invalidation policy (LRU, TTL, size limits). The caller owns eviction.
- Thread-safe writes to the same key. Concurrent writes use last-writer-wins.
- Caching execution outputs or intermediate tensors — this is init-only.

## Architecture

```
┌──────────┐   load_method(cache)   ┌─────────┐   init()   ┌──────────────────┐
│  Module  │ ─────────────────────► │ Program │ ─────────► │      Method      │
└──────────┘                        └─────────┘            │                  │
                                                           │  for each delegate:
                                                           │  ┌──────────────┐│
                                                           │  │ Delegate     ││
                                                           │  │ BackendCache ││
                                                           │  │ (bid, idx)   ││
                                                           │  └──────┬───────┘│
                                                           └─────────┼────────┘
                                                                     │
                                                              load/save/remove
                                                                     │
                                                                     ▼
                                                           ┌──────────────────┐
                                                           │  BackendCache    │
                                                           │  (abstract)      │
                                                           └────────┬─────────┘
                                                                    │
                                                                    ▼
                                                           ┌──────────────────┐
                                                           │ FileBackendCache │
                                                           │ (extension/)     │
                                                           └──────────────────┘
```

### Call flow

The runtime attempts `init_from_cache()` before loading the processed data
segment. This means a cache hit avoids both the data load and the init
computation:

```
BackendDelegate::Init(delegate, program, context, out)
  │
  ├─ if cache available:
  │    result = backend->init_from_cache(context, compile_specs)
  │    ├─ Ok         → return handle, skip data load entirely
  │    ├─ NotSupported → fall through (backend doesn't support caching)
  │    └─ other error → return error
  │
  └─ normal path:
       data = GetProcessedData(delegate, program)
       handle = backend->init(context, &data, compile_specs)
```

### Key scoping

A single `BackendCache` instance is shared across all backends and delegates in
a program. The runtime must prevent key collisions. Rather than concatenating
`"{backend_id}/{delegate_index}/{key}"` into a string buffer (which introduces
silent truncation risk and string manipulation in `runtime/`), the scoping
components are passed separately through the entire stack:

```
BackendCache::load(backend_id, delegate_index, key, alignment)
BackendCache::save(backend_id, delegate_index, key, data, size)
BackendCache::remove(backend_id, delegate_index, key)
```

`DelegateBackendCache` captures `backend_id` and `delegate_index` at construction
time (during `Method::init`) and presents a simple key-only API to backends:

```cpp
DelegateBackendCache delegate_cache(&cache, "XnnpackBackend", 0);
delegate_cache.save("v1/packed_weights", data, size);
// delegates to: cache.save("XnnpackBackend", 0, "v1/packed_weights", data, size)
```

`DelegateBackendCache` is **not** a subclass of `BackendCache`. It's a separate,
simpler type that forwards to the underlying cache with the scoping components
attached. Backends interact with `DelegateBackendCache*` (via
`BackendInitContext::get_cache()`), while cache implementations subclass
`BackendCache`.

## API

### `BackendCache` (abstract, `runtime/backend/backend_cache.h`)

```cpp
class BackendCache {
 public:
  virtual ~BackendCache() = default;

  virtual Result<FreeableBuffer> load(
      const char* backend_id,
      size_t delegate_index,
      const char* key,
      size_t alignment = alignof(std::max_align_t)) const = 0;

  virtual Error save(
      const char* backend_id,
      size_t delegate_index,
      const char* key,
      const void* data,
      size_t size) = 0;

  virtual Error remove(
      const char* backend_id,
      size_t delegate_index,
      const char* key) = 0;
};
```

There is no `contains()` method. Checking existence then loading is a TOCTOU
race — callers should `load()` and handle `Error::NotFound`.

The `alignment` parameter on `load()` lets backends request SIMD-aligned
buffers (e.g., 64-byte for AVX-512 weight packing).

### `DelegateBackendCache` (`runtime/backend/backend_cache.h`)

Wraps a `BackendCache`, capturing `backend_id` and `delegate_index` so that
backends see a simple key-only API:

```cpp
class DelegateBackendCache final {
 public:
  DelegateBackendCache(BackendCache* cache, const char* backend_id, size_t delegate_index);

  Result<FreeableBuffer> load(const char* key, size_t alignment = alignof(std::max_align_t)) const;
  Error save(const char* key, const void* data, size_t size);
  Error remove(const char* key);
};
```

### `BackendInterface::init_from_cache` (`runtime/backend/interface.h`)

```cpp
virtual Result<DelegateHandle*> init_from_cache(
    BackendInitContext& context,
    ArrayRef<CompileSpec> compile_specs) const {
  return Error::NotSupported;  // default: no caching support
}
```

Backends that support caching override this. The default returns
`Error::NotSupported`, which tells the runtime to fall through to the normal
`init()` path. Any other error is treated as fatal.

### `FileBackendCache` (`extension/backend_cache/`)

Concrete implementation that maps cache entries to filesystem paths:

```
{cache_dir}/{backend_id}/{delegate_index}/{key}
```

Key properties:

- **Aligned allocation**: `load()` uses `::operator new(size, align_val_t, nothrow)`
  with matching `::operator delete(ptr, align_val_t)`. This ensures SIMD-aligned
  weight buffers.
- **Atomic writes**: `save()` writes to `{path}.tmp` then renames. Prevents
  corruption if the process is killed mid-write.
- **Concurrency**: Safe for concurrent reads. Concurrent writes to the same key
  use last-writer-wins via atomic rename.
- **`remove()`**: Deletes a single cache entry. Returns `Error::NotFound` if the
  file doesn't exist.

## Threading through the stack

The `BackendCache*` pointer flows from the user down through:

```
Module::load_method(backend_cache=)
  → Program::load_method(backend_cache=)
    → Method::load(backend_cache=)
      → Method::init(backend_cache=)
        → for each delegate:
            DelegateBackendCache(backend_cache, delegate.id, i)
              → BackendInitContext(backend_cache=&delegate_cache)
                → backend->init_from_cache(context, ...)
```

The `DelegateBackendCache` is stack-allocated in `Method::init`'s delegate loop,
so its lifetime covers the `BackendDelegate::Init` call. The `backend_id`
string is owned by the flatbuffer and lives as long as the `Program`.

## Usage

### Loading with a cache

```cpp
FileBackendCache cache("/data/local/tmp/et_cache");
Module module("model.pte");
module.load_method("forward", /*planned_memory=*/nullptr,
                   /*event_tracer=*/nullptr, /*backend_options=*/nullptr,
                   &cache);
```

First load: backends call `init()` normally and can `save()` artifacts to the
cache. Second load: backends' `init_from_cache()` finds the cached data and
returns immediately, skipping the processed data segment load.

### Implementing caching in a backend

```cpp
Result<DelegateHandle*> MyBackend::init_from_cache(
    BackendInitContext& context,
    ArrayRef<CompileSpec> compile_specs) const {
  auto* cache = context.get_cache();  // returns DelegateBackendCache*
  if (!cache) {
    return Error::NotSupported;
  }
  auto result = cache->load("v1/packed_weights", /*alignment=*/64);
  if (!result.ok()) {
    return Error::NotSupported;  // cache miss, fall through to init()
  }
  // Restore from cached data...
  return handle;
}

Result<DelegateHandle*> MyBackend::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {
  // Normal init: process weights...
  auto* cache = context.get_cache();  // returns DelegateBackendCache*
  if (cache) {
    cache->save("v1/packed_weights", packed_data, packed_size);
  }
  return handle;
}
```

## File layout

```
runtime/backend/
  backend_cache.h          # BackendCache + DelegateBackendCache
  backend_init_context.h   # stores DelegateBackendCache*, get_cache()
  interface.h              # init_from_cache() virtual method
  test/
    backend_cache_test.cpp # Unit tests for BackendCache + DelegateBackendCache

runtime/executor/test/
  backend_integration_test.cpp  # Integration tests for cache-first init flow

extension/backend_cache/
  file_backend_cache.h     # FileBackendCache declaration
  file_backend_cache.cpp   # FileBackendCache implementation
  CMakeLists.txt           # extension_backend_cache library
  targets.bzl              # Buck targets
  test/
    file_backend_cache_test.cpp
    CMakeLists.txt
    targets.bzl
```

## Testing

- **`backend_cache_test.cpp`** (unit): Tests `BackendCache` and
  `DelegateBackendCache` with an in-memory implementation. Verifies key scoping
  (delegates and backends are isolated), save/load/remove semantics, and
  overwrite behavior.

- **`file_backend_cache_test.cpp`** (unit): Tests `FileBackendCache` against a
  real temp directory. Covers save/load roundtrip, subdirectory creation,
  overwrite, 1MB data, remove, aligned load (verifies pointer alignment), and
  backend/delegate isolation.

- **`backend_integration_test.cpp`** (integration): Tests the cache-first init
  protocol end-to-end using `StubBackend` with injectable callbacks and real
  `.pte` fixtures loaded via `FileDataLoader`. Parameterized on segments vs
  no-segments. Five cache-specific test cases:
  - **CacheHitSkipsInit**: `init_from_cache` succeeds — `init()` is never
    called and the processed data segment is never loaded (verified via
    `DataLoaderSpy`).
  - **CacheMissFallsThrough**: `init_from_cache` returns `NotSupported` —
    `init()` is called normally with the processed data.
  - **CacheErrorFailsLoad**: `init_from_cache` returns a non-`NotSupported`
    error — `load_method` propagates the error.
  - **NoCacheSkipsInitFromCache**: No `BackendCache` provided —
    `init_from_cache` is never called, `init()` proceeds normally.
  - **CacheAvailableInContext**: `context.get_cache()` returns a non-null
    `DelegateBackendCache*` during `init()` when a cache is provided.

## Design decisions

| Decision | Rationale |
|----------|-----------|
| Separate scoping components instead of string concatenation | Eliminates `snprintf` and fixed-size buffers in `runtime/`. No silent truncation risk. Implementations decide how to use the components. |
| No `contains()` | TOCTOU race: file could be deleted between `contains()` and `load()`. Callers should `load()` and check for `NotFound`. |
| `DelegateBackendCache` is not a subclass of `BackendCache` | Backends should not see the 3-component API. The simple key-only API prevents misuse. Different types enforce this at compile time. |
| `init_from_cache` defaults to `NotSupported` | Existing backends are unaffected. The runtime treats `NotSupported` as "no caching", not as an error. |
| Aligned allocation in `load()` | Backend-packed weights often require SIMD alignment (16, 32, 64 bytes). `std::malloc` only guarantees `max_align_t` (~16 bytes). |
| `nothrow` operator new | ExecuTorch can be built with `-fno-exceptions`. The throwing variant would call `std::terminate` on OOM instead of returning an error. |
| Atomic rename in `save()` | Prevents partially-written cache files if the process crashes. |
| Cache at `Method::init` level, not `Program` level | Different methods may have different delegates. Scoping is per-delegate. |
