# Core AI backend (ExecuTorch)

> ⚠️ **Under construction — not for use.**
> This backend is a work in progress. It is **not** ready for production or
> general use, and provides **no backward- or forward-compatibility guarantees**
> (no BC/FC). APIs, compile specs, serialized manifests, and file layouts may
> change or break at any time without notice. Do not depend on it.

## Running the tests

Install ExecuTorch in editable mode, then the pinned Core AI SDK packages:

```bash
python install_executorch.py --editable
pip install -r backends/apple/coreai/requirements.txt
```

Run the backend's Python tests from the repository root:

```bash
python -m pytest backends/apple/coreai
```

`CoreAIAOTCompileTest` needs `xcrun coreai-build` from the Metal Toolchain
(`xcodebuild -downloadComponent MetalToolchain`) and skips without it.

## Runtime

The delegate manifest describes `.aimodel` source bundles, which require
`inline` packaging, and AOT-compiled `.aimodelc` bundles, which require
`aot_compiled_inline`. Both formats require a `function` name, and ordered
`input_names` and `output_names` matching the converter's bindings. Dictionary
iteration order is not a binding contract. The `files` object maps relative
filenames to byte sizes; `bundle_digests` maps each bundle basename to its
export-time SHA-256 digest. Asset paths, including bundle and directory names,
must remain distinct under case-insensitive, canonically normalized Unicode
comparison. Accepted names are preserved, not rewritten.

Parsing returns an immutable `Manifest` covering all exported bundles. Pure
platform and architecture selection produces an immutable `SelectedBundle`
with one bundle's path, digest, and files. Neither type can be constructed
without validation; filesystem materialization and SDK host probing remain
outside the manifest parser.

### Calling from Swift

When using the Core AI backend, run ExecuTorch model loading, inference and
cache operations on a background `DispatchQueue` or a dedicated worker thread.
Do not call these synchronous APIs directly inside Swift `Task` or
`Task.detached`, because doing so can hang your application.

From async Swift code, submit the entire ExecuTorch call with
`DispatchQueue.async` and await its result using a continuation. Do not block
the Swift task with `DispatchQueue.sync` or a semaphore. Keep input tensors alive
and unchanged through all output copies, even if the waiting task is cancelled.

### AOT architecture selection

Configure AOT export with `AOTCompileConfig`, including the target `platform` and
optional `architectures` list. For example, `AOTCompileConfig(platform="iOS",
architectures=["h17p"])` requests that compiler architecture; omitting the list
lets the compiler emit its supported architectures for the target platform.
These are Core AI architecture names, not CPU names such as `arm64`.

Both formats can require device specialization. Every load first attempts SDK
bookmark restoration. A missing bookmark or SDK-confirmed cache miss falls back
to source materialization and persistent specialization, then binds the requested
function and ordered I/O on that same acquired model.

### Asset storage

The assets root must be an absolute path. The application chooses it, should
reserve it for this backend, and is trusted not to rename, replace or modify its
contents while the backend uses it; the backend does not defend against other
processes of the same user rebinding paths. Written files and changed directories
are synced with `fsync`. Bookmark publication uses `F_FULLFSYNC` on the temporary
file before rename and on the directory afterward. Source-bundle publication
fully flushes the staging directory after syncing its files and subdirectories,
then fully flushes the destination directory after rename. A failure before
rename preserves the old contents; a failure of the final flush reports an error
even though the new complete contents are already visible. Use a local APFS
volume; unsupported full-device flushes are reported as errors, not downgraded
to `fsync`.

Inline bundles are read through `NamedDataMap` and materialized as unchanged
files under the assets root. Named data can be supplied externally; inline
packaging does not require all bytes to reside in one physical PTE. The exporter
computes each bundle digest from all delivered relative filenames and file bytes,
not just the SDK's bytecode-only `main.hash`. Each AOT architecture has an
independent digest; only the selected architecture is materialized. NDS key names
and raw asset bytes are unchanged.

When source recovery is needed, existing bundles are checked for the expected
file set, sizes, entry types and directory structure without reading asset
contents or requesting NDS payloads. Unicode paths are compared in canonical
decomposed form; NDS keys and bytes remain unchanged. Reuse fully flushes the
source root, including after a previous publication's final flush failed.
Missing bundles are written to a staging directory and published with an
exclusive rename; each selected NDS payload is fetched once and checked for size
before writing. If another loader publishes first, its bundle is validated and
used. Incomplete, size-mismatched or otherwise malformed existing bundles fail
normal loading without replacement, since an SDK model may still be using them.
A cache-maintenance caller holding the same entry lock can explicitly replace a
damaged bundle. Replacement preflights the full descriptor-bound tree, then
revalidates entries while removing them without following links. Only
interrupted system calls are retried; other failures are returned to the caller
without automatic retries or repair. Temporary bundles are removed on ordinary
failures before publication; a caller can explicitly retry after addressing the
failure.

These are completeness checks, not full content-integrity verification. Same-size
content changes are not detected, and a complete stored bundle does not cause its
NDS source to be reread. The export-time digest is a trusted artifact identifier;
the runtime does not rehash source or stored bytes. Core AI may reject invalid
contents, but SDK load failures do not automatically evict the extracted copy.

Preparing the assets root sets `NSURLIsExcludedFromBackupKey` on it, which covers
everything beneath it. Ancestors are not modified. This is backup exclusion, not
a control for iCloud Drive synchronization.

Newly created roots request
`NSFileProtectionCompleteUntilFirstUserAuthentication`. On iOS, this allows cache
access while the device is locked after its first unlock following a reboot. New
subdirectories and files inherit their parent's protection. Existing cache
contents are not migrated. Directory preparation skips creation when the root
already exists; child directories are opened first and created only when missing.
Directory creation uses default permissions filtered by the process umask, without
an explicit owner-only restriction. Existing permissions are not changed.
Backup exclusion and required durability flushes still apply on reused paths.

Cache-root preparation owns location and backup/protection policy; descriptor
storage owns no-follow operations, publication, and synchronization. Creation
and enumeration use explicit policies, with a missing child distinguishable
from an error. Fault injection is a separate, thread-local test interface and
compiles away in production.

Each selected bundle has a model key derived from its export digest, bundle
name, and source/AOT mode. A readable variant key combines that model key with
the platform and Core AI device architecture, plus a hash of a versioned
specialization identity supplied as plain data by the SDK boundary. Function
bindings and delivery location are excluded, and no model bytes are hashed at
runtime. Each variant key determines a raw bookmark in `bookmarks`, a bundle
directory in `staging`, and one of 256 shared coordination files in `locks`.
There is no PTE-wide bookmark list or serialized lifecycle record.

A healthy bookmark hit does not inspect staging or consult NDS; the acquired SDK
model is retained through function binding. On a missing bookmark or confirmed
SDK miss, the runtime materializes the source, specializes it and atomically
writes the returned bookmark bytes. After durable publication, it removes the
entire keyed staging directory while holding the per-key lock. Persistent SDK
caching supports inference and bookmark restoration without that extracted copy.
Keep the packaged named data available: a later SDK cache miss requires source
materialization again. There is no secondary source-URL cache lookup.

SDK errors, unreadable bookmarks and publication failures fail loading without
automatic cache deletion. If publication fails after specialization, the SDK
entry and source remain. If staged-source cleanup fails, loading reports the
error but retains the durable bookmark and any remaining staged files. Neither
failure triggers automatic retries, SDK eviction or source repair. A subsequent
bookmark hit can load without inspecting or retrying staging cleanup. A partial
cleanup followed by an SDK miss fails on the damaged staged entry until it is
explicitly cleared. The backend does not track historical or unrecorded SDK
entries.

Raw bookmarks have an 8 MiB backend allocation limit, not an asserted SDK format
maximum, and are published atomically because losing one orphans an SDK entry.
A fixed pool of at most 256 disk locks coordinates loading and eviction across
threads and processes. A SHA-256 hash of the full validated cache key selects a
stripe modulo the configured pool size. Files are created on demand, so new
model versions do not grow the pool beyond 256. Different keys on the same
stripe serialize; different stripes remain independent. Bookmarks and staging
remain separate for every key. Runtime operations hold only one stripe at a
time.

Lock ownership is maintained by the OS, and acquiring a lock does not flush its
file or directory. Stripe files are never removed, including during cache
clearing or after process exit, so waiters and future openers keep using the
same inode. Removing the assets root externally requires all loads, sessions
and maintenance to stop.

## SDK Bridge

The private Swift module `CoreAIBridge` (`runtime/ETCoreAIModel.swift`) wraps
the `CoreAI` framework behind the Objective-C protocols in
`runtime/ETCoreAIBridge.h`. It restores models from bookmarks, specializes source
bundles in the default SDK cache with storage-pressure-only purging, and binds
named functions with ordered inputs and outputs. Each loader owns an immutable
snapshot of its actual specialization options and exposes a stable, versioned
identity derived from that snapshot. Asynchronous SDK calls run in detached
tasks and report through completion blocks; function binding completes
synchronously.

Inputs are borrowed from caller storage through Core AI raw views; only shape
metadata is copied. Outputs retain their SDK storage and expose a copier after
all fallible layout validation finishes. Dense row-major outputs copy with one
`memcpy`, while padded or transposed outputs use strided copies. Only FP16 and
FP32 NDArrays are supported. Callers keep inputs live through output copying and
stage copies before writing destinations that could alias inputs or other output
destinations.

Bridge-owned Objective-C types live in `runtime/ETCoreAITensor.mm`, separately
from the production SDK entry points in `runtime/coreai_bridge.mm`. Both build in
the private `coreai_bridge_obj` object library, while SDK-free tests can link the
types and substitute fake entry points.

`coreai_swift_bridge_test` runs the bridge's XCTests under `xctest` with
`EXECUTORCH_BUILD_TESTS=ON`. It links CoreAI and is registered with CTest only
when the macOS host meets the configured Core AI deployment target.
The test build exports static- and dynamic-shape models for real session tests.
The SDK source-lifetime regression checks live inference and fresh-process
bookmark restoration after removing disposable portable and AOT source copies.
CMake's `PYTHON_EXECUTABLE` must use the Python environment with the pinned Core
AI requirements above. Backend Python changes regenerate the dedicated fixture
directory, replacing previous contents. CTest supplies its location to XCTest;
direct XCTest runs without it skip the session tests. Building only
`coreai_host_test` does not export fixtures or build the Swift test bundle.

## Building

Core AI builds require an Apple SDK 27 or newer, Swift from Xcode 27, and CMake
3.31 or newer with the Ninja or Xcode generator. Execution requires macOS 27 or
iOS 27. `EXECUTORCH_BUILD_COREAI` defaults to `OFF`. Enabling it builds the
private Swift implementation, module `CoreAIBridge`. Only Core AI targets get an
OS 27 minimum; unrelated runtime targets keep the configured deployment target.
Ninja builds use one architecture per build directory. Current runtime support
covers arm64 macOS and iOS device builds. x86_64 builds are blocked by Swift
`Float16` availability, and the tested iOS simulator SDKs do not contain Core AI.

## Host Tests

On macOS, `EXECUTORCH_BUILD_COREAI=ON` with `EXECUTORCH_BUILD_TESTS=ON`
registers the standalone `coreai_host_test` executable with CTest. It tests
manifest parsing, AOT asset selection, cache identity, bookmark I/O, acquisition
coordination, lock contention, and filesystem storage using Foundation, without
Swift or CoreAI SDK calls. Acquisition contention tests run two threads inside a
bounded child process; bookmark lock tests separately cover contention between
processes. The host test target is omitted for Apple cross-builds.

CI builds and runs these tests on macOS 27. To use the same configuration from
the repository root:

```bash
cmake -S . -B <build-dir> -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=27.0 \
  -DEXECUTORCH_BUILD_COREAI=ON -DEXECUTORCH_BUILD_TESTS=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON
cmake --build <build-dir> --target backends/apple/coreai/all
ctest --test-dir <build-dir>/backends/apple/coreai --output-on-failure --no-tests=error
```
