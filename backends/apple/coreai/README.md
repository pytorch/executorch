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

The experimental `CoreAIBackend` runtime supports stateless FP32/FP16 tensor
models in both formats. ExecuTorch tensors must be contiguous. Output shapes
must fit the executor's declared resize and capacity constraints. Stateful
functions, image values, other dtypes, and interleaved layouts are not supported.
Calls on one session must be serialized by the caller.

Supported inputs are borrowed directly from ExecuTorch storage through Core AI
raw views. The bridge copies shape metadata, not tensor bytes, and does not
allocate intermediate input `NSData`, `Data`, or `NDArray` buffers. Each call
wraps `const_data_ptr()` and the current shape's `nbytes()`, not the allocation's
maximum capacity. Outputs use the SDK's actual returned shape; the backend
resizes them within ExecuTorch's declared bounds before copying only the logical
result bytes. Unused capacity is neither wrapped as input nor written as output.
The backend waits for inference and all input access to finish before returning
from `execute`, including on errors. This uses ExecuTorch's normal input-lifetime
contract; callers must not concurrently mutate, release, or rebind the input
storage. Output copiers retain the SDK arrays and copy directly into validated
ExecuTorch destinations, without an intermediate `Data` allocation or payload
copy. If any destination overlaps an input or another destination, all outputs
are staged before writing destinations in order. This preserves input/output
aliasing and overlapping-output semantics. Staging is not zero-initialized.
This does not guarantee that Core AI itself avoids device transfers or internal
copies.

After an execution error, do not use the outputs as inference results. Output
shapes may already have changed, but no result bytes are copied unless every
output passes validation and resizing. Malformed SDK outputs return `Internal`.
Output dtype/rank mismatches and incompatible shapes or capacity return
`InvalidArgument`.

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

At load time, the runtime queries `AIModel.deviceArchitectureName` and selects
the exact matching entry in the manifest's `archs` map. No runtime architecture
option is required, and there is no fallback to another architecture. Platform,
deployment floor, and architecture checks happen before asset reads or storage
creation. Missing-architecture errors identify the device architecture, list
available architectures, and request a matching export. Missing selected files
are reported separately from an absent architecture entry.

Both formats can require device specialization. Every load first attempts SDK
bookmark restoration. A missing bookmark or SDK-confirmed cache miss falls back
to source materialization and persistent specialization, then binds the requested
function and ordered I/O on that same acquired model.

### Asset storage

Backend initialization reads the string runtime spec `coreai_assets_dir`. If
absent, it sets the directory to `executorch_coreai` beneath user-domain
`NSCachesDirectory`. The coordinator and storage helpers always receive that
explicit directory. It contains raw bookmarks, staged bundles and lock files.

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
normal source preparation without replacement, since an SDK model may still be
using them. The acquisition coordinator can explicitly replace a damaged bundle
under the same entry lock and rebuild it from available named data. Replacement
preflights the full descriptor-bound tree, then revalidates entries while
removing them without following links. Named-data and SDK failures are not
retried. Temporary bundles are removed on ordinary failures before publication;
a caller can explicitly retry after addressing the failure.

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

Delegate construction and registration do not create storage directories or load
models. Per-model initialization performs SDK acquisition and function loading,
validating source assets only when recovery requires them. There is no synthetic
zero-input inference at initialization. SDK specialization and function resource
loading can still be substantial.

The runtime always uses `AIModelCache.default` with persistent retention and
stores an opaque bookmark for direct restoration. `coreai_assets_dir` does not
choose the SDK artifact directory. The backend does not inspect SDK-private
files, alter their backup flags, or use an App Group cache.

Each selected bundle has a model key derived from its export digest, bundle
name, and source/AOT mode. A readable variant key combines that model key with
the platform and Core AI device architecture, plus a hash of a versioned
specialization identity supplied as plain data by the SDK boundary. Function
bindings and delivery location are excluded, and no model bytes are hashed at
runtime. Each variant key determines a raw bookmark in `bookmarks`, a bundle
directory in `staging`, and one of 256 shared coordination files in `locks`.
There is no PTE-wide bookmark list or serialized lifecycle record.

A healthy bookmark hit does not consult NDS and attempts to clean leftover
staging. On a missing bookmark or confirmed SDK miss, the coordinator prepares
the source, specializes it and atomically writes the returned bookmark bytes.
After durable publication, it attempts to remove the entire keyed staging
directory while holding the key's stripe lock. Persistent SDK caching supports
inference and bookmark restoration without that extracted copy.
Keep the packaged named data available: a later SDK cache miss requires source
materialization again. There is no secondary source-URL cache lookup.

SDK errors, unreadable bookmarks and publication failures fail loading without
automatic cache deletion. If publication fails after specialization, the SDK
entry and source remain. Cleanup after a bookmark hit or durable publication is
best-effort: failures are logged without discarding the usable model or bookmark.
A later load can finish cleanup or rebuild a partially removed source after an
SDK miss. No cleanup failure triggers SDK eviction or retries within that load.
The backend does not track historical or unrecorded SDK entries.

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

### Runtime options

Pass options through the public `Module` API before loading. In an error-returning
loader using the `executorch::runtime` namespace:

```cpp
BackendOptions<1> options;
ET_CHECK_OK_OR_RETURN_ERROR(options.set_option("coreai_assets_dir", assets_dir));
LoadBackendOptionsMap map;
ET_CHECK_OK_OR_RETURN_ERROR(map.set_options("CoreAIBackend", options.view()));
ET_CHECK_OK_OR_RETURN_ERROR(module.load(map));
```

No options are required when using the default assets root. `coreai_assets_dir`
selects an application-chosen cache directory reserved for this backend. The
supplied path is validated during preflight, but warm bookmark hits need no
source.

Use separate option maps to choose different storage roots for different models.
The path is a root directory, not the bundle itself. ExecuTorch option strings
have a 255-byte limit.

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
bookmark restoration through the bridge after removing disposable portable and
AOT source copies. The AOT case is skipped when the SDK cannot identify the
device architecture. A killed probe can leave an SDK entry if it dies before
saving its bookmark; cleanup never purges unrelated default-cache entries.
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

### Lifecycle logging

Build with `EXECUTORCH_ENABLE_LOGGING=ON` and `EXECUTORCH_LOG_LEVEL=Info` to
observe cache lookup, bookmark restoration hits, missing-bookmark versus
SDK-confirmed misses, specialization start/success, and durable bookmark saves. Each message carries the cache key so events from
multiple partitions can be correlated. These messages do not include bookmark
contents, model data, or per-inference logs.

A cache hit means that the SDK successfully restored a model from its bookmark.
Specialization start means a request to the SDK's persistent default cache; the
SDK does not report whether that request compiled afresh or reused an internal
artifact. The durable-save message confirms local bookmark publication,
including its final directory flush, not independent verification of SDK-private
artifact storage. Failed operations do not emit their success messages.

The delegate does not require `EXECUTORCH_BUILD_EXTENSION_DATA_LOADER`.
Enable it for consumers such as `Module` that load program data.
Link the CMake target `coreaidelegate`, not just its archive filename. Its
transitive dependencies supply Swift and Foundation/CoreAI linkage, and its
link interface retains static backend registration. With
`EXECUTORCH_BUILD_SHARED=ON`, the delegate is a shared library linked to the
single consolidated `executorch_shared` runtime. Otherwise it is static.
Consumers of the shared delegate must use that same shared runtime.
Installed CMake targets resolve framework and Swift runtime dependencies using
the consumer's selected SDK and toolchain, not the producer's Xcode paths.
Select the Apple SDK/toolchain when configuring the consumer project.
SwiftPM and XCFramework distribution are not integrated yet.

The `coreai_runtime_smoke` target checks final linkage of a C++ consumer,
including Swift dependencies and backend registration retention. With
`EXECUTORCH_BUILD_TESTS=ON` it is built; otherwise it is excluded from the
default build. SDK-linked tests are registered with CTest only for native macOS
builds when the host meets the target's deployment minimum (at least 27.0).
Cross-builds and older hosts can build these tests without registering them.
Running the smoke test checks registration and availability; it does not perform
model inference. Do not execute SDK27 binaries on an older host.

## Host Tests

On macOS, `EXECUTORCH_BUILD_COREAI=ON` with `EXECUTORCH_BUILD_TESTS=ON`
registers the standalone `coreai_host_test` executable with CTest. It tests
manifest parsing, AOT asset selection, cache identity, bookmark I/O, acquisition
coordination, lock contention, filesystem storage, and delegate initialization
and execution against a fake bridge. It uses Foundation without Swift or CoreAI
SDK calls. Acquisition contention tests run two threads inside a bounded child
process; bookmark lock tests separately cover contention between processes. The
host test target is omitted for Apple cross-builds. On supported macOS hosts,
CTest also runs the Swift bridge tests, source-lifetime probes, and
`coreai_runtime_smoke`, which link Swift and the Core AI framework.

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

For a compile-only production smoke build, use a fresh build directory without
`EXECUTORCH_BUILD_TESTS`, set `CMAKE_OSX_ARCHITECTURES=arm64`, and build
`coreai_runtime_smoke`. For iOS also set `CMAKE_SYSTEM_NAME=iOS` and
`CMAKE_OSX_SYSROOT=iphoneos`. Do not run the iOS binary on the host or install
a simulator as part of this build.
