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
export-time SHA-256 digest.

### AOT architecture selection

Configure AOT export with `AOTCompileConfig`, including the target `platform` and
optional `architectures` list. For example, `AOTCompileConfig(platform="iOS",
architectures=["h17p"])` requests that compiler architecture; omitting the list
lets the compiler emit its supported architectures for the target platform.
These are Core AI architecture names, not CPU names such as `arm64`.

### Asset storage

The assets root must be an absolute path. The application chooses it, should
reserve it for this backend, and is trusted not to rename, replace or modify its
contents while the backend uses it; the backend does not defend against other
processes of the same user rebinding paths. Written files and changed directories
are synced with `fsync`; atomic publication also uses `F_FULLFSYNC` before its
rename.

Preparing the assets root sets `NSURLIsExcludedFromBackupKey` on it, which covers
everything beneath it. Ancestors are not modified. This is backup exclusion, not
a control for iCloud Drive synchronization.

Each partition derives a key from its selected export digest/bundle, platform,
Core AI device architecture, default SDK cache namespace, versioned default
options and persistent policy. Function bindings and delivery location are
excluded. No model bytes are hashed at runtime. Each key determines a raw
bookmark in the `bookmarks` subdirectory, a bundle directory in `staging`, and an
empty coordination file in `locks`. There is no PTE-wide bookmark list or
serialized lifecycle record.

Raw bookmarks have an 8 MiB backend allocation limit, not an asserted SDK format
maximum, and are published atomically because losing one orphans an SDK entry.
Per-key disk locks coordinate loading and eviction across threads and processes;
different keys remain independent. Lock ownership is maintained by the OS, and
acquiring a lock does not flush its file or directory. Lock files are not
removed by the backend, including after process exit. Removing the assets root
externally requires all loads, sessions and maintenance to stop.

## Host Tests

`EXECUTORCH_BUILD_COREAI=ON` with `EXECUTORCH_BUILD_TESTS=ON` registers the
standalone `coreai_host_test` executable with CTest. It compiles the runtime
sources without Swift or CoreAI, routing any SDK calls through a fake bridge, so
it also runs on macOS 26. Host tests do not establish real SDK behavior.

From the repository root:

```bash
cmake -S . -B <build-dir> -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0 \
  -DEXECUTORCH_BUILD_COREAI=ON -DEXECUTORCH_BUILD_TESTS=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF
cmake --build <build-dir> --target coreai_host_test
ctest --test-dir <build-dir> -R '^coreai_host_test$' --output-on-failure
```
