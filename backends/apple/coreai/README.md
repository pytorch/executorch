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

## Host Tests

On macOS, `EXECUTORCH_BUILD_COREAI=ON` with `EXECUTORCH_BUILD_TESTS=ON`
registers the standalone `coreai_host_test` executable with CTest. It tests
manifest parsing and AOT asset selection using Foundation, without Swift or
CoreAI SDK calls. The host test target is omitted for Apple cross-builds.

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
