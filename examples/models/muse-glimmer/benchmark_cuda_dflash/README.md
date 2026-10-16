<!-- Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. -->

# CUDA DFlash: measured batching branch and four-task HTTP reproduction

Use branch `mg-cuda-dflash-batching-api-201`. Its production source tree is
exactly commit `d530091c08967a9d868b38a120c94145bd108614`; this branch adds only
this portable benchmark directory. It includes CUDA packed batching, INT5
kernels, the Muse Glimmer DFlash adapter over the existing CUDA executor,
committed-only hidden transfers and the emitted-pending-token fast path.
It does not include the experimental `draft_forward_1` through `_4` export
specializations or bounded-context/zero-tail recovery from the newer work.

The four frozen requests and standard-library HTTP client are unchanged from
benchmark package commit `24f47b32811dddf1df8b01d4e8da961393fb55f0`. Only this
README changes. Never execute the fixture source text or generated answers.
Model weights, tokenizers, PTE/PTD, caches, builds and result logs are external;
they are not included in this branch.

## Measured result and limitations

The original SM80 real-HTTP measurement used one GPU, temperature zero,
streaming, `n_draft=3`, three measured repeats and separate warmups. Each group
contained the same four tasks and 2,048 API completion tokens in total:

| Mode | Client concurrency | Median aggregate tokens/s |
| --- | ---: | ---: |
| DFsolo | 1, four requests sequentially | 86.8336096422 |
| batchDF | 4, barrier-released requests | 201.391137643 |

Every request reported `usage.completion_tokens=512` and
`finish_reason=length`. Aggregate throughput is all four API token counts divided
by group makespan, not per-request speed or offline executor timing.
The measured worker SHA256 was
`18a8d4a6e7eacb9d456293bdfdbd0729873cfa425539c9a8aa9c35b1d5223568`.
A rebuild on another machine will not necessarily have the same binary hash.

These are historical measurements with the original export, not a performance
promise for a re-export, another GPU or another workload. This source version's
normal `solo` and `batch-solo` results must be measured afresh; do not combine
older normal-mode numbers with the table above. Concurrent answers varied and
normal versus DFlash answers were not strictly identical. Full numerical parity,
lossless greedy decoding and generated-program correctness are not established.
A near-tied-logit history probe reproduced a difference between small-row A8 and
larger-row A16 target execution; the existing precision policy is unchanged.
There is no new VRAM or RTX 5090 validation claim.

## Checkout and toolchain

Push this branch to your own GitHub repository first, then clone that repository
on the test machine. Clone with submodules and retain the `executorch` directory
basename required by the CMake include root:

```bash
git clone --recurse-submodules --branch mg-cuda-dflash-batching-api-201 \
  "<YOUR_GITHUB_REPOSITORY_URL>" /absolute/path/to/executorch
```

Choose a matching CUDA 13, PyTorch, Python and C++ ABI toolchain for export and
runtime. Use an isolated Python environment installed from this checkout through
its normal packaging; do not redirect a source-only namespace that omits package
resources or keep a stale editable installation from another checkout. Dependencies
must be available before running these commands. This package does not install
tools or download models for you.

The example below targets RTX 5090 (SM120). For SM80 use `CUDA_ARCH=80` and
`TORCH_CUDA_ARCH_LIST=8.0`; for another GPU choose its actual capability. Existing
SM80-native cubins cannot be reused on SM120. Re-export on the test GPU. A
successful SM80 build does not prove another architecture works or has enough
VRAM. Unsupported kernels, ABI mismatches and OOM must fail closed, without
CPU/MLX or non-DFlash fallback.

```bash
set -euo pipefail
export REPO=/absolute/path/to/executorch
export ASSETS=/absolute/path/to/model-assets
export WORK=/absolute/path/to/reproduction-work
export CUDA_HOME=/absolute/path/to/cuda-13
export PYTHON=/absolute/path/to/isolated-env/bin/python
export STB_INCLUDE=/absolute/path/to/pinned-stb
export GPU=0
export CUDA_ARCH=120
export TORCH_CUDA_ARCH_LIST=12.0
export CUDA_VISIBLE_DEVICES="$GPU"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export MAX_JOBS=16
export CMAKE_BUILD_PARALLEL_LEVEL=16
export BENCH="$REPO/examples/models/muse-glimmer/benchmark_cuda_dflash"
export EXPORT="$WORK/export-sm$CUDA_ARCH"
export SDK_BUILD="$REPO/cmake-out"
export SDK="$WORK/runtime-install"
export MUSE_BUILD="$WORK/muse-runtime"
mkdir -p "$WORK" "$WORK/tmp"
export TMPDIR="$WORK/tmp"
test "$(basename "$REPO")" = executorch
cd "$REPO"
export TORCH_CMAKE="$("$PYTHON" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
export PYTHON_SITE="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export LD_LIBRARY_PATH="$SDK/lib:$SDK/lib64:$CUDA_HOME/lib64:$PYTHON_SITE/torch/lib:${LD_LIBRARY_PATH:-}"
```

Record GPU UUID/name, driver, CUDA/PyTorch versions, C++ ABI, compiler,
branch/commit/patch, all production Python origins/hashes, asset/tokenizer/export
hashes, server argv and environment. Verify both the parent exporter and Inductor
workers import this branch's CUDA kernels. Keep everything frozen across modes.

## Build the SDK and worker

These feature flags preserve the validated SDK recipe. Architecture-specific
builds on another machine remain unverified until you execute them.

```bash
cmake -S "$REPO" -B "$SDK_BUILD" -GNinja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$SDK" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
  -DCMAKE_PREFIX_PATH="$CUDA_HOME;$TORCH_CMAKE;$PYTHON_SITE" \
  -DEXECUTORCH_BUILD_CUDA=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_TESTS=ON \
  -DEXECUTORCH_ENABLE_LOGGING=ON -DEXECUTORCH_LOG_LEVEL=Info \
  -DEXECUTORCH_BUILD_PYBIND=OFF -DEXECUTORCH_BUILD_COREML=OFF \
  -DEXECUTORCH_BUILD_QNN=OFF -DEXECUTORCH_BUILD_MLX=OFF \
  -DEXECUTORCH_BUILD_VULKAN=OFF -DEXECUTORCH_BUILD_OPENVINO=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_TRAINING=OFF -DPCRE2_SUPPORT_JIT=OFF \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$SDK_BUILD" --parallel 16
cmake --install "$SDK_BUILD"
cmake -S "$REPO/examples/models/muse-glimmer" -B "$MUSE_BUILD" -GNinja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DCMAKE_PREFIX_PATH="$SDK;$SDK_BUILD/third-party/gflags;$CUDA_HOME;$TORCH_CMAKE" \
  -DEXECUTORCH_BUILD_CUDA=ON -DMUSE_GLIMMER_BUILD_BATCHING=ON \
  -DMUSE_GLIMMER_BUILD_BATCHING_TESTS=ON \
  -DMUSE_GLIMMER_STB_INCLUDE_DIR="$STB_INCLUDE" \
  -DMUSE_GLIMMER_BUILD_CI_BENCHMARK=ON -DMUSE_GLIMMER_BUILD_CI_SMOKE=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$MUSE_BUILD" --parallel 16
ctest --test-dir "$MUSE_BUILD" --output-on-failure
```

Module, LLM/Sampler, flat-tensor and named-data support are required. The SDK
build location above matches the Muse gflags lookup. Supply pinned stb including
the deprecated resize header. The native GTest target uses the checkout's
`third-party/googletest` and propagates `GTEST_HAS_POSIX_RE=0` for tokenizer ABI
compatibility. Obtain missing pinned dependencies before building.

## Check serialization resources, then re-export

A source-only installation can load Python successfully but fail much later when
serializing PTE/PTD. Check the real package resource APIs before the large export;
fix your installation through normal packaging if any resource is absent. Do not
mock these reads or patch the SDK. This check does not establish GPU correctness.

```bash
"$PYTHON" - <<'PY'
import importlib
import importlib.resources

resources = {
    "executorch.exir._serialize": ("program.fbs", "scalar_type.fbs"),
    "executorch.extension.flat_tensor.serialize": ("flat_tensor.fbs", "scalar_type.fbs"),
}
for name, filenames in resources.items():
    package = importlib.import_module(name)
    print(name, package.__file__)
    for filename in filenames:
        data = importlib.resources.files(package).joinpath(filename).read_bytes()
        assert data, (name, filename)
print("PTE/PTD package resources readable")
PY
export FLATC_EXECUTABLE="$SDK_BUILD/third-party/flatc_ep/bin/flatc"
test -x "$FLATC_EXECUTABLE"
"$FLATC_EXECUTABLE" --version
"$PYTHON" -m executorch.examples.models.muse_glimmer.export.export_dflash \
  --target-gguf "$ASSETS/target.gguf" --draft-gguf "$ASSETS/draft.gguf" \
  --backend cuda --activation-dtype bfloat16 --use-offgraph-kv-cache \
  --max-seq-len 8192 --max-step-tokens 4096 \
  --max-cells 65536 --draft-max-cells 65536 --max-draft-owners 4 \
  --forward-sizes 1,2,4,8,16,32,64 --output-dir "$EXPORT"
```

Use this one tapped-target/draft PTE/PTD for every mode, including non-DFlash.
Expected files are `$EXPORT/model.pte` and `$EXPORT/aoti_cuda_blob.ptd`.
Tokenizer inputs are `$ASSETS/tokenizer.json` and its containing HF assets directory
with associated configuration/chat template. Record exact filenames and hashes.
Do not substitute the newer experimental export for this branch's export.

## Four modes and three repeats

Start the normal server in one terminal:

```bash
"$PYTHON" -m executorch.examples.models.muse_glimmer.serving.serve_batching \
  --backend cuda --worker-bin "$MUSE_BUILD/muse_glimmer_batching_worker" \
  --model-path "$EXPORT/model.pte" --data-path "$EXPORT/aoti_cuda_blob.ptd" \
  --tokenizer-path "$ASSETS/tokenizer.json" --hf-tokenizer "$ASSETS" \
  --model-id muse_glimmer --max-context 4095 --max-sessions 4 \
  --max-decode-sequences 4 --max-inflight-requests 4 --prefix-cache-entries 0 \
  --host 127.0.0.1 --port 8000
```

Wait for readiness, then define the client helper in a separate terminal using
the same environment. The mode is only a label, not server configuration.
`--include-usage` adds only `stream_options.include_usage=true` to a deep copy of
the frozen payload and saves the exact wire body. It is required for API token
accounting. Concurrency 1 is sequential; concurrency 4 is a four-thread barrier.

```bash
run_client() {
  "$PYTHON" "$BENCH/compare_http.py" \
    --fixtures "$BENCH/fixtures/requests.json" --base-url http://127.0.0.1:8000 \
    --include-usage --timeout 600 --mode "$1" --concurrency "$2" \
    --output-dir "$WORK/results/$3"
}
mkdir -p "$WORK/results"
run_client solo 1 warmup-solo
run_client batch-solo 4 warmup-batch-solo
for repeat in 1 2 3; do
  run_client solo 1 "solo-$repeat"
  run_client batch-solo 4 "batch-solo-$repeat"
done
```

Stop only your own normal server and worker. Restart with exactly the same server
command, adding `--dflash --dflash-n-draft 3`. This explicit N3 setting is required
for the 201 tokens/s configuration; omitting it selects the default N15.
Keep the same PTE/PTD, runtime, GPU, tokenizer, limits and temperature zero.
Wait for readiness, then run:

```bash
run_client DFsolo 1 warmup-DFsolo
run_client batchDF 4 warmup-batchDF
for repeat in 1 2 3; do
  run_client DFsolo 1 "DFsolo-$repeat"
  run_client batchDF 4 "batchDF-$repeat"
done
```

Warmups are separate and excluded. Use fresh output directories; the client
refuses overwrites. A failed CUDA DFlash co-batch invalidates cache owners and
requires restarting the worker. Do not silently retry, continue affected
sessions, fall back to normal decoding or count failures as valid measurements.
Prefix reuse is disabled. Record any between-repeat restart/warmup protocol and
apply it consistently. In N3, draft still queries 16 positions per owner and
target verification uses four positions per owner.

## Metrics and offline checks

The client saves raw response bytes, SSE events, HTTP status/headers, exact
payloads and monotonic timestamps. Missing/invalid API usage, malformed streams,
missing finish reasons or missing `[DONE]` invalidate a request and suppress group
throughput. Never count chunks, retokenize text or use proposals as completion
counts. TTFT is POST start to first visible reasoning/content delta. TPOT is
first-visible-delta to `[DONE]` divided by API completion tokens minus one, null
for zero/one tokens. These are client metrics, not GPU token timings.

For each mode report all three repeats, median, min/max and sample standard
deviation of aggregate throughput/makespan. Retain per-task TTFT/TPOT, completion
counts, reasoning/content and finish reasons. Do not execute generated programs.
If sampling VRAM, record GPU UUID, idle baseline, other device users, sample
interval and measured-window peaks separately from startup/warmup. Device-wide
NVML usage is not worker-exclusive and can miss transient peaks.

Run the ten offline parser/accounting/mock-transport tests only:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m unittest discover \
  -s "$BENCH" -p test_compare_http.py -v
```

These tests do not build, export, serve, use a GPU or establish model correctness.
Frozen fixture hashes are checked, including the fourth source. Fixture JSON
metadata replaces original machine-specific paths/URLs; all four task IDs,
source hashes and HTTP payloads remain exact. Raw `P*.txt` files are provenance,
not executable scripts. The root BSD license covers this package; no header is
inserted into raw frozen sources because it would change their hashes.

Original pre-portable client SHA256:
`cdc00751df2ac804ac79529290a3c024f2dab3b7b35213c783f8e30a86ce3ecb`.
Original pre-portable helper-test SHA256:
`020d52c03b2498017513afec01690d0057429aa0b34706e8f900d8069bb07350`.
Portable client, tests and fixtures are byte-identical to package commit
`24f47b32811dddf1df8b01d4e8da961393fb55f0`. Scoped lint instructions and
configuration remain in this directory's `validation/` subdirectory. Never apply
a global formatter or validator to frozen export inputs or unrelated checkouts.
