---
name: building
description: Build ExecuTorch from source — Python package, C++ runtime, runners, cross-compilation, and backend-specific builds. Use when compiling anything in the ExecuTorch repo, diagnosing build failures, or setting up platform-specific builds.
---

# Building ExecuTorch

IMPORTANT: Do NOT ask the user questions about environment setup. Detect issues and fix them automatically. The standard path is conda — always use it.

## Step 1: Ensure conda environment (do this silently, fix issues automatically)

```bash
# Check if executorch conda env exists; create if not
conda env list | grep executorch || conda create -yn executorch python=3.12 'cmake<4'
# Activate
conda activate executorch
# Verify — fix automatically if wrong
python --version          # need 3.10–3.13
cmake --version           # need >= 3.24, < 4.0; if cmake >= 4, run: pip install 'cmake>=3.24,<4'
```

If cmake is >= 4.0 inside the conda env, fix it: `pip install 'cmake>=3.24,<4'` — do not ask the user.

Parallel jobs: `$(sysctl -n hw.ncpu)` on macOS, `$(nproc)` on Linux.

## Step 2: Build (route by what the user needs)

### Python package (default — use this unless user asks for something specific)
```bash
conda activate executorch
./install_executorch.sh --editable    # editable install from source
```
This handles everything: submodules, deps, C++ build, Python install. Takes ~10 min on Apple Silicon.

For subsequent rebuilds (deps already present): `pip install -e . --no-build-isolation`

For minimal install (skip example deps): `./install_executorch.sh --minimal`

Enable additional backends:
```bash
CMAKE_ARGS="-DEXECUTORCH_BUILD_COREML=ON -DEXECUTORCH_BUILD_MPS=ON" ./install_executorch.sh --editable
```

Verify: `python -c "from executorch.exir import to_edge_transform_and_lower; print('OK')"`

### LLM / ASR model runner (simplest path for running models)

```bash
conda activate executorch
make <model>-<backend>
```

Available targets (run `make help` for full list):

| Target | Backend | macOS | Linux |
|--------|---------|-------|-------|
| `llama-cpu` | CPU | yes | yes |
| `llama-cuda` | CUDA | — | yes |
| `llama-cuda-debug` | CUDA (debug) | — | yes |
| `llava-cpu` | CPU | yes | yes |
| `whisper-cpu` | CPU | yes | yes |
| `whisper-metal` | Metal | yes | — |
| `whisper-cuda` | CUDA | — | yes |
| `parakeet-cpu` | CPU | yes | yes |
| `parakeet-metal` | Metal | yes | — |
| `parakeet-cuda` | CUDA | — | yes |
| `voxtral-cpu` | CPU | yes | yes |
| `voxtral-cuda` | CUDA | — | yes |
| `voxtral-metal` | Metal | yes | — |
| `voxtral_realtime-cpu` | CPU | yes | yes |
| `voxtral_realtime-cuda` | CUDA | — | yes |
| `voxtral_realtime-metal` | Metal | yes | — |
| `gemma3-cpu` | CPU | yes | yes |
| `gemma3-cuda` | CUDA | — | yes |
| `sortformer-cpu` | CPU | yes | yes |
| `sortformer-cuda` | CUDA | — | yes |
| `silero-vad-cpu` | CPU | yes | yes |
| `clean` | — | yes | yes |

Output: `cmake-out/examples/models/<model>/<runner>`

### C++ runtime (standalone)

**With presets (recommended):**

| Platform | Command |
|----------|---------|
| macOS | `cmake -B cmake-out --preset macos` (uses Xcode generator — requires Xcode) |
| Linux | `cmake -B cmake-out --preset linux -DCMAKE_BUILD_TYPE=Release` |
| Windows | `cmake -B cmake-out --preset windows -T ClangCL` |

Then: `cmake --build cmake-out -j$(sysctl -n hw.ncpu)` (macOS) or `cmake --build cmake-out -j$(nproc)` (Linux)

**LLM libraries via workflow presets** (configure + build + install in one command):
```bash
cmake --workflow --preset llm-release        # CPU
cmake --workflow --preset llm-release-metal  # Metal (macOS)
cmake --workflow --preset llm-release-cuda   # CUDA (Linux)
```

**Manual CMake (custom flags):**
```bash
cmake -B cmake-out \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON
cmake --build cmake-out --parallel "$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
```

Run `cmake --list-presets` to see all available presets.

### Cross-compilation

**iOS/macOS frameworks:**
```bash
./scripts/build_apple_frameworks.sh --coreml --mps --xnnpack
```
Link in Xcode with `-all_load` linker flag.

**Android:**
```bash
export ANDROID_ABIS=arm64-v8a BUILD_AAR_DIR=aar-out
mkdir -p $BUILD_AAR_DIR && sh scripts/build_android_library.sh
```

## Key build options

Most commonly needed flags (full list: `CMakeLists.txt`):

| Flag | What it enables |
|------|-----------------|
| `EXECUTORCH_BUILD_XNNPACK` | XNNPACK CPU backend |
| `EXECUTORCH_BUILD_COREML` | Core ML (macOS/iOS) |
| `EXECUTORCH_BUILD_MPS` | MPS GPU (macOS/iOS) |
| `EXECUTORCH_BUILD_METAL` | Metal compute (macOS, requires EXTENSION_TENSOR) |
| `EXECUTORCH_BUILD_CUDA` | CUDA GPU (Linux, requires EXTENSION_TENSOR) |
| `EXECUTORCH_BUILD_KERNELS_OPTIMIZED` | Optimized kernels |
| `EXECUTORCH_BUILD_KERNELS_QUANTIZED` | Quantized kernels |
| `EXECUTORCH_BUILD_EXTENSION_MODULE` | Module extension (requires DATA_LOADER + FLAT_TENSOR + NAMED_DATA_MAP) |
| `EXECUTORCH_BUILD_EXTENSION_LLM` | LLM extension |
| `EXECUTORCH_BUILD_TESTS` | Unit tests (`ctest --test-dir cmake-out --output-on-failure`) |
| `EXECUTORCH_BUILD_DEVTOOLS` | DevTools (Inspector, ETDump) |
| `EXECUTORCH_OPTIMIZE_SIZE` | Size-optimized build (`-Os`, no exceptions/RTTI) |
| `CMAKE_BUILD_TYPE` | `Release` (default for presets) or `Debug` (5-10x slower) |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Missing headers / `CMakeLists.txt not found` in third-party | `git submodule sync --recursive && git submodule update --init --recursive` |
| Mysterious failures after `git pull` or branch switch | `rm -rf cmake-out/ pip-out/ && git submodule sync && git submodule update --init --recursive` |
| CMake >= 4.0 (too new) | `pip install 'cmake>=3.24,<4'` inside the conda env |
| `externally-managed-environment` / PEP 668 error | You're using system Python, not conda. Activate conda env first. |
| pip conflicts with torch versions | Fresh conda env; or `./install_executorch.sh --use-pt-pinned-commit` |
| Missing `Python.h` (Linux) | `sudo apt install python3.X-dev` |
| Missing operator registrations at runtime | Link kernel libs with `-Wl,-force_load,<lib>` (macOS) or `-Wl,--whole-archive <lib> -Wl,--no-whole-archive` (Linux) |
| `install_executorch.sh` fails on Intel Mac | No prebuilt PyTorch wheels; use `--use-pt-pinned-commit --minimal` |
| XNNPACK build errors about cpuinfo/pthreadpool | Ensure `EXECUTORCH_BUILD_CPUINFO=ON` and `EXECUTORCH_BUILD_PTHREADPOOL=ON` (both ON by default) |
| Duplicate kernel registration abort | Only link one `gen_operators_lib` per target |

## Build output

| Artifact | Location |
|----------|----------|
| Core runtime | `cmake-out/libexecutorch.a` |
| executor_runner | `cmake-out/executor_runner` |
| Model runners | `cmake-out/examples/models/<model>/<runner>` |
| XNNPACK backend | `cmake-out/backends/xnnpack/libxnnpack_backend.a` |
| Python package | `site-packages/executorch` |
| iOS frameworks | `cmake-out/*.xcframework` |
| Android AAR | `aar-out/` |

## Tips
- Always use `Release` for benchmarking; `Debug` is 5–10x slower
- `ccache` is auto-detected if installed (`brew install ccache`)
- `Ninja` is faster than Make (`-G Ninja`) — but `--preset macos` uses Xcode generator
- For LLM workflows, `make <model>-<backend>` is the simplest path
- After `git pull`, clean and re-init submodules before rebuilding
