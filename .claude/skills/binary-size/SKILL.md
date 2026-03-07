---
name: binary-size
description: Analyze and reduce ExecuTorch binary size. Use when investigating binary size, running size tests, or optimizing the runtime for size-constrained deployments.
---

# Binary Size

## Start from the `main` branch of executorch
Ask the user where the executorch repo is.

```bash
git checkout main && git pull
```

## Build and measure baseline
```bash
conda activate executorch
bash test/build_size_test.sh
strip -o /tmp/size_test_stripped cmake-out/test/size_test
strip -o /tmp/size_test_all_ops_stripped cmake-out/test/size_test_all_ops
ls -la /tmp/size_test_stripped /tmp/size_test_all_ops_stripped
```

Produces two binaries:
- `cmake-out/test/size_test` — ExecuTorch runtime without operator implementations
- `cmake-out/test/size_test_all_ops` — ExecuTorch runtime with portable ops

## Analyze with bloaty
```bash
bloaty cmake-out/test/size_test -d symbols -n 30   # by symbol
bloaty cmake-out/test/size_test -d sections        # by ELF section
bloaty <after> -- <before>                          # diff two builds
nm -S <binary> | sort -k2 -rn | head -30           # symbol sizes
strings <binary> | less                             # string literals in .rodata
```

Note: `bloaty -d compileunits` requires debug info (`-g`). The Release build does not include it.

## Key build flags
Set by `test/build_size_test.sh`:
- `CMAKE_BUILD_TYPE=Release`
- `EXECUTORCH_OPTIMIZE_SIZE=ON` — enables `-Os`, `-fno-exceptions`, `-fno-rtti`, unwind table suppression
- `CXXFLAGS="-fno-exceptions -fno-rtti -Wall -Werror"`

## Constraints
- Use **CMake** to build (not Buck)
- **C++17 minimum** language standard
- Must build on **GCC 9** (CI uses `executorch-ubuntu-22.04-gcc9-nopytorch`) and **Clang 12** — avoid compiler-specific flags or pragmas without version guards
- Do not regress existing functionality — run tests for modified files
- Do not change build flags in `build_size_test.sh` for size reductions
- Do not increase latency in the core runtime

## Where to look for size reductions
- `.text`: look for large functions, template bloat, duplicate instantiations
- `.rodata`: verbose error messages, format strings, embedded file paths (`__FILE__`)
- `.eh_frame`: should already be suppressed when `EXECUTORCH_OPTIMIZE_SIZE=ON`
- Static init functions (`nm -S <binary> | grep GLOBAL__sub_I`): use `constexpr` constructors to constant-initialize static arrays
- Logging strings: `ET_LOG_ENABLED=0` in Release eliminates format strings; ensure it propagates to consumers via `PUBLIC` compile definitions on cmake targets
- Inline header functions: watch for compile-define mismatches between library and consumer TUs (e.g. `ET_LOG_ENABLED` set in library but not in consumer)

## For each change
1. Create a branch: `git checkout -b binary-size-<N>`
2. Implement, rebuild, measure stripped sizes
3. Create a separate PR — one logical change per PR
4. Record results in `binary-size-<N>.md`:

| Binary | This change (N vs N-1) | Cumulative (N vs main) |
|---|---|---|
| `size_test` (stripped) | -X | -Y |
| `size_test_all_ops` (stripped) | -X | -Y |

5. Update the CI size threshold in `.github/workflows/pull.yml` if sizes decrease
