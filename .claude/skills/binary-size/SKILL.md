---
name: binary-size
description: Analyze and reduce ExecuTorch binary size. Use when investigating binary size, running size tests, or optimizing the runtime for size-constrained deployments.
---

# Binary Size

## Build and measure
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
bloaty cmake-out/test/size_test -d compileunits   # by source file
bloaty cmake-out/test/size_test -d symbols -n 30   # by symbol
bloaty cmake-out/test/size_test -d sections        # by ELF section
bloaty <after> -- <before>                          # diff two builds
```

Also useful: `nm -S <binary> | sort -k2 -rn | head -30` for symbol sizes.

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
- `.text`: `bloaty -d symbols` — look for large functions, template bloat, duplicate code
- `.rodata`: `strings <binary>` — look for verbose error messages, format strings, file paths
- `.eh_frame`: should be suppressed when `EXECUTORCH_OPTIMIZE_SIZE=ON`
- Static init functions: `nm -S <binary> | grep GLOBAL__sub_I` — constexpr constructors can eliminate these
- Logging: `ET_LOG_ENABLED=0` in Release builds eliminates format strings; ensure it propagates to consumers via `PUBLIC` compile definitions
- Inline header functions: watch for define mismatches between library and consumer TUs

## Document each change
Create `binary-size.md` with:

| Binary | This change (N vs N-1) | Cumulative (N vs main) |
|---|---|---|
| `size_test` (stripped) | -X | -Y |
| `size_test_all_ops` (stripped) | -X | -Y |
