# RISC-V

End-to-end smoke tests that cross-compile ExecuTorch for RISC-V and run a bundled program under QEMU. A `Test_result: PASS` line emitted by the bundled-IO comparison path is the pass criterion.

Part of the RISC-V Support RFC, [pytorch/executorch#18991](https://github.com/pytorch/executorch/issues/18991).

## Quick start (Ubuntu / Debian)

```bash
examples/riscv/setup-linux.sh       # apt: gcc cross riscv64-linux-gnu + qemu-user
examples/riscv/setup-baremetal.sh   # apt: gcc cross riscv64-unknown-elf + qemu-system + picolibc
examples/riscv/run.sh               # export, cross-compile, run under qemu
```

`run.sh` accepts:

| Flag | Values | Default | Notes |
|---|---|---|---|
| `--model=<N>` | `add`, `mv2`, `mobilebert`, `llama2`, `resnet18`, `yolo26` | `add` | which model to export |
| `--quantize` | flag | off | XNNPACK quantizer (requires `--backend=xnnpack`) |
| `--backend=<N>` | `portable`, `xnnpack` | `portable` | xnnpack is linux-only |
| `--os=<N>` | `linux`, `baremetal` | `linux` | qemu-user vs qemu-system + semihosting |
| `--arch=<N>` | `rv32`, `rv64` | `rv64` | valid <os>-<arch> pairs are `linux-rv64`, `baremetal-rv32`, `baremetal-rv64` |
| `--qemu-cpu-ext=<S>` | e.g. `v=true,vlen=128` | empty | extensions appended after the arch base |

## Pipelines

**linux**: `aot_riscv.py` → `cmake --preset riscv64-linux` → `executor_runner` under `qemu-riscv64`. Portable kernels + (optional) XNNPACK delegate.

**baremetal**: `aot_riscv.py` → `cmake -S examples/riscv/baremetal` (standalone project; pulls executorch in via `add_subdirectory`) → `executor_runner_baremetal.elf` under `qemu-system-riscv64 -machine virt -bios none -semihosting-config target=native`.

The baremetal runner embeds the `.bpte` directly in `.rodata` via the same `examples/arm/executor_runner/pte_to_header.py` Cortex-M uses; semihosting SYS_WRITE0 / SYS_EXIT carry log output and exit status to the host.

## CI

`.github/workflows/riscv64.yml` is the entry point; it fans out into `_test_riscv.yml` over a `(model, backend, os, arch, quantize)` matrix and sweeps `qemu-cpu-ext` per backend. Runs on the `executorch-ubuntu-26.04-gcc15` docker image (needed for the `riscv64-unknown-elf` picolibc + libstdc++ packages - see [setup-linux.sh](setup-linux.sh) or [setup-baremetal.sh](setup-baremetal.sh)).
