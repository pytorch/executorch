# RISC-V

Cross-compile `executor_runner` for `riscv64-linux-gnu` and run it under
`qemu-user-static` against a small bundled program. The end-to-end check
mirrors the Arm Cortex-M e2e flow: a `Test_result: PASS` line in stdout from
the bundled-IO comparison path is the pass criterion.

This is the Phase 1 deliverable for the RISC-V Support RFC at
[pytorch/executorch#18991][rfc]. The cross-compile and runner artifacts
(toolchain file, preset, AOT script) are designed to carry over unchanged
to a hardware-runner job once one becomes available; only the invocation
step (qemu-user vs. native) would change.

[rfc]: https://github.com/pytorch/executorch/issues/18991

## Quick start (Ubuntu / Debian)

```bash
examples/riscv/setup.sh        # apt: gcc-riscv64-linux-gnu, qemu-user-static
examples/riscv/run.sh          # export, cross-compile, run under qemu-user
```

The driver does three steps:

1. `python examples/riscv/aot_riscv.py` exports a `torch.add` module to
   `riscv_test/add_riscv.bpte` (a BundledProgram with reference outputs
   embedded for two test cases).
2. `cmake --preset riscv64-linux` configures the cross-build using
   `examples/riscv/riscv64-linux-gnu-toolchain.cmake` and
   `tools/cmake/preset/riscv64_linux.cmake`. `executor_runner` is built
   against portable kernels with `ET_BUNDLE_IO_ENABLED` defined.
3. `qemu-riscv64-static` invokes the runner with `--model_path` pointing at
   the `.bpte`. The runner detects the bundle, runs every embedded test case,
   and emits `Test_result: PASS` (or `FAIL`) per case.

## CI

`.github/workflows/_test_riscv_qemu.yml` is a reusable `workflow_call`
job (mirroring `_test_cortex_m_e2e.yml`) invoked from `pull.yml` to run on
every PR. It runs on the standard `linux.2xlarge` x86_64 runner using the
`executorch-ubuntu-22.04-gcc11` docker image.
