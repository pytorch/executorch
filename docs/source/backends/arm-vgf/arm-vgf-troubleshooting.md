# Arm VGF Troubleshooting

This page describes common issues that you may encounter when using the Arm VGF backend and how to debug and resolve them.

## How do you visualize VGF files

The [VGF Adapter for Model Explorer](https://github.com/arm/vgf-adapter-model-explorer) enables visualization of VGF files and can be useful for debugging.

## Environment preflight commands

The VGF backend provides a preflight helper that can be run before export or runtime execution:

```bash
python -m executorch.backends.arm.vgf.check_env --aot
python -m executorch.backends.arm.vgf.check_env --runtime
python -m executorch.backends.arm.vgf.check_env --host-emulator
python -m executorch.backends.arm.vgf.check_env --source-build --build-dir cmake-out
```

Use `--aot` before export. It checks that the TOSA serializer and ML SDK model converter are available and that the converter can be launched.

Use `--runtime` when debugging Python runtime availability. It checks whether the ExecuTorch runtime backend registry reports VgfBackend as available.

Use `--host-emulator` before host-based emulator runs. It checks runtime availability plus Vulkan SDK and ML emulation layer environment variables.

Use `--source-build --build-dir <dir>` when debugging a source build. It checks for VGF runtime build prerequisites such as `libvgf` and CMake options including `EXECUTORCH_BUILD_VGF` and `EXECUTORCH_BUILD_VULKAN`.

For CI logs or bug reports, add `--json`:

```bash
python -m executorch.backends.arm.vgf.check_env --aot --json
```

## Testing VGF ahead-of-time lowering

The Arm backend includes a lightweight VGF ahead-of-time smoke test that checks
that a small PyTorch model can be exported, partitioned for VGF, lowered through
the shared TOSA pipeline, and converted into an ExecuTorch program.

The test mocks the final VGF `model-converter` invocation, so it does not
require the ML SDK Model Converter, Vulkan runtime, or VKML host-emulation
setup. It is intended to catch integration regressions in the Python AOT
lowering path before running heavier VGF runtime tests.

Run it directly with:

```bash
pytest -q backends/arm/test/misc/test_vgf_smoke.py
```

If using the Arm backend test wrapper, run:

```bash
backends/arm/test/test_arm_backend.sh test_pytest_vgf_smoke
```
