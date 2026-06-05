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
