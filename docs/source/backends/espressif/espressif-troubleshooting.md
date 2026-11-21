# Espressif Troubleshooting

This page describes common issues that you may encounter when using the Espressif backend and how to debug and resolve them.

## Operators run on the CPU instead of ESP-NN

The lowering report lists every operator left on the CPU, with the reason:

```python
edge, report = to_espnn_edge(exported_program)
for op in report["cpu_ops"]:
    print(f"{op['target']:40s} {op['reason']}")
```

- `no int8 lowering for this operator`: the backend doesn't support the operator. See the operator table in {doc}`espressif-partitioner`.
- A constraint message, such as `ceil_mode pooling is not supported`: the operator is supported, but not with these arguments.
- `data movement feeds an operator that runs on the CPU`: a reshape, permute or slice stays with its CPU consumer.
- The operator isn't quantized. Check that the model was quantized with `EspNNQuantizer` and that no per-module configuration leaves that part in float.

To make any CPU fallback an error, for example in a test, lower with `to_espnn_edge(program, allow_cpu_fallback=False)`.

## Lowering fails with "reads tensor N, which nothing writes before it"

The backend checks that every operation in a delegate reads only delegate inputs, constants or tensors written by an earlier operation. This error means the lowering produced an invalid graph, which would otherwise read uninitialized memory on the device. It is a backend bug. Please report it with the model, or a small model that reproduces it.

## Lowering fails with "requantizes ...; back-to-back requantization is not supported"

An identity operator, such as a dropout that was decomposed away, sits between two quantize/dequantize pairs with different quantization parameters. `EspNNQuantizer` gives these operators their input's parameters. This error usually means the model was quantized with a different quantizer, or with a configuration that annotates the operator separately.

## The program fails to load with "Missing operator"

The program needs a CPU operator that the runtime was built without. The runtime logs the operator's name.

- **With the ESP-IDF component:** regenerate the kernel bundle with `tools/gen_kernels.py` from the program you're running. The build checks that the bundle matches the models declared with `executorch_add_model`, but not models loaded at runtime, such as ones delivered over the air. Generate their bundle with `--ops` or `--oplist`.
- **In a source build:** add the operator to `EXECUTORCH_SELECT_OPS_LIST`, or select the operators from the program as shown in {doc}`espressif-overview`. Operators in the `quantized_decomposed::` namespace come from `quantized_ops_lib`, which needs `EXECUTORCH_BUILD_KERNELS_QUANTIZED=ON`.

## The ESP-IDF component fails to configure or link

- **Configure fails because the kernel bundle doesn't match.** A model was re-exported, added or removed, or the bundle was made for another version of the component. The error prints the `gen_kernels.py` command that fixes it. Regenerate the bundle with the `executorch` Python package that matches the component version.
- **Undefined reference to `et_requires_executorch_add_kernels`.** The application calls `et_model` but declares no kernels. Add `executorch_add_kernels`, or `executorch_add_model` with a Python that can import `executorch`, so that the build generates the bundle.

## The example runner reports "Input 0 requires memory-planned storage"

The example runner writes inputs into the program's planned memory, so the program must plan its inputs, which `to_executorch()` does by default. Programs exported for `examples/espressif/camera_stream` leave the input to the application (`MemoryPlanningPass(alloc_graph_input=False)`), so that the camera can share its image buffer with the runtime. Export a separate program with default memory planning for the example runner.

## Kernel registration fails at startup

The runtime logs "The total number of kernels to be registered is larger than the limit". With a selective build, the kernel registry is sized from the selected operators, and `quantized_ops_lib` registers more kernels than that. Build the runtime with an explicit `-DMAX_KERNEL_NUM=256`.

## The firmware fails to link

- **Undefined reference to `sysconf`** from `libpthreadpool.a`: the quantized kernels pull in pthreadpool, which needs POSIX calls that ESP-IDF doesn't provide. Build the runtime with `-DEXECUTORCH_BUILD_PTHREADPOOL=OFF -DEXECUTORCH_BUILD_CPUINFO=OFF`.
- **Undefined reference to `__getreent`** from `posix.cpp` in `libexecutorch_core.a`: the linker used the runtime's default POSIX platform layer instead of the application's. Link the component that defines the `et_pal_*` functions whole-archive (`WHOLE_ARCHIVE` in `idf_component_register`).
- **Undefined references to ExecuTorch symbols from `espnn_runtime`**: the component links the `executorch_core` target that the application imports. Call `find_package(executorch)` with `CMAKE_FIND_PACKAGE_TARGETS_GLOBAL` set to `TRUE`, as `examples/espressif/executor_runner` does.

## The firmware is too big for the app partition

The build fails with `Part 'factory' 0/0 @ 0x10000 size 0x200000 (overflow ...)`. The example project reserves 2 MB for the app, and the program is embedded in it. Use a larger app partition, as described under *Flash and cache settings* in {doc}`espressif-overview`.

## Inference is slower than expected

- **Cache settings.** Check the data cache. With ESP-IDF's default 32 KB data cache, MobileNetV3-Small at 224×224 takes 866 ms on an ESP32-S3. With a 64 KB data cache, 64-byte lines and a 32 KB instruction cache, it takes 649 ms. See *Flash and cache settings* in {doc}`espressif-overview`.
- **Profile.** Enable `CONFIG_ESPNN_DELEGATE_PROFILE` to print each operation's time after a delegate's first run. The sum covers the ESP-NN kernels only, and comparing it with `delegate_stats()` shows the time spent converting inputs and outputs at the delegate boundary.
- **CPU operators.** These run on the portable kernels, which are much slower than ESP-NN. Check `report["cpu_ops"]` from the lowering.

## The delegate isn't found at runtime

If loading fails because `EspNNBackend` isn't registered, the delegate wasn't linked.

- **With the ESP-IDF component:** the delegate is linked only when the kernel bundle records `EspNNBackend` or the project calls `executorch_add_backend(EspNNBackend)`. Regenerate the bundle from a program that was lowered with `to_espnn_edge`, or add the call. The call is needed for models that the build doesn't see, such as ones delivered over the air.
- **In a source build:** the `espnn_runtime` component registers itself from a static initializer, and ESP-IDF links it whole-archive when it is part of the build. Check that the component is in `EXTRA_COMPONENT_DIRS`, or that the example runner was built with `-DET_ESPNN_BACKEND=ON`.

## Results differ from the host

Fully delegated programs match the backend's host emulator exactly. To check a program on the device, export it as a BundleIO program with the emulator's outputs, as `examples/espressif/export_espnn_test_programs.py` does, and run it with the example runner built with `-DET_BUNDLE_IO=ON`.

Programs with CPU regions can differ slightly from the host, because the portable float kernels round differently from PyTorch. When a CPU result sits next to a rounding boundary, it can move a value at the next quantize step by one int8 code.

## The program crashes or hangs in the emulator

ESP-NN's optimized ESP32-S3 kernels use the S3's SIMD instructions. `esp-emu` releases before 0.51.0 don't implement them, and the firmware stops with `Firmware abort: IllegalInstruction`; `examples/espressif/setup.sh` installs 0.51.0. Either re-run the setup script to update the emulator, or build the firmware with ESP-NN's portable C kernels by adding `CONFIG_NN_ANSI_C=y` to the sdkconfig defaults. CI tests both: `.ci/scripts/test_espressif.sh` takes `ESPRESSIF_NN_KERNELS=optimized` or `ansi`.

## The camera example shows a black or frozen preview

At a 20 MHz camera clock, the OV3660 on the ESP32-S3 CAM boards starts dropping and corrupting frames after a few minutes. The board logs `cam_hal: NO-SOI` and frame timeouts, and `dec_err` grows in the results. Keep `CONFIG_CAMERA_STREAM_XCLK_MHZ` at its default of 10 MHz.
