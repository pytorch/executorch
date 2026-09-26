# Minimal Classic ML runtime

This is the reference ExecuTorch runtime integration for single-shot inference
on Arm bare metal. It loads one embedded program, creates the runtime memory
hierarchy, binds tensor inputs, executes once, and retrieves the outputs.

After running `examples/arm/setup.sh`, export the built-in MobileNetV2 model for
Ethos-U55:

```bash
python3 -m backends.arm.scripts.aot_arm_compiler \
  --model_name=mv2 \
  --target=ethos-u55-128 \
  --delegate \
  --quantize \
  --intermediate=cmake-out-arm-classic \
  --output=cmake-out-arm-classic/mv2.pte \
  --system_config=Ethos_U55_High_End_Embedded \
  --memory_mode=Shared_Sram
```

Configure and build the standalone minimal application:

```bash
cmake -S examples/arm/minimal_classic_ml -B cmake-out-arm-classic \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DET_PTE_FILE_PATH="$PWD/cmake-out-arm-classic/mv2.pte" \
  -DSYSTEM_CONFIG=Ethos_U55_High_End_Embedded \
  -DMEMORY_MODE=Shared_Sram
cmake --build cmake-out-arm-classic --target arm_classic_ml_runner -j
```

To load the PTE from a fixed address, set `ET_MODEL_PTE_ADDR`. Set
`ET_MODEL_PTE_SIZE` to the exact PTE size in bytes when it is known; otherwise,
the runner uses the legacy `0x10000000` upper bound. Continue to provide
`ET_PTE_FILE_PATH` for build-time operator selection. In that configuration the
PTE path is not compiled into the ELF. For example:

```bash
cmake -S examples/arm/minimal_classic_ml -B cmake-out-arm-classic \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DET_PTE_FILE_PATH="$PWD/cmake-out-arm-classic/mv2.pte" \
  -DET_MODEL_PTE_ADDR=0x70000000 \
  -DET_MODEL_PTE_SIZE=$(stat -c%s cmake-out-arm-classic/mv2.pte) \
  -DSYSTEM_CONFIG=Ethos_U55_High_End_Embedded \
  -DMEMORY_MODE=Shared_Sram
```

Run the result on Corstone-300:

```bash
backends/arm/scripts/run_fvp.sh \
  --elf=cmake-out-arm-classic/arm_classic_ml_runner \
  --target=ethos-u55-128
```

The target supports the same `SYSTEM_CONFIG`, `MEMORY_MODE`, toolchain, and
allocator-size settings as `arm_executor_runner`. Its input is filled with
deterministic placeholder values; replace that section with the application's
sensor or preprocessing output.

Use `arm_executor_runner` when BundleIO verification, ETDump, semihosted
arbitrary files, profiling, or test instrumentation is required.
