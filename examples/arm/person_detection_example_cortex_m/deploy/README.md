# Person-detection Cortex-M deployment sample

This directory contains an FVP deployment application for Corstone-300 with
Ethos-U55. The FVP Virtual Streaming Interface (VSI4) supplies 320x240 RGB
frames and the modeled MPS3 LCD displays the source with detection boxes.
Image preprocessing, inference, decoding, confidence filtering, and NMS all
run in the target firmware. UART0 is used for diagnostics and optional ETDump
profiles. The runner preloads the PTE in DDR at address `0x72000000`.

For every frame, the application measures `method->execute()` with the PMU
cycle counter and prints `Inference time: N us` on UART0.
Run the commands below from this example's root directory after exporting the
model with `export/export_model.py`. Note that ETDump and cycle counts are for
for demonstration purposes only; the FVP does not report accurate statistics.

## FVP

With the environment setup in the `person_detection_example_cortex_m` folder,
build the firmware:

```sh
cmake -S deploy \
  -B deploy/build-fvp \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/../ethos-u-setup/arm-none-eabi-gcc.cmake" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build deploy/build-fvp --target person_detection
```

Start the FVP with webcam 0:

```sh
python deploy/run_fvp.py
```

Use an image or video file instead with `--input PATH`, or select another
webcam with `--camera INDEX`. A still image is processed once. The FVP LCD
window shows the annotated frame. Stop the simulation with Ctrl-C.

Note that the FVP LCD window requires an accessible graphical session and working
OpenGL drivers.

UART diagnostics can be viewed from another terminal:

```sh
python deploy/host.py
```

`--port` changes the diagnostic UART TCP port and `--vsi-port` changes the
local frame transport port used by the VSI Python adapter.

## ETDump performance report

Generate an ETRecord while exporting the model, then enable profiling:

```sh
python export/export_model.py \
  --etrecord
cmake -S deploy \
  -B deploy/build-fvp \
  -DET_ENABLE_PROFILING=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/../ethos-u-setup/arm-none-eabi-gcc.cmake"
cmake --build deploy/build-fvp \
  --target person_detection
```

The profiling firmware records each `method->execute()` invocation and sends a
checksummed ETDump over UART. The host saves `profile.etdp` in `fvp-results/`
and prints the ExecuTorch Inspector table in CPU cycles. Profiling is disabled
by default because it increases firmware and RAM usage.
