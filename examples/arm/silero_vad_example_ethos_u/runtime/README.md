# Silero VAD Runtime Example

Build the Corstone-320 application, run the exported model, and validate its
speech probabilities.

## Build ExecuTorch for Arm

Complete [setup](../README.md#set-up) and [export](../model_export/README.md)
first. Run all commands from the repository root in the same environment:

```bash
source examples/arm/arm-scratch/setup_path.sh
arm-none-eabi-gcc --version
arm-none-eabi-g++ --version
cmake --preset arm-baremetal \
  -DCMAKE_BUILD_TYPE=Release \
  -DTARGET_CPU=cortex-m85 \
  -B cmake-out-arm -S .
cmake --build cmake-out-arm --target install
```

## Configure and Build

Use the exported PTE and the 40,000-sample validation clip:

```bash
runtime_cmake_args=()
if [ "$(uname -s)" = Darwin ]; then
  runtime_cmake_args+=(-DUART0_BASE=0x49303000)
fi

cmake \
  "${runtime_cmake_args[@]}" \
  -DCMAKE_TOOLCHAIN_FILE="$(pwd)/examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DTARGET_CPU=cortex-m85 \
  -DET_PTE_FILE_PATH="$(pwd)/silero_vad_ethos_u.pte" \
  -DAUDIO_PATH="$(pwd)/silero-vad/validation.wav" \
  -B examples/arm/silero_vad_ethos_u \
  -S examples/arm/silero_vad_example_ethos_u/runtime

cmake --build examples/arm/silero_vad_ethos_u --target silero_vad_ethos_u
```

`AUDIO_PATH` must be a 16 kHz mono 16-bit PCM WAV. The default limit is
40,000 samples (2.5 seconds).

## Run on Corstone-320 FVP

```bash
set -o pipefail
bash backends/arm/scripts/run_fvp.sh \
  --elf=examples/arm/silero_vad_ethos_u/silero_vad_ethos_u \
  --target=ethos-u85-256 \
  --timeout=300 2>&1 | tee fvp.log
```

## Validate

```bash
python examples/arm/silero_vad_example_ethos_u/runtime/compare_vad_probs.py \
  --expected ./expected_probs.bin \
  --actual-log ./fvp.log \
  --atol 0.25 \
  --mean-atol 0.02
```

Expect 79 probability values, zero speech/silence mismatches, and
`Validation passed`.

## Performance

The log reports NPU delegations and performance counters. Reference results
for the supplied clip:

| Metric | Measured value |
| --- | ---: |
| Audio frames | 79 |
| NPU delegations | 158 (2 per frame) |
| NPU active cycles per frame | 223,882 |
| Exported PTE size | 363,392 bytes |

Measured with FVP 11.31.28, a Release build, and default export settings:
ExecuTorch Python package 1.5.1, PyTorch 2.14.0, torchao 0.18.0, Vela 5.1.0,
and tosa-tools 2026.5.0. Keep `--fast` disabled when collecting counters.
FVP CPU cycle counts do not represent hardware latency.

## Optional Binary Dump

For raw probabilities, reconfigure and run in the same shell:

```bash
cmake "${runtime_cmake_args[@]}" \
  -DENABLE_SEMIHOSTING_OUTPUT=ON \
  -B examples/arm/silero_vad_ethos_u \
  -S examples/arm/silero_vad_example_ethos_u/runtime
cmake --build examples/arm/silero_vad_ethos_u --target silero_vad_ethos_u

bash backends/arm/scripts/run_fvp.sh \
  --elf=examples/arm/silero_vad_ethos_u/silero_vad_ethos_u \
  --target=ethos-u85-256 \
  --semihosting-cwd="$(pwd)" \
  --timeout=300
```

This writes `vad_probs.bin`. Validate it by replacing `--actual-log ./fvp.log`
with `--actual ./vad_probs.bin` in the comparison command.
