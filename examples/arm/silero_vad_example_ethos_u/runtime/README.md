# Silero VAD Runtime Example

This bare-metal Corstone-320 application embeds the Ethos-U85 Silero VAD PTE
and a 16 kHz mono WAV file. It passes one 32 ms audio frame to the model at a
time; the ExecuTorch program keeps the recurrent state between calls.

For each frame the application prints the speech probability and decision. It
also merges consecutive speech frames into segments:

```text
PROB 0.000 0.012345 silence
PROB 0.032 0.812345 speech
SEGMENT 0.032 0.640 speech
```

## Build ExecuTorch for Arm

From `examples/arm`:

```bash
cmake --preset arm-baremetal \
  -DCMAKE_BUILD_TYPE=Release \
  -B../../cmake-out-arm ../..
cmake --build ../../cmake-out-arm --target install
```

## Configure and Build

Run from `examples/arm`:

```bash
cmake \
  -DCMAKE_TOOLCHAIN_FILE=$(pwd)/ethos-u-setup/arm-none-eabi-gcc.cmake \
  -DTARGET_CPU=cortex-m85 \
  -DET_PTE_FILE_PATH=/path/to/silero_vad_ethos_u.pte \
  -DAUDIO_PATH=/path/to/validation.wav \
  -Bsilero_vad_ethos_u \
  silero_vad_example_ethos_u/runtime

cmake --build silero_vad_ethos_u --target silero_vad_ethos_u
```

`AUDIO_PATH` must be a 16 kHz mono 16-bit PCM WAV. The default maximum embedded
length is 40,000 samples (2.5 seconds). Set `MAX_AUDIO_SAMPLES` only after
checking the target memory map. The default speech threshold is `0.5`; override
it with `VAD_THRESHOLD` when configuring.

When using FVPs-on-Mac, also configure with `-DUART0_BASE=0x49303000`.

## Run on Corstone-320 FVP

From the ExecuTorch repository root:

```bash
bash backends/arm/scripts/run_fvp.sh \
  --elf examples/arm/silero_vad_ethos_u/silero_vad_ethos_u \
  --target=ethos-u85-256 \
  --timeout=300 2>&1 | tee fvp.log
```

The application sends end-of-transmission after the summary, so the FVP stops
without manual intervention.

## Validate

Compare the serial probabilities and detected speech segments with the
export-time reference:

```bash
python examples/arm/silero_vad_example_ethos_u/runtime/compare_vad_probs.py \
  --expected /path/to/expected_probs.bin \
  --actual-log ./fvp.log \
  --atol 0.25 \
  --mean-atol 0.02
```

The comparison also requires every speech/silence decision to match.

## Optional Binary Dump

The serial log is sufficient for normal validation. For debugging, configure
with `-DENABLE_SEMIHOSTING_OUTPUT=ON` and provide a semihosting directory when
running the FVP:

```bash
bash backends/arm/scripts/run_fvp.sh \
  --elf examples/arm/silero_vad_ethos_u/silero_vad_ethos_u \
  --target=ethos-u85-256 \
  --semihosting-cwd=$(pwd) \
  --timeout=300
```

This writes the raw float32 probabilities to `vad_probs.bin`, avoiding log
parsing and decimal formatting. Validate it by replacing `--actual-log fvp.log`
with `--actual vad_probs.bin` in the comparison command.
