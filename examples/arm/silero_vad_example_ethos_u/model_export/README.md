# Silero VAD Export

Export the 16 kHz Silero VAD model for Corstone-320 / Ethos-U85-256.

## Prepare Inputs

Complete [setup](../README.md#set-up), then run these commands from the
repository root to download the model and sample audio:

```bash
mkdir -p silero-vad
curl --fail --location --output silero-vad/silero_vad.jit \
  https://raw.githubusercontent.com/snakers4/silero-vad/dbacf536adadf42210f37ae50fbaf75f6235b3cf/src/silero_vad/data/silero_vad.jit
curl --fail --location --output silero-vad/test.wav \
  https://raw.githubusercontent.com/snakers4/silero-vad/dbacf536adadf42210f37ae50fbaf75f6235b3cf/tests/data/test.wav
python examples/arm/silero_vad_example_ethos_u/model_export/generate_smoke_audio.py \
  --source-wav silero-vad/test.wav \
  --output silero-vad/validation.wav
```

The helper creates `validation.wav` from the first 40,000 samples (2.5 seconds)
of `test.wav`, which fits the runtime's audio limit and produces 79 frames.
The model and audio come from the MIT-licensed Silero repository.

This example uses overlapping calibration and validation audio to check the
deployment. Use representative calibration audio and separate validation
recordings when measuring model quality.

## Export

```bash
python examples/arm/silero_vad_example_ethos_u/model_export/export_silero_vad_ethos_u.py \
  --jit-model silero-vad/silero_vad.jit \
  --calibration-audio silero-vad/test.wav \
  --validation-audio silero-vad/validation.wav
```

This writes `silero_vad_ethos_u.pte` and `expected_probs.bin` in the repository
root. Use the same `validation.wav` when building the runtime.

## Delegation

`EthosUBackend` subgraphs run on the NPU. Quantization, layout copies, and
recurrent-state writes between delegates run on the CPU. The state persists
between frames; a graph view of one call may not show that connection.

The exporter initializes the recurrent state. The emitter may still print a
mutable-buffer warning; the `warnings.warn(` line is part of Python's warning
output.
