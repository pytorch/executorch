# Silero VAD Export

This flow exports the 16 kHz Silero voice activity detection model for
Corstone-320 / Ethos-U85-256.

Two model forms are used. The Silero JIT archive supplies the trained weights,
while the eager PyTorch implementation in `examples/models/silero_vad` supplies
the graph that ExecuTorch exports. The loader copies the JIT weights into the
eager model and checks that both forms produce matching outputs.

The export script then:

- calibrates PT2E quantization with 16 kHz mono WAV audio;
- derives recurrent-state quantization parameters from the calibration audio
  and stores the state in an internal int8 mutable buffer;
- lowers the quantized model to Ethos-U85-256; and
- writes a `.pte` file and reference probabilities for validation.

## Prepare Inputs

Install ExecuTorch and the Arm dependencies, then download the Silero VAD JIT
archive and sample audio from the tested revision:

```bash
mkdir -p silero-vad
curl --fail --location --output silero-vad/silero_vad.jit \
  https://raw.githubusercontent.com/snakers4/silero-vad/dbacf536adadf42210f37ae50fbaf75f6235b3cf/src/silero_vad/data/silero_vad.jit
curl --fail --location --output silero-vad/test.wav \
  https://raw.githubusercontent.com/snakers4/silero-vad/dbacf536adadf42210f37ae50fbaf75f6235b3cf/tests/data/test.wav
```

The tested revision is licensed under MIT. This example does not redistribute
the model or its weights; users obtain them directly from the Silero repository.

The required files are:

```text
silero-vad/silero_vad.jit
silero-vad/test.wav
```

Calibration and validation audio must be mono, 16 kHz, 16-bit PCM WAV. Use
different, non-overlapping audio for calibration and validation when measuring
model quality.

## Export

Run from the ExecuTorch repository root:

```bash
python examples/arm/silero_vad_example_ethos_u/model_export/export_silero_vad_ethos_u.py \
  --jit-model /path/to/silero-vad/src/silero_vad/data/silero_vad.jit \
  --calibration-audio /path/to/calibration.wav \
  --validation-audio /path/to/validation.wav
```

This writes `silero_vad_ethos_u.pte` and `expected_probs.bin` in the current
directory. Run the script with `--help` to see output paths, frame limits, and
other options.

## Delegation

The exported PTE accepts one `(1, 576)` float audio frame and owns the LSTM
hidden and cell state as an internal mutable buffer. The state is stored as
int8, and the LSTM gate activations and recurrent update are part of the
Ethos-U delegated subgraph. Small quantize/dequantize operations at the model
boundary and the mutable-buffer write remain portable runtime operations.
