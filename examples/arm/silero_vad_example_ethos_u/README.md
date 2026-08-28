# Silero VAD Ethos-U Example

Voice activity detection (VAD) identifies whether each audio chunk contains
speech. Silero VAD processes 512 audio samples per step, preserves recurrent
state between steps, and produces a speech probability for each chunk.

This end-to-end example shows how to:

- load the Silero VAD JIT model, re-export it through PT2E
  quantization, and lower it for Ethos-U85.
- build a bare-metal Corstone-320 application that embeds the exported
  `.pte` and a 16 kHz mono WAV file.
- run the app on the Corstone-320 Fixed Virtual Platform (FVP), print
  speech probabilities over serial, and validate those probabilities on the
  host.

## Layout

- `model_export/README.md` — Model loading, quantization, Ethos-U85
  lowering, `.pte` generation, and reference probability generation.
- `runtime/README.md` — Building the bare-metal app, generating headers
  from the `.pte` and WAV file, running on FVP, and comparing serial
  probability output.
