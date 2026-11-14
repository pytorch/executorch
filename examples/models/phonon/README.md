# Phonon

Text-to-speech at 24 kHz with Gradium's
[Phonon 31M English](https://huggingface.co/gradium/phonon-31m-en), built on
Kyutai's [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) and the Mimi
codec. The weights, tokenizer and voices are Apache-2.0.

```bash
# From the repository root, in a new Python 3.11+ environment. Install
# ExecuTorch from this checkout; it also builds CUDA if nvcc is on PATH,
# Vulkan if glslc is on PATH, and MLX if `xcrun --find metal` succeeds.
./install_executorch.sh
python -m pip install safetensors tokenizers huggingface_hub

# Choose xnnpack (CPU), cuda (NVIDIA GPU), mlx (Apple Silicon), or vulkan (GPU).
# For webgpu (the browser), see WebGPU below.
BACKEND=xnnpack
# Export: FP32 for every backend. Add --dtype bf16 for BF16 on CUDA or MLX.
python -m executorch.examples.models.phonon.export_phonon \
  --backend "$BACKEND" \
  --output-dir "phonon_exports/$BACKEND"

# Build the matching runner (Vulkan requires the Vulkan SDK and glslc).
# Remove cmake-out before building for another backend.
make "phonon-${BACKEND/xnnpack/cpu}"

# Synthesize
DATA_PATH=""
if [ "$BACKEND" = cuda ]; then
  DATA_PATH="phonon_exports/cuda/aoti_cuda_blob.ptd"
fi
cmake-out/examples/models/phonon/phonon_runner \
  --model_path="phonon_exports/$BACKEND/phonon.pte" \
  --data_path="$DATA_PATH" \
  --text="Hello from ExecuTorch." \
  --output=hello.wav
# Prints "Wrote hello.wav: 2.16 s of audio, 27 frames, ..." for an FP32 export.
```

Export downloads `gradium/phonon-31m-en` at a pinned revision (or reads
`--checkpoint-dir`) and writes `phonon.pte`, `phonon.json` (the inputs and
outputs of its four methods), the tokenizer and the default voice.
`--voice_path` takes another voice from
[gradium/phonon-voices](https://huggingface.co/gradium/phonon-voices);
`--temperature` (0.3 by default, 0 for no noise) and `--seed` control sampling.
`run_phonon.py` is the same runner in Python and generates the same audio from
the same program (`python -m executorch.examples.models.phonon.run_phonon
--model phonon_exports/$BACKEND/phonon.pte --text "Hello from ExecuTorch."
--output hello_py.wav`, plus `--data-path` for CUDA).

`--stream` writes 16-bit mono PCM to stdout as each 80 ms frame is decoded,
instead of the WAV file, so playback starts after the first frame:

```bash
cmake-out/examples/models/phonon/phonon_runner --model_path=... --text="..." --stream |
  aplay -q -f S16_LE -r 24000 -c 1     # macOS: play -q -t raw -r 24000 -e signed -b 16 -c 1 -
```

XNNPACK delegates the linear and convolution ops; the flow LM's attention runs
as the ExecuTorch custom op `llama::sdpa_with_kv_cache`.

For XNNPACK, Vulkan, MLX and CUDA, the export rewrites the Mimi decoder's
transposed convolutions as convolutions, which those backends run faster.

`--quantize 4bit` or `--quantize 8bit` quantizes the linear layers of the flow
LM's transformer and flow net with the scheme of the backend; the codec stays in
the export dtype. Without `--quantize`, the export is full precision.

| Backend | `--quantize 4bit` | `--quantize 8bit` | Size of phonon.pte (FP32: about 155 MiB) |
|---------|-------------------|-------------------|------------------------------------------|
| XNNPACK | 8-bit dynamic activations, 4-bit weights | 8-bit dynamic activations, 8-bit weights | 64, 75 MiB |
| CUDA    | 4-bit weights, BF16 | 8-bit weights | 64, 62 MiB with BF16 |
| Vulkan, WebGPU | 4-bit weights | not available | 66 MiB |
| MLX     | asymmetric 4-bit weights | 8-bit weights | 68, 76 MiB; 41, 51 MiB with BF16 |

`--backend cuda --quantize 4bit` exports with `--dtype bf16`, which the int4
kernel needs. A level without a kernel on the backend fails before the download
and names the options that work. 8-bit and the asymmetric 4-bit of CUDA and MLX
keep the timing of FP32. The symmetric 4-bit of XNNPACK, Vulkan and WebGPU
keeps the words but can hold the silence at the end of a sentence for up to two
seconds.

Vulkan and MLX run each method in one partition. Vulkan needs `glslc`, for
example from the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) (on Linux,
extract the tarball and `source setup-env.sh`). MLX needs the full Xcode with
the Metal toolchain, selected with `xcode-select` before
`./install_executorch.sh` (see the [MLX backend](../../../backends/mlx/README.md));
the first run compiles the Metal kernels.

CUDA export requires a CUDA toolkit; with its `nvcc` on `PATH`,
`./install_executorch.sh` installs CUDA-enabled PyTorch and the CUDA backend.
Each method runs in one CUDA partition, with float32 inputs and outputs and
FP32 (or BF16) model computation. Export produces `phonon.pte` and
`aoti_cuda_blob.ptd`; pass the latter through `--data_path` as shown above. If
the runner cannot load a method on a system with an older libstdc++, put the
exporting Python environment's libraries first:
`export LD_LIBRARY_PATH="$(python -c 'import sys; print(sys.prefix)')/lib:$LD_LIBRARY_PATH"`.

BF16 halves the weights, but small differences grow through the autoregressive
loop into a different take of the same text; FP32 matches the official runtime
sample for sample.

WebGPU runs Phonon in the browser on the ExecuTorch Wasm runtime; try it in
the Text to Speech demo of the
[ExecuTorch WebGPU Playground](https://executorch.ai/playground/). Export with
`--backend webgpu --webgpu-compat-ops --output-dir phonon_exports/webgpu`.
The WebGPU backend does not delegate `aten.elu` or 1-D transposed convolution
yet; `--webgpu-compat-ops` replaces them (and `native_layer_norm` and
`aten.silu`) with equivalent ops, so every method but the text embedding runs
on the GPU. Without it, those ops run on the CPU.

Phonon reads English only; spell out numbers and symbols. The tests
(`python -m pytest examples/models/phonon/tests -q`) compare the exported
programs of a small random model with a reference implementation of the
official runtime (see [NOTICE](NOTICE)). To check your own export, synthesize
"The quick brown fox jumps over the lazy dog." with `--temperature=0`: every
FP32 backend gives 37 frames (2.96 s), as the official runtime does.
