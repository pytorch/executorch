# Whisper Runner

This directory hosts a lightweight C++ helper that drives Whisper models
exported to ExecuTorch. The `AsrRunner` owns the `Module` instance that
wraps a bundled `.pte` program and optional `.ptd` weight file, loads the
`encoder` and `text_decoder` methods, and exposes a `transcribe()` loop that
streams decoded text pieces through a callback.

The runner assumes:
- `model.pte` contains both Whisper encoder and decoder entry points named
  `encoder` and `text_decoder`.
- (Optional) Depending on export configurations, model weights can be optionally stored in a companion
  `model.ptd`. The runner will load the file if present.
- A tokenizer JSON compatible with the ExecuTorch tokenizers shim is available.

Audio preprocessing is not part of the runner itself. To transform raw audio
into the mel features expected by the encoder, reuse the pattern in
`examples/models/voxtral/multimodal.cpp`, which loads a `preprocessor.pte`
module to generate the spectrogram tensor.

## Build

Currently we have CUDA and Metal build support.

For CPU:
```
make whisper-cpu
```

For CUDA:
```
make whisper-cuda
```

For Metal:
```
make whisper-metal
```

## Usage

### Export Whisper Model

Use [Optimum-ExecuTorch](https://github.com/huggingface/optimum-executorch) to export a Whisper model from Hugging Face:

#### CUDA backend:

```bash
optimum-cli export executorch \
    --model openai/whisper-small \
    --task automatic-speech-recognition \
    --recipe cuda \
    --dtype bfloat16 \
    --device cuda \
    --output_dir ./
```

This command generates:
- `model.pte` — Compiled Whisper model
- `aoti_cuda_blob.ptd` — Weight data file for CUDA backend

#### Metal backend:

```bash
optimum-cli export executorch \
    --model openai/whisper-small \
    --task automatic-speech-recognition \
    --recipe metal \
    --dtype bfloat16 \
    --output_dir ./
```

This command generates:
- `model.pte` — Compiled Whisper model
- `aoti_metal_blob.ptd` — Weight data file for Metal backend

### Preprocessor

Export a preprocessor to convert raw audio to mel-spectrograms:

```bash
# Use --feature_size 128 for whisper-large-v3 and whisper-large-v3-turbo
python -m executorch.extension.audio.mel_spectrogram \
    --feature_size 80 \
    --stack_output \
    --max_audio_len 300 \
    --output_file whisper_preprocessor.pte
```

### Quantization

Export quantized models to reduce size and improve performance (Not enabled for Metal yet):

```bash
# 4-bit tile packed quantization for encoder
optimum-cli export executorch \
    --model openai/whisper-small \
    --task automatic-speech-recognition \
    --recipe cuda \
    --dtype bfloat16 \
    --device cuda \
    --qlinear 4w \
    --qlinear_encoder 4w \
    --qlinear_packing_format tile_packed_to_4d \
    --qlinear_encoder_packing_format tile_packed_to_4d \
    --output_dir ./
```


### Download Tokenizer

Download the tokenizer files required for inference according to your model version:

**For Whisper Small:**
```bash
curl -L https://huggingface.co/openai/whisper-small/resolve/main/tokenizer.json -o tokenizer.json
curl -L https://huggingface.co/openai/whisper-small/resolve/main/tokenizer_config.json -o tokenizer_config.json
curl -L https://huggingface.co/openai/whisper-small/resolve/main/special_tokens_map.json -o special_tokens_map.json
```

**For Whisper Large v2:**
```bash
curl -L https://huggingface.co/openai/whisper-large-v2/resolve/main/tokenizer.json -o tokenizer.json
curl -L https://huggingface.co/openai/whisper-large-v2/resolve/main/tokenizer_config.json -o tokenizer_config.json
curl -L https://huggingface.co/openai/whisper-large-v2/resolve/main/special_tokens_map.json -o special_tokens_map.json
```

### Prepare Audio

Generate test audio or use an existing WAV file. The model expects 16kHz mono audio.

```bash
# Generate sample audio using librispeech dataset
python -c "from datasets import load_dataset; import soundfile as sf; sample = load_dataset('distil-whisper/librispeech_long', 'clean', split='validation')[0]['audio']; sf.write('output.wav', sample['array'][:sample['sampling_rate']*30], sample['sampling_rate'])"
```

### Run Inference

After building the runner (see [Build](#build) section), execute it with the exported model and audio:

#### CUDA backend:

```bash
# Set library path for CUDA dependencies
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

# Run the Whisper runner
cmake-out/examples/models/whisper/whisper_runner \
    --model_path model.pte \
    --data_path aoti_cuda_blob.ptd \
    --tokenizer_path ./ \
    --audio_path output.wav \
    --processor_path whisper_preprocessor.pte \
    --temperature 0
```

#### Metal backend:

```bash
# Run the Whisper runner
cmake-out/examples/models/whisper/whisper_runner \
    --model_path model.pte \
    --data_path aoti_metal_blob.ptd \
    --tokenizer_path ./ \
    --audio_path output.wav \
    --processor_path whisper_preprocessor.pte \
    --temperature 0
```
