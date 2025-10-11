# Summary

This example demonstrates how to export and run Mistral's [Voxtral](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) audio multimodal model locally on ExecuTorch.

# Exporting the model
To export the model, we use [Optimum ExecuTorch](https://github.com/huggingface/optimum-executorch), a repo that enables exporting models straight from the source - from HuggingFace's Transformers repo.

## Setting up Optimum ExecuTorch
Install through pip package:
```
pip install optimum-executorch
```

Or install from source:
```
git clone https://github.com/huggingface/optimum-executorch.git
cd optimum-executorch
python install_dev.py
```

## Using the export CLI
We export Voxtral using the Optimum CLI, which will export `model.pte` to the `voxtral` output directory:
```
optimum-cli export executorch \
  --model "mistralai/Voxtral-Mini-3B-2507" \
  --task "multimodal-text-to-text" \
  --recipe "xnnpack" \
  --use_custom_sdpa \
  --use_custom_kv_cache \
  --max_seq_len 2048 \
  --qlinear 8da4w \
  --qlinear_encoder 8da4w \
  --qembedding 4w \
  --output_dir="voxtral"
```

This exports Voxtral with XNNPack backend acceleration and 4-bit weight/8-bit activation linear quantization.

## CUDA Support
If your environment has CUDA support, you can enable the runner to run on CUDA for improved performance. Follow the export and runtime commands below:

**Note:** We are currently working on quantization support for CUDA. Currently, only bfloat16 dtype is supported for CUDA execution.

### Exporting with CUDA
```
optimum-cli export executorch \
  --model "mistralai/Voxtral-Mini-3B-2507" \
  --task "multimodal-text-to-text" \
  --recipe "cuda" \
  --dtype bfloat16 \
  --device cuda \
  --max_seq_len 1024 \
  --output_dir="voxtral"
```

This will generate:
- `model.pte` - The exported model
- `aoti_cuda_blob.ptd` - The CUDA kernel blob required for runtime

See the "Building the multimodal runner" section below for instructions on building with CUDA support, and the "Running the model" section for runtime instructions.

# Running the model
To run the model, we will use the Voxtral runner, which utilizes ExecuTorch's MultiModal runner API.
The Voxtral runner will do the following things:

- Audio Input:
   - Option A:  Pass raw audio data from a `.wav` file into the exported preprocessor to produce a mel spectrogram tensor.
   - Option B:  If starting directly with an already processed audio input tensor (preprocessed mel spectrogram), format the inputs to the multimodal runner (metadata tokens, audio tokens, text tokens, etc.).
- Feed the formatted inputs to the multimodal modal runner.


## Exporting the audio preprocessor
The exported model takes in a mel spectrogram input tensor as its audio inputs.
We provide a simple way to transform raw audio data into a mel spectrogram by exporting a version of Voxtral's audio preprocessor used directly by Transformers.

```
# Export a preprocessor that can handle audio up to 5 mins (300s).
python -m executorch.extension.audio.mel_spectrogram --feature_size 128 --stack_output --max_audio_len 300 --output_file voxtral_preprocessor.pte
```

## Building the multimodal runner

### Building for CPU (XNNPack)
```
# Build and install ExecuTorch
cmake --preset llm -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=cmake-out -DEXECUTORCH_ENABLE_LOGGING=ON && cmake --build cmake-out -j16 --target install --config Release

# Build and install Voxtral runner
cmake -DCMAKE_INSTALL_PREFIX=cmake-out -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -Bcmake-out/examples/models/voxtral examples/models/voxtral && cmake --build cmake-out/examples/models/voxtral -j16 --config Release
```

### Building for CUDA
```
# Install ExecuTorch with CUDA support
CMAKE_ARGS="-DEXECUTORCH_BUILD_CUDA=ON" ./install_executorch.sh

# Build the multimodal runner with CUDA
cmake --preset llm \
      -DEXECUTORCH_BUILD_CUDA=ON \
      -DCMAKE_INSTALL_PREFIX=cmake-out \
      -DCMAKE_BUILD_TYPE=Release \
      -Bcmake-out -S.
cmake --build cmake-out -j16 --target install --config Release

cmake -DEXECUTORCH_BUILD_CUDA=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -Sexamples/models/voxtral \
      -Bcmake-out/examples/models/voxtral/
cmake --build cmake-out/examples/models/voxtral --target voxtral_runner --config Release
```

## Running the model
You can download the `tekken.json` tokenizer from [Voxtral's HuggingFace repo](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507).

### Running with raw audio (.wav file)
For raw audio files (`.wav`), you must provide a preprocessor to convert the audio into mel spectrogram format:
```
./cmake-out/examples/models/voxtral/voxtral_runner \
  --model_path path/to/model.pte \
  --tokenizer_path path/to/tekken.json \
  --prompt "What can you tell me about this audio?" \
  --audio_path path/to/audio_input.wav \
  --processor_path path/to/voxtral_preprocessor.pte
```

### Running with preprocessed audio (.bin file)
If you already have a preprocessed mel spectrogram saved as a `.bin` file, you can skip the preprocessor:
```
./cmake-out/examples/models/voxtral/voxtral_runner \
  --model_path path/to/model.pte \
  --tokenizer_path path/to/tekken.json \
  --prompt "What can you tell me about this audio?" \
  --audio_path path/to/preprocessed_audio.bin
```


**For CUDA:** Add the `--data_path` argument to provide the CUDA kernel blob to the commands above:
```
  --data_path path/to/aoti_cuda_blob.ptd
```

Example output:
```
The speaker in this audio seems to be talking about their concerns about a device called the model or maybe they're just talking about the model in general. They mention that the model was trained with the speaker for inference, which suggests that
 the model was trained based on the speaker's data or instructions. They also mention that the volume is quite small, which could imply that the speaker is trying to control the volume of the model's output, likely because they are concerned about how loud the model's responses might
PyTorchObserver {"prompt_tokens":388,"generated_tokens":99,"model_load_start_ms":0,"model_load_end_ms":0,"inference_start_ms":1756351346381,"inference_end_ms":1756351362602,"prompt_eval_end_ms":1756351351435,"first_token_ms":1756351351435,"aggregate_sampling_time_ms":99,"SCALING_FACTOR_UNITS_PER_SECOND":1000}
I 00:00:24.036773 executorch:stats.h:104]       Prompt Tokens: 388    Generated Tokens: 99
I 00:00:24.036800 executorch:stats.h:110]       Model Load Time:                0.000000 (seconds)
I 00:00:24.036805 executorch:stats.h:117]       Total inference time:           16.221000 (seconds)              Rate:  6.103200 (tokens/second)
I 00:00:24.036815 executorch:stats.h:127]               Prompt evaluation:      5.054000 (seconds)               Rate:  76.770875 (tokens/second)
I 00:00:24.036819 executorch:stats.h:136]               Generated 99 tokens:    11.167000 (seconds)              Rate:  8.865407 (tokens/second)
I 00:00:24.036822 executorch:stats.h:147]       Time to first generated token:  5.054000 (seconds)
I 00:00:24.036828 executorch:stats.h:153]       Sampling time over 487 tokens:  0.099000 (seconds)
```

You can easily produce an `.bin` for the audio input in Python like this:
```
# t = some torch.Tensor
with open("tensor.bin", "wb") as f:
    f.write(t.numpy().tobytes())
```

You can also produce raw audio file as follows (for Option A):

```
ffmpeg -i audio.mp3 -f f32le -acodec pcm_f32le -ar 16000 audio_input.bin
```
