# Summary

This example demonstrates how to export and run Mistral's [Voxtral](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) audio multimodal model locally on ExecuTorch.

# Exporting the model
To export the model, we use [Optimum ExecuTorch](https://github.com/huggingface/optimum-executorch), a repo that enables exporting models straight from the source - from HuggingFace's Transformers repo.

## Setting up Optimum ExecuTorch
Install through pip package:
```
pip install optimum-excecutorch
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
optimum-cli export executorch
  --model "mistralai/Voxtral-Mini-3B-2507"
  --task "multimodal-text-to-text"
  --recipe "xnnpack"
  --use_custom_sdpa
  --use_custom_kv_cache
  --qlinear 8da4w
  --qembedding 4w
  --output_dir="voxtral
```

This exports Voxtral with XNNPack backend acceleration and 4-bit weight/8-bit activation linear quantization.

# [Optional] Exporting the audio preprocessor
The exported model takes in a mel spectrogram input tensor as its audio inputs.
We provide a simple way to transform raw audio data into a mel spectrogram by exporting a version of Voxtral's audio preprocessor used directly by Transformers.

```
python -m executorch.extension.audio.mel_spectrogram --feature_size 128 --output_file voxtral_preprocessor.pte
```

# Running the model
To run the model, we will use the Voxtral runner, which utilizes ExecuTorch's MultiModal runner API.
The Voxtral runner will do the following things:
1. [Optional] Pass the raw audio tensor into exported preprocessor to produce a mel spectrogram tensor.
2. [If starting directly with an already processed audio input tensor, starts here] Formats the inputs to the multimodal runner (metadata tokens, audio tokens, text tokens, etc.).
3. Feed the formatted inputs to the multimodal modal runner.

## Building the multimodal runner
```
# Build and install ExecuTorch
cmake --preset llm -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=cmake-out -DEXECUTORCH_ENABLE_LOGGING=ON && cmake --build cmake-out -j16 --target install --config Release

# Build and install Voxtral runner
cmake -DCMAKE_INSTALL_PREFIX=cmake-out -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -Bcmake-out/examples/models/voxtral examples/models/voxtral && cmake --build cmake-out/examples/models/voxtral -j16 --config Release
```

## Running the model
You can download the `tekken.json` tokenizer from [Voxtral's HuggingFace repo](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507).
```
./cmake-out/examples/models/voxtral/voxtral_runner
  --model_path voxtral/model.pte
  --tokenizer_path path/to/tekken.json
  --prompt "What can you tell me about this audio?"
  --audio_path ~/models/voxtral/audio_input.bin
```

You can easily produce an `.bin` for the audio input in Python like this:
```
# t = some torch.Tensor
with open("tensor.bin", "wb") as f:
    f.write(t.numpy().tobytes())
```
