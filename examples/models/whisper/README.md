# Whisper Runner

This directory hosts a lightweight C++ helper that drives Whisper models
exported to ExecuTorch. The `AsrRunner` owns the `Module` instance that
wraps a bundled `.pte` program and optional `.ptd` weight file, loads the
`encoder` and `text_decoder` methods, and exposes a `transcribe()` loop that
streams decoded text pieces through a callback.

The runner assumes:
- `model.pte` contains both Whisper encoder and decoder entry points named
  `encoder` and `text_decoder`.
- External parameters (for example KV cache blocks) are stored in a companion
  `model.ptd`.
- A tokenizer JSON compatible with the ExecuTorch tokenizers shim is available.

Audio preprocessing is not part of the runner itself. To transform raw audio
into the mel features expected by the encoder, reuse the pattern in
`examples/models/voxtral/multimodal.cpp`, which loads a `preprocessor.pte`
module to generate the spectrogram tensor.

## Build

```bash
# Install ExecuTorch libraries:
cmake --preset llm -DEXECUTORCH_BUILD_CUDA=ON -DCMAKE_INSTALL_PREFIX=cmake-out -DCMAKE_BUILD_TYPE=Release . -Bcmake-out
cmake --build cmake-out -j$(nproc) --target install --config Release

# Build the runner:
cmake \
  -B cmake-out/examples/models/whisper \
  -S examples/models/whisper
cmake --build cmake-out/examples/models/whisper -j
```

The first cmake command build produces a static library named `extension_asr_runner`. The second cmake command links it into your
application together with the standard ExecuTorch runtime libraries and the
tokenizer target (`tokenizers::tokenizers`).

## Usage

```cpp
#include <executorch/extension/asr/runner.h>
#include <executorch/extension/tensor/tensor_ptr.h>

using executorch::extension::llm::AsrRunner;
using executorch::extension::llm::AsrTranscribeConfig;

AsrRunner runner("model.pte", "model.ptd", "tokenizer.json");
ET_CHECK_OK(runner.load());

// `features` is the mel spectrogram tensor produced by the preprocessor.
executorch::aten::Tensor features = load_features_somehow();

AsrTranscribeConfig config;
config.max_new_tokens = 128; // stop after 128 generated tokens
config.temperature = 0.7f;  // optional: enable stochastic sampling
config.decoder_start_token_id = 50257; // override the BOS token id

auto tokens_result = runner.transcribe(
    features,
    config,
    [](const std::string& piece) {
      std::cout << piece;
    });

if (!tokens_result.ok()) {
  ET_LOG(Error, "Transcription failed: %d", static_cast<int>(tokens_result.error()));
}
```

`transcribe()` returns the full token history (prompt + generated tokens) and
invokes the callback every time a new token is emitted. Provide a non-empty
`decoder_input_ids` vector if you want to seed the decoder with a custom prompt,
and override `AsrTranscribeConfig::eos_token_ids` when the model exposes
custom termination ids.
