# Whisper Runner

This directory hosts a lightweight C++ helper that drives Whisper models
exported to ExecuTorch. The `WhisperRunner` owns the `Module` instance that
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
cmake -G Ninja \
  -B cmake-out/examples/models/whisper \
  -S examples/models/whisper
cmake --build cmake-out/examples/models/whisper -j
```

The build produces a static library named `whisper_runner`. Link it into your
application together with the standard ExecuTorch runtime libraries and the
tokenizer target (`tokenizers::tokenizers`).

## Usage

```cpp
#include <executorch/examples/models/whisper/runner.h>
#include <executorch/extension/tensor/tensor_ptr.h>

using example::WhisperRunner;
using example::WhisperTranscribeConfig;

WhisperRunner runner("model.pte", "model.ptd", "tokenizer.json");
ET_CHECK_OK(runner.load());

// `features` is the mel spectrogram tensor produced by the preprocessor.
executorch::aten::Tensor features = load_features_somehow();

WhisperTranscribeConfig config;
config.max_new_tokens = 128; // stop after 128 generated tokens
config.temperature = 0.7f;  // optional: enable stochastic sampling

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
and override `WhisperTranscribeConfig::eos_token_ids` when the model exposes
custom termination ids.
