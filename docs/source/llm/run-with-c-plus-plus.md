# Running LLMs with C++

This guide explains how to use ExecuTorch's C++ runner library to run LLM models that have been exported to the `.pte` format. The runner library provides a high-level API for text generation with LLMs, handling tokenization, inference, and token generation.

## Prerequisites

Before you begin, make sure you have:

1. A model exported to `.pte` format using the `export_llm` API as described in [Exporting popular LLMs out of the box](export-llm.md) or [Exporting custom LLMs](export-custom-llm.md).
   - Please also see [Model Metadata](#model-metadata) section for important metadata to be serialized into `.pte`.
2. A tokenizer file compatible with your model
   - For HuggingFace tokenizers, this is a JSON file `tokenizer.json`
   - For SentencePiece tokenizers, this is a `tokenizer.model` file and normally lives alongside the weights file
3. CMake and a C++ compiler installed
   - CMake version 3.29 or higher
   - g++ or clang compiler

## Model Metadata

The metadata includes several important configuration parameters to be included during export step, which will be used by the runner library:

1. **`enable_dynamic_shape`**: Whether the model supports dynamic input shapes
2. **`max_seq_len`**: Maximum sequence length the model can handle
3. **`max_context_len`**: Maximum context length for KV cache
4. **`use_kv_cache`**: Whether the model uses KV cache for efficient generation
6. **`get_bos_id`**: Beginning-of-sequence token ID
7. **`get_eos_ids`**: End-of-sequence token IDs

### Adding Metadata During Export

To ensure your model has the necessary metadata, you can specify it during export using the `metadata` parameter in the export configuration:

```python
# export_llm
python -m extension.llm.export.export_llm \
  --config path/to/config.yaml \
  +base.metadata='{"get_bos_id":128000, "get_eos_ids":[128009, 128001], "get_max_context_len":4096}'
```
## Building the Runner Library

The ExecuTorch LLM runner library can be built using CMake. To integrate it into your project:

1. Add ExecuTorch as a dependency in your CMake project
2. Enable the required components (extension_module, extension_tensor, etc.)
3. Link your application against the `extension_llm_runner` library

Here's a simplified example of the CMake configuration:

```cmake
# Enable required components
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_MODULE ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_TENSOR ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER ON)

# Add ExecuTorch as a dependency
add_subdirectory(executorch)

# Link against the LLM runner library
target_link_libraries(your_app PRIVATE extension_llm_runner)
```

## Building the Llama Runner

ExecuTorch provides a complete example of a C++ runner for Llama models in the [`examples/models/llama`](https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md#step-3-run-on-your-computer-to-validate) directory. This runner demonstrates how to use the LLM runner library to run Llama models exported to the `.pte` format.

Please note that this runner library is not limited to Llama models and can be used with any text-only decoder-only LLM model that has been exported to the `.pte`.

## Basic Usage Example

Here's a simplified example of using the runner:

```cpp
#include <executorch/extension/llm/runner/text_llm_runner.h>

using namespace executorch::extension::llm;

int main() {
  // Load tokenizer and create runner
  auto tokenizer = load_tokenizer("path/to/tokenizer.json", nullptr, std::nullopt, 0, 0);
  auto runner = create_text_llm_runner("path/to/model.pte", std::move(tokenizer));

  // Load the model
  runner->load();

  // Configure generation
  GenerationConfig config;
  config.max_new_tokens = 100;
  config.temperature = 0.8f;

  // Generate text with streaming output
  runner->generate("Hello, world!", config,
    [](const std::string& token) { std::cout << token << std::flush; },
    nullptr);

  return 0;
}
```

## The Runner API Architecture

The ExecuTorch LLM runner library is designed with a modular architecture that separates concerns between different components of the text generation pipeline.

### IRunner Interface

The `IRunner` interface (`irunner.h`) defines the core functionality for LLM text generation. This interface serves as the primary abstraction for interacting with LLM models:

```cpp
class IRunner {
public:
  virtual ~IRunner() = default;
  virtual bool is_loaded() const = 0;
  virtual runtime::Error load() = 0;
  virtual runtime::Error generate(...) = 0;
  virtual runtime::Error generate_from_pos(...) = 0;
  virtual void stop() = 0;
};
```

Let's examine each method in detail:

```c++
bool is_loaded() const
```

Checks if the model and all necessary resources have been loaded into memory and are ready for inference. This method is useful for verifying the runner's state before attempting to generate text.

```c++
runtime::Error load()
```

Loads the model and prepares it for inference. This includes:
- Loading the model weights from the `.pte` file
- Initializing any necessary buffers or caches
- Preparing the execution environment

This method should be called before any generation attempts. It returns an `Error` object indicating success or failure.

```c++
runtime::Error generate(
   const std::string& prompt,
   const GenerationConfig& config,
   std::function<void(const std::string&)> token_callback,
   std::function<void(const Stats&)> stats_callback)
```
The primary method for text generation. It takes:

- `prompt`: The input text to generate from
- `config`: Configuration parameters controlling the generation process
- `token_callback`: A callback function that receives each generated token as a string
- `stats_callback`: A callback function that receives performance statistics after generation completes

The token callback is called for each token as it's generated, allowing for streaming output. The stats callback provides detailed performance metrics after generation completes.

```c++
runtime::Error generate_from_pos(
   const std::string& prompt,
   int64_t start_pos,
   const GenerationConfig& config,
   std::function<void(const std::string&)> token_callback,
   std::function<void(const Stats&)> stats_callback)
```

An advanced version of `generate()` that allows starting generation from a specific position in the KV cache. This is useful for continuing generation from a previous state.

```c++
void stop()
```

Immediately stops the generation loop. This is typically called from another thread to interrupt a long-running generation.

### GenerationConfig Structure

The `GenerationConfig` struct controls various aspects of the generation process:

```cpp
struct GenerationConfig {
  bool echo = true;                // Whether to echo the input prompt in the output
  int32_t max_new_tokens = -1;     // Maximum number of new tokens to generate
  bool warming = false;            // Whether this is a warmup run
  int32_t seq_len = -1;            // Maximum number of total tokens
  float temperature = 0.8f;        // Temperature for sampling
  int32_t num_bos = 0;             // Number of BOS tokens to add
  int32_t num_eos = 0;             // Number of EOS tokens to add

  // Helper method to resolve the actual max_new_tokens based on constraints
  int32_t resolve_max_new_tokens(int32_t max_context_len, int32_t num_prompt_tokens) const;
};
```

The `resolve_max_new_tokens` method handles the logic of determining how many tokens can be generated based on:
- The model's maximum context length
- The number of tokens in the prompt
- The user-specified maximum sequence length and maximum new tokens

### Implementation Components

The runner library consists of several specialized components that work together:

#### TextLLMRunner

The main implementation of the `IRunner` interface that orchestrates the text generation process. It manages:

1. Tokenization of input text
2. Prefilling the KV cache with prompt tokens
3. Generating new tokens one by one
4. Collecting performance statistics

#### TextPrefiller

Responsible for processing the initial prompt tokens and filling the KV cache. Key features:

- Efficiently processes large prompts
- Handles dynamic sequence lengths
- Supports parallel prefilling for performance optimization

#### TextTokenGenerator

Generates new tokens one by one in an autoregressive manner. It:

- Manages the token generation loop
- Applies temperature-based sampling
- Detects end-of-sequence conditions
- Streams tokens as they're generated

#### TextDecoderRunner

Interfaces with the ExecuTorch Module to run the model forward pass. It:

- Manages inputs and outputs to the model
- Handles KV cache updates
- Converts logits to tokens via sampling

## Tokenizer Support

The runner library supports multiple tokenizer formats through a unified interface:

```cpp
std::unique_ptr<tokenizers::Tokenizer> tokenizer = load_tokenizer(
    tokenizer_path,  // Path to tokenizer file
    nullptr,         // Optional special tokens
    std::nullopt,    // Optional regex pattern (for TikToken)
    0,               // BOS token index
    0                // EOS token index
);
```

Supported tokenizer formats include:

1. **HuggingFace Tokenizers**: JSON format tokenizers
2. **SentencePiece**: `.model` format tokenizers
3. **TikToken**: BPE tokenizers
4. **Llama2c**: BPE tokenizers in the Llama2.c format

For custom tokenizers, you can find implementations in the [meta-pytorch/tokenizers](https://github.com/meta-pytorch/tokenizers) repository.


## Other APIs

### Model Warmup

For more accurate timing and optimal performance, you should perform a warmup run before actual inference:

```cpp
runner->warmup("Hello world", 10);  // Generate 10 tokens as warmup
```

During warmup:

1. A special `GenerationConfig` is created with:
   - `echo = false`: The prompt is not included in the output
   - `warming = true`: Indicates this is a warmup run
   - `max_new_tokens`: Set to the specified number of tokens to generate

2. The model runs through the entire generation pipeline:
   - Loading the model (if not already loaded)
   - Tokenizing the prompt
   - Prefilling the KV cache
   - Generating the specified number of tokens

3. Special behavior during warmup:
   - Tokens are not displayed to the console
   - The runner logs "Doing a warmup run..." and "Warmup run finished!" messages

4. After warmup:
   - The `Stats` object is reset to clear performance metrics
   - The model remains loaded and ready for actual inference

Warmup is particularly important for accurate benchmarking as the first inference often includes one-time initialization costs that would skew performance measurements.

### Memory Usage Monitoring

You can monitor memory usage with the `Stats` object:

```cpp
std::cout << "RSS after loading: " << get_rss_bytes() / 1024.0 / 1024.0 << " MiB" << std::endl;
```
