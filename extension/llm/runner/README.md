# LLM Runner Framework for ExecuTorch

This directory contains the LLM Runner framework for ExecuTorch, providing high-level C++ APIs for running Large Language Models with both text-only and multimodal capabilities.

## Overview

The LLM Runner framework provides two main runner classes:

- **TextLLMRunner**: For text-only language models (e.g., Llama, GPT, etc.)
- **MultimodalRunner**: For multimodal models that can process text, images, and audio (e.g., LLaVA, CLIP-based models)

Both runners are built on a modular architecture with dependency injection, providing clean separation of concerns and efficient resource management.

## Architecture Overview

## MultimodalRunner Architecture

The MultimodalRunner supports mixed inputs (text, images, audio) and generates text outputs:

```
MultimodalRunner Supported Model Architecture:
┌─────────────────────────────────────────────────────────────────────────┐
│                        Multimodal LLM Architecture                      │
└─────────────────────────────────────────────────────────────────────────┘
   Input: std::vector<MultimodalInput>
          ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
          │     Image       │  │     Audio       │  │      Text       │
          │    [224x        │  │    [16kHz       │  │     "What"      │
          │     224x3]      │  │     audio]      │  │                 │
          └─────────────────┘  └─────────────────┘  └─────────────────┘
                   │                    │                    │
                   ▼                    ▼                    ▼
          ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ ◄─┐
          │     Encoder     │  │     Encoder     │  │ Text Tokenizer  │   │
          │   (Vision)      │  │   (Audio)       │  │   & Embedding   │   │
          │                 │  │                 │  │                 │   │
          │ pixels → embed  │  │ waveform→embed  │  │ tokens → embed  │   │
          └─────────────────┘  └─────────────────┘  └─────────────────┘   │
                   │                    │                    │            │
                   ▼                    ▼                    ▼            │
          ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
          │     [D_emb]     │  │     [D_emb]     │  │     [D_emb]     │   │
          │    Embedding    │  │    Embedding    │  │    Embedding    │   │
          └─────────────────┘  └─────────────────┘  └─────────────────┘   │
                   │                    │                    │            │
                   └────────────────────┼────────────────────┘            │
                                        │                                 │
                                        ▼                                 │
                   ┌─────────────────────────────┐                        │
                   │      Text Decoder Block     │                        │
                   │    (Transformer Layers)     │                        │
                   │                             │                        │
                   │  ┌─────────────────────┐    │                        │
                   │  │   Self-Attention    │    │                        │
                   │  │   + Feed Forward    │    │                        │
                   │  │   (with KV Cache)   │    │                        │
                   │  └─────────────────────┘    │                        │
                   │           │                 │                        │
                   │           ▼                 │                        │
                   │    Token Generation         │                        │
                   │    (pos_ tracking)          │                        │
                   └─────────────────────────────┘                        │
                                  │───────────────────────────────────────┘
                                  │          (Autoregressive)
                                  ▼
                         ┌─────────────────┐
                         │  Generated Text │
                         │ "This image     │
                         │  shows a cat    │
                         │  sitting..."    │
                         └─────────────────┘
```

## Key Features

### TextLLMRunner
- **Text-only processing**: Optimized for pure language models
- **Efficient tokenization**: Support for multiple tokenizer formats
- **KV cache management**: Automatic position tracking for efficient inference
- **Streaming generation**: Token-by-token callbacks for real-time output
- **Configuration-driven**: Comprehensive control via `GenerationConfig`

### MultimodalRunner
- **Mixed input support**: Process text, images, and audio in any order
- **Type-safe inputs**: `MultimodalInput` class with compile-time type checking
- **Modular encoders**: Separate processing pipelines for different modalities
- **Unified generation**: Single API for complex multimodal workflows
- **Extensible design**: Easy to add support for new modalities

## Quick Start

### TextLLMRunner Example

```cpp
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>

int main() {
    // Load tokenizer and create runner
    auto tokenizer = load_tokenizer("tokenizer.bin");
    auto runner = create_text_llm_runner("model.pte", std::move(tokenizer));

    // Configure generation
    GenerationConfig config;
    config.max_new_tokens = 100;
    config.temperature = 0.7f;
    config.echo = true;

    // Set up callbacks
    auto token_callback = [](const std::string& token) {
        std::cout << token << std::flush;
    };

    // Generate text
    auto error = runner->generate(
        "Hello, how are you?",  // prompt
        config,                 // configuration
        token_callback         // token callback
    );

    return error == executorch::runtime::Error::Ok ? 0 : 1;
}
```

### MultimodalRunner Example

```cpp
#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>

int main() {
    // Load tokenizer and create runner
    auto tokenizer = load_tokenizer("tokenizer.bin");
    auto runner = create_multimodal_runner("model.pte", std::move(tokenizer));

    // Create multimodal inputs
    std::vector<MultimodalInput> inputs;
    inputs.emplace_back(make_text_input("What do you see in this image?"));

    // Load and add image
    Image image = load_image("photo.jpg");  // Your image loading function
    inputs.emplace_back(make_image_input(std::move(image)));

    // Configure generation
    GenerationConfig config;
    config.max_new_tokens = 150;
    config.temperature = 0.7f;
    config.echo = true;

    // Set up callbacks
    auto token_callback = [](const std::string& token) {
        std::cout << token << std::flush;
    };

    auto stats_callback = [](const Stats& stats) {
        std::cout << "\nGenerated " << stats.num_generated_tokens << " tokens" << std::endl;
    };

    // Generate text
    auto error = runner->generate(inputs, config, token_callback, stats_callback);

    return error == executorch::runtime::Error::Ok ? 0 : 1;
}
```

## Python API

The LLM Runner framework provides Python bindings for easy integration with Python applications. The Python API mirrors the C++ interface while providing Pythonic convenience features like torch tensor support and Hugging Face compatibility.

### Installation

Build the Python bindings as part of the ExecuTorch build:

```bash
# Build from source with Python bindings enabled:
# In executorch root directory
bash install_executorch.sh
```

### Quick Start Examples

#### Basic Multimodal Generation

```python
from executorch.extension.llm.runner import (
    GenerationConfig, MultimodalRunner, 
    make_text_input, make_image_input, make_audio_input
)
import torch

# Create a multimodal runner
runner = MultimodalRunner(
    model_path="/path/to/model.pte",
    tokenizer_path="/path/to/tokenizer.bin"
)

# Create multimodal inputs
inputs = []
inputs.append(make_text_input("What do you see in this image?"))

# Add image from torch tensor (supports both CHW and HWC formats)
image_tensor = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)  # CHW format
inputs.append(make_image_input(image_tensor))

# Configure generation
config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.7,
    echo=False
)

# Generate with streaming output
def token_callback(token: str):
    print(token, end='', flush=True)

def stats_callback(stats):
    print(f"\n[Stats] Generated {stats.num_generated_tokens} tokens")
    inference_time = stats.inference_end_ms - stats.inference_start_ms
    if inference_time > 0:
        tokens_per_sec = stats.num_generated_tokens * 1000 / inference_time
        print(f"[Stats] Speed: {tokens_per_sec:.1f} tokens/sec")

runner.generate(inputs, config, token_callback, stats_callback)
```

#### Working with Different Input Types

```python
from executorch.extension.llm.runner import (
    MultimodalRunner, GenerationConfig,
    make_text_input, make_token_input, make_image_input, 
    make_audio_input, make_raw_audio_input
)
import torch

runner = MultimodalRunner("model.pte", "tokenizer.bin")

# 1. Text input
text_input = make_text_input("Analyze this multimodal content:")

# 2. Pre-tokenized input (useful for chat templates)
token_ids = [1, 15043, 445, 2420]  # Example token IDs
token_input = make_token_input(token_ids)

# 3. Image input from torch tensor
# Supports multiple formats: (H,W,C), (C,H,W), (1,H,W,C), (1,C,H,W)
image_hwc = torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8)  # HWC
image_input = make_image_input(image_hwc)

# Float tensors also supported for normalized images
image_float = torch.rand(3, 224, 224, dtype=torch.float32)  # CHW, normalized
image_input_float = make_image_input(image_float)

# 4. Preprocessed audio input (e.g., mel spectrograms)
audio_features = torch.rand(1, 80, 100, dtype=torch.float32)  # (batch, n_bins, n_frames)
audio_input = make_audio_input(audio_features)

# 5. Raw audio input (for models with built-in audio processing)
raw_audio = torch.randint(0, 255, (1, 1, 16000), dtype=torch.uint8)  # (batch, channels, samples)
raw_audio_input = make_raw_audio_input(raw_audio)

# Combine inputs and generate
inputs = [text_input, image_input, audio_input]
config = GenerationConfig(max_new_tokens=50, temperature=0.8)
response = runner.generate_text(inputs, config)
print(f"Response: {response}")
```

#### Hugging Face Integration

```python
from executorch.extension.llm.runner import MultimodalRunner, GenerationConfig
from transformers import AutoProcessor
from PIL import Image
import torch

# Load HF processor for your model
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Create runner
runner = MultimodalRunner("llava_model.pte", "tokenizer.bin")

# Process inputs with HF processor
image = Image.open("photo.jpg")
conversation = [
    {"role": "user", "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image"}
    ]}
]

# Apply chat template and process
prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
inputs_hf = processor(prompt, image, return_tensors="pt")

# Generate using HF inputs directly
config = GenerationConfig(max_new_tokens=100, temperature=0.7)
runner.generate_hf(
    inputs_hf, 
    config, 
    image_token_id=processor.tokenizer.convert_tokens_to_ids("<image>"),
    token_callback=lambda token: print(token, end='', flush=True)
)
```

#### Chat Session with State Management

```python
from executorch.extension.llm.runner import MultimodalRunner, GenerationConfig, make_text_input

class ChatSession:
    def __init__(self, model_path: str, tokenizer_path: str):
        self.runner = MultimodalRunner(model_path, tokenizer_path)
        self.config = GenerationConfig(max_new_tokens=150, temperature=0.7, echo=False)
        
    def send_message(self, message: str) -> str:
        """Send a message and get response"""
        inputs = [make_text_input(message)]
        response = self.runner.generate_text(inputs, self.config)
        return response
        
    def send_multimodal(self, text: str, image_tensor: torch.Tensor) -> str:
        """Send text + image and get response"""
        inputs = [
            make_text_input(text),
            make_image_input(image_tensor)
        ]
        response = self.runner.generate_text(inputs, self.config)
        return response
        
    def reset_conversation(self):
        """Reset the conversation state"""
        self.runner.reset()

# Usage
chat = ChatSession("model.pte", "tokenizer.bin")
print(chat.send_message("Hello! How are you?"))

# Continue conversation (KV cache maintains context)
print(chat.send_message("What's the weather like?"))

# Reset when starting new conversation
chat.reset_conversation()
```

### Python API Classes

#### GenerationConfig
```python
from executorch.extension.llm.runner import GenerationConfig

# Create with defaults
config = GenerationConfig()

# Or specify parameters
config = GenerationConfig(
    max_new_tokens=100,    # Maximum tokens to generate (-1 = auto)
    temperature=0.8,       # Sampling temperature (0.0 = deterministic)
    echo=True,            # Echo input prompt in output
    seq_len=2048,         # Maximum sequence length (-1 = auto)
    num_bos=0,            # Number of BOS tokens
    num_eos=0             # Number of EOS tokens
)

# Modify after creation
config.temperature = 0.5
config.max_new_tokens = 50
```

#### MultimodalInput Types
```python
from executorch.extension.llm.runner import (
    MultimodalInput, make_text_input, make_token_input, 
    make_image_input, make_audio_input
)

# Text input
text_input = make_text_input("Hello, world!")
print(text_input.is_text())  # True
print(text_input.get_text())  # "Hello, world!"

# Token input (pre-tokenized)
token_input = make_token_input([1, 2, 3, 4])
print(token_input.is_tokens())  # True
print(token_input.get_tokens())  # [1, 2, 3, 4]

# Image input from torch tensor
import torch
image_tensor = torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8)
image_input = make_image_input(image_tensor)
print(image_input.is_image())  # True
image = image_input.get_image()
print(f"Image: {image.width}x{image.height}x{image.channels}")

# Check input types safely
if text_input.is_text():
    text = text_input.get_text()
elif text_input.is_image():
    image = text_input.get_image()
```

#### Stats and Performance Monitoring
```python
def detailed_stats_callback(stats):
    """Comprehensive stats monitoring"""
    print(f"\n=== Generation Statistics ===")
    print(f"Prompt tokens: {stats.num_prompt_tokens}")
    print(f"Generated tokens: {stats.num_generated_tokens}")
    
    # Timing breakdown
    model_load_time = stats.model_load_end_ms - stats.model_load_start_ms
    if model_load_time > 0:
        print(f"Model load time: {model_load_time}ms")
    
    inference_time = stats.inference_end_ms - stats.inference_start_ms
    if inference_time > 0:
        print(f"Total inference time: {inference_time}ms")
        
        # Calculate throughput
        tokens_per_sec = stats.num_generated_tokens * 1000 / inference_time
        print(f"Generation speed: {tokens_per_sec:.1f} tokens/sec")
    
    # Time to first token
    if stats.first_token_ms > stats.inference_start_ms:
        ttft = stats.first_token_ms - stats.inference_start_ms
        print(f"Time to first token: {ttft}ms")
    
    # Export to JSON for logging
    json_stats = stats.to_json_string()
    print(f"JSON stats: {json_stats}")

# Use in generation
runner.generate(inputs, config, token_callback, detailed_stats_callback)
```

### Error Handling

```python
from executorch.extension.llm.runner import MultimodalRunner, GenerationConfig
import torch

try:
    runner = MultimodalRunner("model.pte", "tokenizer.bin")
    
    # Invalid image tensor will raise RuntimeError
    invalid_image = torch.rand(2, 224, 224, 3)  # Wrong number of dimensions
    inputs = [make_image_input(invalid_image)]
    
    config = GenerationConfig(max_new_tokens=50)
    runner.generate_text(inputs, config)
    
except RuntimeError as e:
    print(f"Generation failed: {e}")
    
except FileNotFoundError as e:
    print(f"Model or tokenizer file not found: {e}")
```

For more C++ API documentation and implementation details, see the [Core Components](#core-components) section below.

## Core Components

### Component Architecture

```

       ┌─────────────────┐
       │     IRunner     │
       │   <<interface>> │
       │                 │
       │ + is_loaded()   │
       │ + load()        │
       │ + generate()    │
       │ + stop()        │
       └─────────────────┘
              △
              │
              │ implements
              │
              │
              │
              │
       ┌──────┴──────────┐          ┌─────────────────┐
       │ TextLLMRunner   │          │MultimodalRunner │
       │                 │          │                 │
       │ - tokenizer_    │          │ - tokenizer_    │
 ┌─────┼ - module_       │          │ - module_       ┼─────┐
 │ ┌───┼ - stats_        │          │ - stats_        ┼───┐ │
 │ │ ┌─┼ - metadata_     │          │ - metadata_     ┼─┐ │ │
 │ │ │ │ - temperature_  │          │ - pos_          │ │ │ │
 │ │ │ └─────────────────┘          └─────────────────┘ │ │ │
 │ │ │                                                  │ │ │
 │ │ │                                                  │ │ │
 │ │ │                                                  │ │ │
 │ │ │               ┌─────────────────┐                │ │ │
 │ │ │               │TextTokenGenerat-│                │ │ │
 │ │ │               │or               │                │ │ │
 │ │ │               │                 │                │ │ │
 │ │ │               │ - tokenizer_*   │                │ │ │
 │ │ │  consists     │ - text_decoder_ │    consists    │ │ │
 │ │ └──────────────►│   runner_       │◄───────────────┘ │ │
 │ │                 │ - eos_ids_      │                  │ │
 │ │                 │ - use_kv_cache_ │                  │ │
 │ │                 │ - stats_*       │                  │ │
 │ │                 │                 │                  │ │
 │ │consists         │ + generate()    │         consists │ │
 │ │                 └────────┬────────┘                  │ │
 │ │           ┌──────────────┴───────────────┐           │ │
 │ │           ▼            uses              ▼           │ │
 │ │   ┌─────────────────┐          ┌─────────────────┐   │ │
 │ │   │TextDecoderRunner│          │MultimodalDecode-│   │ │
 │ │   │                 │          │rRunner          │   │ │
 │ │   │ - module_*      │ extends  │ - module_*      │   │ │
 │ └──►│ - should_stop_  │◄─────────┼ - should_stop_  │◄──┘ │
 │     │                 │          │                 │     │
 │     │ + step()        │          │ + step()        │     │
 │     │ + logits_to_    │          │ + logits_to_    │     │
 │     │   token()       │          │   token()       │     │
 │     └─────────────────┘          └─────────────────┘     │
 │             ▲                             ▲              │
 │             │           uses              │              │
 │consists     ├─────────────────────────────┤              │
 │     ┌───────┴─────────┐                   │              │
 │     │  TextPrefiller  │                   │      consists│
 │     │                 │          ┌────────┴────────┐     │
 │     │ - text_decoder_ │          │ MultimodalPrefi-│     │
 │     │   runner_       │          │ller             │     │
 └────►│ - use_kv_cache_ │          │ - module_*      │     │
       │ - enable_       │          │                 │◄────┘
       │   parallel_     │          │ + prefill()     │
       │   prefill_      │          │ + logits_to_    │
       │                 │          │   token()       │
       │ + prefill()     │          └─────────────────┘
       ├─────────────────┘
```

### 1. Tokenizer
**Purpose**: Converts between text and token IDs

**Supported Formats**:
- HF JSON (Hugging Face tokenizer format)
- TikToken (OpenAI's tokenizer format)
- SentencePiece (Google's tokenizer format)
- BPE (Byte-pair encoding tokenizer)

**Key Methods**:
```cpp
virtual Result<std::vector<uint64_t>> encode(const std::string& text, int8_t bos = 1, int8_t eos = 0) = 0;
virtual Result<std::string> decode(uint64_t prev_token, uint64_t token) = 0;
virtual uint64_t bos_tok() const = 0;
virtual uint64_t eos_tok() const = 0;
```

### 2. TextDecoderRunner
**Purpose**: Executes the transformer decoder part of the model

**Key Responsibilities**:
- Executes transformer decoder layers
- Manages KV cache during inference
- Handles both prefill and decode phases
- Provides low-level model execution interface

### 3. TextPrefiller
**Purpose**: Handles the prefill phase for text inputs

**Key Features**:
- Parallel token processing for efficiency
- KV cache management
- Batch processing support
- Integration with tokenizer

**Configuration**:
```cpp
TextPrefiller(
    TextDecoderRunner* text_decoder_runner,
    bool use_kv_cache,
    bool enable_parallel_prefill,
    int64_t max_seq_len
);
```

### 4. ImagePrefiller (MultimodalRunner only)
**Purpose**: Processes image inputs through vision encoders

**Key Features**:
- Vision encoder integration
- Pixel data to embedding conversion
- Multiple image format support
- KV cache integration

**Image Format**:
```cpp
struct Image {
    int32_t width;
    int32_t height;
    int32_t channels;
    std::vector<uint8_t> data;  // Raw pixel data
};
```

### 5. TextTokenGenerator
**Purpose**: Handles autoregressive token generation

**Key Features**:
- Temperature-based sampling
- EOS token detection
- Token-by-token callbacks
- Performance statistics tracking

**Usage**:
```cpp
int64_t num_tokens = text_token_generator->generate(
    {start_token},           // Initial tokens
    current_pos,             // Starting position
    max_new_tokens,          // Maximum tokens to generate
    temperature,             // Sampling temperature
    token_callback           // Callback for each token
);
```

### 6. GenerationConfig
**Purpose**: Comprehensive configuration for text generation

**Key Parameters**:
```cpp
struct GenerationConfig {
    int32_t max_new_tokens = -1;    // Max tokens to generate (-1 = use available)
    int32_t seq_len = 1024;         // Total sequence length
    float temperature = 0.8f;       // Sampling temperature
    bool echo = true;               // Echo input prompt
    int8_t num_bos = 1;            // Number of BOS tokens
    int8_t num_eos = 1;            // Number of EOS tokens
    bool warming = false;           // Warmup run flag
};
```

### 7. MultimodalInput (MultimodalRunner only)
**Purpose**: Type-safe wrapper for mixed input types

**Key Features**:
- `std::variant<std::string, Image>` internally
- Type-safe access methods
- Exception-based and safe access patterns
- Move semantics for efficiency

**API**:
```cpp
// Type checking
bool is_text() const;
bool is_image() const;

// Direct access (throws on type mismatch)
const std::string& get_text() const;
const Image& get_image() const;

// Safe access (returns nullptr on type mismatch)
const std::string* try_get_text() const;
const Image* try_get_image() const;

// Factory functions
MultimodalInput make_text_input(const std::string& text);
MultimodalInput make_image_input(Image&& image);
```

## Helper Functions

The framework provides utility functions in `llm_runner_helper.h`:

### load_tokenizer()
```cpp
std::unique_ptr<tokenizers::Tokenizer> load_tokenizer(
    const std::string& tokenizer_path,
    std::unique_ptr<std::vector<std::string>> special_tokens = nullptr,
    std::optional<std::string> pattern = std::nullopt,
    size_t bos_token_index = 0,
    size_t eos_token_index = 1
);
```

### create_text_llm_runner()
```cpp
std::unique_ptr<TextLLMRunner> create_text_llm_runner(
    const std::string& model_path,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::optional<const std::string> data_path = std::nullopt,
    float temperature = -1.0f
);
```

### create_multimodal_runner()
```cpp
std::unique_ptr<MultimodalRunner> create_multimodal_runner(
    const std::string& model_path,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::optional<const std::string> data_path = std::nullopt,
    float temperature = 0.8f
);
```

### get_llm_metadata()
```cpp
std::unordered_map<std::string, int64_t> get_llm_metadata(
    tokenizers::Tokenizer* tokenizer,
    Module* module
);
```

## Configuration and Tuning

### Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_new_tokens` | `int32_t` | `-1` | Maximum new tokens to generate (-1 = use available context) |
| `seq_len` | `int32_t` | `1024` | Total sequence length including prompt |
| `temperature` | `float` | `0.8f` | Sampling temperature (0.0 = deterministic, 1.0+ = creative) |
| `echo` | `bool` | `true` | Whether to echo the input prompt |
| `num_bos` | `int8_t` | `1` | Number of beginning-of-sequence tokens |
| `num_eos` | `int8_t` | `1` | Number of end-of-sequence tokens |
| `warming` | `bool` | `false` | Whether this is a warmup run |

### Performance Tuning

**Memory Optimization**:
- Use KV cache for efficient autoregressive generation
- Enable parallel prefill for faster prompt processing
- Set appropriate `seq_len` based on available memory

**Sampling Strategies**:
- Low temperature (0.1-0.3) for factual, deterministic output
- High temperature (0.7-1.0) for creative, diverse output
- Set `max_new_tokens` to prevent runaway generation

**Monitoring**:
```cpp
auto stats_callback = [](const Stats& stats) {
    std::cout << "Model load time: "
              << (stats.model_load_end_ms - stats.model_load_start_ms) << "ms" << std::endl;
    std::cout << "Inference time: "
              << (stats.inference_end_ms - stats.inference_start_ms) << "ms" << std::endl;
    std::cout << "Tokens/second: " << stats.tokens_per_second() << std::endl;
};
```

## Supported Models

### TextLLMRunner
- **Llama family**: Llama 2, Llama 3, Code Llama
- **GPT models**: GPT-2, GPT-3.5, GPT-4 (compatible architectures)
- **Phi models**: Phi-3-mini and variants
- **Custom models**: Any transformer-based text generation model

### MultimodalRunner

**Note**: The MultimodalRunner currently supports **EarlyFusion** model architectures only. EarlyFusion is a type of fused model architecture where pretrained encoder(s) are combined with a pretrained decoder (LLM) at the model input and not in internal layers. This is a popular architecture for multimodal models, with a full overview available in [The Evolution of Multimodal Model Architectures](https://arxiv.org/abs/2405.17927). This module works both for decoders in which the encoder tokens are inside the vocab and outside the vocab.

**Supported EarlyFusion Models**:
- **LLaVA**: Large Language and Vision Assistant
- **CLIP-based models**: Contrastive Language-Image Pre-training
- **Gemma3 4B**: Multimodal variant with vision capabilities
- **Voxtral**: Audio-text multimodal models
- **Custom EarlyFusion models**: Any model with separate encoders that fuse at the input level

**DeepFusion Models (Not Currently Supported)**:
DeepFusion is another popular model architecture type where a pretrained encoder is combined with a pretrained decoder (LLM) in the internal decoder layers. A common deep fusion architecture is to fuse the encoder input into the decoder with interspersed cross-attention layers. DeepFusion models are currently out of scope because they require significant model definition rewrites to work with torch.export.

**Examples of DeepFusion models (not supported)**:
- **Llama 3.2 Vision**: Uses cross-attention layers for vision-text fusion
- **Other cross-attention based multimodal models**

For DeepFusion support, consider using the model's native inference framework or wait for future ExecuTorch updates that may include DeepFusion architecture support.

## Building and Integration

### CMake Integration
```cmake
find_package(executorch REQUIRED)
target_link_libraries(your_target
    executorch::extension_llm_runner
    executorch::extension_module
)
```

### Required Headers
```cpp
// For TextLLMRunner
#include <executorch/extension/llm/runner/text_llm_runner.h>

// For MultimodalRunner
#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <executorch/extension/llm/runner/multimodal_input.h>

// Helper functions
#include <executorch/extension/llm/runner/llm_runner_helper.h>

// Configuration
#include <executorch/extension/llm/runner/irunner.h>
```

## Advanced Usage

### Custom Sampling
```cpp
// Custom temperature per generation
GenerationConfig config;
config.temperature = 0.1f;  // Very deterministic
runner->generate(factual_prompt, config, callback);

config.temperature = 1.2f;  // Very creative
runner->generate(creative_prompt, config, callback);
```

### Memory Monitoring
```cpp
#include <executorch/extension/llm/runner/util.h>

auto stats_callback = [](const Stats& stats) {
    double rss_mb = get_rss_bytes() / 1024.0 / 1024.0;
    std::cout << "RSS: " << rss_mb << " MiB" << std::endl;
};
```
