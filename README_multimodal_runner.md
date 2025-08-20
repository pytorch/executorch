# Multimodal Runner Example

This directory contains example scripts that demonstrate how to use the new multimodal runner introduced in commit `83749ae59dfdb898129c7dbae9a0b78564298326`.

## Files Created

1. **`run_multimodal_runner.cpp`** - Complete multimodal runner example with image support
2. **`CMakeLists_multimodal_example.txt`** - CMake configuration for the example
3. **`build_multimodal_example.sh`** - Build script using CMake

## Prerequisites

### 1. Build ExecutorTorch

Before using these examples, you need to build ExecutorTorch first:

```bash
# Install dependencies (Python packages)
./install_requirements.sh

# Build ExecutorTorch with LLM support
mkdir build
cd build

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release \
      -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
      -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
      -DEXECUTORCH_BUILD_TOKENIZERS=ON \
      ..

# Build (use -j$(nproc) for parallel build)
make -j$(nproc)

cd ..
```

### 2. Get Model and Tokenizer Files

You need:
- A `.pte` file (PyTorch Exported model for ExecutorTorch)
- A tokenizer file (supports various formats: `.model`, `.json`, etc.)

Based on the files in your directory, you already have:
- `*.pte` files (models)
- `tokenizer.model*` files

## Building the Example

```bash
# Make the build script executable
chmod +x build_multimodal_example.sh

# Run the build script
./build_multimodal_example.sh
```

## Running the Example

The multimodal runner supports multiple usage modes:

```bash
# Text only (creates test gradient image automatically)
./build_example/run_multimodal_runner <model.pte> <tokenizer_path>

# With PPM image
./build_example/run_multimodal_runner <model.pte> <tokenizer_path> image.ppm ppm

# With raw RGB data (224x224x3 bytes)
./build_example/run_multimodal_runner <model.pte> <tokenizer_path> image.raw raw

# Examples with your files
./build_example/run_multimodal_runner qwen3-0_6b_x.pte tokenizer.model
./build_example/run_multimodal_runner qwen3-0_6b_x.pte tokenizer.model image.ppm ppm
./build_example/run_multimodal_runner qwen3-0_6b_x.pte tokenizer.model photo.raw raw
```

## Code Overview

### Key Components Used

1. **MultimodalRunner** - Main runner class for multimodal models
2. **MultimodalInput** - Type-safe input wrapper for text and images
3. **GenerationConfig** - Configuration for text generation parameters
4. **Image struct** - Simple image container (width, height, channels, data)

### Basic Usage Pattern

```cpp
// 1. Load tokenizer
auto tokenizer = load_tokenizer("tokenizer.model");

// 2. Create multimodal runner
auto runner = create_multimodal_runner("model.pte", std::move(tokenizer));

// 3. Load the model
runner->load();

// 4. Create inputs (mix of text and images)
std::vector<MultimodalInput> inputs;
inputs.emplace_back(make_text_input("What do you see?"));
inputs.emplace_back(make_image_input(std::move(image)));

// 5. Configure generation
GenerationConfig config;
config.max_new_tokens = 100;
config.temperature = 0.7f;

// 6. Generate text with callbacks
runner->generate(inputs, config, token_callback, stats_callback);
```

## Supported Model Architectures

According to the multimodal runner documentation, it supports **EarlyFusion** models only:

### Supported:
- LLaVA (Large Language and Vision Assistant)
- CLIP-based vision-language models
- Gemma3 4B multimodal variant
- Voxtral (Audio-text models)
- Custom EarlyFusion models

### Not Supported:
- DeepFusion models (like Llama 3.2 Vision with cross-attention layers)

## Image Formats

The multimodal runner supports:
- **PPM P6 format** - Standard image format
- **Raw RGB data** - Raw byte data (assumes 224x224x3)
- **Test images** - Generated gradient pattern for testing

For production use, consider integrating with:
- OpenCV for comprehensive image format support
- STB image library for lightweight image loading
- Custom image preprocessing pipelines

## Troubleshooting

### Build Issues

1. **Headers not found**: Ensure ExecutorTorch is built first
2. **Libraries not found**: Check that build/lib exists and contains the required .so files
3. **Linking errors**: Verify all dependencies are built (extension_llm_runner, tokenizers, etc.)

### Runtime Issues

1. **Model load fails**: Check that the .pte file is compatible with multimodal runner
2. **Tokenizer load fails**: Verify tokenizer format is supported (SentencePiece, HF JSON, TikToken, BPE)
3. **Generation fails**: Check model compatibility and input format

### Model Compatibility

Ensure your model uses an EarlyFusion architecture. If using a DeepFusion model, you'll need to use a different inference approach or convert to an EarlyFusion format.

## Additional Resources

- ExecutorTorch documentation: [Getting Started](https://pytorch.org/executorch/stable/)
- Multimodal Runner source: `extension/llm/runner/multimodal_runner.h`
- LLM examples: `examples/llm_manual/`
- Full README: `extension/llm/runner/README.md`