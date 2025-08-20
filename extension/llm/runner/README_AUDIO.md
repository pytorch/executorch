# Multimodal Runner with Audio Support

This document describes the audio support that has been added to the ExecuteTorch multimodal runner.

## Overview

The multimodal runner now supports three types of input modalities:
- **Text**: String inputs for prompts and questions
- **Image**: Visual data in NCHW format (existing functionality)
- **Audio**: Raw audio data with sample rate and channel information (new functionality)

## Audio Support Implementation

### New Components

1. **Audio Struct** (`audio.h`): Represents audio data with the following fields:
   - `data`: Vector of float values representing raw audio samples
   - `sample_rate`: Audio sample rate in Hz (e.g., 16000, 44100)
   - `channels`: Number of audio channels (1 for mono, 2 for stereo)
   - `num_samples`: Total number of audio samples

2. **Enhanced MultimodalInput** (`multimodal_input.h`): Now supports audio inputs
   - Added `AUDIO` to the `Type` enum
   - Added audio constructors and getter methods
   - Added convenience factory functions: `make_audio_input()`

### Usage Example

```cpp
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/audio.h>

// Create audio data
Audio audio;
audio.data = {0.1f, 0.2f, -0.1f, 0.3f}; // Raw audio samples
audio.sample_rate = 16000; // 16kHz
audio.channels = 1; // Mono
audio.num_samples = 4;

// Create multimodal inputs
std::vector<MultimodalInput> inputs;
inputs.emplace_back(make_text_input("Transcribe this audio:"));
inputs.emplace_back(make_audio_input(std::move(audio)));

// Use with multimodal runner
runner->generate(inputs, config, token_callback, stats_callback);
```

## Example Script

The `multimodal_example.cpp` script demonstrates how to use the multimodal runner with audio support:

### Features
- Load audio data from .pt tensor files
- Combine text prompts with audio inputs
- Configurable generation parameters
- Command-line interface for easy testing

### Building the Example

```bash
# Make the build script executable (if not already)
chmod +x build_multimodal_example.sh

# Build the example
./build_multimodal_example.sh
```

### Running the Example

```bash
# Basic usage with text only
./build_example/multimodal_example model.pte tokenizer.json --text "Hello, world!"

# With audio input from a .pt tensor file
./build_example/multimodal_example model.pte tokenizer.json \
  --text "Transcribe this audio:" \
  --audio audio_data.pt

# Show help
./build_example/multimodal_example --help
```

### Command Line Options

- `--audio <path>`: Path to a .pt tensor file containing audio data
- `--text <text>`: Text input for the model (default: "Process this input:")
- `--help`: Show help message

## Audio Data Format

The example script expects audio data in .pt tensor files containing raw float32 values. For production use, you may need to implement proper PyTorch tensor parsing or use a different audio loading mechanism.

### Preparing Audio Data

To create compatible audio tensor files, you can use PyTorch:

```python
import torch
import torchaudio

# Load audio file
waveform, sample_rate = torchaudio.load("audio.wav")

# Ensure mono and correct sample rate
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

# Resample if needed
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = resampler(waveform)

# Save as tensor
torch.save(waveform.squeeze(), "audio_data.pt")
```

## Integration with Existing Models

To use audio inputs with your multimodal model, ensure that:

1. Your model supports audio embeddings alongside text and image inputs
2. The model's encoder can process audio tensors appropriately
3. The multimodal prefiller handles audio inputs correctly

## Notes

- The audio support is designed to be extensible and follows the same patterns as the existing image support
- Audio data is currently expected as raw float values, but this can be adapted for different audio formats
- The sample rate and channel information should match your model's expectations
- For production use, consider adding audio format validation and conversion utilities