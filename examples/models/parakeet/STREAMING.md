# Parakeet Streaming Transcription Runner

Real-time speech recognition using Parakeet TDT with microphone input.

## Overview

The streaming runner captures audio from a microphone (e.g., Logitech camera/headset) and performs real-time transcription using chunked processing. Audio is processed in configurable chunks with historical context for improved accuracy across chunk boundaries.

## Architecture

```
Microphone → AudioStream → Chunked Buffer → Preprocessor → Encoder → Decoder → Text
                              ↑
                       Left Context (10s)
```

**Key Components:**

1. **AudioStream**: PortAudio-based microphone interface with callback API
2. **StreamingTranscriber**: Manages chunked audio processing and decoder state
3. **Context Management**: Maintains left context window for seamless transcription

**Default Configuration:**
- Chunk size: 2 seconds
- Left context: 10 seconds (historical audio for better accuracy)
- Sample rate: 16kHz (matches model requirement)
- Channels: Mono

## Prerequisites

Install PortAudio:

```bash
# Linux
sudo apt-get install portaudio19-dev

# macOS
brew install portaudio
```

## Building

From the executorch root directory:

```bash
# CPU/XNNPACK build
make parakeet-cpu

# Metal build (macOS)
make parakeet-metal

# CUDA build (Linux)
make parakeet-cuda
```

The build will automatically detect PortAudio and build `parakeet_streaming_runner` if available.

## Usage

### List Available Microphones

```bash
./cmake-out/examples/models/parakeet/parakeet_streaming_runner \
  --list_devices
```

Example output:
```
Available audio input devices:
  [0] Built-in Microphone
  [1] Logitech C920 HD Pro Webcam
  [2] External USB Microphone
```

### Start Streaming Transcription

```bash
# Use default microphone
./cmake-out/examples/models/parakeet/parakeet_streaming_runner \
  --model_path examples/models/parakeet/parakeet_tdt_exports/model.pte \
  --tokenizer_path examples/models/parakeet/parakeet_tdt_exports/tokenizer.model

# Specify device (e.g., Logitech camera)
./cmake-out/examples/models/parakeet/parakeet_streaming_runner \
  --model_path examples/models/parakeet/parakeet_tdt_exports/model.pte \
  --tokenizer_path examples/models/parakeet/parakeet_tdt_exports/tokenizer.model \
  --device_index 1

# CUDA backend with custom chunk size
./cmake-out/examples/models/parakeet/parakeet_streaming_runner \
  --model_path examples/models/parakeet/parakeet_cuda/model.pte \
  --data_path examples/models/parakeet/parakeet_cuda/aoti_cuda_blob.ptd \
  --tokenizer_path examples/models/parakeet/parakeet_cuda/tokenizer.model \
  --chunk_seconds 1.5 \
  --left_context_seconds 8.0
```

### Command-Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model_path` | Path to .pte model file | `parakeet.pte` |
| `--tokenizer_path` | Path to tokenizer file | `tokenizer.model` |
| `--data_path` | Path to .ptd data file (CUDA only) | - |
| `--device_index` | Microphone device index (-1 = default) | -1 |
| `--list_devices` | List available devices and exit | false |
| `--chunk_seconds` | Audio chunk duration | 2.0 |
| `--left_context_seconds` | Left context duration | 10.0 |

## How It Works

### Streaming Pipeline

1. **Audio Capture**: PortAudio callback receives audio frames (512 samples/chunk)
2. **Buffering**: Frames accumulate until chunk size reached
3. **Context Window**: Include left context from previous audio
4. **Processing**:
   - Preprocessor: Convert audio → mel spectrogram
   - Encoder: Mel → acoustic features
   - Decoder: Greedy decoding with TDT (Token-and-Duration Transducer)
5. **State Management**: Decoder LSTM state persists across chunks
6. **Output**: Transcription updates in real-time

### Context Management

Left context improves accuracy by providing historical audio information:

```
Chunk N-1: [...........] (processed, stored in buffer)
Chunk N:   [===========] (current chunk to process)
           ↑
    Include 10s of context from previous audio
```

### Decoder State Persistence

The decoder RNN state (hidden and cell states) persists across chunks, allowing the model to maintain linguistic context throughout the entire audio stream.

## API Design

The streaming runner provides a clean separation of concerns:

### AudioStream Interface

```cpp
class AudioStream {
  bool open(int device_index = -1);
  bool start();
  bool stop();
  void set_callback(AudioCallback callback);
  static std::vector<std::string> list_devices();
};
```

### StreamingTranscriber

```cpp
class StreamingTranscriber {
  bool initialize();
  void process_audio_chunk(const float* samples, size_t num_samples);
  std::string get_current_text() const;
};
```

### Usage Pattern

```cpp
// Create components
auto audio_stream = create_audio_stream(config);
StreamingTranscriber transcriber(model, tokenizer, ...);
transcriber.initialize();

// Connect audio to transcriber
audio_stream->set_callback([&](const float* samples, size_t n) {
  transcriber.process_audio_chunk(samples, n);
});

// Start streaming
audio_stream->open();
audio_stream->start();

// Get results anytime
std::string text = transcriber.get_current_text();
```

## Performance Considerations

**Latency**: ~2 seconds (chunk duration) + processing time

**Chunk Size Trade-offs**:
- Smaller chunks (0.5-1s): Lower latency, less context, lower accuracy
- Larger chunks (2-4s): Higher latency, more context, better accuracy

**Left Context**:
- More context (10-20s): Better accuracy, higher memory usage
- Less context (5s): Lower memory, potential accuracy loss at boundaries

**Threading**: Audio callback runs on audio thread; chunk processing on separate threads to avoid blocking

## Troubleshooting

**No audio devices found**: Check microphone permissions and connections

**PortAudio errors**: Verify installation with `pkg-config --modversion portaudio-2.0`

**High latency**: Reduce `--chunk_seconds` (may impact accuracy)

**Poor accuracy**: Increase `--left_context_seconds` or `--chunk_seconds`

**Choppy transcription**: Model processing slower than real-time; try CUDA/Metal backend or quantization

## Example: Logitech Camera Setup

```bash
# 1. List devices to find your Logitech camera
./parakeet_streaming_runner --list_devices

# Output shows:
# [2] Logitech C920 HD Pro Webcam

# 2. Run with that device
./parakeet_streaming_runner \
  --model_path model.pte \
  --tokenizer_path tokenizer.model \
  --device_index 2
```

## References

- Parakeet TDT model: [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- Streaming configuration based on NeMo's `speech_to_text_streaming_infer_rnnt.py`
- PortAudio: [portaudio.com](http://www.portaudio.com/)
