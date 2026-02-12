#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace parakeet {

// Audio configuration for streaming
struct AudioStreamConfig {
  int32_t sample_rate = 16000;
  int32_t channels = 1;
  int32_t frames_per_buffer = 512;
};

// Callback invoked when audio data is available
// Parameters: audio samples (float), number of samples
using AudioCallback = std::function<void(const float*, size_t)>;

// Interface for audio stream from microphone
class AudioStream {
 public:
  virtual ~AudioStream() = default;

  // Open and start capturing audio from microphone
  // device_index: -1 for default device, or specific device ID
  virtual bool open(int device_index = -1) = 0;

  // Start streaming audio
  virtual bool start() = 0;

  // Stop streaming audio
  virtual bool stop() = 0;

  // Close the audio stream
  virtual void close() = 0;

  // Check if stream is active
  virtual bool is_active() const = 0;

  // Set callback for audio data
  virtual void set_callback(AudioCallback callback) = 0;

  // Get current configuration
  virtual const AudioStreamConfig& get_config() const = 0;

  // List available audio input devices
  static std::vector<std::string> list_devices();
};

// Factory function to create an audio stream
std::unique_ptr<AudioStream> create_audio_stream(
    const AudioStreamConfig& config);

} // namespace parakeet
