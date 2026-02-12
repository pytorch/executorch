#include "audio_stream.h"

#include <portaudio.h>
#include <cstring>
#include <iostream>
#include <mutex>

namespace parakeet {

class PortAudioStream : public AudioStream {
 public:
  explicit PortAudioStream(const AudioStreamConfig& config)
      : config_(config), stream_(nullptr), callback_(nullptr) {}

  ~PortAudioStream() override {
    close();
  }

  bool open(int device_index = -1) override {
    if (stream_ != nullptr) {
      std::cerr << "Stream already open" << std::endl;
      return false;
    }

    PaError err = Pa_Initialize();
    if (err != paNoError) {
      std::cerr << "PortAudio init failed: " << Pa_GetErrorText(err)
                << std::endl;
      return false;
    }

    PaStreamParameters input_params;
    std::memset(&input_params, 0, sizeof(input_params));

    if (device_index == -1) {
      input_params.device = Pa_GetDefaultInputDevice();
    } else {
      input_params.device = device_index;
    }

    if (input_params.device == paNoDevice) {
      std::cerr << "No default input device" << std::endl;
      Pa_Terminate();
      return false;
    }

    const PaDeviceInfo* device_info = Pa_GetDeviceInfo(input_params.device);
    std::cout << "Using input device: " << device_info->name << std::endl;

    input_params.channelCount = config_.channels;
    input_params.sampleFormat = paFloat32;
    input_params.suggestedLatency =
        device_info->defaultLowInputLatency;
    input_params.hostApiSpecificStreamInfo = nullptr;

    err = Pa_OpenStream(
        &stream_,
        &input_params,
        nullptr, // no output
        config_.sample_rate,
        config_.frames_per_buffer,
        paClipOff,
        &PortAudioStream::pa_callback,
        this);

    if (err != paNoError) {
      std::cerr << "Failed to open stream: " << Pa_GetErrorText(err)
                << std::endl;
      Pa_Terminate();
      return false;
    }

    return true;
  }

  bool start() override {
    if (stream_ == nullptr) {
      std::cerr << "Stream not open" << std::endl;
      return false;
    }

    PaError err = Pa_StartStream(stream_);
    if (err != paNoError) {
      std::cerr << "Failed to start stream: " << Pa_GetErrorText(err)
                << std::endl;
      return false;
    }

    return true;
  }

  bool stop() override {
    if (stream_ == nullptr) {
      return true;
    }

    PaError err = Pa_StopStream(stream_);
    if (err != paNoError) {
      std::cerr << "Failed to stop stream: " << Pa_GetErrorText(err)
                << std::endl;
      return false;
    }

    return true;
  }

  void close() override {
    if (stream_ != nullptr) {
      Pa_CloseStream(stream_);
      stream_ = nullptr;
    }
    Pa_Terminate();
  }

  bool is_active() const override {
    if (stream_ == nullptr) {
      return false;
    }
    return Pa_IsStreamActive(stream_) == 1;
  }

  void set_callback(AudioCallback callback) override {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    callback_ = callback;
  }

  const AudioStreamConfig& get_config() const override {
    return config_;
  }

 private:
  static int pa_callback(
      const void* input_buffer,
      void* output_buffer,
      unsigned long frames_per_buffer,
      const PaStreamCallbackTimeInfo* time_info,
      PaStreamCallbackFlags status_flags,
      void* user_data) {
    auto* stream = static_cast<PortAudioStream*>(user_data);

    if (input_buffer != nullptr) {
      std::lock_guard<std::mutex> lock(stream->callback_mutex_);
      if (stream->callback_) {
        const float* samples = static_cast<const float*>(input_buffer);
        stream->callback_(samples, frames_per_buffer);
      }
    }

    return paContinue;
  }

  AudioStreamConfig config_;
  PaStream* stream_;
  AudioCallback callback_;
  std::mutex callback_mutex_;
};

std::unique_ptr<AudioStream> create_audio_stream(
    const AudioStreamConfig& config) {
  return std::make_unique<PortAudioStream>(config);
}

std::vector<std::string> AudioStream::list_devices() {
  std::vector<std::string> devices;

  PaError err = Pa_Initialize();
  if (err != paNoError) {
    std::cerr << "PortAudio init failed: " << Pa_GetErrorText(err)
              << std::endl;
    return devices;
  }

  int num_devices = Pa_GetDeviceCount();
  if (num_devices < 0) {
    std::cerr << "Failed to get device count" << std::endl;
    Pa_Terminate();
    return devices;
  }

  for (int i = 0; i < num_devices; i++) {
    const PaDeviceInfo* device_info = Pa_GetDeviceInfo(i);
    if (device_info->maxInputChannels > 0) {
      devices.push_back(
          std::string("[") + std::to_string(i) + "] " + device_info->name);
    }
  }

  Pa_Terminate();
  return devices;
}

} // namespace parakeet
