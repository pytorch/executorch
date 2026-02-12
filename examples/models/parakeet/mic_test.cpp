// Minimal mic test - records 3 seconds and prints audio stats
#include <portaudio.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

static std::vector<float> g_audio;

static int callback(
    const void* input,
    void* /*output*/,
    unsigned long frames,
    const PaStreamCallbackTimeInfo* /*time_info*/,
    PaStreamCallbackFlags flags,
    void* /*user_data*/) {
  if (input) {
    const float* samples = static_cast<const float*>(input);
    g_audio.insert(g_audio.end(), samples, samples + frames);
  }
  return paContinue;
}

int main() {
  Pa_Initialize();

  int num_devices = Pa_GetDeviceCount();
  std::cout << "PortAudio devices (" << num_devices << "):" << std::endl;
  for (int i = 0; i < num_devices; i++) {
    const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
    if (info->maxInputChannels > 0) {
      std::cout << "  [" << i << "] " << info->name
                << " (in=" << info->maxInputChannels
                << " rate=" << info->defaultSampleRate << ")" << std::endl;
    }
  }

  PaDeviceIndex dev = Pa_GetDefaultInputDevice();
  std::cout << "\nDefault input device: " << dev << std::endl;

  const PaDeviceInfo* info = Pa_GetDeviceInfo(dev);
  std::cout << "Device: " << info->name << std::endl;
  std::cout << "Default sample rate: " << info->defaultSampleRate << std::endl;

  // Try device's native sample rate first
  double sample_rate = info->defaultSampleRate;

  PaStreamParameters params;
  std::memset(&params, 0, sizeof(params));
  params.device = dev;
  params.channelCount = 1;
  params.sampleFormat = paFloat32;
  params.suggestedLatency = info->defaultLowInputLatency;

  // Check if 16000 is supported
  PaError supported = Pa_IsFormatSupported(&params, nullptr, 16000);
  std::cout << "16000 Hz supported: " << (supported == paFormatIsSupported ? "yes" : "no") << std::endl;

  supported = Pa_IsFormatSupported(&params, nullptr, sample_rate);
  std::cout << sample_rate << " Hz supported: " << (supported == paFormatIsSupported ? "yes" : "no") << std::endl;

  PaStream* stream = nullptr;
  PaError err = Pa_OpenStream(
      &stream, &params, nullptr, sample_rate, 512, paClipOff, callback, nullptr);
  if (err != paNoError) {
    std::cerr << "Open failed: " << Pa_GetErrorText(err) << std::endl;
    // Retry with 16000
    sample_rate = 16000;
    err = Pa_OpenStream(
        &stream, &params, nullptr, sample_rate, 512, paClipOff, callback, nullptr);
    if (err != paNoError) {
      std::cerr << "Open at 16000 also failed: " << Pa_GetErrorText(err) << std::endl;
      Pa_Terminate();
      return 1;
    }
  }

  std::cout << "\nRecording at " << sample_rate << " Hz for 3 seconds..." << std::endl;
  Pa_StartStream(stream);

  Pa_Sleep(3000);

  Pa_StopStream(stream);
  Pa_CloseStream(stream);
  Pa_Terminate();

  std::cout << "Collected " << g_audio.size() << " samples ("
            << g_audio.size() / sample_rate << "s)" << std::endl;

  float peak = 0;
  double sum_sq = 0;
  for (float s : g_audio) {
    float a = std::abs(s);
    if (a > peak) peak = a;
    sum_sq += (double)s * s;
  }
  double rms = g_audio.empty() ? 0 : std::sqrt(sum_sq / g_audio.size());

  std::cout << "peak=" << peak << " rms=" << rms << std::endl;

  if (peak < 0.001f) {
    std::cout << "SILENCE - mic is not working" << std::endl;
  } else {
    std::cout << "OK - mic is capturing audio" << std::endl;
  }

  return 0;
}
