/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "reverb_audio.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace reverb {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSilenceDb = -50.0;

uint32_t read_u32(const uint8_t* p) {
  return p[0] | (p[1] << 8) | (p[2] << 16) |
      (static_cast<uint32_t>(p[3]) << 24);
}

uint16_t read_u16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

void write_u32(std::ofstream& out, uint32_t value) {
  const uint8_t bytes[4] = {
      static_cast<uint8_t>(value),
      static_cast<uint8_t>(value >> 8),
      static_cast<uint8_t>(value >> 16),
      static_cast<uint8_t>(value >> 24)};
  out.write(reinterpret_cast<const char*>(bytes), 4);
}

void write_u16(std::ofstream& out, uint16_t value) {
  const uint8_t bytes[2] = {
      static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8)};
  out.write(reinterpret_cast<const char*>(bytes), 2);
}

// The level of every whole 10 ms window, in dBFS.
std::vector<double> window_levels(const std::vector<double>& x, int window) {
  const size_t count = x.size() / window;
  std::vector<double> levels(count);
  for (size_t w = 0; w < count; ++w) {
    double energy = 0.0;
    for (int i = 0; i < window; ++i) {
      const double v = x[w * window + i];
      energy += v * v;
    }
    levels[w] = 20.0 * std::log10(std::sqrt(energy / window) + 1e-10);
  }
  return levels;
}

std::vector<size_t> loud_windows(const std::vector<double>& levels) {
  std::vector<size_t> loud;
  for (size_t i = 0; i < levels.size(); ++i) {
    if (levels[i] >= kSilenceDb) {
      loud.push_back(i);
    }
  }
  return loud;
}

// The modified Bessel function of the first kind, order zero.
double bessel_i0(double x) {
  double sum = 1.0;
  double term = 1.0;
  const double half_squared = x * x / 4.0;
  for (int k = 1; k < 200; ++k) {
    term *= half_squared / (static_cast<double>(k) * k);
    sum += term;
    if (term < sum * 1e-17) {
      break;
    }
  }
  return sum;
}

} // namespace

Audio read_wav(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  std::vector<uint8_t> data(
      (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (data.size() < 12 || std::memcmp(data.data(), "RIFF", 4) ||
      std::memcmp(data.data() + 8, "WAVE", 4)) {
    throw std::runtime_error(path + " is not a WAV file");
  }
  uint16_t format = 0, channels = 0, bits = 0;
  uint32_t rate = 0;
  const uint8_t* samples = nullptr;
  size_t samples_bytes = 0;
  size_t at = 12;
  while (at + 8 <= data.size()) {
    const uint32_t size = read_u32(data.data() + at + 4);
    const uint8_t* body = data.data() + at + 8;
    const size_t available = std::min<size_t>(size, data.size() - at - 8);
    if (!std::memcmp(data.data() + at, "fmt ", 4) && available >= 16) {
      format = read_u16(body);
      channels = read_u16(body + 2);
      rate = read_u32(body + 4);
      bits = read_u16(body + 14);
      if (format == 0xFFFE && available >= 26) {
        format = read_u16(body + 24); // WAVE_FORMAT_EXTENSIBLE sub-format
      }
    } else if (!std::memcmp(data.data() + at, "data", 4)) {
      samples = body;
      samples_bytes = available;
    }
    at += 8 + size + (size & 1);
  }
  const bool pcm =
      format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32);
  const bool floating = format == 3 && bits == 32;
  if (!samples || channels == 0 || rate == 0 || !(pcm || floating)) {
    throw std::runtime_error(
        path + ": expected PCM 8/16/24/32-bit or float32 WAV data");
  }
  const size_t width = bits / 8;
  const size_t frames = samples_bytes / (width * channels);
  Audio audio;
  audio.sample_rate = static_cast<int>(rate);
  audio.samples.resize(frames);
  for (size_t f = 0; f < frames; ++f) {
    double sum = 0.0;
    for (size_t c = 0; c < channels; ++c) {
      const uint8_t* p = samples + (f * channels + c) * width;
      double v = 0.0;
      if (floating) {
        float x;
        std::memcpy(&x, p, 4);
        v = x;
      } else if (bits == 8) {
        v = (static_cast<int>(p[0]) - 128) / 128.0;
      } else if (bits == 16) {
        v = static_cast<int16_t>(read_u16(p)) / 32768.0;
      } else if (bits == 24) {
        int32_t x = p[0] | (p[1] << 8) | (p[2] << 16);
        if (x & 0x800000) {
          x |= ~0xFFFFFF;
        }
        v = x / 8388608.0;
      } else {
        v = static_cast<int32_t>(read_u32(p)) / 2147483648.0;
      }
      sum += v;
    }
    audio.samples[f] = static_cast<float>(sum / channels);
  }
  return audio;
}

void write_wav(
    const std::string& path,
    const std::vector<float>& samples,
    int sample_rate) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("cannot write " + path);
  }
  const uint32_t data_bytes = static_cast<uint32_t>(samples.size() * 2);
  out.write("RIFF", 4);
  write_u32(out, 36 + data_bytes);
  out.write("WAVEfmt ", 8);
  write_u32(out, 16);
  write_u16(out, 1);
  write_u16(out, 1);
  write_u32(out, static_cast<uint32_t>(sample_rate));
  write_u32(out, static_cast<uint32_t>(sample_rate) * 2);
  write_u16(out, 2);
  write_u16(out, 16);
  out.write("data", 4);
  write_u32(out, data_bytes);
  for (float sample : samples) {
    const float clipped = std::max(-1.0f, std::min(1.0f, sample));
    write_u16(
        out,
        static_cast<uint16_t>(
            static_cast<int16_t>(std::lrint(clipped * 32767.0f))));
  }
  if (!out) {
    throw std::runtime_error("failed to write " + path);
  }
}

std::vector<float>
resample(const std::vector<float>& samples, int from_rate, int to_rate) {
  if (from_rate <= 0 || to_rate <= 0) {
    throw std::invalid_argument("sample rates must be positive");
  }
  if (from_rate == to_rate) {
    return samples;
  }
  // librosa's kaiser_best, as torchaudio parameterises it.
  constexpr int kWidth = 64;
  constexpr double kRolloff = 0.9475937167399596;
  constexpr double kBeta = 14.769656459379392;
  const int gcd = std::gcd(from_rate, to_rate);
  const int orig = from_rate / gcd;
  const int next = to_rate / gcd;
  const double base = std::min(orig, next) * kRolloff;
  const int width = static_cast<int>(std::ceil(kWidth * orig / base));
  const int taps = 2 * width + orig;

  // kernels[j][k] weighs input sample i * orig + k - width for output i * next
  // + j.
  std::vector<float> kernels(static_cast<size_t>(next) * taps);
  const double i0_beta = bessel_i0(kBeta);
  for (int j = 0; j < next; ++j) {
    for (int k = 0; k < taps; ++k) {
      double t = (-static_cast<double>(j) / next +
                  static_cast<double>(k - width) / orig) *
          base;
      t = std::max(
          -static_cast<double>(kWidth),
          std::min(static_cast<double>(kWidth), t));
      const double ratio = t / kWidth;
      const double window =
          bessel_i0(kBeta * std::sqrt(std::max(0.0, 1.0 - ratio * ratio))) /
          i0_beta;
      const double x = t * kPi;
      const double sinc = x == 0.0 ? 1.0 : std::sin(x) / x;
      kernels[static_cast<size_t>(j) * taps + k] =
          static_cast<float>(sinc * window * base / orig);
    }
  }

  const int64_t length = static_cast<int64_t>(samples.size());
  const int64_t target =
      (static_cast<int64_t>(next) * length + orig - 1) / orig;
  std::vector<float> out(static_cast<size_t>(target));
  for (int64_t n = 0; n < target; ++n) {
    const int64_t i = n / next;
    const int j = static_cast<int>(n % next);
    const float* kernel = kernels.data() + static_cast<size_t>(j) * taps;
    double acc = 0.0;
    for (int k = 0; k < taps; ++k) {
      const int64_t src = i * orig + k - width;
      if (src >= 0 && src < length) {
        acc += static_cast<double>(kernel[k]) * samples[src];
      }
    }
    out[n] = static_cast<float>(acc);
  }
  return out;
}

void raise_quiet(std::vector<float>& samples) {
  constexpr double kQuietDb = -40.0;
  constexpr double kLevelDb = -20.0;
  constexpr double kPeak = 0.99;
  if (samples.empty()) {
    return;
  }
  double energy = 0.0;
  double peak = 0.0;
  for (float v : samples) {
    energy += static_cast<double>(v) * v;
    peak = std::max(peak, static_cast<double>(std::fabs(v)));
  }
  const double rms = std::sqrt(energy / samples.size());
  if (rms == 0.0 || 20.0 * std::log10(rms) >= kQuietDb) {
    return;
  }
  const double gain =
      std::min(std::pow(10.0, kLevelDb / 20.0) / rms, kPeak / peak);
  for (float& v : samples) {
    v = static_cast<float>(v * gain);
  }
}

std::vector<float>
trim_silence(const std::vector<float>& samples, int sample_rate, float keep) {
  const int window = sample_rate / 100;
  if (window <= 0) {
    return samples;
  }
  const std::vector<double> x(samples.begin(), samples.end());
  const std::vector<double> levels = window_levels(x, window);
  const std::vector<size_t> loud = loud_windows(levels);
  if (loud.empty()) {
    return samples;
  }
  // Sounds: runs of loud windows separated by over 200 ms of silence.
  std::vector<std::pair<size_t, size_t>> sounds = {{loud[0], loud[0]}};
  for (size_t i = 1; i < loud.size(); ++i) {
    if (loud[i] - loud[i - 1] > 20) {
      sounds.push_back({loud[i], loud[i]});
    } else {
      sounds.back().second = loud[i];
    }
  }
  // A sound of at most 120 ms (12 loud windows) alone at an edge is a codec
  // burst rather than speech.
  const auto loud_in = [&](const std::pair<size_t, size_t>& sound) {
    return std::count_if(loud.begin(), loud.end(), [&](size_t w) {
      return w >= sound.first && w <= sound.second;
    });
  };
  size_t first = 0, last = sounds.size() - 1;
  if (sounds.size() > 1 && loud_in(sounds[first]) <= 12) {
    ++first;
  }
  if (last > first && loud_in(sounds[last]) <= 12) {
    --last;
  }
  const int64_t margin = std::lround(keep * 100.0);
  const int64_t start =
      std::max<int64_t>(static_cast<int64_t>(sounds[first].first) - margin, 0) *
      window;
  const int64_t end =
      std::min<int64_t>(
          static_cast<int64_t>(sounds[last].second) + 1 + margin,
          static_cast<int64_t>(levels.size())) *
      window;
  return std::vector<float>(samples.begin() + start, samples.begin() + end);
}

double speech_length(
    const std::vector<float>& samples,
    int sample_rate,
    double max_pause,
    double lead,
    double trail) {
  const int window = sample_rate / 100;
  const size_t count = window > 0 ? samples.size() / window : 0;
  if (count == 0) {
    return static_cast<double>(samples.size()) / sample_rate;
  }
  std::vector<double> x(samples.begin(), samples.begin() + count * window);
  double energy = 0.0;
  for (double v : x) {
    energy += v * v;
  }
  const double scale = 0.1 / std::max(std::sqrt(energy / x.size()), 1e-8);
  for (double& v : x) {
    v *= scale;
  }
  const std::vector<size_t> loud = loud_windows(window_levels(x, window));
  if (loud.empty()) {
    return static_cast<double>(samples.size()) / sample_rate;
  }
  const int64_t pause_cap = std::llround(max_pause * 100.0);
  int64_t kept = static_cast<int64_t>(loud.size());
  for (size_t i = 1; i < loud.size(); ++i) {
    kept += std::min<int64_t>(
        static_cast<int64_t>(loud[i] - loud[i - 1]) - 1, pause_cap);
  }
  const int64_t first = static_cast<int64_t>(loud.front());
  const int64_t last = static_cast<int64_t>(loud.back());
  const int64_t edges = std::min<int64_t>(first, std::llround(lead * 100.0)) +
      std::min<int64_t>(static_cast<int64_t>(count) - 1 - last,
                        std::llround(trail * 100.0));
  return static_cast<double>(kept + edges) / 100.0;
}

} // namespace reverb
