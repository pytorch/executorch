/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Microbenchmark for the ImageProcessor reuse APIs: process_into (BGRA/RGBA)
// and process_yuv_into (NV12/NV21). Sweeps common input sizes x target sizes
// and, per cell, runs variants over resize mode (stretch/letterbox), execution
// path (CPU/GPU/size-default), color format, orientation (upright + 90) and the
// allocating process() vs process_into().
//
// Configurable (filters default to all):
//   --format=bgra|rgba|nv12|nv21   restrict to one color / YUV format
//   --unit=cpu|gpu|default         restrict to one execution path
//   --out=PATH                     write the results table to PATH (else
//   stdout)
// The input-size sweep and the rotation variant always run.
//
// On Apple the GPU rows use the CoreImage path; on portable backends every row
// runs the CPU pipeline. Input is a synthetic gradient; a row that fails to
// process is reported as ERROR rather than timed.
//
// Run (write a results file, then diff two with compare_benchmarks.py):
//   buck2 run -c cxx.extra_cxxflags=-Os \
//     fbsource//xplat/executorch/extension/image/benchmark:image_processor_benchmark
//     \
//     -- --out=/tmp/neon.txt

#include <executorch/extension/image/image_processor.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/platform/platform.h>

using namespace executorch::extension::image;
using executorch::extension::make_tensor_ptr;
using executorch::runtime::Error;

namespace {

constexpr int kWarmup = 10;
constexpr int kIters = 100;

// CLI filters (empty == no filter / all); set from argv in main().
std::string g_format;
std::string g_unit;

// Results sink: a file when --out=PATH is given, else stdout.
FILE* g_out = stdout;

const char* unit_name(int64_t gpu_min_input_pixels) {
  if (gpu_min_input_pixels == ImageProcessorConfig::kGpuNever) {
    return "cpu";
  }
  if (gpu_min_input_pixels == ImageProcessorConfig::kGpuAlways) {
    return "gpu";
  }
  return "default";
}

bool unit_ok(int64_t gpu_min_input_pixels) {
  return g_unit.empty() || g_unit == unit_name(gpu_min_input_pixels);
}

bool color_format_ok(ColorFormat f) {
  return g_format.empty() ||
      g_format == (f == ColorFormat::BGRA ? "bgra" : "rgba");
}

bool yuv_format_ok(YUVFormat f) {
  return g_format.empty() ||
      g_format == (f == YUVFormat::NV12 ? "nv12" : "nv21");
}

// "===" section banner + column legend; per-size cells below use "---".
void print_banner(const char* title) {
  const std::string bar(96, '=');
  std::fprintf(
      g_out,
      "\n%s\n%s\ncols: mean / median / p95 / stddev (ms), %d warmup + %d iters\n%s\n",
      bar.c_str(),
      title,
      kWarmup,
      kIters,
      bar.c_str());
}

void print_usage() {
  std::printf(
      "usage: image_processor_benchmark [--format=bgra|rgba|nv12|nv21] "
      "[--unit=cpu|gpu|default] [--out=PATH]\n"
      "  Filters default to all. --out writes the results table to PATH "
      "(stdout otherwise).\n"
      "  The input-size sweep and rotation always run.\n");
}

// Synthetic interleaved 4-byte-per-pixel input with a deterministic gradient.
std::vector<uint8_t> make_input(int32_t w, int32_t h) {
  std::vector<uint8_t> img(static_cast<size_t>(w) * h * 4);
  for (int32_t y = 0; y < h; ++y) {
    for (int32_t x = 0; x < w; ++x) {
      uint8_t* px = img.data() + (static_cast<size_t>(y) * w + x) * 4;
      px[0] = static_cast<uint8_t>(x);
      px[1] = static_cast<uint8_t>(y);
      px[2] = static_cast<uint8_t>(x + y);
      px[3] = 255;
    }
  }
  return img;
}

struct Stats {
  double mean, median, p95, stddev;
};

template <typename F>
Stats bench(F&& f) {
  for (int i = 0; i < kWarmup; ++i) {
    f();
  }
  std::vector<double> samples;
  samples.reserve(kIters);
  for (int i = 0; i < kIters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    const auto t1 = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  const double mean = sum / samples.size();
  const double var = std::accumulate(
      samples.begin(), samples.end(), 0.0, [mean](double acc, double s) {
        return acc + (s - mean) * (s - mean);
      });
  std::sort(samples.begin(), samples.end());
  assert(!samples.empty());
  return Stats{
      mean,
      samples[samples.size() / 2],
      samples[static_cast<size_t>(samples.size() * 0.95)],
      std::sqrt(var / samples.size())};
}

void print_row(const char* label, const Stats& s) {
  std::fprintf(
      g_out,
      "%-34s mean=%7.3f ms  median=%7.3f ms  p95=%7.3f ms  stddev=%6.3f ms\n",
      label,
      s.mean,
      s.median,
      s.p95,
      s.stddev);
}

void print_error(const char* label, Error err) {
  std::fprintf(
      g_out, "%-34s ERROR (0x%x)\n", label, static_cast<unsigned>(err));
}

void run_case(
    const char* label,
    int32_t in_w,
    int32_t in_h,
    int32_t target,
    ColorFormat format,
    ResizeMode mode,
    Orientation orientation,
    int64_t gpu_min_input_pixels,
    NormalizedRect roi,
    bool allocating) {
  ImageProcessorConfig config;
  config.target_width = target;
  config.target_height = target;
  config.resize_mode = mode;
  config.gpu_min_input_pixels = gpu_min_input_pixels;
  ImageProcessor proc(config);

  const auto input = make_input(in_w, in_h);
  auto out = make_tensor_ptr(
      {1, 3, target, target},
      std::vector<float>(static_cast<size_t>(3) * target * target));

  // One untimed call to surface any error and to fault in lazy state. The
  // allocating variant times process() (a fresh output tensor per call) to
  // expose the per-call allocation that process_into() avoids.
  const Error err = allocating
      ? proc.process(
                input.data(), in_w, in_h, in_w * 4, format, orientation, roi)
            .error()
      : proc.process_into(
            input.data(), in_w, in_h, in_w * 4, format, *out, orientation, roi);
  if (err != Error::Ok) {
    print_error(label, err);
    return;
  }

  const Stats s = bench([&] {
    if (allocating) {
      (void)proc.process(
          input.data(), in_w, in_h, in_w * 4, format, orientation, roi);
    } else {
      (void)proc.process_into(
          input.data(), in_w, in_h, in_w * 4, format, *out, orientation, roi);
    }
  });
  print_row(label, s);
}

// Per-cell variant: a labeled (format, mode, orientation, path, roi)
// combination. `allocating` times the allocating process() instead of
// process_into(). `always` runs the row regardless of the --format/--unit
// filters (used to keep the rotation variant running in every invocation).
struct Variant {
  const char* label;
  ColorFormat format;
  ResizeMode mode;
  Orientation orientation;
  int64_t gpu_min_input_pixels;
  NormalizedRect roi;
  bool allocating = false;
  bool always = false;
};

// Semi-planar YUV (NV12/NV21) synthetic input: full-res Y plane + half-res
// interleaved chroma, tight strides (== width). Dimensions must be even.
struct YuvInput {
  std::vector<uint8_t> y;
  std::vector<uint8_t> uv;
};

YuvInput make_yuv_input(int32_t w, int32_t h) {
  YuvInput in;
  in.y.resize(static_cast<size_t>(w) * h);
  for (int32_t yy = 0; yy < h; ++yy) {
    for (int32_t x = 0; x < w; ++x) {
      in.y[static_cast<size_t>(yy) * w + x] = static_cast<uint8_t>(x + yy);
    }
  }
  in.uv.resize(static_cast<size_t>(w) * (h / 2));
  for (int32_t r = 0; r < h / 2; ++r) {
    for (int32_t c = 0; c < w / 2; ++c) {
      uint8_t* px = in.uv.data() + static_cast<size_t>(r) * w + c * 2;
      px[0] = 128;
      px[1] = 128;
    }
  }
  return in;
}

void run_yuv_case(
    const char* label,
    int32_t in_w,
    int32_t in_h,
    int32_t target,
    YUVFormat format,
    ResizeMode mode,
    Orientation orientation,
    int64_t gpu_min_input_pixels,
    YUVRange range) {
  ImageProcessorConfig config;
  config.target_width = target;
  config.target_height = target;
  config.resize_mode = mode;
  config.gpu_min_input_pixels = gpu_min_input_pixels;
  ImageProcessor proc(config);

  const YuvInput in = make_yuv_input(in_w, in_h);
  auto out = make_tensor_ptr(
      {1, 3, target, target},
      std::vector<float>(static_cast<size_t>(3) * target * target));

  // y_stride and uv_stride are tight (== in_w).
  const Error err = proc.process_yuv_into(
      in.y.data(),
      in_w,
      in.uv.data(),
      in_w,
      in_w,
      in_h,
      format,
      *out,
      orientation,
      kFullImage,
      range);
  if (err != Error::Ok) {
    print_error(label, err);
    return;
  }

  const Stats s = bench([&] {
    (void)proc.process_yuv_into(
        in.y.data(),
        in_w,
        in.uv.data(),
        in_w,
        in_w,
        in_h,
        format,
        *out,
        orientation,
        kFullImage,
        range);
  });
  print_row(label, s);
}

// Per-cell YUV variant: (format, mode, orientation, path, range).
struct YuvVariant {
  const char* label;
  YUVFormat format;
  ResizeMode mode;
  Orientation orientation;
  int64_t gpu_min_input_pixels;
  YUVRange range;
};

struct Size {
  int32_t w, h;
  const char* label;
};

} // namespace

int main(int argc, char** argv) {
  et_pal_init();

  std::string out_path;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      print_usage();
      return 0;
    } else if (a.rfind("--format=", 0) == 0) {
      g_format = a.substr(std::string("--format=").size());
    } else if (a.rfind("--unit=", 0) == 0) {
      g_unit = a.substr(std::string("--unit=").size());
    } else if (a.rfind("--out=", 0) == 0) {
      out_path = a.substr(std::string("--out=").size());
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
      print_usage();
      return 2;
    }
  }
  if (g_format == "all") {
    g_format.clear();
  }
  if (g_unit == "all") {
    g_unit.clear();
  }
  if (!g_format.empty() && g_format != "bgra" && g_format != "rgba" &&
      g_format != "nv12" && g_format != "nv21") {
    std::fprintf(stderr, "invalid --format: %s\n", g_format.c_str());
    print_usage();
    return 2;
  }
  if (!g_unit.empty() && g_unit != "cpu" && g_unit != "gpu" &&
      g_unit != "default") {
    std::fprintf(stderr, "invalid --unit: %s\n", g_unit.c_str());
    print_usage();
    return 2;
  }
  if (!out_path.empty()) {
    g_out = std::fopen(out_path.c_str(), "w");
    if (g_out == nullptr) {
      std::fprintf(stderr, "could not open --out file: %s\n", out_path.c_str());
      return 2;
    }
    std::fprintf(stderr, "writing results to %s\n", out_path.c_str());
  }

  const std::array<Size, 4> inputs = {{
      {854, 480, "480p"},
      {1280, 720, "720p"},
      {1920, 1080, "1080p"},
      {3840, 2160, "4K"},
  }};
  const std::array<int32_t, 3> targets = {224, 256, 512};

  constexpr int64_t kCpu = ImageProcessorConfig::kGpuNever;
  constexpr int64_t kGpu = ImageProcessorConfig::kGpuAlways;
  constexpr int64_t kDefault = ImageProcessorConfig::kDefaultGpuMinInputPixels;

  std::fprintf(
      g_out,
      "filters: format=%s unit=%s\n",
      g_format.empty() ? "all" : g_format.c_str(),
      g_unit.empty() ? "all" : g_unit.c_str());

  const std::array<Variant, 9> variants = {{
      {"BGRA stretch   UP   CPU",
       ColorFormat::BGRA,
       ResizeMode::STRETCH,
       Orientation::UP,
       kCpu,
       kFullImage},
      {"BGRA stretch   UP   GPU",
       ColorFormat::BGRA,
       ResizeMode::STRETCH,
       Orientation::UP,
       kGpu,
       kFullImage},
      {"BGRA stretch   UP   def",
       ColorFormat::BGRA,
       ResizeMode::STRETCH,
       Orientation::UP,
       kDefault,
       kFullImage},
      {"BGRA letterbox UP   CPU",
       ColorFormat::BGRA,
       ResizeMode::LETTERBOX,
       Orientation::UP,
       kCpu,
       kFullImage},
      {"BGRA letterbox UP   GPU",
       ColorFormat::BGRA,
       ResizeMode::LETTERBOX,
       Orientation::UP,
       kGpu,
       kFullImage},
      {"RGBA stretch   UP   CPU",
       ColorFormat::RGBA,
       ResizeMode::STRETCH,
       Orientation::UP,
       kCpu,
       kFullImage},
      {"BGRA stretch   90   CPU",
       ColorFormat::BGRA,
       ResizeMode::STRETCH,
       Orientation::RIGHT,
       kCpu,
       kFullImage,
       /*allocating=*/false,
       /*always=*/true},
      {"BGRA stretch   ROI  CPU",
       ColorFormat::BGRA,
       ResizeMode::STRETCH,
       Orientation::UP,
       kCpu,
       NormalizedRect{0.25f, 0.25f, 0.5f, 0.5f}},
      {"BGRA stretch   UP   CPU alloc",
       ColorFormat::BGRA,
       ResizeMode::STRETCH,
       Orientation::UP,
       kCpu,
       kFullImage,
       /*allocating=*/true},
  }};

  auto color_included = [](const Variant& v) {
    return v.always ||
        (color_format_ok(v.format) && unit_ok(v.gpu_min_input_pixels));
  };

  const bool any_color =
      std::any_of(variants.begin(), variants.end(), color_included);
  if (any_color) {
    print_banner("ImageProcessor::process_into  (BGRA / RGBA)");
    for (const Size& in : inputs) {
      for (int32_t target : targets) {
        std::fprintf(
            g_out,
            "\n[%s %dx%d -> %dx%d]\n",
            in.label,
            in.w,
            in.h,
            target,
            target);
        std::fprintf(g_out, "%s\n", std::string(96, '-').c_str());
        for (const Variant& v : variants) {
          if (!color_included(v)) {
            continue;
          }
          run_case(
              v.label,
              in.w,
              in.h,
              target,
              v.format,
              v.mode,
              v.orientation,
              v.gpu_min_input_pixels,
              v.roi,
              v.allocating);
        }
      }
    }
  }

  const std::array<YuvVariant, 6> yuv_variants = {{
      {"NV12 stretch   UP   CPU",
       YUVFormat::NV12,
       ResizeMode::STRETCH,
       Orientation::UP,
       kCpu,
       YUVRange::VIDEO},
      {"NV12 stretch   UP   GPU",
       YUVFormat::NV12,
       ResizeMode::STRETCH,
       Orientation::UP,
       kGpu,
       YUVRange::VIDEO},
      {"NV12 stretch   UP   def",
       YUVFormat::NV12,
       ResizeMode::STRETCH,
       Orientation::UP,
       kDefault,
       YUVRange::VIDEO},
      {"NV12 letterbox UP   CPU",
       YUVFormat::NV12,
       ResizeMode::LETTERBOX,
       Orientation::UP,
       kCpu,
       YUVRange::VIDEO},
      {"NV21 stretch   UP   CPU",
       YUVFormat::NV21,
       ResizeMode::STRETCH,
       Orientation::UP,
       kCpu,
       YUVRange::VIDEO},
      {"NV12 stretch   UP   CPU(full)",
       YUVFormat::NV12,
       ResizeMode::STRETCH,
       Orientation::UP,
       kCpu,
       YUVRange::FULL},
  }};

  auto yuv_included = [](const YuvVariant& v) {
    return yuv_format_ok(v.format) && unit_ok(v.gpu_min_input_pixels);
  };

  const bool any_yuv =
      std::any_of(yuv_variants.begin(), yuv_variants.end(), yuv_included);
  if (any_yuv) {
    print_banner("ImageProcessor::process_yuv_into  (NV12 / NV21)");
    for (const Size& in : inputs) {
      for (int32_t target : targets) {
        std::fprintf(
            g_out,
            "\n[%s %dx%d -> %dx%d]\n",
            in.label,
            in.w,
            in.h,
            target,
            target);
        std::fprintf(g_out, "%s\n", std::string(96, '-').c_str());
        for (const YuvVariant& v : yuv_variants) {
          if (!yuv_included(v)) {
            continue;
          }
          run_yuv_case(
              v.label,
              in.w,
              in.h,
              target,
              v.format,
              v.mode,
              v.orientation,
              v.gpu_min_input_pixels,
              v.range);
        }
      }
    }
  }
  if (g_out != stdout) {
    std::fclose(g_out);
  }
  return 0;
}
