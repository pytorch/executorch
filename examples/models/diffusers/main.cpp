/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/mlx/runtime/backend_options.h>
#include <executorch/extension/diffusers/runner/diffusion_runner.h>
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/backend/options.h>

#include <gflags/gflags.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

DEFINE_string(model_path, "", "Path to the diffusion program (.pte).");
DEFINE_string(data_path, "", "Path to external weights (.ptd), if any.");
DEFINE_string(
    model_dir,
    "",
    "Hugging Face snapshot directory with the tokenizers and scheduler.");
DEFINE_string(prompt, "", "Prompt.");
DEFINE_string(negative_prompt, "", "Negative prompt.");
DEFINE_string(output, "output.png", "Output PNG path.");
DEFINE_int32(steps, 0, "Denoising steps; defaults to the program's value.");
DEFINE_double(cfg_scale, 0, "CFG scale; defaults to the program's value.");
DEFINE_double(
    guidance_scale,
    0,
    "Guidance input; defaults to the program's value.");
DEFINE_uint64(seed, 0, "Noise seed.");
DEFINE_int64(height, 0, "Image height; defaults to the exported height.");
DEFINE_int64(width, 0, "Image width; defaults to the exported width.");
DEFINE_string(
    image,
    "",
    "Input image for image-to-image; the program needs an encode method.");
DEFINE_double(
    strength,
    0.8,
    "Share of the denoising schedule run on the input image, in (0, 1].");
DEFINE_int32(
    preview_every,
    0,
    "Write the decoded latents to --output every N steps; 0 disables.");

namespace {

bool is_set(const char* flag) {
  return !gflags::GetCommandLineFlagInfoOrDie(flag).is_default;
}

// Overwrites path in place, so a viewer that has it open sees each update.
bool write_png(
    const std::string& path,
    const executorch::extension::diffusers::GeneratedImage& image) {
  return stbi_write_png(
      path.c_str(),
      static_cast<int>(image.width),
      static_cast<int>(image.height),
      3,
      image.rgb.data(),
      static_cast<int>(image.width * 3));
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_path.empty() || FLAGS_model_dir.empty() ||
      FLAGS_prompt.empty()) {
    std::cerr << "--model_path, --model_dir and --prompt are required"
              << std::endl;
    return 1;
  }
  if (FLAGS_preview_every < 0) {
    std::cerr << "--preview_every must not be negative" << std::endl;
    return 1;
  }

  using executorch::extension::diffusers::DiffusionRunner;
  using executorch::extension::diffusers::GeneratedImage;
  using executorch::extension::diffusers::GenerationConfig;
  GenerationConfig config;
  config.seed = FLAGS_seed;
  if (is_set("negative_prompt")) {
    config.negative_prompt = FLAGS_negative_prompt;
  }
  if (is_set("steps")) {
    config.steps = FLAGS_steps;
  }
  if (is_set("cfg_scale")) {
    config.cfg_scale = FLAGS_cfg_scale;
  }
  if (is_set("guidance_scale")) {
    config.guidance_scale = FLAGS_guidance_scale;
  }
  if (is_set("height")) {
    config.height = FLAGS_height;
  }
  if (is_set("width")) {
    config.width = FLAGS_width;
  }
  if (!FLAGS_image.empty()) {
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* pixels =
        stbi_load(FLAGS_image.c_str(), &width, &height, &channels, 3);
    if (pixels == nullptr) {
      std::cerr << "Failed to read " << FLAGS_image << std::endl;
      return 1;
    }
    // (1, 3, height, width) with values in [0, 1].
    executorch::extension::diffusers::HostTensor image{
        {1, 3, height, width},
        std::vector<float>(static_cast<size_t>(3) * height * width)};
    const size_t plane = static_cast<size_t>(height) * width;
    for (size_t pixel = 0; pixel < plane; ++pixel) {
      for (size_t channel = 0; channel < 3; ++channel) {
        image.data[channel * plane + pixel] =
            pixels[pixel * 3 + channel] / 255.0f;
      }
    }
    stbi_image_free(pixels);
    config.image = std::move(image);
    config.strength = FLAGS_strength;
    if (!config.height) {
      config.height = height;
    }
    if (!config.width) {
      config.width = width;
    }
  }
  config.preview_interval = FLAGS_preview_every;
  // Seconds since the previous progress line: a step's denoising, or a
  // preview's decode.
  auto mark = std::chrono::steady_clock::now();
  const auto lap = [&mark]() {
    const auto now = std::chrono::steady_clock::now();
    const std::chrono::duration<double> seconds = now - mark;
    mark = now;
    return seconds.count();
  };
  config.on_preview =
      [&lap](size_t step, size_t total, const GeneratedImage& preview) {
        const double decode_seconds = lap();
        if (write_png(FLAGS_output, preview)) {
          std::cout << "Preview of step " << step << "/" << total
                    << " written to " << FLAGS_output << " (decode "
                    << decode_seconds << " s, write " << lap() << " s)"
                    << std::endl;
        }
      };

  // Release the MLX buffer cache after every method call, so buffers sized for
  // one method are not kept while the next runs.
  executorch::runtime::BackendOptions<1> mlx_options;
  executorch::runtime::LoadBackendOptionsMap backend_options;
  if (mlx_options.set_option(
          executorch::backends::mlx::kClearCacheIntervalKey, 1) !=
          executorch::runtime::Error::Ok ||
      backend_options.set_options(
          executorch::backends::mlx::kMLXBackendId, mlx_options.view()) !=
          executorch::runtime::Error::Ok) {
    std::cerr << "Failed to set MLX options" << std::endl;
    return 1;
  }
  auto runner = DiffusionRunner::load(
      FLAGS_model_path, FLAGS_model_dir, FLAGS_data_path, &backend_options);
  if (!runner.ok()) {
    std::cerr << "Failed to load " << FLAGS_model_path << std::endl;
    return 1;
  }
  const auto start = std::chrono::steady_clock::now();
  mark = start;
  auto image = runner.get()->generate(
      FLAGS_prompt, config, [&lap](size_t step, size_t total) {
        std::cout << "Step " << step << "/" << total << " (" << lap() << " s)"
                  << std::endl;
      });
  if (!image.ok()) {
    std::cerr << "Generation failed" << std::endl;
    return 1;
  }
  const std::chrono::duration<double> elapsed =
      std::chrono::steady_clock::now() - start;
  std::cout << "Generated in " << elapsed.count() << " seconds" << std::endl;
  if (!write_png(FLAGS_output, image.get())) {
    std::cerr << "Failed to write " << FLAGS_output << std::endl;
    return 1;
  }
  std::cout << "Saved " << FLAGS_output << std::endl;
  return 0;
}
