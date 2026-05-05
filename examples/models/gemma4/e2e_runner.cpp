/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @lint-ignore-every CLANGTIDY facebook-hte-Deprecated
 */

#include <gflags/gflags.h>

#include <iostream>
#include <thread>

#include <executorch/examples/models/gemma4/image_utils.h>
#include <executorch/examples/models/gemma4/runner/gemma4_runner.h>
#include <executorch/extension/llm/runner/wav_loader.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/platform/log.h>

#include <stb_image.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

using executorch::examples::gemma4::Gemma4Runner;
using executorch::examples::gemma4::Gemma4Stats;
using executorch::examples::gemma4::GenerationConfig;
using executorch::examples::gemma4::patchify_rgb_image;

DEFINE_string(model_path, "gemma4.pte", "Single PTE model path.");
DEFINE_string(tokenizer_path, "tokenizer.model", "Tokenizer model path.");
DEFINE_string(audio_path, "", "Path to WAV audio file (16kHz, 16-bit PCM).");
DEFINE_string(image_path, "", "Path to image file (JPEG/PNG).");
DEFINE_string(
    prompt,
    "Transcribe the following audio:",
    "Prompt for generation.");
DEFINE_int32(max_new_tokens, 100, "Maximum tokens to generate.");
DEFINE_int32(max_vision_tokens, 140, "Maximum soft tokens for vision encoder.");
DEFINE_double(temperature, 0.0, "Sampling temperature (0.0 = greedy).");
DEFINE_int32(cpu_threads, -1, "Number of CPU threads. -1 = auto-detect.");
DEFINE_bool(
    enable_workspace_sharing,
    true,
    "Enable XNNPACK PerModel workspace sharing + weight cache. "
    "Pass --noenable_workspace_sharing to disable for debugging.");

int32_t main(int32_t argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

#if defined(ET_USE_THREADPOOL)
  uint32_t num_threads = FLAGS_cpu_threads == -1
      ? ::executorch::extension::cpuinfo::get_num_performant_cores()
      : static_cast<uint32_t>(FLAGS_cpu_threads);
  if (num_threads == 0) {
    num_threads = std::min(std::thread::hardware_concurrency(), 8u);
  }
  ET_LOG(Info, "Setting threadpool to %d threads", (int)num_threads);
  ::executorch::extension::threadpool::get_threadpool()
      ->_unsafe_reset_threadpool(num_threads);
#endif

  Gemma4Stats stats;
  stats.on_load_begin();

  Gemma4Runner runner(
      FLAGS_model_path, FLAGS_tokenizer_path, FLAGS_enable_workspace_sharing);
  auto err = runner.load();
  ET_CHECK_MSG(err == executorch::runtime::Error::Ok, "Failed to load model");

  stats.on_load_end();

  GenerationConfig config;
  config.max_new_tokens = FLAGS_max_new_tokens;
  config.temperature = static_cast<float>(FLAGS_temperature);

  auto token_cb = [](const std::string& tok) {
    std::cout << tok << std::flush;
  };

  bool has_audio = !FLAGS_audio_path.empty();
  bool has_image = !FLAGS_image_path.empty();
  ET_CHECK_MSG(
      !(has_audio && has_image),
      "Cannot specify both --audio_path and --image_path");

  if (has_audio) {
    auto audio_data =
        executorch::extension::llm::load_wav_audio_data(FLAGS_audio_path);
    ET_CHECK_MSG(!audio_data.empty(), "Failed to load audio file");

    int64_t actual_samples = static_cast<int64_t>(audio_data.size());
    if (actual_samples > 480000) {
      audio_data.resize(480000);
      actual_samples = 480000;
    }

    int64_t padded_len = ((actual_samples + 127) / 128) * 128;
    audio_data.resize(padded_len, 0.0f);

    auto waveform = executorch::extension::from_blob(
        audio_data.data(),
        {static_cast<int32_t>(padded_len)},
        executorch::aten::ScalarType::Float);

    auto result = runner.generate(
        waveform, actual_samples, FLAGS_prompt, config, token_cb, &stats);
    ET_CHECK_MSG(result.ok(), "Audio generation failed");

  } else if (has_image) {
    int img_w, img_h, img_c;
    unsigned char* img_data =
        stbi_load(FLAGS_image_path.c_str(), &img_w, &img_h, &img_c, 3);
    ET_CHECK_MSG(img_data != nullptr, "Failed to load image");

    auto image_data =
        patchify_rgb_image(img_data, img_w, img_h, FLAGS_max_vision_tokens);
    stbi_image_free(img_data);

    auto result = runner.generate_vision(
        image_data.pixel_values,
        image_data.pixel_position_ids,
        FLAGS_prompt,
        config,
        token_cb,
        &stats);
    ET_CHECK_MSG(result.ok(), "Vision generation failed");

  } else {
    auto result = runner.generate_text(FLAGS_prompt, config, token_cb, &stats);
    ET_CHECK_MSG(result.ok(), "Text generation failed");
  }

  std::cout << std::endl;
  std::cerr << stats.report();

  return 0;
}
