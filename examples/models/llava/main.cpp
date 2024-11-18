/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/llava/runner/llava_runner.h>
#include <gflags/gflags.h>
#ifndef LLAVA_NO_TORCH_DUMMY_IMAGE
#include <torch/torch.h>
#else
#include <algorithm> // std::fill
#endif

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

DEFINE_string(
    model_path,
    "llava.pte",
    "Model serialized in flatbuffer format.");

DEFINE_string(tokenizer_path, "tokenizer.bin", "Tokenizer stuff.");

DEFINE_string(prompt, "The answer to the ultimate question is", "Prompt.");

DEFINE_string(
    image_path,
    "",
    "The path to a .pt file, a serialized torch tensor for an image, longest edge resized to 336.");

DEFINE_double(
    temperature,
    0.8f,
    "Temperature; Default is 0.8f. 0 = greedy argmax sampling (deterministic). Lower temperature = more deterministic");

DEFINE_int32(
    seq_len,
    1024,
    "Total number of tokens to generate (prompt + output). Defaults to max_seq_len. If the number of input tokens + seq_len > max_seq_len, the output will be truncated to max_seq_len tokens.");

DEFINE_int32(
    cpu_threads,
    -1,
    "Number of CPU threads for inference. Defaults to -1, which implies we'll use a heuristic to derive the # of performant cores for a specific device.");

using executorch::extension::llm::Image;

int32_t main(int32_t argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Create a loader to get the data of the program file. There are other
  // DataLoaders that use mmap() or point32_t to data that's already in memory,
  // and users can create their own DataLoaders to load from arbitrary sources.
  const char* model_path = FLAGS_model_path.c_str();

  const char* tokenizer_path = FLAGS_tokenizer_path.c_str();

  const char* prompt = FLAGS_prompt.c_str();

  std::string image_path = FLAGS_image_path;

  double temperature = FLAGS_temperature;

  int32_t seq_len = FLAGS_seq_len;

  int32_t cpu_threads = FLAGS_cpu_threads;

#if defined(ET_USE_THREADPOOL)
  uint32_t num_performant_cores = cpu_threads == -1
      ? ::executorch::extension::cpuinfo::get_num_performant_cores()
      : static_cast<uint32_t>(cpu_threads);
  ET_LOG(
      Info, "Resetting threadpool with num threads = %d", num_performant_cores);
  if (num_performant_cores > 0) {
    ::executorch::extension::threadpool::get_threadpool()
        ->_unsafe_reset_threadpool(num_performant_cores);
  }
#endif
  // create llama runner
  example::LlavaRunner runner(model_path, tokenizer_path, temperature);

  // read image and resize the longest edge to 336
  std::vector<uint8_t> image_data;

#ifdef LLAVA_NO_TORCH_DUMMY_IMAGE
  // Work without torch using a random data
  image_data.resize(3 * 240 * 336);
  std::fill(image_data.begin(), image_data.end(), 0); // black
  std::array<int32_t, 3> image_shape = {3, 240, 336};
  std::vector<Image> images = {
      {.data = image_data, .width = image_shape[2], .height = image_shape[1]}};
#else //  LLAVA_NO_TORCH_DUMMY_IMAGE
  //   cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  //   int longest_edge = std::max(image.rows, image.cols);
  //   float scale_factor = 336.0f / longest_edge;
  //   cv::Size new_size(image.cols * scale_factor, image.rows * scale_factor);
  //   cv::Mat resized_image;
  //   cv::resize(image, resized_image, new_size);
  //   image_data.assign(resized_image.datastart, resized_image.dataend);
  torch::Tensor image_tensor;
  torch::load(image_tensor, image_path); // CHW
  ET_LOG(
      Info,
      "image size(0): %" PRId64 ", size(1): %" PRId64 ", size(2): %" PRId64,
      image_tensor.size(0),
      image_tensor.size(1),
      image_tensor.size(2));
  image_data.assign(
      image_tensor.data_ptr<uint8_t>(),
      image_tensor.data_ptr<uint8_t>() + image_tensor.numel());
  std::vector<Image> images = {
      {.data = image_data,
       .width = static_cast<int32_t>(image_tensor.size(2)),
       .height = static_cast<int32_t>(image_tensor.size(1))}};
#endif // LLAVA_NO_TORCH_DUMMY_IMAGE

  // generate
  runner.generate(std::move(images), prompt, seq_len);
  return 0;
}
