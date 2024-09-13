/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/llava/runner/llava_runner.h>
#include <gflags/gflags.h>
#ifndef BUILD_LLAVA_RUNNER_WITHOUT_TORCH
#include <torch/torch.h>
#endif
#include <fstream>
#include <string>

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
    "The path to a .pt file, a serialized torch tensor for an image. Only work if compiled without BUILD_LLAVA_RUNNER_WITHOUT_TORCH.");

DEFINE_bool(
    is_csv_image,
    false,
    "If true, the image is a csv file, otherwise it is assumed to be a torch saved .pt file.");

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

int get_images(
    const std::string image_path,
    std::vector<torch::executor::Image>& images,
    bool is_csv) {
  std::vector<uint8_t> image_data;
  if (is_csv) {
    // Work without torch, parse a csv file
    // see image_util.py file for csv format
    std::array<int32_t, 3> image_shape;
    std::ifstream csv_image(image_path, std::ios::in | std::ios::binary);

    if (csv_image.is_open()) {
      std::string item;

      // Parse csv header, first number of dims
      std::getline(csv_image, item, ',');
      uint32_t dims = std::stoul(item);
      if (dims != 3) {
        ET_LOG(Error, "csv image dims != 3");
        return 1;
      }

      // Parse csv header, next shape
      uint32_t numel = 1;
      for (uint32_t i = 0; i < dims; i++) {
        std::getline(csv_image, item, ',');
        image_shape[i] = std::stoul(item);
        numel *= image_shape[i];
      }
      ET_LOG(
          Info,
          "csv image shape: %u, %u, %u, numel: %u",
          image_shape[0],
          image_shape[1],
          image_shape[2],
          numel);

      // Parse csv header, data type
      std::getline(csv_image, item, '\n');
      uint32_t data_type = std::stoul(item);
      if (data_type != sizeof(uint8_t)) {
        ET_LOG(Error, "csv image data type != uint8");
        return 1;
      }

      // Read csv data
      image_data.resize(numel);
      csv_image.read((char*)image_data.data(), numel);
      if (static_cast<uint32_t>(csv_image.gcount()) != numel) {
        ET_LOG(
            Error,
            "Failed to read csv image data, expected %u bytes, read %u bytes",
            numel,
            static_cast<uint32_t>(csv_image.gcount()));
        return 1;
      }
      images.push_back(
          {.data = image_data,
           .width = image_shape[2],
           .height = image_shape[1]});
      csv_image.close();
    } else {
      ET_LOG(
          Error, "Failed to open input csv image file: %s", image_path.c_str());
      return 1;
    }
  } else { // is_csv == false
#ifndef BUILD_LLAVA_RUNNER_WITHOUT_TORCH
    // Work with torch, load a serialized torch tensor
    torch::Tensor image_tensor;
    torch::load(image_tensor, image_path); // CHW
    ET_LOG(
        Info,
        "tensor image size(0): %lld, size(1): %lld, size(2): %lld, numel: %lld",
        image_tensor.size(0),
        image_tensor.size(1),
        image_tensor.size(2),
        image_tensor.numel());
    image_data.assign(
        image_tensor.data_ptr<uint8_t>(),
        image_tensor.data_ptr<uint8_t>() + image_tensor.numel());
    images.push_back(
        {.data = image_data,
         .width = static_cast<int32_t>(image_tensor.size(2)),
         .height = static_cast<int32_t>(image_tensor.size(1))});
#else
    ET_LOG(
        Error,
        "BUILD_LLAVA_RUNNER_WITHOUT_TORCH is defined, cannot load pt image.");
    return 1;
#endif // BUILD_LLAVA_RUNNER_WITHOUT_TORCH
  }
  return 0;
}

int32_t main(int32_t argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Create a loader to get the data of the program file. There are other
  // DataLoaders that use mmap() or point32_t to data that's already in memory,
  // and users can create their own DataLoaders to load from arbitrary sources.
  const char* model_path = FLAGS_model_path.c_str();

  const char* tokenizer_path = FLAGS_tokenizer_path.c_str();

  const char* prompt = FLAGS_prompt.c_str();

  double temperature = FLAGS_temperature;

  int32_t seq_len = FLAGS_seq_len;

  int32_t cpu_threads = FLAGS_cpu_threads;

#if defined(ET_USE_THREADPOOL)
  uint32_t num_performant_cores = cpu_threads == -1
      ? torch::executorch::cpuinfo::get_num_performant_cores()
      : static_cast<uint32_t>(cpu_threads);
  ET_LOG(
      Info, "Resetting threadpool with num threads = %d", num_performant_cores);
  if (num_performant_cores > 0) {
    torch::executorch::threadpool::get_threadpool()->_unsafe_reset_threadpool(
        num_performant_cores);
  }
#endif
  // create llama runner
  example::LlavaRunner runner(model_path, tokenizer_path, temperature);

  // read image
  std::vector<torch::executor::Image> images;

  std::string image_path;
  if (FLAGS_image_path == "") {
    ET_LOG(Error, "image path is empty.");
    return 1;
  }
#ifdef BUILD_LLAVA_RUNNER_WITHOUT_TORCH
  if (FLAGS_is_csv_image == false) {
    ET_LOG(
        Error,
        "pt image is not supported when compiled without torch. Only provide a csv image and set `is_csv_image` flag to true.");
    return 1;
  }
#endif // BUILD_LLAVA_RUNNER_WITHOUT_TORCH

  int ret = get_images(FLAGS_image_path, images, FLAGS_is_csv_image);
  if (ret != 0) {
    return ret;
  }

  // generate
  runner.generate(std::move(images), prompt, seq_len);
  return 0;
}
