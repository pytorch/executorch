/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * Runs one SmolVLA plan with Qualcomm AI Engine Direct: reads the
 * SmolVLAInputs raw files, runs the Vision Encoder once per camera, the Prefix
 * Model once and the Action Expert for every Euler step, then writes the final
 * action chunk. Every shape and byte size comes from the PTE MethodMeta.
 */

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/platform/runtime.h>
#include <gflags/gflags.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

DEFINE_string(
    vision_pte,
    "smolvla_vision_encoder.pte",
    "SmolVLA Vision Encoder PTE.");
DEFINE_string(prefix_pte, "smolvla_prefix_model.pte", "SmolVLA Prefix PTE.");
DEFINE_string(
    action_pte,
    "smolvla_action_expert.pte",
    "SmolVLA Action Expert PTE.");
DEFINE_string(
    input_dir,
    "inputs",
    "Directory holding one <SmolVLAInputs field>.raw file per input.");
DEFINE_string(
    output_path,
    "action_chunk.raw",
    "Output file for the final action chunk.");
DEFINE_int32(num_steps, 10, "Number of flow matching Euler steps.");

namespace {

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::MethodMeta;
using executorch::runtime::Tag;
using executorch::runtime::TensorInfo;
using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point since) {
  return std::chrono::duration<double, std::milli>(Clock::now() - since)
      .count();
}

constexpr char kForward[] = "forward";
// Prefix Model inputs.
constexpr size_t kPrefixEmbeddings = 0;
constexpr size_t kPrefixAttentionMask = 1;
constexpr size_t kPrefixPositionIds = 2;
// Action Expert inputs: x_t, euler_step, K/V caches, attention mask, positions.
constexpr size_t kActionXt = 0;
constexpr size_t kActionEulerStep = 1;
constexpr size_t kActionCacheOffset = 2;

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::string describe(const TensorInfo& info) {
  std::ostringstream stream;
  stream << executorch::runtime::toString(info.scalar_type()) << "[";
  for (size_t i = 0; i < info.sizes().size(); ++i) {
    stream << (i == 0 ? "" : ", ") << info.sizes()[i];
  }
  stream << "] " << info.nbytes() << " bytes";
  return stream.str();
}

bool same_contract(const TensorInfo& left, const TensorInfo& right) {
  return left.scalar_type() == right.scalar_type() &&
      left.nbytes() == right.nbytes() &&
      std::equal(
             left.sizes().begin(),
             left.sizes().end(),
             right.sizes().begin(),
             right.sizes().end()) &&
      std::equal(
             left.dim_order().begin(),
             left.dim_order().end(),
             right.dim_order().begin(),
             right.dim_order().end());
}

size_t file_size(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  require(file.is_open(), "Could not open " + path);
  return static_cast<size_t>(file.tellg());
}

std::vector<uint8_t> read_raw(const std::string& path, size_t nbytes) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  require(file.is_open(), "Could not open " + path);
  const auto size = static_cast<size_t>(file.tellg());
  require(
      size == nbytes,
      path + " must be " + std::to_string(nbytes) + " bytes, got " +
          std::to_string(size));
  file.seekg(0);
  std::vector<uint8_t> data(nbytes);
  file.read(
      reinterpret_cast<char*>(data.data()),
      static_cast<std::streamsize>(nbytes));
  require(file.good(), "Could not read " + path);
  return data;
}

TensorPtr make_tensor(const TensorInfo& info, std::vector<uint8_t> data) {
  return executorch::extension::make_tensor_ptr(
      std::vector<executorch::aten::SizesType>(
          info.sizes().begin(), info.sizes().end()),
      std::move(data),
      std::vector<executorch::aten::DimOrderType>(
          info.dim_order().begin(), info.dim_order().end()),
      {},
      info.scalar_type(),
      executorch::aten::TensorShapeDynamism::STATIC);
}

class Stage {
 public:
  Stage(std::string name, const std::string& path) : name_(std::move(name)) {
    const auto load_started_at = Clock::now();
    module_ = std::make_unique<Module>(
        path, Module::LoadMode::MmapUseMlockIgnoreErrors);
    require(
        module_->load_method(kForward) == Error::Ok,
        name_ + " failed to load " + path);
    const auto meta = module_->method_meta(kForward);
    require(meta.ok(), name_ + " has no forward metadata");
    meta_ = std::make_unique<MethodMeta>(meta.get());
    for (size_t i = 0; i < meta_->num_inputs(); ++i) {
      require(
          meta_->input_tag(i).ok() && meta_->input_tag(i).get() == Tag::Tensor,
          name_ + " input " + std::to_string(i) + " is not a tensor");
    }
    for (size_t i = 0; i < meta_->num_outputs(); ++i) {
      require(
          meta_->output_tag(i).ok() &&
              meta_->output_tag(i).get() == Tag::Tensor,
          name_ + " output " + std::to_string(i) + " is not a tensor");
    }
    load_ms_ = elapsed_ms(load_started_at);
  }

  size_t num_inputs() const {
    return meta_->num_inputs();
  }

  size_t num_outputs() const {
    return meta_->num_outputs();
  }

  TensorInfo input(size_t index) const {
    return meta_->input_tensor_meta(index).get();
  }

  TensorInfo output(size_t index) const {
    return meta_->output_tensor_meta(index).get();
  }

  std::vector<EValue> run(const std::vector<EValue>& inputs) {
    const auto started_at = Clock::now();
    const auto result = module_->execute(kForward, inputs);
    execute_ms_ += elapsed_ms(started_at);
    ++num_executions_;
    require(
        result.ok(),
        name_ + " execution failed with error 0x" +
            std::to_string(static_cast<unsigned int>(result.error())));
    return result.get();
  }

  double execute_ms() const {
    return execute_ms_;
  }

  void print_timing() const {
    std::cout << "[qnn_smolvla_runner] " << name_ << " load_ms=" << load_ms_
              << " execute_ms=" << execute_ms_
              << " executions=" << num_executions_ << " mean_execute_ms="
              << execute_ms_ / std::max<size_t>(num_executions_, 1)
              << std::endl;
  }

 private:
  std::string name_;
  std::unique_ptr<Module> module_;
  std::unique_ptr<MethodMeta> meta_;
  double load_ms_ = 0.0;
  double execute_ms_ = 0.0;
  size_t num_executions_ = 0;
};

void validate(const Stage& vision, const Stage& prefix, const Stage& action) {
  require(
      vision.num_inputs() == 1 && vision.num_outputs() == 1,
      "Vision Encoder must be 1 -> 1");
  require(
      prefix.num_inputs() == kPrefixPositionIds + 1,
      "Prefix Model must take embeddings, attention mask, and position ids");
  const size_t num_caches = prefix.num_outputs();
  require(
      action.num_inputs() == kActionCacheOffset + num_caches + 2 &&
          action.num_outputs() == 1,
      "Action Expert must take x_t, euler_step, every Prefix K/V output, "
      "attention mask, and position ids");

  require(
      vision.output(0).scalar_type() == ScalarType::Float &&
          prefix.input(kPrefixEmbeddings).scalar_type() == ScalarType::Float,
      "Vision output " + describe(vision.output(0)) +
          " and Prefix embeddings " + describe(prefix.input(kPrefixEmbeddings)) +
          " must be float");
  for (size_t i = 0; i < num_caches; ++i) {
    require(
        same_contract(prefix.output(i), action.input(kActionCacheOffset + i)),
        "Prefix output " + std::to_string(i) + " " +
            describe(prefix.output(i)) + " does not match Action Expert input " +
            std::to_string(kActionCacheOffset + i) + " " +
            describe(action.input(kActionCacheOffset + i)));
  }
  require(
      same_contract(action.output(0), action.input(kActionXt)) &&
          action.output(0).scalar_type() == ScalarType::Float,
      "Action Expert output " + describe(action.output(0)) +
          " must match float x_t " + describe(action.input(kActionXt)));
  require(
      action.input(kActionEulerStep).scalar_type() == ScalarType::Int &&
          action.input(kActionEulerStep).nbytes() == sizeof(int32_t),
      "Action Expert euler_step must be one int32, got " +
          describe(action.input(kActionEulerStep)));
}

std::string input_path(const std::string& name) {
  return FLAGS_input_dir + "/" + name + ".raw";
}

TensorPtr read_tensor(const TensorInfo& info, const std::string& name) {
  return make_tensor(info, read_raw(input_path(name), info.nbytes()));
}

void run_plan() {
  Stage vision("Vision Encoder", FLAGS_vision_pte);
  Stage prefix("Prefix Model", FLAGS_prefix_pte);
  Stage action("Action Expert", FLAGS_action_pte);
  validate(vision, prefix, action);
  const auto started_at = Clock::now();

  // Vision Encoder, once per camera; the camera count follows pixel_values.
  const size_t camera_bytes = vision.input(0).nbytes();
  const size_t embedding_bytes = vision.output(0).nbytes();
  const size_t prefix_bytes = prefix.input(kPrefixEmbeddings).nbytes();
  const size_t pixel_bytes = file_size(input_path("pixel_values"));
  const size_t num_cameras = pixel_bytes / camera_bytes;
  require(
      num_cameras > 0 && pixel_bytes == num_cameras * camera_bytes &&
          num_cameras * embedding_bytes < prefix_bytes,
      "pixel_values.raw holds " + std::to_string(pixel_bytes) +
          " bytes; expected a whole number of " + describe(vision.input(0)) +
          " cameras whose embeddings fit in " +
          describe(prefix.input(kPrefixEmbeddings)));
  const auto pixel_values = read_raw(input_path("pixel_values"), pixel_bytes);
  std::vector<uint8_t> prefix_embeddings(prefix_bytes);
  for (size_t camera = 0; camera < num_cameras; ++camera) {
    const auto begin = pixel_values.begin() + camera * camera_bytes;
    auto image = make_tensor(
        vision.input(0), std::vector<uint8_t>(begin, begin + camera_bytes));
    const auto outputs = vision.run({EValue(*image)});
    std::memcpy(
        prefix_embeddings.data() + camera * embedding_bytes,
        outputs[0].toTensor().const_data_ptr(),
        embedding_bytes);
  }

  // Prefix Model on [camera embeddings ; language and state prefix tail].
  const size_t tail_offset = num_cameras * embedding_bytes;
  const auto prefix_tail =
      read_raw(input_path("prefix_tail"), prefix_bytes - tail_offset);
  std::memcpy(
      prefix_embeddings.data() + tail_offset,
      prefix_tail.data(),
      prefix_tail.size());
  auto embeddings = make_tensor(
      prefix.input(kPrefixEmbeddings), std::move(prefix_embeddings));
  auto prefix_mask = read_tensor(
      prefix.input(kPrefixAttentionMask), "prefix_attention_mask");
  auto prefix_positions =
      read_tensor(prefix.input(kPrefixPositionIds), "prefix_position_ids");
  const auto prefix_caches = prefix.run(
      {EValue(*embeddings), EValue(*prefix_mask), EValue(*prefix_positions)});

  // Action Expert Euler loop, mirroring SmolVLAFlowMatching.forward.
  const size_t num_caches = prefix.num_outputs();
  auto x_t = read_tensor(action.input(kActionXt), "noise");
  auto euler_step = make_tensor(
      action.input(kActionEulerStep), std::vector<uint8_t>(sizeof(int32_t)));
  auto action_mask = read_tensor(
      action.input(kActionCacheOffset + num_caches), "action_attention_mask");
  auto suffix_positions = read_tensor(
      action.input(kActionCacheOffset + num_caches + 1), "suffix_position_ids");
  std::vector<EValue> action_inputs = {EValue(*x_t), EValue(*euler_step)};
  action_inputs.insert(
      action_inputs.end(), prefix_caches.begin(), prefix_caches.end());
  action_inputs.emplace_back(*action_mask);
  action_inputs.emplace_back(*suffix_positions);

  float* actions = x_t->mutable_data_ptr<float>();
  const size_t num_values = x_t->numel();
  for (int32_t step = 0; step < FLAGS_num_steps; ++step) {
    *euler_step->mutable_data_ptr<int32_t>() = step;
    const auto outputs = action.run(action_inputs);
    const float* velocity = outputs[0].toTensor().const_data_ptr<float>();
    for (size_t i = 0; i < num_values; ++i) {
      actions[i] -= velocity[i] / FLAGS_num_steps;
    }
  }

  const double plan_ms = elapsed_ms(started_at);
  std::ofstream output(FLAGS_output_path, std::ios::binary);
  output.write(
      reinterpret_cast<const char*>(actions),
      static_cast<std::streamsize>(x_t->nbytes()));
  require(output.good(), "Could not write " + FLAGS_output_path);
  vision.print_timing();
  prefix.print_timing();
  action.print_timing();
  // e2e.py parses this line for the device-side timing.
  std::cout << "[qnn_smolvla_runner] plan_ms=" << plan_ms << " execute_ms="
            << vision.execute_ms() + prefix.execute_ms() + action.execute_ms()
            << std::endl;
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  executorch::runtime::runtime_init();
  try {
    require(FLAGS_num_steps > 0, "--num_steps must be positive");
    run_plan();
  } catch (const std::exception& error) {
    std::cerr << "qnn_smolvla_runner: " << error.what() << std::endl;
    return 1;
  }
  return 0;
}
