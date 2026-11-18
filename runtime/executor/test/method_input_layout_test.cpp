/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/test/managed_memory_manager.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::ET_RUNTIME_NAMESPACE::Method;
using executorch::ET_RUNTIME_NAMESPACE::Program;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Result;
using executorch::runtime::testing::ManagedMemoryManager;
using executorch::runtime::testing::TensorFactory;
using torch::executor::util::FileDataLoader;

namespace {

constexpr size_t kDefaultNonConstMemBytes = 32 * 1024U;
constexpr size_t kDefaultRuntimeMemBytes = 32 * 1024U;

// The ModuleChannelsLast input is (N, C, H, W) = (1, 3, 4, 5) in channels last.
const std::vector<int32_t> kSizes = {1, 3, 4, 5};
const std::vector<uint8_t> kChannelsLast = {0, 2, 3, 1};

// Every element is distinct, unlike an all-ones input, so data read through the
// wrong layout cannot produce the expected output.
std::vector<float> Iota(size_t numel, float start = 0) {
  std::vector<float> values(numel);
  for (size_t i = 0; i < numel; ++i) {
    values[i] = start + static_cast<float>(i);
  }
  return values;
}

std::vector<float> Doubled(const std::vector<float>& values) {
  std::vector<float> doubled;
  for (float value : values) {
    doubled.push_back(2 * value);
  }
  return doubled;
}

// Moves `logical` (row-major order) into the memory order of `dim_order`.
std::vector<float> ToMemoryOrder(
    const std::vector<float>& logical,
    const std::vector<int32_t>& sizes,
    const std::vector<uint8_t>& dim_order) {
  const size_t rank = sizes.size();
  std::vector<int64_t> strides(rank);
  strides[dim_order[rank - 1]] = 1;
  for (int64_t i = static_cast<int64_t>(rank) - 2; i >= 0; --i) {
    strides[dim_order[i]] = strides[dim_order[i + 1]] * sizes[dim_order[i + 1]];
  }
  std::vector<float> memory(logical.size());
  std::vector<int64_t> index(rank, 0);
  for (size_t linear = 0; linear < logical.size(); ++linear) {
    int64_t offset = 0;
    for (size_t d = 0; d < rank; ++d) {
      offset += index[d] * strides[d];
    }
    memory[offset] = logical[linear];
    for (int64_t d = static_cast<int64_t>(rank) - 1; d >= 0; --d) {
      if (++index[d] < sizes[d]) {
        break;
      }
      index[d] = 0;
    }
  }
  return memory;
}

// Reads the elements of `t` in row-major order, following its strides.
std::vector<float> LogicalValues(const Tensor& t) {
  const int64_t rank = static_cast<int64_t>(t.dim());
  const float* data = t.const_data_ptr<float>();
  std::vector<float> values;
  std::vector<int64_t> index(rank, 0);
  for (int64_t linear = 0; linear < static_cast<int64_t>(t.numel()); ++linear) {
    int64_t offset = 0;
    for (int64_t d = 0; d < rank; ++d) {
      offset += index[d] * t.strides()[d];
    }
    values.push_back(data[offset]);
    for (int64_t d = rank - 1; d >= 0; --d) {
      if (++index[d] < t.size(d)) {
        break;
      }
      index[d] = 0;
    }
  }
  return values;
}

} // namespace

class MethodInputLayoutTest : public ::testing::Test {
 protected:
  void load_program(const char* path, const char* module_name) {
    Result<FileDataLoader> loader = FileDataLoader::from(path);
    ASSERT_EQ(loader.error(), Error::Ok);
    loaders_.insert(
        {module_name,
         std::make_unique<FileDataLoader>(std::move(loader.get()))});

    Result<Program> program = Program::load(
        loaders_[module_name].get(),
        Program::Verification::InternalConsistency);
    ASSERT_EQ(program.error(), Error::Ok);
    programs_.insert(
        {module_name, std::make_unique<Program>(std::move(program.get()))});
  }

  void SetUp() override {
    executorch::runtime::runtime_init();

    load_program(std::getenv("ET_MODULE_ADD_PATH"), "add");
    load_program(std::getenv("ET_MODULE_CHANNELS_LAST_PATH"), "channels_last");
  }

  Tensor channels_last_input(const std::vector<float>& logical) {
    return tf_.make_with_dimorder(
        kSizes, ToMemoryOrder(logical, kSizes, kChannelsLast), kChannelsLast);
  }

  // Keeps the tensors it makes alive.
  TensorFactory<ScalarType::Float> tf_;

 private:
  // Must outlive the programs, but tests shouldn't need to touch it.
  std::unordered_map<std::string, std::unique_ptr<FileDataLoader>> loaders_;

 protected:
  std::unordered_map<std::string, std::unique_ptr<Program>> programs_;
};

TEST_F(MethodInputLayoutTest, ChannelsLastInputKeepsItsValues) {
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<Method> method =
      programs_["channels_last"]->load_method("forward", &mmm.get());
  ASSERT_EQ(method.error(), Error::Ok);

  const std::vector<float> logical = Iota(60);
  // The test only discriminates if the layouts really differ.
  ASSERT_NE(ToMemoryOrder(logical, kSizes, kChannelsLast), logical);

  ASSERT_EQ(
      method->set_input(EValue(channels_last_input(logical)), 0), Error::Ok);
  ASSERT_EQ(method->execute(), Error::Ok);

  EXPECT_EQ(LogicalValues(method->get_output(0).toTensor()), Doubled(logical));
}

TEST_F(MethodInputLayoutTest, ChannelsLastInputCanBeReplaced) {
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<Method> method =
      programs_["channels_last"]->load_method("forward", &mmm.get());
  ASSERT_EQ(method.error(), Error::Ok);

  for (float start : {0.f, 100.f, -30.f}) {
    const std::vector<float> logical = Iota(60, start);
    ASSERT_EQ(
        method->set_input(EValue(channels_last_input(logical)), 0), Error::Ok);
    ASSERT_EQ(method->execute(), Error::Ok);
    EXPECT_EQ(
        LogicalValues(method->get_output(0).toTensor()), Doubled(logical));
  }
}

TEST_F(MethodInputLayoutTest, DefaultContiguousInputIsAcceptedForChannelsLast) {
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<Method> method =
      programs_["channels_last"]->load_method("forward", &mmm.get());
  ASSERT_EQ(method.error(), Error::Ok);

  const std::vector<float> logical = Iota(60);
  ASSERT_EQ(method->set_input(EValue(tf_.make(kSizes, logical)), 0), Error::Ok);
  ASSERT_EQ(method->execute(), Error::Ok);

#ifdef USE_ATEN_LIB
  // An ATen tensor reports the strides its data really has, so it is read as
  // given. Portable mode instead treats the data as already in the method's
  // layout.
  EXPECT_EQ(LogicalValues(method->get_output(0).toTensor()), Doubled(logical));
#endif
}

TEST_F(MethodInputLayoutTest, MismatchedExplicitLayoutIsRejected) {
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<Method> method =
      programs_["channels_last"]->load_method("forward", &mmm.get());
  ASSERT_EQ(method.error(), Error::Ok);

  const std::vector<float> logical = Iota(60);
  const std::vector<uint8_t> swapped_hw = {0, 1, 3, 2};
  Tensor input = tf_.make_with_dimorder(
      kSizes, ToMemoryOrder(logical, kSizes, swapped_hw), swapped_hw);
  EXPECT_EQ(method->set_input(EValue(input), 0), Error::InvalidArgument);

  // A rejected input must not leave the method unable to run a valid one.
  ASSERT_EQ(
      method->set_input(EValue(channels_last_input(logical)), 0), Error::Ok);
  ASSERT_EQ(method->execute(), Error::Ok);
  EXPECT_EQ(LogicalValues(method->get_output(0).toTensor()), Doubled(logical));
}

TEST_F(MethodInputLayoutTest, ContiguousMethodRejectsTransposedInput) {
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<Method> method = programs_["add"]->load_method("forward", &mmm.get());
  ASSERT_EQ(method.error(), Error::Ok);

  const std::vector<int32_t> sizes = {2, 2};
  const std::vector<float> x = {1, 2, 3, 4};
  const std::vector<float> y = {10, 20, 30, 40};

  Tensor transposed = tf_.make_with_dimorder(
      sizes, ToMemoryOrder(x, sizes, {1, 0}), /*dim_order=*/{1, 0});
  EXPECT_EQ(method->set_input(EValue(transposed), 0), Error::InvalidArgument);

  // A rejected input must not leave the method unable to run valid ones.
  ASSERT_EQ(method->set_input(EValue(tf_.make(sizes, x)), 0), Error::Ok);
  ASSERT_EQ(method->set_input(EValue(tf_.make(sizes, y)), 1), Error::Ok);
  ASSERT_EQ(method->set_input(EValue(1.0), 2), Error::Ok);
  ASSERT_EQ(method->execute(), Error::Ok);
  EXPECT_EQ(
      LogicalValues(method->get_output(0).toTensor()),
      (std::vector<float>{11, 22, 33, 44}));
}
