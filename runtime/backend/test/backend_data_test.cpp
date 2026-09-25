/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/backend/interface.h>

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <new>
#include <optional>
#include <utility>

using executorch::runtime::ArrayRef;
using executorch::runtime::BackendData;
using executorch::runtime::BackendDataInitContext;
using executorch::runtime::BackendDataInput;
using executorch::runtime::BackendDataWriter;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::BackendInterface;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::NamedBackendData;
using executorch::runtime::Result;
using executorch::runtime::Span;

namespace {

class TestBackendBase : public BackendInterface {
 public:
  bool is_available() const override {
    return true;
  }

  Result<DelegateHandle*> init(
      BackendInitContext&,
      FreeableBuffer*,
      ArrayRef<CompileSpec>) const override {
    return nullptr;
  }

  Error execute(BackendExecutionContext&, DelegateHandle*, Span<EValue*>)
      const override {
    return Error::Ok;
  }
};

class EmptyDataInitContext final : public BackendDataInitContext {
 public:
  explicit EmptyDataInitContext(executorch::runtime::MemoryAllocator* allocator)
      : BackendDataInitContext(allocator, nullptr, nullptr) {}

  Result<std::optional<BackendDataInput>> next_backend_data() override {
    return std::optional<BackendDataInput>{};
  }
};

class RecordingWriter final : public BackendDataWriter {
 public:
  Error write_named_data(Span<const NamedBackendData> data) override {
    if (data.size() != 2 || data[0].data.bytes.size() != 2 ||
        data[1].data.bytes.size() != 0) {
      return Error::InvalidArgument;
    }
    named_keys_match = std::strcmp(data[0].key, "weight") == 0 &&
        std::strcmp(data[1].key, "bias") == 0;
    named_alignment = data[0].data.alignment.value_or(0);
    std::memcpy(
        named_value.data(),
        data[0].data.bytes.data(),
        data[0].data.bytes.size());
    ++named_write_count;
    return Error::Ok;
  }

  size_t named_write_count{0};
  bool named_keys_match{false};
  size_t named_alignment{0};
  std::array<uint8_t, 2> named_value{};
};

class TrackingDataInitContext final : public BackendDataInitContext {
 public:
  explicit TrackingDataInitContext(
      executorch::runtime::MemoryAllocator* allocator)
      : BackendDataInitContext(allocator, nullptr, nullptr) {}

  Result<std::optional<BackendDataInput>> next_backend_data() override {
    if (index_ == inputs_.size()) {
      return std::optional<BackendDataInput>{};
    }

    BackendDataInput input{
        method_names_[index_],
        Span<const uint8_t>(inputs_[index_].data(), inputs_[index_].size()),
        ArrayRef<CompileSpec>()};
    ++index_;
    return std::optional<BackendDataInput>{std::move(input)};
  }

  size_t yielded() const {
    return index_;
  }

 private:
  const std::array<std::array<uint8_t, 2>, 2> inputs_{{{1, 2}, {3, 4}}};
  const std::array<const char*, 2> method_names_{{"prefill", "decode"}};
  size_t index_{0};
};

class PreparingBackend final : public TestBackendBase {
 public:
  Error initialize_backend_data(
      BackendDataInitContext& context,
      BackendDataWriter& output) const override {
    while (true) {
      auto input_result = context.next_backend_data();
      if (!input_result.ok()) {
        return input_result.error();
      }
      std::optional<BackendDataInput> input_optional =
          std::move(input_result.get());
      if (!input_optional.has_value()) {
        break;
      }

      (void)input_optional;
    }

    auto* packed = context.get_temp_allocator()->allocateList<uint8_t>(2);
    auto* values =
        context.get_temp_allocator()->allocateList<NamedBackendData>(2);
    auto* weight_key = context.get_temp_allocator()->allocateList<char>(7);
    auto* bias_key = context.get_temp_allocator()->allocateList<char>(5);
    if (packed == nullptr || values == nullptr || weight_key == nullptr ||
        bias_key == nullptr) {
      return Error::MemoryAllocationFailed;
    }
    packed[0] = 5;
    packed[1] = 6;
    std::memcpy(weight_key, "weight", 7);
    std::memcpy(bias_key, "bias", 5);
    new (&values[0]) NamedBackendData{
        weight_key, BackendData{Span<const uint8_t>(packed, 2), 128}};
    new (&values[1]) NamedBackendData{
        bias_key,
        BackendData{Span<const uint8_t>(nullptr, static_cast<size_t>(0)), 128}};
    return output.write_named_data(Span<const NamedBackendData>(values, 2));
  }
};

TEST(BackendDataTest, DefaultImplementationIsNotSupported) {
  std::array<uint8_t, 64> allocator_storage{};
  executorch::runtime::MemoryAllocator allocator(
      allocator_storage.size(), allocator_storage.data());
  EmptyDataInitContext context(&allocator);
  RecordingWriter writer;
  TestBackendBase backend;

  EXPECT_EQ(
      backend.initialize_backend_data(context, writer), Error::NotSupported);
}

TEST(BackendDataTest, BackendConsumesInputsLazilyAndEmitsLogicalOutputs) {
  std::array<uint8_t, 1024> allocator_storage{};
  executorch::runtime::MemoryAllocator allocator(
      allocator_storage.size(), allocator_storage.data());
  TrackingDataInitContext context(&allocator);
  RecordingWriter writer;
  PreparingBackend backend;

  ASSERT_EQ(backend.initialize_backend_data(context, writer), Error::Ok);
  EXPECT_EQ(context.yielded(), 2U);
  EXPECT_EQ(writer.named_write_count, 1U);
  EXPECT_TRUE(writer.named_keys_match);
  EXPECT_EQ(writer.named_alignment, 128U);
  EXPECT_EQ(writer.named_value, (std::array<uint8_t, 2>{5, 6}));
}

} // namespace
