/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include <executorch/backends/arm/runtime/EthosUBackend_Internal.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/core/tensor_layout.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::runtime::Error;

namespace executorch::backends::arm {

// cppcheck-suppress unusedFunction
Error platform_init(
    executorch::runtime::ArrayRef<executorch::runtime::CompileSpec>,
    executorch::runtime::MemoryAllocator*,
    ExecutionHandle*) {
  return Error::Ok;
}

// cppcheck-suppress unusedFunction
void platform_destroy(PlatformState*) {}

// cppcheck-suppress unusedFunction
bool needs_scratch_allocation() {
  return false;
}

// cppcheck-suppress unusedFunction
Error platform_execute(
    executorch::runtime::BackendExecutionContext&,
    const ExecutionHandle*,
    IOPlan&) {
  return Error::Ok;
}

} // namespace executorch::backends::arm

namespace {

using executorch::backends::arm::ExecutionHandle;
using executorch::backends::arm::kVelaExternalBlockReference;
using executorch::backends::arm::VelaBinBlock;
using executorch::runtime::BackendInitContext;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::NamedDataMap;
using executorch::runtime::Result;
using executorch::runtime::TensorLayout;

constexpr std::string_view kCommandKey =
    "g123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
constexpr std::string_view kWeightKey =
    "h123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

void write_external_block(
    uint8_t* block,
    std::string_view name,
    std::string_view key) {
  std::memcpy(block, name.data(), name.size());
  const uint32_t size = key.size();
  for (size_t i = 0; i < sizeof(size); ++i) {
    block[offsetof(VelaBinBlock, size) + i] = (size >> (8 * i)) & 0xff;
  }
  block[offsetof(VelaBinBlock, external)] = kVelaExternalBlockReference;
  std::memcpy(block + offsetof(VelaBinBlock, data), key.data(), key.size());
}

class OwningNamedDataMap final : public NamedDataMap {
 public:
  struct Entry {
    std::string_view key;
    void* data;
    size_t size;
  };

  OwningNamedDataMap(Entry command, Entry weights, int& releases)
      : entries_{command, weights}, releases_(releases) {}

  Result<const TensorLayout> get_tensor_layout(
      std::string_view) const override {
    return Error::NotFound;
  }

  Result<FreeableBuffer> get_data(std::string_view key) const override {
    for (const auto& entry : entries_) {
      if (entry.key == key) {
        return FreeableBuffer(
            entry.data,
            entry.size,
            [](void* context, void* data, size_t size) {
              ++*static_cast<int*>(context);
              std::memset(data, 0, size);
            },
            &releases_);
      }
    }
    return Error::NotFound;
  }

  Error load_data_into(std::string_view, void*, size_t) const override {
    return Error::NotImplemented;
  }

  Result<uint32_t> get_num_keys() const override {
    return entries_.size();
  }

  Result<const char*> get_key(uint32_t index) const override {
    return index < entries_.size()
        ? Result<const char*>(entries_[index].key.data())
        : Result<const char*>(Error::NotFound);
  }

 private:
  std::array<Entry, 2> entries_;
  int& releases_;
};

// cppcheck-suppress syntaxError
TEST(EthosUBackendInitTest, KeepsExternalPayloadsUntilDestroy) {
  executorch::runtime::runtime_init();
  alignas(16) std::array<uint8_t, 4> command{'C', 'O', 'P', '1'};
  alignas(16) std::array<uint8_t, 4> weights{1, 2, 3, 4};
  constexpr size_t block_size = sizeof(VelaBinBlock) + kCommandKey.size();
  alignas(16) std::array<uint8_t, 2 * sizeof(VelaBinBlock) + 2 * block_size>
      stream{};
  std::memcpy(stream.data(), "vela_bin_stream", 15);
  write_external_block(
      stream.data() + sizeof(VelaBinBlock), "cmd_data", kCommandKey);
  write_external_block(
      stream.data() + sizeof(VelaBinBlock) + block_size,
      "weight_data",
      kWeightKey);
  std::memcpy(
      stream.data() + sizeof(VelaBinBlock) + 2 * block_size,
      "vela_end_stream",
      15);

  int releases = 0;
  OwningNamedDataMap data_map(
      {kCommandKey, command.data(), command.size()},
      {kWeightKey, weights.data(), weights.size()},
      releases);
  alignas(16) std::array<uint8_t, 2048> allocator_storage{};
  MemoryAllocator allocator(allocator_storage.size(), allocator_storage.data());
  BackendInitContext context(&allocator, nullptr, nullptr, &data_map);
  FreeableBuffer processed(stream.data(), stream.size(), nullptr);
  const auto* backend = executorch::runtime::get_backend_class("EthosUBackend");
  ASSERT_NE(backend, nullptr);

  auto result = backend->init(context, &processed, {});
  ASSERT_TRUE(result.ok());
  auto* handle = static_cast<ExecutionHandle*>(result.get());
  EXPECT_EQ(releases, 0);
  EXPECT_EQ(handle->handles.cmd_data, reinterpret_cast<char*>(command.data()));
  EXPECT_EQ(std::memcmp(handle->handles.cmd_data, "COP1", command.size()), 0);
  EXPECT_EQ(
      handle->handles.weight_data, reinterpret_cast<char*>(weights.data()));
  EXPECT_EQ(
      std::memcmp(
          handle->handles.weight_data, "\x01\x02\x03\x04", weights.size()),
      0);

  backend->destroy(handle);
  EXPECT_EQ(releases, 2);
  EXPECT_EQ(command[0], 0);
  EXPECT_EQ(weights[0], 0);
}

} // namespace
