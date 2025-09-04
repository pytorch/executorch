/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/named_data_map/pt2_archive_data_map.h>

#include <executorch/extension/data_loader/mmap_data_loader.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>

#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/test/managed_memory_manager.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::extension::MmapDataLoader;
using executorch::extension::PT2ArchiveDataMap;
using executorch::runtime::DataLoader;
using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Method;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::TensorLayout;

using executorch::runtime::testing::ManagedMemoryManager;

constexpr size_t kDefaultNonConstMemBytes = 32 * 1024U;
constexpr size_t kDefaultRuntimeMemBytes = 32 * 1024U;

class PT2ArchiveDataMapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();

    archive_path_ = std::getenv("TEST_LINEAR_PT2");
    const char* model_path = std::getenv("ET_MODULE_LINEAR_PATH");

    Result<MmapDataLoader> loader = MmapDataLoader::from(model_path);
    program_loader_ = std::make_unique<MmapDataLoader>(std::move(loader.get()));

    Result<Program> program = Program::load(
        program_loader_.get(), Program::Verification::InternalConsistency);
    ASSERT_EQ(program.error(), Error::Ok);
    program_ = std::make_unique<Program>(std::move(program.get()));
  }
  static inline std::string archive_path_;
  std::unique_ptr<MmapDataLoader> program_loader_;
  std::unique_ptr<Program> program_;
};

TEST_F(PT2ArchiveDataMapTest, TestLoad) {
  auto archive_data_map = PT2ArchiveDataMap::load(archive_path_);
  EXPECT_EQ(archive_data_map.error(), Error::Ok);
}

TEST_F(PT2ArchiveDataMapTest, GetTensorLayout) {
  auto data_map = PT2ArchiveDataMap::load(archive_path_);
  EXPECT_EQ(data_map.error(), Error::Ok);

  Result<const TensorLayout> const a_res =
      data_map->get_tensor_layout("linear.weight");
  EXPECT_EQ(a_res.error(), Error::Ok);
  const TensorLayout& a = a_res.get();
  EXPECT_EQ(a.scalar_type(), executorch::aten::ScalarType::Float);
  auto sizes_a = a.sizes();
  EXPECT_EQ(sizes_a.size(), 2);
  EXPECT_EQ(sizes_a[0], 3);
  EXPECT_EQ(sizes_a[1], 3);

  Result<const TensorLayout> const b_res =
      data_map->get_tensor_layout("linear.bias");
  EXPECT_EQ(b_res.error(), Error::Ok);

  const TensorLayout& b = b_res.get();
  EXPECT_EQ(b.scalar_type(), executorch::aten::ScalarType::Float);
  auto sizes_b = b.sizes();
  EXPECT_EQ(sizes_b.size(), 1);
  EXPECT_EQ(sizes_b[0], 3);
}

TEST_F(PT2ArchiveDataMapTest, GetTensorData) {
  auto data_map = PT2ArchiveDataMap::load(archive_path_);
  EXPECT_EQ(data_map.error(), Error::Ok);

  Result<FreeableBuffer> weight = data_map->get_data("linear.weight");
  EXPECT_EQ(weight.error(), Error::Ok);
  FreeableBuffer& a = weight.get();
  EXPECT_EQ(a.size(), 36);

  Result<FreeableBuffer> bias = data_map->get_data("linear.bias");
  EXPECT_EQ(bias.error(), Error::Ok);
  FreeableBuffer& b = bias.get();
  EXPECT_EQ(b.size(), 12);

  weight->Free();
  bias->Free();
}

TEST_F(PT2ArchiveDataMapTest, GetKeys) {
  auto data_map = PT2ArchiveDataMap::load(archive_path_);
  EXPECT_EQ(data_map.error(), Error::Ok);

  EXPECT_EQ(data_map->get_num_keys().get(), 2);
  EXPECT_EQ(strcmp(data_map->get_key(0).get(), "linear.weight"), 0);
  EXPECT_EQ(strcmp(data_map->get_key(1).get(), "linear.bias"), 0);
}

TEST_F(PT2ArchiveDataMapTest, E2E) {
  auto data_map = PT2ArchiveDataMap::load(archive_path_);
  EXPECT_EQ(data_map.error(), Error::Ok);
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);

  std::unique_ptr<executorch::runtime::NamedDataMap> data_map_ptr =
      std::make_unique<PT2ArchiveDataMap>(std::move(data_map.get()));
  Result<Method> method =
      program_->load_method("forward", &mmm.get(), nullptr, data_map_ptr.get());
  ASSERT_EQ(method.error(), Error::Ok);
}
