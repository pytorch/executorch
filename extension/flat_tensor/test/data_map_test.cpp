/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/flat_tensor/named_data_map/data_map.h>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>

#include <executorch/extension/flat_tensor/serialize/flat_tensor_header.h>
#include <executorch/extension/flat_tensor/serialize/schema_generated.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::extension::DataMap;
using executorch::extension::FlatTensorHeader;
using executorch::runtime::DataLoader;
using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;
using executorch::runtime::TensorLayout;
using torch::executor::util::FileDataLoader;

class DataMapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();
  }
};

TEST_F(DataMapTest, LoadDataMap) {
  const char* path = std::getenv("ET_MODULE_LINEAR_DATA");
  Result<FileDataLoader> loader = FileDataLoader::from(path);
  ASSERT_EQ(loader.error(), Error::Ok);

  Result<FreeableBuffer> header = loader->load(
      /*offset=*/0,
      FlatTensorHeader::kNumHeadBytes,
      /*segment_info=*/
      DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::External));

  ASSERT_EQ(header.error(), Error::Ok);

  auto data_map_loader_ =
      std::make_unique<FileDataLoader>(std::move(loader.get()));

  Result<DataMap> data_map = DataMap::load(data_map_loader_.get());
  EXPECT_EQ(data_map.error(), Error::Ok);

  // Check tensor metadata.
  Result<const TensorLayout> const_a_res = data_map->get_metadata("a");
  assert(const_a_res.ok());

  const TensorLayout const_a = const_a_res.get();
  EXPECT_EQ(const_a.scalar_type(), executorch::aten::ScalarType::Float);
  auto sizes_a = const_a.sizes();
  EXPECT_EQ(sizes_a.size(), 2);
  EXPECT_EQ(sizes_a[0], 2);
  EXPECT_EQ(sizes_a[1], 2);
  auto dim_order_a = const_a.dim_order();
  EXPECT_EQ(dim_order_a.size(), 2);
  EXPECT_EQ(dim_order_a[0], 0);
  EXPECT_EQ(dim_order_a[1], 1);

  Result<const TensorLayout> const_b_res = data_map->get_metadata("b");
  assert(const_b_res.ok());

  const TensorLayout const_b = const_b_res.get();
  EXPECT_EQ(const_b.scalar_type(), executorch::aten::ScalarType::Float);
  auto sizes_b = const_b.sizes();
  EXPECT_EQ(sizes_b.size(), 2);
  EXPECT_EQ(sizes_b[0], 2);
  EXPECT_EQ(sizes_b[1], 2);
  auto dim_order_b = const_b.dim_order();
  EXPECT_EQ(dim_order_b.size(), 2);
  EXPECT_EQ(dim_order_b[0], 0);
  EXPECT_EQ(dim_order_b[1], 1);

  // Check tensor data.
  Result<FreeableBuffer> data_a_res = data_map->get_data("a");
  assert(data_a_res.ok());
  // Check we have the correct tensor data.
  FreeableBuffer data_a = std::move(data_a_res.get());
  EXPECT_EQ(data_a.size(), 16);

  Result<FreeableBuffer> data_b_res = data_map->get_data("b");
  assert(data_b_res.ok());
  // Check we have the correct tensor data.
  FreeableBuffer data_b = std::move(data_b_res.get());
  EXPECT_EQ(data_b.size(), 16);
}
