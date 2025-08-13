/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/flat_tensor/flat_tensor_data_map.h>
#include <executorch/extension/flat_tensor/serialize/flat_tensor_generated.h>
#include <executorch/extension/flat_tensor/serialize/flat_tensor_header.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::extension::FlatTensorDataMap;
using executorch::extension::FlatTensorHeader;
using executorch::runtime::DataLoader;
using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;
using executorch::runtime::TensorLayout;
using torch::executor::util::FileDataLoader;

class FlatTensorDataMapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();

    // Load data map. The eager addmul model is defined at:
    // //executorch/test/models/export_program.py
    const char* path = std::getenv("ET_MODULE_ADD_MUL_DATA_PATH");
    Result<FileDataLoader> loader = FileDataLoader::from(path);
    ASSERT_EQ(loader.error(), Error::Ok);

    data_map_loader_ =
        std::make_unique<FileDataLoader>(std::move(loader.get()));
  }
  std::unique_ptr<FileDataLoader> data_map_loader_;
};

TEST_F(FlatTensorDataMapTest, LoadFlatTensorDataMap) {
  Result<FlatTensorDataMap> data_map =
      FlatTensorDataMap::load(data_map_loader_.get());
  EXPECT_EQ(data_map.error(), Error::Ok);
}

TEST_F(FlatTensorDataMapTest, FlatTensorDataMap_GetMetadata) {
  Result<FlatTensorDataMap> data_map =
      FlatTensorDataMap::load(data_map_loader_.get());
  EXPECT_EQ(data_map.error(), Error::Ok);

  // Check tensor layouts are correct.
  // From //executorch/test/models/linear_model.py, we have the tensors
  // self.a = 3 * torch.ones(2, 2, dtype=torch.float)
  // self.b = 2 * torch.ones(2, 2, dtype=torch.float)
  Result<const TensorLayout> const_a_res = data_map->get_tensor_layout("a");
  ASSERT_EQ(Error::Ok, const_a_res.error());

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

  Result<const TensorLayout> const_b_res = data_map->get_tensor_layout("b");
  ASSERT_EQ(Error::Ok, const_b_res.error());

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

  // Check get_tensor_layout fails when key is not found.
  Result<const TensorLayout> const_c_res = data_map->get_tensor_layout("c");
  EXPECT_EQ(const_c_res.error(), Error::NotFound);
}

TEST_F(FlatTensorDataMapTest, FlatTensorDataMap_GetData) {
  Result<FlatTensorDataMap> data_map =
      FlatTensorDataMap::load(data_map_loader_.get());
  EXPECT_EQ(data_map.error(), Error::Ok);

  // Check tensor data sizes are correct.
  Result<FreeableBuffer> data_a_res = data_map->get_data("a");
  ASSERT_EQ(Error::Ok, data_a_res.error());
  FreeableBuffer data_a = std::move(data_a_res.get());
  EXPECT_EQ(data_a.size(), 16);

  Result<FreeableBuffer> data_b_res = data_map->get_data("b");
  ASSERT_EQ(Error::Ok, data_b_res.error());
  FreeableBuffer data_b = std::move(data_b_res.get());
  EXPECT_EQ(data_b.size(), 16);

  // Check get_data fails when key is not found.
  Result<FreeableBuffer> data_c_res = data_map->get_data("c");
  EXPECT_EQ(data_c_res.error(), Error::NotFound);
}

TEST_F(FlatTensorDataMapTest, FlatTensorDataMap_Keys) {
  Result<FlatTensorDataMap> data_map =
      FlatTensorDataMap::load(data_map_loader_.get());
  EXPECT_EQ(data_map.error(), Error::Ok);

  // Check num tensors is 2.
  Result<uint32_t> num_tensors_res = data_map->get_num_keys();
  ASSERT_EQ(Error::Ok, num_tensors_res.error());
  EXPECT_EQ(num_tensors_res.get(), 2);

  // Check get_key returns the correct keys.
  Result<const char*> key0_res = data_map->get_key(0);
  ASSERT_EQ(Error::Ok, key0_res.error());
  EXPECT_EQ(strcmp(key0_res.get(), "a"), 0);

  Result<const char*> key1_res = data_map->get_key(1);
  ASSERT_EQ(Error::Ok, key1_res.error());
  EXPECT_EQ(strcmp(key1_res.get(), "b"), 0);

  // Check get_key fails when out of bounds.
  Result<const char*> key2_res = data_map->get_key(2);
  EXPECT_EQ(key2_res.error(), Error::InvalidArgument);
}

TEST_F(FlatTensorDataMapTest, FlatTensorDataMap_LoadInto) {
  Result<FlatTensorDataMap> data_map =
      FlatTensorDataMap::load(data_map_loader_.get());
  EXPECT_EQ(data_map.error(), Error::Ok);

  // get the metadata
  auto meta_data_res = data_map->get_tensor_layout("a");
  ASSERT_EQ(meta_data_res.error(), Error::Ok);

  // get data blob
  void* data = malloc(meta_data_res->nbytes());
  auto load_into_error =
      data_map->load_data_into("a", data, meta_data_res->nbytes());
  ASSERT_EQ(load_into_error, Error::Ok);

  // Check tensor data is correct.
  float* data_a = static_cast<float*>(data);
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(data_a[i], 3.0);
  }
  free(data);
}
