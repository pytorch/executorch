/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/flat_tensor/flat_tensor_data_map.h>
#include <executorch/extension/training/module/state_dict_util.h>

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

class LoadStateDictTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();

    // Load data map.
    // The eager linear model is defined at:
    // //executorch/test/models/linear_model.py
    const char* path = std::getenv("ET_MODULE_ADD_MUL_DATA_PATH");
    Result<FileDataLoader> loader = FileDataLoader::from(path);
    ASSERT_EQ(loader.error(), Error::Ok);

    Result<FreeableBuffer> header = loader->load(
        /*offset=*/0,
        FlatTensorHeader::kNumHeadBytes,
        /*segment_info=*/
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));

    ASSERT_EQ(header.error(), Error::Ok);

    data_map_loader_ =
        std::make_unique<FileDataLoader>(std::move(loader.get()));
  }
  std::unique_ptr<FileDataLoader> data_map_loader_;
};

TEST_F(LoadStateDictTest, LoadDataMap) {
  Result<FlatTensorDataMap> data_map =
      FlatTensorDataMap::load(data_map_loader_.get());
  EXPECT_EQ(data_map.error(), Error::Ok);

  auto state_dict =
      executorch::extension::training::load_state_dict(data_map.get());
  ASSERT_TRUE(state_dict.ok());

  EXPECT_EQ(state_dict->size(), 2);
  EXPECT_EQ(state_dict->at("a")->sizes().size(), 2);
  EXPECT_EQ(state_dict->at("a")->sizes()[0], 2);
  EXPECT_EQ(state_dict->at("a")->sizes()[1], 2);
  EXPECT_EQ(
      state_dict->at("a")->scalar_type(), torch::executor::ScalarType::Float);
  EXPECT_EQ(state_dict->at("a")->dim(), 2);
  EXPECT_EQ(state_dict->at("a")->const_data_ptr<float>()[0], 3.f);
  EXPECT_EQ(state_dict->at("a")->const_data_ptr<float>()[1], 3.f);
  EXPECT_EQ(state_dict->at("a")->const_data_ptr<float>()[2], 3.f);
  EXPECT_EQ(state_dict->at("a")->const_data_ptr<float>()[3], 3.f);

  EXPECT_EQ(state_dict->size(), 2);
  EXPECT_EQ(state_dict->at("b")->sizes().size(), 2);
  EXPECT_EQ(state_dict->at("b")->sizes()[0], 2);
  EXPECT_EQ(state_dict->at("b")->sizes()[1], 2);
  EXPECT_EQ(
      state_dict->at("b")->scalar_type(), torch::executor::ScalarType::Float);
  EXPECT_EQ(state_dict->at("b")->dim(), 2);
  EXPECT_EQ(state_dict->at("b")->const_data_ptr<float>()[0], 2.f);
  EXPECT_EQ(state_dict->at("b")->const_data_ptr<float>()[1], 2.f);
  EXPECT_EQ(state_dict->at("b")->const_data_ptr<float>()[2], 2.f);
  EXPECT_EQ(state_dict->at("b")->const_data_ptr<float>()[3], 2.f);
}
