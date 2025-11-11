/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/flat_tensor/flat_tensor_data_map.h>
#include <executorch/extension/named_data_map/merged_data_map.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

#include <memory>
#include <unordered_map>
#include <vector>

using namespace ::testing;
using executorch::extension::FileDataLoader;
using executorch::extension::FlatTensorDataMap;
using executorch::extension::merged_data_map::MergedDataMap;
using executorch::runtime::DataLoader;
using executorch::runtime::Error;
using executorch::runtime::NamedDataMap;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::TensorLayout;

class MergedDataMapTest : public ::testing::Test {
 protected:
  void load_flat_tensor_data_map(const char* path, const char* module_name) {
    Result<FileDataLoader> loader = FileDataLoader::from(path);
    ASSERT_EQ(loader.error(), Error::Ok);
    loaders_.emplace(
        module_name, std::make_unique<FileDataLoader>(std::move(loader.get())));

    Result<FlatTensorDataMap> data_map =
        FlatTensorDataMap::load(loaders_[module_name].get());
    EXPECT_EQ(data_map.error(), Error::Ok);

    data_maps_.emplace(
        module_name,
        std::make_unique<FlatTensorDataMap>(std::move(data_map.get())));
  }

  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();

    // Load FlatTensor data maps.
    // The eager addmul and linear models are defined at:
    // //executorch/test/models/export_program.py
    load_flat_tensor_data_map(
        std::getenv("ET_MODULE_ADD_MUL_DATA_PATH"), "addmul");
    load_flat_tensor_data_map(
        std::getenv("ET_MODULE_LINEAR_DATA_PATH"), "linear");
  }

 private:
  // Must outlive data_maps_, but tests shouldn't need to touch it.
  std::unordered_map<std::string, std::unique_ptr<FileDataLoader>> loaders_;

 protected:
  std::unordered_map<std::string, std::unique_ptr<NamedDataMap>> data_maps_;
};

// Check that two tensor layouts are equivalent.
void check_tensor_layout(TensorLayout& layout1, TensorLayout& layout2) {
  EXPECT_EQ(layout1.scalar_type(), layout2.scalar_type());
  EXPECT_EQ(layout1.nbytes(), layout2.nbytes());
  EXPECT_EQ(layout1.sizes().size(), layout2.sizes().size());
  for (auto i : c10::irange(layout1.sizes().size())) {
    EXPECT_EQ(layout1.sizes()[i], layout2.sizes()[i]);
  }
  EXPECT_EQ(layout1.dim_order().size(), layout2.dim_order().size());
  for (auto i : c10::irange(layout1.dim_order().size())) {
    EXPECT_EQ(layout1.dim_order()[i], layout2.dim_order()[i]);
  }
}

// Given that ndm is part of merged, check that all the API calls on ndm produce
// the same results as merged.
void compare_ndm_api_calls(
    const NamedDataMap* ndm,
    const NamedDataMap* merged) {
  uint32_t num_keys = ndm->get_num_keys().get();
  for (auto i : c10::irange(num_keys)) {
    auto key = ndm->get_key(i).get();

    // Compare get_tensor_layout.
    auto ndm_meta = ndm->get_tensor_layout(key).get();
    auto merged_meta = merged->get_tensor_layout(key).get();
    check_tensor_layout(ndm_meta, merged_meta);

    // Compare get_data.
    auto ndm_data = ndm->get_data(key);
    auto merged_data = merged->get_data(key);
    EXPECT_EQ(ndm_data.get().size(), merged_data.get().size());
    for (auto j : c10::irange(ndm_meta.nbytes())) {
      EXPECT_EQ(
          ((uint8_t*)ndm_data.get().data())[j],
          ((uint8_t*)merged_data.get().data())[j]);
    }
    ndm_data->Free();
    merged_data->Free();

    // Compare load_into.
    auto nbytes = ndm_meta.nbytes();
    auto ndm_buffer = std::make_unique<uint8_t[]>(nbytes);
    auto ndm_load_into = ndm->load_data_into(key, ndm_buffer.get(), nbytes);
    EXPECT_EQ(ndm_load_into, Error::Ok);
    auto merged_buffer = std::make_unique<uint8_t[]>(nbytes);
    auto merged_load_into =
        merged->load_data_into(key, merged_buffer.get(), nbytes);
    EXPECT_EQ(merged_load_into, Error::Ok);
    for (auto j : c10::irange(ndm_meta.nbytes())) {
      EXPECT_EQ(
          ((uint8_t*)merged_buffer.get())[j],
          ((uint8_t*)merged_buffer.get())[j]);
    }
  }
}

TEST_F(MergedDataMapTest, LoadNullDataMap) {
  Result<MergedDataMap> merged_map = MergedDataMap::load({nullptr, nullptr});
  EXPECT_EQ(merged_map.error(), Error::InvalidArgument);
}

TEST_F(MergedDataMapTest, LoadSingleDataMap) {
  std::vector<const NamedDataMap*> ndms = {data_maps_["addmul"].get(), nullptr};
  Result<MergedDataMap> merged_map =
      MergedDataMap::load(Span<const NamedDataMap*>(ndms.data(), ndms.size()));
  EXPECT_EQ(merged_map.error(), Error::Ok);

  // Num keys.
  EXPECT_EQ(
      merged_map->get_num_keys().get(),
      data_maps_["addmul"]->get_num_keys().get());

  // API calls produce equivalent results.
  compare_ndm_api_calls(data_maps_["addmul"].get(), &merged_map.get());
}

TEST_F(MergedDataMapTest, LoadDuplicateDataMapsFail) {
  std::vector<const NamedDataMap*> ndms = {
      data_maps_["addmul"].get(), data_maps_["addmul"].get()};
  Result<MergedDataMap> merged_map =
      MergedDataMap::load(Span<const NamedDataMap*>(ndms.data(), ndms.size()));
  EXPECT_EQ(merged_map.error(), Error::InvalidArgument);
}

TEST_F(MergedDataMapTest, CheckDataMapContents) {
  std::vector<const NamedDataMap*> ndms = {
      data_maps_["addmul"].get(), data_maps_["linear"].get()};
  Result<MergedDataMap> merged_map =
      MergedDataMap::load(Span<const NamedDataMap*>(ndms.data(), ndms.size()));
  EXPECT_EQ(merged_map.error(), Error::Ok);

  // Num keys.
  size_t addmul_num_keys = data_maps_["addmul"]->get_num_keys().get();
  size_t linear_num_keys = data_maps_["linear"]->get_num_keys().get();
  EXPECT_EQ(
      merged_map->get_num_keys().get(), addmul_num_keys + linear_num_keys);

  // API calls produce equivalent results.
  compare_ndm_api_calls(data_maps_["addmul"].get(), &merged_map.get());
  compare_ndm_api_calls(data_maps_["linear"].get(), &merged_map.get());
}
