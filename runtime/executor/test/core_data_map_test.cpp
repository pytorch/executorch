/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/testing_util/temp_file.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/executor/core_data_map.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/schema/program_generated.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::extension::FileDataLoader;
using executorch::extension::testing::TempFile;
using executorch::runtime::CoreDataMap;
using executorch::runtime::DataLoader;
using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;

class CoreDataMapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();

    // Create a sample Program with only named_data and segments. Technically
    // not a valid Program; only used to test the CoreDataMap.
    // Create named data.
    const flatbuffers::Offset<executorch_flatbuffer::NamedData>
        named_data_arr[4] = {
            executorch_flatbuffer::CreateNamedDataDirect(
                builder_, "key0", /*segment_index=*/0),
            executorch_flatbuffer::CreateNamedDataDirect(
                builder_, "key1", /*segment_index=*/1),
            executorch_flatbuffer::CreateNamedDataDirect(
                builder_, "key2", /*segment_index=*/0),
            executorch_flatbuffer::CreateNamedDataDirect(
                builder_, "key_invalid", /*segment_index=*/10),
        };
    const auto named_data = builder_.CreateVector(named_data_arr, 4);

    // Create segments.
    const flatbuffers::Offset<executorch_flatbuffer::DataSegment>
        segment_arr[2] = {
            executorch_flatbuffer::CreateDataSegment(
                builder_, /*offset=*/0, /*size=*/SEGMENT_SIZES[0]),
            executorch_flatbuffer::CreateDataSegment(
                builder_,
                /*offset=*/SEGMENT_ALIGNMENT * 2,
                /*size=*/SEGMENT_SIZES[1])};
    const auto segments = builder_.CreateVector(segment_arr, 2);

    // Create Program.
    const auto program = executorch_flatbuffer::CreateProgram(
        builder_, 0, 0, 0, 0, segments, 0, 0, named_data);

    builder_.Finish(program);
    program_ = executorch_flatbuffer::GetProgram(builder_.GetBufferPointer());

    // Create sample segment data.
    uint8_t sample_data[64] = {};
    for (int i = 0; i < SEGMENT_SIZES[0]; i++) {
      sample_data[i] = 1;
    }
    for (int i = SEGMENT_OFFSETS[1]; i < SEGMENT_OFFSETS[1] + SEGMENT_SIZES[1];
         i++) {
      sample_data[i] = 2;
    }
    TempFile tf(sample_data, sizeof(sample_data));

    // Wrap the sample data in a loader.
    Result<FileDataLoader> loader = FileDataLoader::from(tf.path().c_str(), 16);
    ASSERT_EQ(loader.error(), Error::Ok);
    data_map_loader_ =
        std::make_unique<FileDataLoader>(std::move(loader.get()));
  }

  // Program builder constants.
  int SEGMENT_ALIGNMENT = 16;
  int SEGMENT_SIZES[2] = {17, 8};
  int SEGMENT_OFFSETS[2] = {0, SEGMENT_ALIGNMENT * 2};

  // Program builder.
  flatbuffers::FlatBufferBuilder builder_;
  const executorch_flatbuffer::Program* program_;

  // Data loader for the sample data.
  std::unique_ptr<FileDataLoader> data_map_loader_;
};

TEST_F(CoreDataMapTest, CoreDataMap_Load) {
  Result<CoreDataMap> data_map = CoreDataMap::load(
      data_map_loader_.get(), 0, program_->named_data(), program_->segments());
  EXPECT_EQ(data_map.error(), Error::Ok);
}

TEST_F(CoreDataMapTest, CoreDataMap_LoadFail) {
  Result<CoreDataMap> data_map = CoreDataMap::load(
      nullptr, 0, program_->named_data(), program_->segments());
  EXPECT_EQ(data_map.error(), Error::InvalidArgument);
}

TEST_F(CoreDataMapTest, CoreDataMap_UnimplementedMethods) {
  Result<CoreDataMap> data_map = CoreDataMap::load(
      data_map_loader_.get(), 0, program_->named_data(), program_->segments());
  EXPECT_EQ(data_map.error(), Error::Ok);

  // Check get_metadata is not implemented.
  auto result = data_map->get_metadata("sample_key");
  EXPECT_EQ(result.error(), Error::NotImplemented);

  // Check load_data_into is not implemented.
  auto err = data_map->load_data_into("sample_key", nullptr, 0);
  EXPECT_EQ(err, Error::NotImplemented);
}

TEST_F(CoreDataMapTest, CoreDataMap_Keys) {
  Result<CoreDataMap> data_map = CoreDataMap::load(
      data_map_loader_.get(), 0, program_->named_data(), program_->segments());
  EXPECT_EQ(data_map.error(), Error::Ok);

  // Check get_num_keys.
  auto num_keys = data_map->get_num_keys();
  EXPECT_EQ(num_keys.error(), Error::Ok);
  EXPECT_EQ(num_keys.get(), 4);

  // Check get_key_at.
  auto key0 = data_map->get_key(0);
  EXPECT_EQ(strcmp(key0.get(), "key0"), 0);
  auto key1 = data_map->get_key(1);
  EXPECT_EQ(strcmp(key1.get(), "key1"), 0);
  auto key2 = data_map->get_key(2);
  EXPECT_EQ(strcmp(key2.get(), "key2"), 0);

  // Invalid key exists. Note: practically, a PTE should not have invalid keys.
  auto key_invalid = data_map->get_key(3);
  EXPECT_EQ(strcmp(key_invalid.get(), "key_invalid"), 0);

  // Throw error on non-existent key.
  auto nonexistent_key = data_map->get_key(10);
  EXPECT_EQ(nonexistent_key.error(), Error::InvalidArgument);
}

TEST_F(CoreDataMapTest, CoreDataMap_GetData) {
  Result<CoreDataMap> data_map = CoreDataMap::load(
      data_map_loader_.get(), 0, program_->named_data(), program_->segments());
  EXPECT_EQ(data_map.error(), Error::Ok);

  auto data0 = data_map->get_data("key0");
  EXPECT_EQ(data0.error(), Error::Ok);
  EXPECT_EQ(data0.get().size(), SEGMENT_SIZES[0]);
  const uint8_t* values0 = static_cast<const uint8_t*>(data0.get().data());
  for (int i = 0; i < data0.get().size(); i++) {
    EXPECT_EQ(values0[i], 1);
  }

  auto data1 = data_map->get_data("key1");
  EXPECT_EQ(data1.error(), Error::Ok);
  EXPECT_EQ(data1.get().size(), SEGMENT_SIZES[1]);
  const uint8_t* values1 = static_cast<const uint8_t*>(data1.get().data());
  for (int i = 0; i < data1.get().size(); i++) {
    EXPECT_EQ(values1[i], 2);
  }

  // Expect the same as data0.
  auto data2 = data_map->get_data("key2");
  EXPECT_EQ(data0.error(), Error::Ok);
  EXPECT_EQ(data0.get().size(), SEGMENT_SIZES[0]);
  const uint8_t* values2 = static_cast<const uint8_t*>(data0.get().data());
  for (int i = 0; i < data2.get().size(); i++) {
    EXPECT_EQ(values2[i], 1);
  }

  // Throw error, as key_invalid contains segment_index=10, which
  // is out of range for segments.size()=2.
  auto data_invalid = data_map->get_data("key_invalid");
  EXPECT_EQ(data_invalid.error(), Error::InvalidArgument);

  // Throw error on nonexistent key.
  auto data_nonexistent = data_map->get_data("nonexistent_key");
  EXPECT_EQ(data_nonexistent.error(), Error::NotFound);
}
