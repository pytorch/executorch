/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/executor/pte_data_map.h>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/testing_util/temp_file.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/schema/program_generated.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::extension::FileDataLoader;
using executorch::extension::testing::TempFile;
using executorch::runtime::DataLoader;
using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;
using executorch::runtime::internal::PteDataMap;

class PteDataMapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();

    // Create a sample Program with only named_data and segments. Technically
    // not a valid Program; only used to test the PteDataMap.
    // Create named data.
    std::array<const flatbuffers::Offset<executorch_flatbuffer::NamedData>, 4>
        named_data_arr = {
            executorch_flatbuffer::CreateNamedDataDirect(
                builder_, "key0", /*segment_index=*/0),
            executorch_flatbuffer::CreateNamedDataDirect(
                builder_, "key1", /*segment_index=*/1),
            // Note: key2 points to the same segment as key0.
            executorch_flatbuffer::CreateNamedDataDirect(
                builder_, "key2", /*segment_index=*/0),
            // This is invalid, as segment_index=10 is out of range when the
            // number of segments is 2.
            executorch_flatbuffer::CreateNamedDataDirect(
                builder_, "key_invalid", /*segment_index=*/10),
        };
    const auto named_data =
        builder_.CreateVector(named_data_arr.data(), named_data_arr.size());

    // Create segments.
    std::array<const flatbuffers::Offset<executorch_flatbuffer::DataSegment>, 2>
        segment_arr = {// @lint-ignore CLANGTIDY facebook-hte-BadArgumentComment
                       executorch_flatbuffer::CreateDataSegment(
                           builder_, /*offset=*/0, /*size=*/kSegmentSizes[0]),
                       // @lint-ignore CLANGTIDY facebook-hte-BadArgumentComment
                       executorch_flatbuffer::CreateDataSegment(
                           builder_,
                           /*offset=*/kSegmentAlignment * 2,
                           /*size=*/kSegmentSizes[1])};
    const auto segments =
        builder_.CreateVector(segment_arr.data(), segment_arr.size());

    // Create Program.
    const auto program = executorch_flatbuffer::CreateProgram(
        builder_, 0, 0, 0, 0, segments, 0, 0, named_data);

    builder_.Finish(program);
    program_ = executorch_flatbuffer::GetProgram(builder_.GetBufferPointer());

    // Create sample segment data.
    for (int i = 0; i < kSegmentSizes[0]; i++) {
      sample_data_[i] = 1;
    }
    for (int i = kSegmentOffsets[1]; i < kSegmentOffsets[1] + kSegmentSizes[1];
         i++) {
      sample_data_[i] = 2;
    }
    TempFile tf(sample_data_.data(), sizeof(sample_data_));

    // Wrap the sample data in a loader.
    Result<FileDataLoader> loader =
        FileDataLoader::from(tf.path().c_str(), kSegmentAlignment);
    ASSERT_EQ(loader.error(), Error::Ok);
    data_map_loader_ =
        std::make_unique<FileDataLoader>(std::move(loader.get()));
  }

  // Program builder constants.
  static constexpr int kSegmentAlignment = 16;
  static constexpr std::array<int, 2> kSegmentSizes{17, 8};
  static constexpr std::array<int, 2> kSegmentOffsets{0, kSegmentAlignment * 2};
  std::array<uint8_t, 64> sample_data_;

  // Program builder.
  flatbuffers::FlatBufferBuilder builder_;
  const executorch_flatbuffer::Program* program_;

  // Data loader for the sample data.
  std::unique_ptr<FileDataLoader> data_map_loader_;
};

TEST_F(PteDataMapTest, Load) {
  Result<PteDataMap> data_map = PteDataMap::create(
      data_map_loader_.get(), 0, program_->named_data(), program_->segments());
  ASSERT_TRUE(data_map.ok());
}

TEST_F(PteDataMapTest, LoadFail) {
  Result<PteDataMap> data_map = PteDataMap::create(
      /*loader=*/nullptr,
      /*segment_base_offset=*/0,
      program_->named_data(),
      program_->segments());
  EXPECT_EQ(data_map.error(), Error::InvalidArgument);
}

TEST_F(PteDataMapTest, UnimplementedMethods) {
  Result<PteDataMap> data_map = PteDataMap::create(
      data_map_loader_.get(), 0, program_->named_data(), program_->segments());
  ;

  // Check get_tensor_layout is not implemented.
  auto result = data_map->get_tensor_layout("sample_key");
  EXPECT_EQ(result.error(), Error::NotImplemented);

  // Check load_data_into is not implemented.
  auto err = data_map->load_data_into("sample_key", nullptr, 0);
  EXPECT_EQ(err, Error::NotImplemented);
}

TEST_F(PteDataMapTest, Keys) {
  Result<PteDataMap> data_map = PteDataMap::create(
      data_map_loader_.get(), 0, program_->named_data(), program_->segments());
  ASSERT_TRUE(data_map.ok());

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

  // This key is invalid because it points to a segment_index=10, which is out
  // of range for this example with segment size=2.
  // Note: practically, a PTE should not have invalid keys.
  auto key_invalid = data_map->get_key(3);
  EXPECT_EQ(strcmp(key_invalid.get(), "key_invalid"), 0);

  // Returns an error on non-existent key.
  auto nonexistent_key = data_map->get_key(10);
  EXPECT_EQ(nonexistent_key.error(), Error::InvalidArgument);
}

TEST_F(PteDataMapTest, GetData) {
  Result<PteDataMap> data_map = PteDataMap::create(
      data_map_loader_.get(), 0, program_->named_data(), program_->segments());
  ASSERT_TRUE(data_map.ok());

  Result<FreeableBuffer> data0 = data_map->get_data("key0");
  EXPECT_EQ(data0.error(), Error::Ok);
  EXPECT_EQ(data0.get().size(), kSegmentSizes[0]);
  EXPECT_EQ(
      memcmp(data0.get().data(), sample_data_.data(), data0.get().size()), 0);

  Result<FreeableBuffer> data1 = data_map->get_data("key1");
  EXPECT_EQ(data1.error(), Error::Ok);
  EXPECT_EQ(data1.get().size(), kSegmentSizes[1]);
  EXPECT_EQ(
      memcmp(
          data1.get().data(),
          sample_data_.data() + kSegmentOffsets[1],
          data1.get().size()),
      0);

  Result<FreeableBuffer> data2 = data_map->get_data("key2");
  EXPECT_EQ(data2.error(), Error::Ok);
  // Expect the same values as data0, as key0 and key2 point to the same
  // segment.
  EXPECT_EQ(data2.get().size(), kSegmentSizes[0]);
  EXPECT_EQ(
      memcmp(data2.get().data(), sample_data_.data(), data2.get().size()), 0);

  // Free data.
  data0->Free();
  data1->Free();
  data2->Free();

  // Returns an error, as key_invalid contains segment_index=10, which
  // is out of range for segments.size()=2.
  Result<FreeableBuffer> data_invalid = data_map->get_data("key_invalid");
  EXPECT_EQ(data_invalid.error(), Error::InvalidArgument);

  // Returns an error on nonexistent key.
  Result<FreeableBuffer> data_nonexistent =
      data_map->get_data("nonexistent_key");
  EXPECT_EQ(data_nonexistent.error(), Error::NotFound);
}

TEST_F(PteDataMapTest, FreeAndReload) {
  // Load a key, free it, and then load it again, and ensure that the
  // core data map can return a new FreeableBuffer with the same data.
  Result<PteDataMap> data_map = PteDataMap::create(
      data_map_loader_.get(), 0, program_->named_data(), program_->segments());
  ASSERT_TRUE(data_map.ok());

  // Load data0.
  Result<FreeableBuffer> data0 = data_map->get_data("key0");
  EXPECT_EQ(data0.error(), Error::Ok);
  EXPECT_EQ(data0.get().size(), kSegmentSizes[0]);
  EXPECT_EQ(
      memcmp(data0.get().data(), sample_data_.data(), data0.get().size()), 0);
  data0->Free();

  // Reload data0, ensure that the core data map can return a new
  // FreeableBuffer with the same data.
  Result<FreeableBuffer> data0_reload = data_map->get_data("key0");
  EXPECT_EQ(data0_reload.error(), Error::Ok);
  EXPECT_EQ(data0_reload.get().size(), kSegmentSizes[0]);
  EXPECT_EQ(
      memcmp(
          data0_reload.get().data(),
          sample_data_.data(),
          data0_reload.get().size()),
      0);
  data0_reload->Free();
}

TEST_F(PteDataMapTest, ReloadAndFree) {
  // Load the same key multiple times, and then free one and ensure that the
  // data in the other is still valid.
  Result<PteDataMap> data_map = PteDataMap::create(
      data_map_loader_.get(), 0, program_->named_data(), program_->segments());
  ASSERT_TRUE(data_map.ok());

  // Load data0.
  Result<FreeableBuffer> data0 = data_map->get_data("key0");
  EXPECT_EQ(data0.error(), Error::Ok);
  EXPECT_EQ(data0.get().size(), kSegmentSizes[0]);
  EXPECT_EQ(
      memcmp(data0.get().data(), sample_data_.data(), data0.get().size()), 0);

  // Reload data0.
  Result<FreeableBuffer> data0_reload = data_map->get_data("key0");
  EXPECT_EQ(data0_reload.error(), Error::Ok);
  EXPECT_EQ(data0_reload.get().size(), kSegmentSizes[0]);
  EXPECT_EQ(
      memcmp(
          data0_reload.get().data(),
          sample_data_.data(),
          data0_reload.get().size()),
      0);

  // Free data0 and check that data0_reload is still valid.
  data0->Free();
  EXPECT_EQ(data0_reload.get().size(), kSegmentSizes[0]);
  EXPECT_EQ(
      memcmp(
          data0_reload.get().data(),
          sample_data_.data(),
          data0_reload.get().size()),
      0);

  // Free data_reload0.
  data0_reload->Free();
}
