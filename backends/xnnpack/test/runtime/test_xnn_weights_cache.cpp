/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/runtime/XNNWeightsCache.h>

#include <executorch/runtime/executor/pte_data_map.h>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/testing_util/temp_file.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/schema/program_generated.h>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xnnpack.h>
#include <atomic>
#include <fstream>
#include <mutex>
#include <thread>

using executorch::backends::xnnpack::delegate::XNNWeightsCache;
using executorch::extension::FileDataLoader;
using executorch::extension::testing::TempFile;
using executorch::runtime::DataLoader;
using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Result;
using executorch::runtime::internal::PteDataMap;

class XNNWeightsCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Creating a NamedDataMap from scratch is a little bit convoluted, so
    // we copied a lot of setup from test_pte_data_map.cpp

    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();

    // Create a sample Program with only named_data and segments. Technically
    // not a valid Program; only used to test the PteDataMap.
    // Create named data.
    std::array<const flatbuffers::Offset<executorch_flatbuffer::NamedData>, 2>
        named_data_arr = {
            executorch_flatbuffer::CreateNamedDataDirect(
                builder_, "weight", /*segment_index=*/0),
            executorch_flatbuffer::CreateNamedDataDirect(
                builder_, "bias", /*segment_index=*/1),
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

    Result<PteDataMap> data_map = PteDataMap::create(
        data_map_loader_.get(),
        0,
        program_->named_data(),
        program_->segments());
    ASSERT_EQ(data_map.error(), Error::Ok);
    data_map_ = std::make_unique<PteDataMap>(std::move(data_map.get()));

    memory_allocator_ = std::make_unique<MemoryAllocator>(
        memory_allocator_data_.size(), memory_allocator_data_.data());

    xnn_status status = xnn_initialize(nullptr);
    ASSERT_EQ(status, xnn_status_success);
  }

  void BuildAndRunGraphWithWeightsCache(
      XNNWeightsCache& weight_cache,
      const std::vector<size_t>& batches,
      size_t input_channels,
      size_t output_channels,
      float* input_data,
      float* output_data) {
    // Defining subgraph
    xnn_subgraph_t subgraph_ptr = nullptr;
    xnn_status status = xnn_create_subgraph(
        /*external_value_ids=*/2,
        /*flags=*/0,
        &subgraph_ptr);
    ASSERT_EQ(status, xnn_status_success);
    std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph(
        subgraph_ptr, &xnn_delete_subgraph);

    // Define tensors
    // Define input
    uint32_t input_id;
    std::vector<size_t> input_dims(batches);
    input_dims.push_back(input_channels);
    status = xnn_define_tensor_value(
        subgraph_ptr,
        xnn_datatype_fp32,
        input_dims.size(),
        input_dims.data(),
        nullptr,
        0,
        XNN_VALUE_FLAG_EXTERNAL_INPUT,
        &input_id);

    // Define weight
    uint32_t weight_id;
    Result<const uint8_t*> weight_pointer =
        weight_cache.load_unpacked_data("weight");
    ASSERT_TRUE(weight_pointer.ok());
    ASSERT_TRUE(weight_pointer.get() != nullptr);
    std::vector<size_t> weight_dims{output_channels, input_channels};
    status = xnn_define_tensor_value(
        subgraph_ptr,
        xnn_datatype_fp32,
        weight_dims.size(),
        weight_dims.data(),
        weight_pointer.get(),
        XNN_INVALID_VALUE_ID,
        0,
        &weight_id);
    ASSERT_EQ(status, xnn_status_success);

    // Define bias
    uint32_t bias_id;
    Result<const uint8_t*> bias_pointer =
        weight_cache.load_unpacked_data("bias");
    ASSERT_TRUE(bias_pointer.ok());
    std::vector<size_t> bias_dims{output_channels};
    status = xnn_define_tensor_value(
        subgraph_ptr,
        xnn_datatype_fp32,
        bias_dims.size(),
        bias_dims.data(),
        bias_pointer.get(),
        XNN_INVALID_VALUE_ID,
        0,
        &bias_id);

    // Define output tensor
    uint32_t output_id;
    std::vector<size_t> output_dims(batches);
    output_dims.push_back(output_channels);
    status = xnn_define_tensor_value(
        subgraph_ptr,
        xnn_datatype_fp32,
        output_dims.size(),
        output_dims.data(),
        nullptr,
        1,
        XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
        &output_id);

    // create xecond fully connected
    status = xnn_define_fully_connected(
        subgraph_ptr,
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        input_id,
        weight_id,
        bias_id,
        output_id,
        0);
    // Create and Pack Weights
    xnn_runtime_t runtime_ptr = nullptr;
    status = xnn_create_runtime_v3(
        subgraph_ptr, weight_cache.get(), nullptr, 0, &runtime_ptr);
    Result<std::vector<std::string>> packed_weights_added =
        weight_cache.finalize_for_runtime();
    ASSERT_TRUE(packed_weights_added.ok());
    ASSERT_EQ(packed_weights_added.get().size(), 1);
    ASSERT_EQ(packed_weights_added.get()[0], "weightbias");

    auto runtime = std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)>(
        runtime_ptr, xnn_delete_runtime);

    const std::array<xnn_external_value, 2> external = {
        xnn_external_value{0, input_data},
        xnn_external_value{1, output_data},
    };

    status = xnn_reshape_runtime(runtime.get());
    status =
        xnn_setup_runtime_v2(runtime.get(), external.size(), external.data());

    ASSERT_EQ(status, xnn_status_success);
    status = xnn_invoke_runtime(runtime.get());
    ASSERT_EQ(status, xnn_status_success);
  }

  // Program builder constants.
  static constexpr int kSegmentAlignment = 16;
  static constexpr std::array<int, 2> kSegmentSizes{384, 128};
  static constexpr std::array<int, 2> kSegmentOffsets{0, kSegmentAlignment * 2};
  std::array<uint8_t, 512> sample_data_;

  // Program builder.
  flatbuffers::FlatBufferBuilder builder_;
  const executorch_flatbuffer::Program* program_;

  // Data loader for the sample data.
  std::unique_ptr<FileDataLoader> data_map_loader_;

  // PteDataMap
  std::unique_ptr<PteDataMap> data_map_;

  // MemoryAllocator
  std::array<uint8_t, 200> memory_allocator_data_;
  std::unique_ptr<MemoryAllocator> memory_allocator_;
};

TEST_F(XNNWeightsCacheTest, ReusePackedWeights) {
  XNNWeightsCache weight_cache;
  size_t padding = 32;

  std::vector<size_t> batches{1, 2, 3};
  size_t num_batches = 1;
  for (size_t batch_dim : batches) {
    num_batches *= batch_dim;
  }
  size_t input_channels = 3;
  size_t output_channels = 4;
  std::vector<float> input_tensor(num_batches * input_channels + padding, 1.0f);
  std::vector<float> output_tensor(num_batches * output_channels, 0.0f);
  float* input_data = input_tensor.data();
  float* output_data = output_tensor.data();
  weight_cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
  BuildAndRunGraphWithWeightsCache(
      weight_cache,
      batches,
      input_channels,
      output_channels,
      input_data,
      output_data);

  weight_cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
  BuildAndRunGraphWithWeightsCache(
      weight_cache,
      batches,
      input_channels,
      output_channels,
      input_data,
      output_data);
  ASSERT_EQ(weight_cache.get_num_unpacked_data(), 0);
  weight_cache.delete_packed_data(weight_cache.get_packed_data_names());
  std::vector<std::string> packed_data_names =
      weight_cache.get_packed_data_names();
  // Packed Data Still exists because it has a ref count of 2
  ASSERT_EQ(packed_data_names.size(), 1);
  weight_cache.delete_packed_data(weight_cache.get_packed_data_names());
  packed_data_names = weight_cache.get_packed_data_names();
  ASSERT_EQ(packed_data_names.size(), 0);
}

#ifndef _WIN32
// Verify pack-and-run works when packed weight allocations go to a
// MAP_SHARED file instead of heap. The cache path is unique per test so
// flock won't collide.
TEST_F(XNNWeightsCacheTest, PackedWeightsToMmapFile) {
  std::string cache_path = std::string("/tmp/xnn_weights_cache_test_") +
      std::to_string(::getpid()) + ".packed_cache";
  // Ensure cleanup if a previous run left a file behind.
  ::unlink(cache_path.c_str());

  XNNWeightsCache weight_cache;
  weight_cache.set_packed_cache_path(cache_path);

  std::vector<size_t> batches{1, 2, 3};
  size_t num_batches = 1;
  for (size_t batch_dim : batches) {
    num_batches *= batch_dim;
  }
  size_t input_channels = 3;
  size_t output_channels = 4;
  size_t padding = 32;
  std::vector<float> input_tensor(num_batches * input_channels + padding, 1.0f);
  std::vector<float> output_tensor(num_batches * output_channels, 0.0f);

  weight_cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
  BuildAndRunGraphWithWeightsCache(
      weight_cache,
      batches,
      input_channels,
      output_channels,
      input_tensor.data(),
      output_tensor.data());

  // The cache file should have been created and contain packed weight bytes.
  struct stat st {};
  ASSERT_EQ(::stat(cache_path.c_str(), &st), 0);
  ASSERT_GT(st.st_size, 0);

  // delete_packed_data should release the mmap region without crashing.
  weight_cache.delete_packed_data(weight_cache.get_packed_data_names());
  ASSERT_EQ(weight_cache.get_packed_data_names().size(), 0);

  ::unlink(cache_path.c_str());
}

// A second XNNWeightsCache pointing at the same cache file while the first
// one still holds it must not corrupt the first instance's mmaps. The
// second one falls back to heap and runs to completion.
TEST_F(XNNWeightsCacheTest, PackedWeightsMmapPathLockCollision) {
  std::string cache_path = std::string("/tmp/xnn_weights_cache_collision_") +
      std::to_string(::getpid()) + ".packed_cache";
  ::unlink(cache_path.c_str());

  XNNWeightsCache cache_a;
  cache_a.set_packed_cache_path(cache_path);
  cache_a.initialize_for_runtime(memory_allocator_.get(), data_map_.get());

  // Second cache holding the same path before cache_a is destroyed.
  XNNWeightsCache cache_b;
  cache_b.set_packed_cache_path(cache_path);
  // Must not throw / abort — should log and fall back to heap.
  Error err =
      cache_b.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
  ASSERT_EQ(err, Error::Ok);

  ::unlink(cache_path.c_str());
}

// Verify load_packed_cache produces byte-identical inference results to
// a fresh build of the same graph. Guards against weight pointers being
// mis-mapped after cache load.
TEST_F(XNNWeightsCacheTest, SaveAndLoad_PreservesInferenceOutput) {
  std::string cache_path = std::string("/tmp/xnn_weights_cache_output_") +
      std::to_string(::getpid()) + ".packed_cache";
  ::unlink(cache_path.c_str());

  std::vector<size_t> batches{1, 2, 3};
  size_t input_channels = 3;
  size_t output_channels = 4;
  size_t num_batches = 1 * 2 * 3;
  size_t padding = 32;
  std::vector<float> input_tensor(num_batches * input_channels + padding, 1.0f);

  // Run 1: no cache file (pure heap pack).
  std::vector<float> output_baseline(num_batches * output_channels, 0.0f);
  {
    XNNWeightsCache cache;
    cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
    BuildAndRunGraphWithWeightsCache(
        cache,
        batches,
        input_channels,
        output_channels,
        input_tensor.data(),
        output_baseline.data());
  }

  // Run 2: file-backed mmap path, save trailer.
  {
    XNNWeightsCache cache;
    cache.set_packed_cache_path(cache_path);
    cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
    std::vector<float> output_write(num_batches * output_channels, 0.0f);
    BuildAndRunGraphWithWeightsCache(
        cache,
        batches,
        input_channels,
        output_channels,
        input_tensor.data(),
        output_write.data());
    ASSERT_EQ(cache.save_packed_index(), Error::Ok);
    EXPECT_EQ(output_write, output_baseline);
  }

  // Run 3: fresh instance loads from disk; output must match.
  {
    XNNWeightsCache cache;
    cache.set_packed_cache_path(cache_path);
    cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
    ASSERT_GT(cache.get_packed_data_names().size(), 0u);
    std::vector<float> output_load(num_batches * output_channels, 0.0f);
    BuildAndRunGraphWithWeightsCache(
        cache,
        batches,
        input_channels,
        output_channels,
        input_tensor.data(),
        output_load.data());
    EXPECT_EQ(output_load, output_baseline);
  }

  ::unlink(cache_path.c_str());
}

// Corrupted cache file must not crash; load_packed_cache returns false and
// the next init falls through to the fresh-build path that overwrites it.
TEST_F(XNNWeightsCacheTest, LoadPackedCache_RejectsCorruptTrailer) {
  std::string cache_path = std::string("/tmp/xnn_weights_cache_corrupt_") +
      std::to_string(::getpid()) + ".packed_cache";
  ::unlink(cache_path.c_str());

  // Write a file with valid size but garbage trailer.
  {
    std::ofstream f(cache_path, std::ios::binary);
    std::vector<char> garbage(1024, '\xCC');
    f.write(garbage.data(), garbage.size());
  }

  XNNWeightsCache cache;
  cache.set_packed_cache_path(cache_path);
  // Must not crash; load returns false → falls through to fresh build.
  Error err =
      cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
  ASSERT_EQ(err, Error::Ok);

  // Fresh build still works.
  std::vector<size_t> batches{1, 2, 3};
  size_t input_channels = 3;
  size_t output_channels = 4;
  size_t num_batches = 1 * 2 * 3;
  size_t padding = 32;
  std::vector<float> input(num_batches * input_channels + padding, 1.0f);
  std::vector<float> output(num_batches * output_channels, 0.0f);
  BuildAndRunGraphWithWeightsCache(
      cache,
      batches,
      input_channels,
      output_channels,
      input.data(),
      output.data());

  ::unlink(cache_path.c_str());
}

// Repeated init+run+save cycles on the same file must not grow the cache
// file. Guards against the regression where each PTE init re-packed weights
// and appended a fresh copy (+500 MB per inference observed in production).
TEST_F(XNNWeightsCacheTest, MultiSessionLoad_DoesNotGrowCacheFile) {
  std::string cache_path = std::string("/tmp/xnn_weights_cache_nogrow_") +
      std::to_string(::getpid()) + ".packed_cache";
  ::unlink(cache_path.c_str());

  std::vector<size_t> batches{1, 2, 3};
  size_t input_channels = 3;
  size_t output_channels = 4;
  size_t num_batches = 1 * 2 * 3;
  size_t padding = 32;
  std::vector<float> input(num_batches * input_channels + padding, 1.0f);
  std::vector<float> output(num_batches * output_channels, 0.0f);

  // Cycle 1: fresh write of cache.
  off_t size_after_first_save = 0;
  {
    XNNWeightsCache cache;
    cache.set_packed_cache_path(cache_path);
    cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
    BuildAndRunGraphWithWeightsCache(
        cache,
        batches,
        input_channels,
        output_channels,
        input.data(),
        output.data());
    ASSERT_EQ(cache.save_packed_index(), Error::Ok);
    struct stat st {};
    ASSERT_EQ(::stat(cache_path.c_str(), &st), 0);
    size_after_first_save = st.st_size;
    ASSERT_GT(size_after_first_save, 0);
  }

  // Cycle 2: fresh instance loads from disk, runs, saves. No new weights
  // were packed → file must be byte-for-byte identical in length.
  {
    XNNWeightsCache cache;
    cache.set_packed_cache_path(cache_path);
    cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
    ASSERT_GT(cache.get_packed_data_names().size(), 0u);
    BuildAndRunGraphWithWeightsCache(
        cache,
        batches,
        input_channels,
        output_channels,
        input.data(),
        output.data());
    ASSERT_EQ(cache.save_packed_index(), Error::Ok);
  }
  {
    struct stat st {};
    ASSERT_EQ(::stat(cache_path.c_str(), &st), 0);
    EXPECT_EQ(st.st_size, size_after_first_save);
  }

  // Cycle 3: simulate PTE destroy + recreate inside the same instance.
  // delete_packed_data on from_load entries must not erase metadata, so
  // the second init's look_up still hits → no new file append.
  {
    XNNWeightsCache cache;
    cache.set_packed_cache_path(cache_path);
    cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
    BuildAndRunGraphWithWeightsCache(
        cache,
        batches,
        input_channels,
        output_channels,
        input.data(),
        output.data());
    cache.delete_packed_data(cache.get_packed_data_names());
    cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
    BuildAndRunGraphWithWeightsCache(
        cache,
        batches,
        input_channels,
        output_channels,
        input.data(),
        output.data());
    ASSERT_EQ(cache.save_packed_index(), Error::Ok);
  }
  {
    struct stat st {};
    ASSERT_EQ(::stat(cache_path.c_str(), &st), 0);
    EXPECT_EQ(st.st_size, size_after_first_save);
  }

  ::unlink(cache_path.c_str());
}

// After loading from disk, delete_packed_data must skip from_load entries
// so the next init still hits the cache. Bug would re-pack weights from
// scratch each time the backend destroys + recreates a delegate.
TEST_F(
    XNNWeightsCacheTest,
    DeletePackedData_OnFromLoadEntries_PreservesMetadata) {
  std::string cache_path = std::string("/tmp/xnn_weights_cache_fromload_") +
      std::to_string(::getpid()) + ".packed_cache";
  ::unlink(cache_path.c_str());

  std::vector<size_t> batches{1, 2, 3};
  size_t input_channels = 3;
  size_t output_channels = 4;
  size_t num_batches = 1 * 2 * 3;
  size_t padding = 32;
  std::vector<float> input(num_batches * input_channels + padding, 1.0f);
  std::vector<float> output(num_batches * output_channels, 0.0f);

  // Seed the cache file.
  {
    XNNWeightsCache cache;
    cache.set_packed_cache_path(cache_path);
    cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
    BuildAndRunGraphWithWeightsCache(
        cache,
        batches,
        input_channels,
        output_channels,
        input.data(),
        output.data());
    ASSERT_EQ(cache.save_packed_index(), Error::Ok);
  }

  // Fresh instance: all populated entries are from_load=true.
  XNNWeightsCache cache;
  cache.set_packed_cache_path(cache_path);
  cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
  size_t loaded_count = cache.get_packed_data_names().size();
  ASSERT_GT(loaded_count, 0u);

  BuildAndRunGraphWithWeightsCache(
      cache,
      batches,
      input_channels,
      output_channels,
      input.data(),
      output.data());

  // Repeated delete must never erase from_load entries — contrast with
  // ReusePackedWeights where two delete calls drop the count to 0.
  for (int i = 0; i < 5; ++i) {
    cache.delete_packed_data(cache.get_packed_data_names());
    EXPECT_EQ(cache.get_packed_data_names().size(), loaded_count)
        << "from_load entries should survive delete; iteration " << i;
  }

  ::unlink(cache_path.c_str());
}

// A model with multiple PTE/method delegates initializes the cache
// sequentially before any one is destroyed. The second PTE's init must
// see the first PTE's packed entries already in the map → look_up hits,
// no new reserve_space, file does not grow per PTE.
TEST_F(XNNWeightsCacheTest, MultiplePTEsInSameInstance_NoFileGrowth) {
  std::string cache_path = std::string("/tmp/xnn_weights_cache_multipte_") +
      std::to_string(::getpid()) + ".packed_cache";
  ::unlink(cache_path.c_str());

  std::vector<size_t> batches{1, 2, 3};
  size_t input_channels = 3;
  size_t output_channels = 4;
  size_t num_batches = 1 * 2 * 3;
  size_t padding = 32;
  std::vector<float> input(num_batches * input_channels + padding, 1.0f);
  std::vector<float> out_pte1(num_batches * output_channels, 0.0f);
  std::vector<float> out_pte2(num_batches * output_channels, 0.0f);

  XNNWeightsCache cache;
  cache.set_packed_cache_path(cache_path);

  // PTE 1: fresh pack + save.
  cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
  BuildAndRunGraphWithWeightsCache(
      cache,
      batches,
      input_channels,
      output_channels,
      input.data(),
      out_pte1.data());
  ASSERT_EQ(cache.save_packed_index(), Error::Ok);

  off_t size_after_pte1 = 0;
  {
    struct stat st {};
    ASSERT_EQ(::stat(cache_path.c_str(), &st), 0);
    size_after_pte1 = st.st_size;
    ASSERT_GT(size_after_pte1, 0);
  }
  size_t names_after_pte1 = cache.get_packed_data_names().size();
  ASSERT_GT(names_after_pte1, 0u);

  // PTE 2: sibling delegate, NO destroy between. look_up must hit the
  // entry from PTE 1 → no new reserve_space → file size unchanged after
  // save.
  cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
  BuildAndRunGraphWithWeightsCache(
      cache,
      batches,
      input_channels,
      output_channels,
      input.data(),
      out_pte2.data());
  ASSERT_EQ(cache.save_packed_index(), Error::Ok);

  {
    struct stat st {};
    ASSERT_EQ(::stat(cache_path.c_str(), &st), 0);
    EXPECT_EQ(st.st_size, size_after_pte1)
        << "PTE 2 with same weights must not append to the cache file";
  }
  EXPECT_EQ(cache.get_packed_data_names().size(), names_after_pte1);

  // Both PTEs produced the same output for the same input (correctness).
  EXPECT_EQ(out_pte1, out_pte2);

  // PTE 3: third sibling. Still no growth.
  std::vector<float> out_pte3(num_batches * output_channels, 0.0f);
  cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
  BuildAndRunGraphWithWeightsCache(
      cache,
      batches,
      input_channels,
      output_channels,
      input.data(),
      out_pte3.data());
  ASSERT_EQ(cache.save_packed_index(), Error::Ok);
  {
    struct stat st {};
    ASSERT_EQ(::stat(cache_path.c_str(), &st), 0);
    EXPECT_EQ(st.st_size, size_after_pte1);
  }
  EXPECT_EQ(out_pte3, out_pte1);

  ::unlink(cache_path.c_str());
}

// save_packed_index must be a true no-op when no new reserve_space happened
// since the last save — same content but writing would still bump mtime,
// making the cache file look modified on every model load.
TEST_F(XNNWeightsCacheTest, SavePackedIndex_NoNewReserves_IsNoOp) {
  std::string cache_path = std::string("/tmp/xnn_weights_cache_noop_") +
      std::to_string(::getpid()) + ".packed_cache";
  ::unlink(cache_path.c_str());

  std::vector<size_t> batches{1, 2, 3};
  size_t input_channels = 3;
  size_t output_channels = 4;
  size_t num_batches = 1 * 2 * 3;
  size_t padding = 32;
  std::vector<float> input(num_batches * input_channels + padding, 1.0f);
  std::vector<float> output(num_batches * output_channels, 0.0f);

  // Seed cache + first save.
  XNNWeightsCache cache;
  cache.set_packed_cache_path(cache_path);
  cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
  BuildAndRunGraphWithWeightsCache(
      cache,
      batches,
      input_channels,
      output_channels,
      input.data(),
      output.data());
  ASSERT_EQ(cache.save_packed_index(), Error::Ok);

  // Force an old mtime so any real write is detectable as a forward jump,
  // without relying on wall-clock granularity / sleep (sleeps are flaky and
  // forbidden by lint).
  const struct timespec old_times[2] = {
      {1000000, 0}, // atime
      {1000000, 0}, // mtime
  };
  ASSERT_EQ(::utimensat(AT_FDCWD, cache_path.c_str(), old_times, 0), 0);

  struct stat st_before {};
  ASSERT_EQ(::stat(cache_path.c_str(), &st_before), 0);

  // Second save with no intervening reserve_space → no-op short-circuit.
  ASSERT_EQ(cache.save_packed_index(), Error::Ok);

  struct stat st_after {};
  ASSERT_EQ(::stat(cache_path.c_str(), &st_after), 0);
  EXPECT_EQ(st_before.st_size, st_after.st_size);
  EXPECT_EQ(st_before.st_mtime, st_after.st_mtime);

  ::unlink(cache_path.c_str());
}

// Stress test for gjcomer's V6 review concern: concurrent
// `set_packed_cache_path` + `save_packed_index` against the shared cache
// must not crash or leave the on-disk file inconsistent under the lock
// discipline that XNNPACKBackend uses (single mutex around the cache).
// This does NOT exercise concurrent runtime creation — XNNPACK's runtime
// init itself is not thread-safe and would require XNNPACKBackend
// machinery to test properly.
TEST_F(XNNWeightsCacheTest, ConcurrentOptionsAndSave_NoCrash_FileStable) {
  std::string cache_path = std::string("/tmp/xnn_weights_cache_concurrent_") +
      std::to_string(::getpid()) + ".packed_cache";
  ::unlink(cache_path.c_str());

  // Seed a populated cache + initial save so subsequent save_packed_index
  // calls hit the no-op short-circuit path (the case most prone to race).
  std::vector<size_t> batches{1, 2, 3};
  size_t input_channels = 3;
  size_t output_channels = 4;
  size_t num_batches = 1 * 2 * 3;
  size_t padding = 32;
  std::vector<float> input(num_batches * input_channels + padding, 1.0f);
  std::vector<float> output(num_batches * output_channels, 0.0f);
  XNNWeightsCache cache;
  cache.set_packed_cache_path(cache_path);
  cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
  BuildAndRunGraphWithWeightsCache(
      cache,
      batches,
      input_channels,
      output_channels,
      input.data(),
      output.data());
  ASSERT_EQ(cache.save_packed_index(), Error::Ok);

  struct stat st_baseline {};
  ASSERT_EQ(::stat(cache_path.c_str(), &st_baseline), 0);

  // Lock discipline matches XNNPACKBackend's `weights_cache_mutex_`: every
  // cache mutation is serialized. Threads spam set_packed_cache_path and
  // save_packed_index under the shared lock for ~25 iterations each.
  std::mutex cache_mu;
  constexpr int kThreads = 4;
  constexpr int kIters = 25;
  std::atomic<int> failure_count{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&]() {
      for (int i = 0; i < kIters; ++i) {
        try {
          const std::lock_guard<std::mutex> lock(cache_mu);
          // Re-set the same path — should be benign / a stable no-op.
          cache.set_packed_cache_path(cache_path);
          // No new reserves between calls → save short-circuits.
          (void)cache.save_packed_index();
        } catch (const std::exception&) {
          failure_count.fetch_add(1);
        }
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }

  EXPECT_EQ(failure_count.load(), 0);

  // File must not balloon: every iteration's save is a no-op.
  struct stat st_after {};
  ASSERT_EQ(::stat(cache_path.c_str(), &st_after), 0);
  EXPECT_EQ(st_after.st_size, st_baseline.st_size);

  ::unlink(cache_path.c_str());
}

#endif
