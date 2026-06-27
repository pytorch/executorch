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

  // delete_packed_data on entries that have been persisted to disk
  // (from_load=true after the auto-save in finalize_for_runtime) is a
  // ref_count decrement, not a metadata erase. The entries remain so a
  // subsequent loadModel can look them up without re-packing — matches
  // production semantics where cache survives unload/reload.
  weight_cache.delete_packed_data(weight_cache.get_packed_data_names());
  ASSERT_GT(weight_cache.get_packed_data_names().size(), 0);

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

namespace {

// Little-endian decode helpers matching XNNWeightsCache's on-disk format.
uint32_t read_le_u32(const uint8_t* p) {
  uint32_t v = 0;
  for (int i = 0; i < 4; ++i) {
    v |= static_cast<uint32_t>(p[i]) << (8 * i);
  }
  return v;
}
uint64_t read_le_u64(const uint8_t* p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) {
    v |= static_cast<uint64_t>(p[i]) << (8 * i);
  }
  return v;
}
void write_le_u32(std::ostream& f, uint32_t v) {
  for (int i = 0; i < 4; ++i) {
    char b = static_cast<char>((v >> (8 * i)) & 0xff);
    f.write(&b, 1);
  }
}
void write_le_u64(std::ostream& f, uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    char b = static_cast<char>((v >> (8 * i)) & 0xff);
    f.write(&b, 1);
  }
}

} // namespace

// A cache file written by older code (kCacheVersion=1) carries no per-entry
// seed field. Loading such a file with the current schema would yield
// entries with seed=0 and mismatch every fresh look_up. The version bump
// must reject it outright so the next init re-packs from scratch.
TEST_F(XNNWeightsCacheTest, LoadPackedCache_RejectsV1Format) {
  std::string cache_path = std::string("/tmp/xnn_weights_cache_v1_") +
      std::to_string(::getpid()) + ".packed_cache";
  ::unlink(cache_path.c_str());

  // v1 layout: 64 bytes of dummy data, then 20-byte footer with version=1.
  {
    std::ofstream f(cache_path, std::ios::binary);
    std::vector<char> data(64, 0);
    f.write(data.data(), data.size());
    write_le_u64(f, 64); // index_start
    write_le_u32(f, 0); // entry_count
    write_le_u32(f, 0x58505743); // kCacheMagic "XPWC"
    write_le_u32(f, 1); // OLD kCacheVersion = 1
  }

  XNNWeightsCache cache;
  cache.set_packed_cache_path(cache_path);
  Error err =
      cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
  ASSERT_EQ(err, Error::Ok);
  // Version mismatch → load_packed_cache returned false → no entries.
  EXPECT_EQ(cache.get_packed_data_names().size(), 0u);

  ::unlink(cache_path.c_str());
}

// Verify save_packed_index writes the schema version 2 footer and embeds a
// 4-byte seed field in each entry record. Guards against future refactors
// silently dropping the seed write.
TEST_F(XNNWeightsCacheTest, SavePackedIndex_EntryFormatIncludesSeed) {
  std::string cache_path = std::string("/tmp/xnn_weights_cache_format_") +
      std::to_string(::getpid()) + ".packed_cache";
  ::unlink(cache_path.c_str());

  std::vector<size_t> batches{1, 2, 3};
  size_t input_channels = 3;
  size_t output_channels = 4;
  size_t num_batches = 1 * 2 * 3;
  size_t padding = 32;
  std::vector<float> input(num_batches * input_channels + padding, 1.0f);
  std::vector<float> output(num_batches * output_channels, 0.0f);

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

  // Parse footer at file_size - 20.
  std::ifstream f(cache_path, std::ios::binary);
  ASSERT_TRUE(f.is_open());
  f.seekg(0, std::ios::end);
  size_t file_size = f.tellg();
  ASSERT_GE(file_size, 24u);

  uint8_t footer[20];
  f.seekg(file_size - 20);
  f.read(reinterpret_cast<char*>(footer), 20);
  uint32_t magic = read_le_u32(footer + 12);
  uint32_t version = read_le_u32(footer + 16);
  EXPECT_EQ(magic, 0x58505743u);
  EXPECT_EQ(version, 2u);

  // Walk first entry:
  // [name_len:u32][name][file_offset:u64][data_size:u64][seed:u32]
  uint64_t index_start = read_le_u64(footer);
  uint32_t entry_count = read_le_u32(footer + 8);
  ASSERT_GT(entry_count, 0u);

  f.seekg(index_start);
  uint8_t name_len_buf[4];
  f.read(reinterpret_cast<char*>(name_len_buf), 4);
  uint32_t name_len = read_le_u32(name_len_buf);

  // The seed field sits at index_start + 4 + name_len + 8 + 8.
  f.seekg(index_start + 4 + name_len + 8 + 8);
  uint8_t seed_buf[4];
  f.read(reinterpret_cast<char*>(seed_buf), 4);
  // XNNPACK ukernel seeds are non-zero in practice. The signal here is
  // simply that 4 well-formed bytes follow the size field — confirming
  // the new entry layout was written, not the legacy 16-byte tail.
  uint32_t stored_seed = read_le_u32(seed_buf);
  EXPECT_NE(stored_seed, 0u);

  ::unlink(cache_path.c_str());
}

// After loading a cache file whose entry seed has been tampered with
// (simulating an XNNPACK upgrade where the same ukernel now emits a
// different seed), the next inference must produce correct output. Either
// look_up's seed check or look_up_or_insert's memcmp fallback drives the
// re-pack; this test exercises the end-to-end safety net.
TEST_F(
    XNNWeightsCacheTest,
    LoadPackedCache_CorruptedSeed_ProducesCorrectOutput) {
  std::string cache_path = std::string("/tmp/xnn_weights_cache_badseed_") +
      std::to_string(::getpid()) + ".packed_cache";
  ::unlink(cache_path.c_str());

  std::vector<size_t> batches{1, 2, 3};
  size_t input_channels = 3;
  size_t output_channels = 4;
  size_t num_batches = 1 * 2 * 3;
  size_t padding = 32;
  std::vector<float> input(num_batches * input_channels + padding, 1.0f);

  // Baseline: fresh pack, heap-only, no cache file.
  std::vector<float> baseline(num_batches * output_channels, 0.0f);
  {
    XNNWeightsCache cache;
    cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
    BuildAndRunGraphWithWeightsCache(
        cache,
        batches,
        input_channels,
        output_channels,
        input.data(),
        baseline.data());
  }

  // Write a valid cache file.
  {
    XNNWeightsCache cache;
    cache.set_packed_cache_path(cache_path);
    cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
    std::vector<float> out(num_batches * output_channels, 0.0f);
    BuildAndRunGraphWithWeightsCache(
        cache,
        batches,
        input_channels,
        output_channels,
        input.data(),
        out.data());
    ASSERT_EQ(cache.save_packed_index(), Error::Ok);
  }

  // Corrupt the seed field of the first entry to a value no real ukernel
  // would emit (0xDEADBEEF).
  {
    std::fstream f(cache_path, std::ios::binary | std::ios::in | std::ios::out);
    ASSERT_TRUE(f.is_open());
    f.seekg(0, std::ios::end);
    size_t file_size = f.tellg();
    ASSERT_GE(file_size, 24u);

    uint8_t footer_buf[20];
    f.seekg(file_size - 20);
    f.read(reinterpret_cast<char*>(footer_buf), 20);
    uint64_t index_start = read_le_u64(footer_buf);
    uint32_t entry_count = read_le_u32(footer_buf + 8);
    ASSERT_GT(entry_count, 0u);

    f.seekg(index_start);
    uint8_t name_len_buf[4];
    f.read(reinterpret_cast<char*>(name_len_buf), 4);
    uint32_t name_len = read_le_u32(name_len_buf);

    size_t seed_offset = index_start + 4 + name_len + 8 + 8;
    f.seekp(seed_offset);
    uint32_t corrupted = 0xDEADBEEFu;
    f.write(reinterpret_cast<const char*>(&corrupted), 4);
    f.close();
  }

  // Reload and run. Output must still match baseline.
  std::vector<float> after_corruption(num_batches * output_channels, 0.0f);
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
        after_corruption.data());
  }

  EXPECT_EQ(after_corruption, baseline);

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

// Test seam: XNNWeightsCache befriends this type (see XNNWeightsCache.h) so
// the test can drive the private fresh-write path and inspect the
// offset->pointer table directly.
namespace executorch {
namespace backends {
namespace xnnpack {
namespace delegate {
class XNNWeightsCacheTestPeer {
 public:
  static void reset_for_fresh_write(XNNWeightsCache& c) {
    c.reset_for_fresh_write();
  }
  static void* offset_to_addr(XNNWeightsCache& c, size_t offset) {
    return XNNWeightsCache::offset_to_addr(&c, offset);
  }
  static size_t offset_of(XNNWeightsCache& c, const std::string& name) {
    return c.name_to_packed_data_metadata_.at(name).offset;
  }
};
} // namespace delegate
} // namespace xnnpack
} // namespace backends
} // namespace executorch

using executorch::backends::xnnpack::delegate::XNNWeightsCacheTestPeer;

TEST_F(XNNWeightsCacheTest, OffsetToAddr_AfterResetForFreshWrite_NotDangling) {
  std::string cache_path = std::string("/tmp/xnn_weights_cache_dangling_") +
      std::to_string(::getpid()) + ".packed_cache";
  ::unlink(cache_path.c_str());

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
  ASSERT_EQ(cache.get_packed_data_names().size(), 1u);

  size_t offset = XNNWeightsCacheTestPeer::offset_of(cache, "weightbias");
  void* before = XNNWeightsCacheTestPeer::offset_to_addr(cache, offset);
  ASSERT_NE(before, nullptr);
  // The pointer is into a live MAP_SHARED region and is readable now.
  ASSERT_NO_FATAL_FAILURE({
    volatile char c = *static_cast<volatile char*>(before);
    (void)c;
  });

  // Fresh-write reset: munmaps the region backing `before`.
  XNNWeightsCacheTestPeer::reset_for_fresh_write(cache);

  // The slot must no longer reference the unmapped region. A nullptr return is
  // safe (look_up_or_insert treats it as a miss); a non-null return would be a
  // dangling pointer into munmapped memory.
  void* after = XNNWeightsCacheTestPeer::offset_to_addr(cache, offset);
  EXPECT_EQ(after, nullptr)
      << "offset_to_addr returned a dangling pointer into a region that "
         "reset_for_fresh_write() already munmapped";

  ::unlink(cache_path.c_str());
}

TEST_F(XNNWeightsCacheTest, LoadPackedCache_RejectsMidTrailerTruncation) {
  std::string cache_path = std::string("/tmp/xnn_weights_cache_midtrunc_") +
      std::to_string(::getpid()) + ".packed_cache";
  ::unlink(cache_path.c_str());

  // Layout (little-endian, matching XNNWeightsCache's on-disk format):
  //   [0, 16)   16 bytes of packed-data region (entry 0's bytes live here)
  //   [16, 41)  entry 0: name_len(4)=1, name(1)="w",
  //             file_offset(8)=0, data_size(8)=8, seed(4)=0x12345678
  //   [41, 43)  2 trailing bytes  <- < 4 bytes, so the load loop bails here
  //   [43, 63)  footer: index_start(8)=16, entry_count(4)=2,
  //             magic(4)="XPWC", version(4)=2
  // The footer is fully valid and claims 2 entries, but only 1 is present.
  {
    std::ofstream f(cache_path, std::ios::binary);
    std::vector<char> data_region(16, 0);
    f.write(data_region.data(), data_region.size());

    // entry 0
    write_le_u32(f, 1); // name_len
    f.put('w'); // name
    write_le_u64(f, 0); // file_offset
    write_le_u64(f, 8); // data_size (<= index_start - file_offset)
    write_le_u32(f, 0x12345678u); // seed

    // 2 trailing bytes: leaves < 4 bytes before the footer.
    f.put('\0');
    f.put('\0');

    // footer
    write_le_u64(f, 16); // index_start
    write_le_u32(f, 2); // entry_count (claims 2, only 1 present)
    write_le_u32(f, 0x58505743u); // kCacheMagic "XPWC"
    write_le_u32(f, 2); // kCacheVersion
  }

  XNNWeightsCache cache;
  cache.set_packed_cache_path(cache_path);
  Error err =
      cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
  ASSERT_EQ(err, Error::Ok);

  EXPECT_EQ(cache.get_packed_data_names().size(), 0u)
      << "mid-trailer-truncated cache (footer entry_count=2, only 1 entry "
         "present) was wrongly accepted as a valid partial cache";

  ::unlink(cache_path.c_str());
}

#endif
