/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/runtime/XNNWeightsCache.h>
#include <executorch/backends/xnnpack/runtime/XNNWeightsCacheManager.h>

#include <executorch/runtime/executor/pte_data_map.h>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/flat_tensor/serialize/flat_tensor_generated.h>
#include <executorch/extension/flat_tensor/serialize/flat_tensor_header.h>
#include <executorch/extension/testing_util/temp_file.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/schema/program_generated.h>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xnnpack.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iterator>
#include <mutex>
#include <thread>

using executorch::backends::xnnpack::delegate::XNNWeightsCache;
using executorch::extension::FileDataLoader;
using executorch::extension::testing::TempFile;
using executorch::runtime::Error;
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

  // One "app launch": a fresh cache instance bound to cache_path loads the
  // file if it exists, packs-and-runs the graph, and persists the index.
  void RunLaunchWithCacheFile(
      const std::string& cache_path,
      std::vector<float>& output) {
    const std::vector<size_t> batches{1, 2, 3};
    const size_t num_batches = 1 * 2 * 3;
    const size_t padding = 32;
    std::vector<float> input(
        num_batches * kLaunchInputChannels + padding, 1.0f);
    output.assign(num_batches * kLaunchOutputChannels, 0.0f);

    XNNWeightsCache cache;
    cache.set_packed_cache_path(cache_path);
    ASSERT_EQ(
        cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get()),
        Error::Ok);
    BuildAndRunGraphWithWeightsCache(
        cache,
        batches,
        kLaunchInputChannels,
        kLaunchOutputChannels,
        input.data(),
        output.data());
    ASSERT_EQ(cache.save_packed_index(), Error::Ok);
  }

  // Packs and runs the test graph through a cache the caller has initialized.
  void PackAndRun(XNNWeightsCache& cache) {
    const std::vector<size_t> batches{1, 2, 3};
    const size_t num_batches = 1 * 2 * 3;
    const size_t padding = 32;
    std::vector<float> input(
        num_batches * kLaunchInputChannels + padding, 1.0f);
    std::vector<float> output(num_batches * kLaunchOutputChannels, 0.0f);
    BuildAndRunGraphWithWeightsCache(
        cache,
        batches,
        kLaunchInputChannels,
        kLaunchOutputChannels,
        input.data(),
        output.data());
  }

  static constexpr size_t kLaunchInputChannels = 3;
  static constexpr size_t kLaunchOutputChannels = 4;

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

TEST_F(XNNWeightsCacheTest, TakenUnpackedDataSurvivesFinalize) {
  // XNNPACK does not pack every constant it is given: PReLU slopes, for one,
  // stay as the unpacked buffer the subgraph points at. Those buffers are taken
  // out of the cache while the graph is built, and finalize_for_runtime must
  // leave them alone while still freeing the ones it packed.
  XNNWeightsCache cache;
  cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());

  Result<const uint8_t*> packed = cache.load_unpacked_data("weight");
  ASSERT_EQ(packed.error(), Error::Ok);
  // The index the caller would take immediately before defining the value it
  // wants to keep, which is what XNNCompiler passes down.
  size_t first_index = cache.get_num_unpacked_data();
  Result<const uint8_t*> retained_load = cache.load_unpacked_data("bias");
  ASSERT_EQ(retained_load.error(), Error::Ok);
  ASSERT_EQ(cache.get_num_unpacked_data(), first_index + 1);

  std::vector<executorch::runtime::FreeableBuffer> retained;
  cache.take_unpacked_data_from(first_index, retained);
  ASSERT_EQ(retained.size(), 1u);
  ASSERT_EQ(retained[0].size(), static_cast<size_t>(kSegmentSizes[1]));

  Result<std::vector<std::string>> names = cache.finalize_for_runtime();
  ASSERT_EQ(names.error(), Error::Ok);

  // Still readable, and still the "bias" segment, which SetUp fills with 2s.
  ASSERT_NE(retained[0].data(), nullptr);
  const uint8_t* data = static_cast<const uint8_t*>(retained[0].data());
  for (size_t i = 0; i < retained[0].size(); i++) {
    ASSERT_EQ(data[i], 2) << "byte " << i << " was freed or overwritten";
  }
}

TEST_F(XNNWeightsCacheTest, InitializeForRuntimeClearsLeftoverUnpackedData) {
  // A compile that bails out after loading a constant never reaches
  // finalize_for_runtime. This instance outlives any one model, so the next
  // initialize has to drop what the failed compile left behind.
  XNNWeightsCache cache;
  cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
  Result<const uint8_t*> loaded = cache.load_unpacked_data("weight");
  ASSERT_EQ(loaded.error(), Error::Ok);
  ASSERT_EQ(cache.get_num_unpacked_data(), 1u);

  // No finalize_for_runtime(): this is the failed-compile path.
  cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
  ASSERT_EQ(cache.get_num_unpacked_data(), 0u);
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

// Locate the 4-byte seed field of the first index entry by walking the
// footer [index_start:u64][entry_count:u32][magic:u32][version:u32] at the
// end of the file, then the entry record
// [name_len:u32][name][file_offset:u64][data_size:u64][seed:u32].
// Returns 0 when the file is too small or carries no entries.
size_t find_first_entry_seed_offset(std::istream& f) {
  f.seekg(0, std::ios::end);
  const size_t file_size = f.tellg();
  if (file_size < 24) {
    return 0;
  }
  uint8_t footer[20];
  f.seekg(file_size - 20);
  f.read(reinterpret_cast<char*>(footer), 20);
  if (read_le_u32(footer + 8) == 0) {
    return 0;
  }
  const uint64_t index_start = read_le_u64(footer);
  f.seekg(index_start);
  uint8_t name_len_buf[4];
  f.read(reinterpret_cast<char*>(name_len_buf), 4);
  return index_start + 4 + read_le_u32(name_len_buf) + 8 + 8;
}

uint32_t read_first_entry_seed(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  const size_t seed_offset = find_first_entry_seed_offset(f);
  if (seed_offset == 0) {
    return 0;
  }
  uint8_t seed_buf[4];
  f.seekg(seed_offset);
  f.read(reinterpret_cast<char*>(seed_buf), 4);
  return read_le_u32(seed_buf);
}

void write_first_entry_seed(const std::string& path, uint32_t seed) {
  std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
  const size_t seed_offset = find_first_entry_seed_offset(f);
  if (seed_offset == 0) {
    return;
  }
  f.seekp(seed_offset);
  write_le_u32(f, seed);
}

off_t file_size_of(const std::string& path) {
  struct stat st {};
  return ::stat(path.c_str(), &st) == 0 ? st.st_size : -1;
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
  write_first_entry_seed(cache_path, 0xDEADBEEFu);
  ASSERT_EQ(read_first_entry_seed(cache_path), 0xDEADBEEFu);

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

// An XNNPACK upgrade changes the per-ukernel seed, so every loaded entry
// misses look_up and is re-packed. That re-pack must land in the cache file,
// not on the heap: only the file-backed path lets save_packed_index write
// back the refreshed seed. On the heap path the file keeps the stale seed
// and the same re-pack repeats into anonymous dirty memory every launch.
TEST_F(XNNWeightsCacheTest, SeedMismatch_RepackIsPersistedToFile) {
  const std::string cache_path = std::string("/tmp/xnn_weights_cache_reseed_") +
      std::to_string(::getpid()) + ".packed_cache";
  ::unlink(cache_path.c_str());

  std::vector<float> output;
  RunLaunchWithCacheFile(cache_path, output);
  const off_t size_after_first_launch = file_size_of(cache_path);
  ASSERT_GT(size_after_first_launch, 0);

  const uint32_t real_seed = read_first_entry_seed(cache_path);
  ASSERT_NE(real_seed, 0u);

  // Simulate the upgrade: the persisted seed no longer matches what the
  // current ukernel reports for the same weights.
  write_first_entry_seed(cache_path, 0xDEADBEEFu);
  ASSERT_EQ(read_first_entry_seed(cache_path), 0xDEADBEEFu);

  std::vector<float> output_after_upgrade;
  RunLaunchWithCacheFile(cache_path, output_after_upgrade);

  EXPECT_EQ(read_first_entry_seed(cache_path), real_seed)
      << "seed-mismatch re-pack must be written back to the cache file";
  EXPECT_GT(file_size_of(cache_path), size_after_first_launch)
      << "the re-packed weights must be appended to the file, not left on heap";
  EXPECT_EQ(output_after_upgrade, output);

  ::unlink(cache_path.c_str());
}

// The point of persisting the re-pack is convergence: exactly one launch
// pays for the rebuild, and every launch after it is a plain cache hit —
// no new packing, no file growth. Before the fix each launch re-packed into
// heap and the file never stopped being stale.
TEST_F(XNNWeightsCacheTest, SeedMismatch_CacheConvergesAfterOneRebuild) {
  const std::string cache_path =
      std::string("/tmp/xnn_weights_cache_converge_") +
      std::to_string(::getpid()) + ".packed_cache";
  ::unlink(cache_path.c_str());

  std::vector<float> baseline;
  RunLaunchWithCacheFile(cache_path, baseline);
  const uint32_t real_seed = read_first_entry_seed(cache_path);
  ASSERT_NE(real_seed, 0u);

  write_first_entry_seed(cache_path, 0xDEADBEEFu);

  // Rebuild launch: absorbs the one-time re-pack.
  std::vector<float> output;
  RunLaunchWithCacheFile(cache_path, output);
  const off_t size_after_rebuild = file_size_of(cache_path);
  ASSERT_GT(size_after_rebuild, 0);

  // Steady state: three more launches must be pure hits.
  for (int launch = 0; launch < 3; ++launch) {
    RunLaunchWithCacheFile(cache_path, output);
    EXPECT_EQ(file_size_of(cache_path), size_after_rebuild)
        << "cache did not converge; launch " << launch << " re-packed";
    EXPECT_EQ(read_first_entry_seed(cache_path), real_seed);
    EXPECT_EQ(output, baseline);
  }

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

namespace {

std::vector<char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<char>(
      (std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void write_file(const std::string& path, const std::vector<char>& bytes) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f.write(bytes.data(), bytes.size());
}

// A program flatbuffer with the given file identifier, padded to `size` the
// way segments follow it in a real .pte.
std::vector<char> program_file(const char* identifier, size_t size) {
  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(
      executorch_flatbuffer::CreateProgram(fbb, /*version=*/0), identifier);
  std::vector<char> bytes(
      reinterpret_cast<const char*>(fbb.GetBufferPointer()),
      reinterpret_cast<const char*>(fbb.GetBufferPointer()) + fbb.GetSize());
  bytes.resize(size, '\x5a');
  return bytes;
}

void put_le32(char* dst, uint32_t value) {
  for (int i = 0; i < 4; ++i) {
    dst[i] = static_cast<char>((value >> (8 * i)) & 0xff);
  }
}

// A file of `size` filler bytes whose head reads as the current program
// version, with `root_offset` as its flatbuffer root offset.
std::vector<char> leftover_with_head(size_t size, uint32_t root_offset) {
  std::vector<char> bytes(size, '\x33');
  put_le32(bytes.data(), root_offset);
  std::memcpy(bytes.data() + 4, executorch_flatbuffer::ProgramIdentifier(), 4);
  return bytes;
}

// The start of a .ptd file, from the flat tensor schema's own constants (the
// cache spells them out): flatbuffer root offset, the file identifier, then
// the flat tensor header magic at its offset. The root offset points past the
// header's head bytes, as in a real .ptd, where the root table follows the
// header.
std::vector<char> data_file(size_t size) {
  using executorch::extension::FlatTensorHeader;
  std::vector<char> bytes(size, '\x5a');
  put_le32(bytes.data(), FlatTensorHeader::kNumHeadBytes);
  std::memcpy(
      bytes.data() + 4, flat_tensor_flatbuffer::FlatTensorIdentifier(), 4);
  std::memcpy(
      bytes.data() + FlatTensorHeader::kHeaderOffset,
      FlatTensorHeader::kMagic,
      FlatTensorHeader::kMagicSize);
  return bytes;
}

// Unlinks a temporary path when it goes out of scope, so a test that stops at
// a fatal assertion still removes it. Also removes a leftover from an earlier
// run up front.
struct TempPath {
  explicit TempPath(const std::string& name)
      : path(
            "/tmp/xnn_weights_cache_" + name + "_" +
            std::to_string(::getpid())) {
    ::unlink(path.c_str());
  }
  ~TempPath() {
    ::unlink(path.c_str());
  }
  TempPath(const TempPath&) = delete;
  TempPath& operator=(const TempPath&) = delete;
  TempPath(TempPath&&) = delete;
  TempPath& operator=(TempPath&&) = delete;

  const std::string path;
};

// Closes a descriptor when it goes out of scope.
struct ScopedFd {
  explicit ScopedFd(int fd) : fd(fd) {}
  ~ScopedFd() {
    if (fd >= 0) {
      ::close(fd);
    }
  }
  ScopedFd(const ScopedFd&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;
  ScopedFd(ScopedFd&&) = delete;
  ScopedFd& operator=(ScopedFd&&) = delete;

  const int fd;
};

// An ExecuTorch file at the cache path, mapped read-only the way the runtime
// maps a loaded program. Reading through `map` would SIGBUS if the file were
// cut. The mapping, descriptor and file are released when it goes out of
// scope.
struct MappedModel {
  MappedModel(const std::string& name, std::vector<char> file_bytes)
      : file(name + ".pte"), bytes(std::move(file_bytes)) {
    write_file(file.path, bytes);
    fd = ::open(file.path.c_str(), O_RDONLY);
    if (fd >= 0) {
      map = ::mmap(nullptr, bytes.size(), PROT_READ, MAP_SHARED, fd, 0);
    }
  }
  // A current-version program.
  explicit MappedModel(const std::string& name)
      : MappedModel(
            name,
            program_file(
                executorch_flatbuffer::ProgramIdentifier(),
                64 * 1024)) {}
  ~MappedModel() {
    if (map != MAP_FAILED) {
      ::munmap(map, bytes.size());
    }
    if (fd >= 0) {
      ::close(fd);
    }
  }
  MappedModel(const MappedModel&) = delete;
  MappedModel& operator=(const MappedModel&) = delete;
  MappedModel(MappedModel&&) = delete;
  MappedModel& operator=(MappedModel&&) = delete;

  TempPath file;
  const std::string& path = file.path;
  std::vector<char> bytes;
  int fd = -1;
  void* map = MAP_FAILED;
};

} // namespace

// A cache path that names an ExecuTorch file must leave it byte for byte
// unchanged, even while it is mapped (as a loaded model is), and packing falls
// back to heap. Covers the current program version, another program version,
// and a .ptd data file.
TEST_F(XNNWeightsCacheTest, ExecuTorchFilesAtCachePathAreNeverModified) {
  const struct {
    const char* name;
    std::vector<char> bytes;
  } cases[] = {
      {"program",
       program_file(executorch_flatbuffer::ProgramIdentifier(), 64 * 1024)},
      {"other_program_version", program_file("ET99", 64 * 1024)},
      {"data", data_file(64 * 1024)},
      {"root_at_loader_bound", leftover_with_head(64 * 1024, 64 * 1024 - 4)},
  };
  for (const auto& test_case : cases) {
    MappedModel model(std::string("et_") + test_case.name, test_case.bytes);
    ASSERT_NE(model.map, MAP_FAILED) << test_case.name;

    std::vector<float> output;
    RunLaunchWithCacheFile(model.path, output);
    XNNWeightsCache cache;
    cache.set_packed_cache_path(model.path);
    cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
    EXPECT_EQ(
        cache.stats().state,
        executorch::backends::xnnpack::delegate::PackedCacheState::HeapFallback)
        << test_case.name;
    EXPECT_EQ(cache.stats().last_errno, EEXIST) << test_case.name;
    ASSERT_EQ(read_file(model.path), test_case.bytes) << test_case.name;
    // Reading through the mapping would SIGBUS if the file had been cut.
    EXPECT_EQ(
        std::memcmp(model.map, test_case.bytes.data(), test_case.bytes.size()),
        0)
        << test_case.name;
  }
}

// Unchanged behavior: a file this cache left without a trailer (a process
// killed before its first save leaves packs only) is rebuilt, and the rebuilt
// cache loads on the next launch.
TEST_F(XNNWeightsCacheTest, FooterlessLeftoverIsRebuilt) {
  const TempPath file("leftover.packed_cache");
  const std::string& path = file.path;
  write_file(path, std::vector<char>(2 * 4096, '\x33'));

  std::vector<float> output;
  RunLaunchWithCacheFile(path, output);
  XNNWeightsCache cache;
  cache.set_packed_cache_path(path);
  cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
  EXPECT_GT(cache.get_packed_data_names().size(), 0u);
}

// A leftover whose bytes 4-5 happen to read "ET" is not a program unless two
// version digits follow, so it is still rebuilt.
TEST_F(XNNWeightsCacheTest, LeftoverThatOnlyStartsLikeAProgramIsRebuilt) {
  const TempPath file("etlike.packed_cache");
  const std::string& path = file.path;
  std::vector<char> leftover(2 * 4096, '\x33');
  std::memcpy(leftover.data() + 4, "ETx1", 4);
  write_file(path, leftover);

  std::vector<float> output;
  RunLaunchWithCacheFile(path, output);
  XNNWeightsCache cache;
  cache.set_packed_cache_path(path);
  cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
  EXPECT_GT(cache.get_packed_data_names().size(), 0u);
}

// A leftover whose bytes 4-7 read like a program identifier is still rebuilt
// when its root offset couldn't belong to a program the loader would accept:
// inside the root offset and identifier, or with no room for the root table
// inside the file. Packed weights left without a trailer usually look like
// this.
TEST_F(XNNWeightsCacheTest, LeftoverWithAnImpossibleRootOffsetIsRebuilt) {
  constexpr size_t kSize = 2 * 4096;
  const struct {
    const char* name;
    uint32_t root_offset;
  } cases[] = {
      {"zero", 0},
      {"inside_identifier", 4},
      {"past_bound", kSize - 3},
      {"past_end", 0x7fffffff},
  };
  for (const auto& test_case : cases) {
    const TempPath file(std::string("badroot_") + test_case.name);
    write_file(file.path, leftover_with_head(kSize, test_case.root_offset));

    std::vector<float> output;
    RunLaunchWithCacheFile(file.path, output);
    XNNWeightsCache cache;
    cache.set_packed_cache_path(file.path);
    cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
    EXPECT_GT(cache.get_packed_data_names().size(), 0u) << test_case.name;
  }
}

// Unchanged behavior: any other file that fails the cache check is replaced
// by a cache, as before. Only ExecuTorch program and data files are spared.
TEST_F(XNNWeightsCacheTest, OtherFileFailingTheCheckIsReplaced) {
  const TempPath file("other.packed_cache");
  const std::string& path = file.path;
  write_file(path, std::vector<char>(8192, '\xcc'));

  std::vector<float> output;
  RunLaunchWithCacheFile(path, output);
  XNNWeightsCache cache;
  cache.set_packed_cache_path(path);
  cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
  EXPECT_GT(cache.get_packed_data_names().size(), 0u);
}

// Unchanged behavior: a file too short to hold a program or data file head is
// rebuilt, even if its first bytes look like a program. The head is read until
// end of file, so "short" here means the file really ends.
TEST_F(XNNWeightsCacheTest, FileShorterThanAProgramHeadIsRebuilt) {
  const TempPath file("tiny.packed_cache");
  const std::string& path = file.path;
  std::vector<char> tiny(11, '\x33');
  std::memcpy(tiny.data() + 4, executorch_flatbuffer::ProgramIdentifier(), 4);
  write_file(path, tiny);

  std::vector<float> output;
  RunLaunchWithCacheFile(path, output);
  XNNWeightsCache cache;
  cache.set_packed_cache_path(path);
  cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
  EXPECT_GT(cache.get_packed_data_names().size(), 0u);
}

// The first init refuses the model at the cache path and packs to heap. Those
// heap entries stay in the instance, so the next init on the same instance
// takes the branch that reopens the file for writing; a save after it must not
// touch the model. (Before the refusal stuck, that reopen skipped the check
// and the save cut the model to a 20 byte footer.)
TEST_F(XNNWeightsCacheTest, ReinitAfterRefusalThenSaveLeavesTheModelUnchanged) {
  MappedModel model("reinit_save");
  ASSERT_NE(model.map, MAP_FAILED);

  XNNWeightsCache cache;
  cache.set_packed_cache_path(model.path);
  ASSERT_EQ(
      cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get()),
      Error::Ok);
  PackAndRun(cache);
  ASSERT_FALSE(cache.get_packed_data_names().empty());

  ASSERT_EQ(
      cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get()),
      Error::Ok);
  EXPECT_EQ(cache.save_packed_index(), Error::Ok);

  EXPECT_EQ(
      cache.stats().state,
      executorch::backends::xnnpack::delegate::PackedCacheState::HeapFallback);
  EXPECT_EQ(cache.stats().last_errno, EEXIST);
  ASSERT_EQ(read_file(model.path), model.bytes);
  EXPECT_EQ(std::memcmp(model.map, model.bytes.data(), model.bytes.size()), 0);
}

// Same sequence, then a new pack: it goes to heap, not into the model file.
TEST_F(XNNWeightsCacheTest, ReinitAfterRefusalPacksNewWeightsOnHeap) {
  MappedModel model("reinit_pack");
  ASSERT_NE(model.map, MAP_FAILED);

  XNNWeightsCache cache;
  cache.set_packed_cache_path(model.path);
  ASSERT_EQ(
      cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get()),
      Error::Ok);
  PackAndRun(cache);
  ASSERT_EQ(
      cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get()),
      Error::Ok);

  const auto before = cache.stats();
  xnn_weights_cache_t provider = cache.get();
  ASSERT_NE(provider->reserve_space(provider->context, 4096), nullptr);
  const auto after = cache.stats();

  EXPECT_EQ(after.mapped_bytes, before.mapped_bytes);
  EXPECT_EQ(after.heap_bytes, before.heap_bytes + 4096);
  ASSERT_EQ(read_file(model.path), model.bytes);
  EXPECT_EQ(std::memcmp(model.map, model.bytes.data(), model.bytes.size()), 0);
}

// The whole sequence through the manager, which hands one instance to every
// delegate on a path (another partition, another method, a second load):
// the first delegate's init refuses the model and packs named weights to heap,
// a second delegate gets the same instance and initializes it again, a third
// init adds a new pack, then an explicit save and the manager's save_all. The
// model and a live mapping of it stay unchanged throughout.
TEST_F(XNNWeightsCacheTest, SharedInstanceNeverWritesARefusedModel) {
  MappedModel model("shared");
  ASSERT_NE(model.map, MAP_FAILED);
  executorch::backends::xnnpack::XNNWeightsCacheManager manager;

  auto first = manager.get_or_create(model.path);
  ASSERT_TRUE(first.ok());
  {
    std::lock_guard<std::mutex> lock(first.get()->mutex());
    ASSERT_EQ(
        first.get()->initialize_for_runtime(
            memory_allocator_.get(), data_map_.get()),
        Error::Ok);
    PackAndRun(*first.get());
    ASSERT_FALSE(first.get()->get_packed_data_names().empty());
  }

  auto second = manager.get_or_create(model.path);
  ASSERT_TRUE(second.ok());
  ASSERT_EQ(second.get(), first.get());
  XNNWeightsCache& cache = *second.get();
  {
    std::lock_guard<std::mutex> lock(cache.mutex());
    ASSERT_EQ(
        cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get()),
        Error::Ok);
    PackAndRun(cache);
    ASSERT_EQ(
        cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get()),
        Error::Ok);
    xnn_weights_cache_t provider = cache.get();
    ASSERT_NE(provider->reserve_space(provider->context, 4096), nullptr);
    EXPECT_EQ(cache.save_packed_index(), Error::Ok);
  }
  EXPECT_EQ(manager.save_all(), Error::Ok);

  EXPECT_EQ(cache.stats().mapped_bytes, 0);
  EXPECT_EQ(cache.stats().last_errno, EEXIST);
  ASSERT_EQ(read_file(model.path), model.bytes);
  EXPECT_EQ(std::memcmp(model.map, model.bytes.data(), model.bytes.size()), 0);
}

// The refusal is about the file at one path: pointing an instance that holds
// no packed weights at another path lets it write a cache there again.
TEST_F(XNNWeightsCacheTest, NewCachePathClearsTheRefusal) {
  MappedModel model("new_path");
  ASSERT_NE(model.map, MAP_FAILED);
  const TempPath cache_file("new.packed_cache");

  XNNWeightsCache cache;
  cache.set_packed_cache_path(model.path);
  ASSERT_EQ(
      cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get()),
      Error::Ok);
  ASSERT_TRUE(cache.get_packed_data_names().empty());

  cache.set_packed_cache_path(cache_file.path);
  ASSERT_EQ(
      cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get()),
      Error::Ok);
  const auto before = cache.stats();
  xnn_weights_cache_t provider = cache.get();
  ASSERT_NE(provider->reserve_space(provider->context, 4096), nullptr);

  EXPECT_EQ(cache.stats().mapped_bytes, before.mapped_bytes + 4096);
  EXPECT_EQ(read_file(model.path), model.bytes);
}

// An instance that holds packed weights and is pointed at another path never
// accepted the file there, so it packs in memory and leaves that file alone
// rather than opening it for writing.
TEST_F(XNNWeightsCacheTest, NewCachePathWithWeightsInMemoryPacksOnHeap) {
  MappedModel model("new_path_heap");
  ASSERT_NE(model.map, MAP_FAILED);
  const TempPath other("new_path_heap_other.packed_cache");
  const std::vector<char> other_bytes(8192, '\xcc');
  write_file(other.path, other_bytes);

  XNNWeightsCache cache;
  cache.set_packed_cache_path(model.path);
  ASSERT_EQ(
      cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get()),
      Error::Ok);
  PackAndRun(cache);
  ASSERT_FALSE(cache.get_packed_data_names().empty());

  cache.set_packed_cache_path(other.path);
  ASSERT_EQ(
      cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get()),
      Error::Ok);
  const auto before = cache.stats();
  xnn_weights_cache_t provider = cache.get();
  ASSERT_NE(provider->reserve_space(provider->context, 4096), nullptr);

  EXPECT_EQ(cache.stats().mapped_bytes, before.mapped_bytes);
  EXPECT_EQ(cache.stats().heap_bytes, before.heap_bytes + 4096);
  EXPECT_EQ(read_file(other.path), other_bytes);
  EXPECT_EQ(read_file(model.path), model.bytes);
}

// The weights this process packs sit at the start of its own cache file and
// can happen to read like a program or data file head, with a root offset the
// head check accepts. The cache must keep using that file: the save writes
// the trailer, the next launch loads it, and that launch can still write to
// it. The next launch is made to re-pack (a stale seed in the index), and the
// re-pack only lands in the file if that launch reopened it for writing. A
// third launch then loads it with nothing refused.
TEST_F(
    XNNWeightsCacheTest,
    CacheWhoseWeightsStartLikeAnExecuTorchFileKeepsWorking) {
  const struct {
    const char* name;
    const char* bytes_4_to_11;
  } cases[] = {
      {"program", "ET12\x33\x33\x33\x33"},
      {"data", "FT01FH01"},
  };
  for (const auto& test_case : cases) {
    const TempPath file(std::string("weights_head_") + test_case.name);
    std::vector<float> output;
    RunLaunchWithCacheFile(file.path, output);
    const uint32_t real_seed = read_first_entry_seed(file.path);
    ASSERT_NE(real_seed, 0u);

    std::array<char, 12> head{};
    put_le32(head.data(), 8);
    std::memcpy(head.data() + 4, test_case.bytes_4_to_11, 8);
    {
      const ScopedFd writer(::open(file.path.c_str(), O_WRONLY));
      ASSERT_GE(writer.fd, 0);
      ASSERT_EQ(::pwrite(writer.fd, head.data(), head.size(), 0), 12);
    }
    write_first_entry_seed(file.path, 0xDEADBEEFu);
    const off_t size_before = file_size_of(file.path);

    std::vector<float> output_after;
    RunLaunchWithCacheFile(file.path, output_after);
    EXPECT_EQ(read_first_entry_seed(file.path), real_seed) << test_case.name;
    EXPECT_GT(file_size_of(file.path), size_before) << test_case.name;

    XNNWeightsCache third;
    third.set_packed_cache_path(file.path);
    ASSERT_EQ(
        third.initialize_for_runtime(memory_allocator_.get(), data_map_.get()),
        Error::Ok);
    EXPECT_GT(third.get_packed_data_names().size(), 0u) << test_case.name;
    EXPECT_EQ(
        third.stats().failure,
        executorch::backends::xnnpack::delegate::PackedCacheFailure::None)
        << test_case.name;
    EXPECT_EQ(
        third.stats().state,
        executorch::backends::xnnpack::delegate::PackedCacheState::FileBacked)
        << test_case.name;
  }
}

// Trust comes from the file's own trailer, read on the descriptor that would
// be written. After a load, a model renamed over the path isn't written when
// the instance reopens the path: that init packs in memory and leaves the
// model alone. The refusal doesn't stick, so a cache renamed back is used
// again.
TEST_F(XNNWeightsCacheTest, ModelRenamedOverALoadedCacheIsNotWritten) {
  const TempPath file("renamed_over.packed_cache");
  std::vector<float> output;
  RunLaunchWithCacheFile(file.path, output);
  const std::vector<char> cache_bytes = read_file(file.path);

  // The first init can't take the write lock, so the instance holds the
  // loaded entries with no write descriptor, and the next init reopens.
  const ScopedFd holder(::open(file.path.c_str(), O_RDONLY));
  ASSERT_GE(holder.fd, 0);
  ASSERT_EQ(::flock(holder.fd, LOCK_SH), 0);
  XNNWeightsCache cache;
  cache.set_packed_cache_path(file.path);
  ASSERT_EQ(
      cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get()),
      Error::Ok);
  ASSERT_GT(cache.get_packed_data_names().size(), 0u);
  ASSERT_EQ(cache.stats().last_errno, EWOULDBLOCK);
  ASSERT_EQ(::flock(holder.fd, LOCK_UN), 0);

  MappedModel model("renamed_over_model");
  ASSERT_NE(model.map, MAP_FAILED);
  ASSERT_EQ(::rename(model.path.c_str(), file.path.c_str()), 0);
  ASSERT_EQ(
      cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get()),
      Error::Ok);
  EXPECT_EQ(cache.stats().last_errno, 0);
  ASSERT_EQ(read_file(file.path), model.bytes);
  EXPECT_EQ(std::memcmp(model.map, model.bytes.data(), model.bytes.size()), 0);

  const TempPath copy("renamed_over_copy.packed_cache");
  write_file(copy.path, cache_bytes);
  ASSERT_EQ(::rename(copy.path.c_str(), file.path.c_str()), 0);
  ASSERT_EQ(
      cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get()),
      Error::Ok);
  // The cache reopened the file and holds its exclusive lock.
  const ScopedFd probe(::open(file.path.c_str(), O_RDONLY));
  ASSERT_GE(probe.fd, 0);
  errno = 0;
  EXPECT_EQ(::flock(probe.fd, LOCK_SH | LOCK_NB), -1);
  EXPECT_EQ(errno, EWOULDBLOCK);
}

// A later init of the same instance reopens a file it accepted without a
// content check, even when the file starts like a program. Here the first
// init loads the cache but can't take the write lock (the test holds a shared
// lock), so the entries stay in memory with no write descriptor. Once the
// lock is free, the next init takes the reopen branch. Holding the exclusive
// lock afterwards shows the cache reopened its file.
TEST_F(XNNWeightsCacheTest, ReinitReopensItsOwnLoadedFile) {
  const TempPath file("reopen_own.packed_cache");
  {
    XNNWeightsCache writer;
    writer.set_packed_cache_path(file.path);
    ASSERT_EQ(
        writer.initialize_for_runtime(memory_allocator_.get(), data_map_.get()),
        Error::Ok);
    PackAndRun(writer);
  }
  // The cache's first packed bytes now read like a program head, with a root
  // offset the open-time check accepts. The trailer is untouched.
  std::array<char, 8> head{};
  put_le32(head.data(), 8);
  std::memcpy(head.data() + 4, executorch_flatbuffer::ProgramIdentifier(), 4);
  const ScopedFd holder(::open(file.path.c_str(), O_RDWR));
  ASSERT_GE(holder.fd, 0);
  ASSERT_EQ(::pwrite(holder.fd, head.data(), head.size(), 0), 8);
  ASSERT_EQ(::flock(holder.fd, LOCK_SH), 0);

  XNNWeightsCache cache;
  cache.set_packed_cache_path(file.path);
  ASSERT_EQ(
      cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get()),
      Error::Ok);
  ASSERT_GT(cache.get_packed_data_names().size(), 0u);
  ASSERT_EQ(cache.stats().last_errno, EWOULDBLOCK);

  ASSERT_EQ(::flock(holder.fd, LOCK_UN), 0);
  ASSERT_EQ(
      cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get()),
      Error::Ok);
  EXPECT_NE(cache.stats().last_errno, EEXIST);
  errno = 0;
  EXPECT_EQ(::flock(holder.fd, LOCK_SH | LOCK_NB), -1);
  EXPECT_EQ(errno, EWOULDBLOCK);
}

// A cache path whose start can't be read is left alone. A FIFO makes pread
// fail (ESPIPE): it is refused, not truncated, and weights pack to heap.
TEST_F(XNNWeightsCacheTest, UnreadableFileAtCachePathIsLeftAlone) {
  const TempPath file("fifo.packed_cache");
  const std::string& path = file.path;
  ASSERT_EQ(::mkfifo(path.c_str(), 0600), 0);
  // Keep both ends open so the cache's own opens of the FIFO don't block
  // waiting for the other end.
  const ScopedFd hold(::open(path.c_str(), O_RDWR | O_NONBLOCK));
  ASSERT_GE(hold.fd, 0);

  XNNWeightsCache cache;
  cache.set_packed_cache_path(path);
  ASSERT_EQ(
      cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get()),
      Error::Ok);
  PackAndRun(cache);

  const auto stats = cache.stats();
  EXPECT_EQ(
      stats.state,
      executorch::backends::xnnpack::delegate::PackedCacheState::HeapFallback);
  EXPECT_EQ(
      stats.failure,
      executorch::backends::xnnpack::delegate::PackedCacheFailure::OpenFailed);
  EXPECT_EQ(stats.last_errno, ESPIPE);
  EXPECT_EQ(stats.mapped_bytes, 0);
  EXPECT_GT(stats.heap_bytes, 0);
  struct stat st {};
  ASSERT_EQ(::stat(path.c_str(), &st), 0);
  EXPECT_TRUE(S_ISFIFO(st.st_mode));
}

// Unchanged behavior: a real cache stays file-backed across a re-init on the
// same instance. New packs still go to the file, saves still succeed, and the
// file still loads as a cache.
TEST_F(XNNWeightsCacheTest, NormalCacheIsUnaffectedByReinit) {
  const TempPath file("reinit_ok.packed_cache");
  const std::string& path = file.path;
  {
    XNNWeightsCache cache;
    cache.set_packed_cache_path(path);
    ASSERT_EQ(
        cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get()),
        Error::Ok);
    PackAndRun(cache);
    ASSERT_EQ(cache.save_packed_index(), Error::Ok);

    ASSERT_EQ(
        cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get()),
        Error::Ok);
    const auto before = cache.stats();
    xnn_weights_cache_t provider = cache.get();
    ASSERT_NE(provider->reserve_space(provider->context, 4096), nullptr);
    const auto after = cache.stats();
    EXPECT_EQ(after.mapped_bytes, before.mapped_bytes + 4096);
    EXPECT_EQ(after.heap_bytes, before.heap_bytes);
    EXPECT_EQ(cache.save_packed_index(), Error::Ok);
    EXPECT_EQ(
        cache.stats().state,
        executorch::backends::xnnpack::delegate::PackedCacheState::FileBacked);
  }

  XNNWeightsCache reloaded;
  reloaded.set_packed_cache_path(path);
  ASSERT_EQ(
      reloaded.initialize_for_runtime(memory_allocator_.get(), data_map_.get()),
      Error::Ok);
  EXPECT_GT(reloaded.get_packed_data_names().size(), 0u);
  EXPECT_EQ(
      reloaded.stats().state,
      executorch::backends::xnnpack::delegate::PackedCacheState::FileBacked);
  EXPECT_EQ(
      reloaded.stats().failure,
      executorch::backends::xnnpack::delegate::PackedCacheFailure::None);
}

namespace {

// A file served through read_fully_at's read_at hook, so a test can make
// pread return short reads, EINTR or an error.
struct FakeFile {
  std::vector<uint8_t> bytes;
  size_t max_chunk = SIZE_MAX;
  int eintr_left = 0;
  int fail_errno = 0;
  int calls = 0;
};

// The file the fake reads, set by each test. A function-local static keeps it
// out of namespace scope.
FakeFile*& fake_file() {
  static FakeFile* file = nullptr;
  return file;
}

// Points the fake at a test's file for the scope, and clears it after so no
// later read follows a pointer to a finished test's local.
struct ScopedFakeFile {
  explicit ScopedFakeFile(FakeFile& f) {
    fake_file() = &f;
  }
  ~ScopedFakeFile() {
    fake_file() = nullptr;
  }
  ScopedFakeFile(const ScopedFakeFile&) = delete;
  ScopedFakeFile& operator=(const ScopedFakeFile&) = delete;
  ScopedFakeFile(ScopedFakeFile&&) = delete;
  ScopedFakeFile& operator=(ScopedFakeFile&&) = delete;
};

ssize_t fake_read_at(int /*fd*/, void* buf, size_t n, off_t offset) {
  FakeFile& f = *fake_file();
  ++f.calls;
  if (f.eintr_left > 0) {
    --f.eintr_left;
    errno = EINTR;
    return -1;
  }
  if (f.fail_errno != 0) {
    errno = f.fail_errno;
    return -1;
  }
  const size_t pos = static_cast<size_t>(offset);
  if (pos >= f.bytes.size()) {
    return 0;
  }
  const size_t k = std::min({n, f.max_chunk, f.bytes.size() - pos});
  std::memcpy(buf, f.bytes.data() + pos, k);
  return static_cast<ssize_t>(k);
}

std::vector<uint8_t> counting_bytes(size_t n) {
  std::vector<uint8_t> bytes(n);
  for (size_t i = 0; i < n; ++i) {
    bytes[i] = static_cast<uint8_t>(i);
  }
  return bytes;
}

} // namespace

using executorch::backends::xnnpack::delegate::internal::read_fully_at;

// pread may return fewer bytes than asked without reaching end of file. The
// cache file check reads its 12 bytes through read_fully_at, which keeps
// reading, so a long model is never mistaken for a file shorter than the head.
TEST(XNNWeightsCacheReadTest, ShortReadsOfALongerFileAreCompleted) {
  FakeFile f;
  f.bytes = counting_bytes(64);
  f.max_chunk = 5;
  const ScopedFakeFile scope(f);
  std::array<uint8_t, 12> head{};

  EXPECT_EQ(read_fully_at(3, head.data(), head.size(), 0, &fake_read_at), 12);
  EXPECT_TRUE(std::equal(head.begin(), head.end(), f.bytes.begin()));
  EXPECT_EQ(f.calls, 3);
}

TEST(XNNWeightsCacheReadTest, EintrIsRetried) {
  FakeFile f;
  f.bytes = counting_bytes(64);
  f.eintr_left = 2;
  const ScopedFakeFile scope(f);
  std::array<uint8_t, 12> head{};

  EXPECT_EQ(read_fully_at(3, head.data(), head.size(), 0, &fake_read_at), 12);
  EXPECT_TRUE(std::equal(head.begin(), head.end(), f.bytes.begin()));
}

TEST(XNNWeightsCacheReadTest, StopsShortOnlyAtEndOfFile) {
  FakeFile f;
  f.bytes = counting_bytes(7);
  f.max_chunk = 3;
  const ScopedFakeFile scope(f);
  std::array<uint8_t, 12> head{};

  EXPECT_EQ(read_fully_at(3, head.data(), head.size(), 0, &fake_read_at), 7);
  EXPECT_TRUE(std::equal(head.begin(), head.begin() + 7, f.bytes.begin()));
}

TEST(XNNWeightsCacheReadTest, ReportsOtherErrors) {
  FakeFile f;
  f.bytes = counting_bytes(64);
  f.fail_errno = EIO;
  const ScopedFakeFile scope(f);
  std::array<uint8_t, 12> head{};

  errno = 0;
  EXPECT_EQ(read_fully_at(3, head.data(), head.size(), 0, &fake_read_at), -1);
  EXPECT_EQ(errno, EIO);
}

#endif
