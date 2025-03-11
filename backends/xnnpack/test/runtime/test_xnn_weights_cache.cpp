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
#include <gtest/gtest.h>
#include <xnnpack.h>

using executorch::backends::xnnpack::delegate::XNNWeightsCache;
using executorch::runtime::MemoryAllocator;
using executorch::extension::FileDataLoader;
using executorch::extension::testing::TempFile;
using executorch::runtime::DataLoader;
using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
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
      data_map_loader_.get(), 0, program_->named_data(), program_->segments());
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
    float* output_data
  ){
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
    Result<const uint8_t*> weight_pointer = weight_cache.load_unpacked_data(
      "weight"
    );
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
    Result<const uint8_t*> bias_pointer = weight_cache.load_unpacked_data(
      "bias"
    );
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
    Result<std::vector<std::string>> packed_weights_added = weight_cache.finalize_for_runtime();
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
      output_data
    );

    weight_cache.initialize_for_runtime(memory_allocator_.get(), data_map_.get());
    BuildAndRunGraphWithWeightsCache(
      weight_cache,
      batches,
      input_channels,
      output_channels,
      input_data,
      output_data
    );
    ASSERT_EQ(weight_cache.get_num_unpacked_data(), 0);
    weight_cache.delete_packed_data(weight_cache.get_packed_data_names());
    std::vector<std::string> packed_data_names = weight_cache.get_packed_data_names();
    // check packed data names have been deleted
    ASSERT_EQ(packed_data_names.size(), 0);
}
