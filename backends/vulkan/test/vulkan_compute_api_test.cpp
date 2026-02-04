/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <bitset>
#include <iomanip>
#include <utility>
#include <vector>

#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/StagingUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/QPackUtils.h>

#include <executorch/backends/vulkan/test/utils/test_utils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/DispatchNode.h>

using namespace vkcompute;
using namespace vkcompute::api;

std::vector<float>
transpose_matrix(std::vector<float>& mat, const int H, const int W) {
  std::vector<float> out(W * H);
  for (int out_y = 0; out_y < H; ++out_y) {
    for (int out_x = 0; out_x < W; ++out_x) {
      out[out_x * H + out_y] = mat[out_y * W + out_x];
    }
  }
  return out;
}

std::vector<float> compute_reference_matmul(
    std::vector<float>& mat1,
    std::vector<float>& mat2,
    const int M,
    const int K,
    const int N) {
  std::vector<float> out(M * N);
  for (int out_y = 0; out_y < M; ++out_y) {
    for (int out_x = 0; out_x < N; ++out_x) {
      out[out_y * N + out_x] = 0;
      for (int k = 0; k < K; ++k) {
        out[out_y * N + out_x] += mat1[out_y * K + k] * mat2[k * N + out_x];
      }
    }
  }
  return out;
}

std::vector<std::vector<int64_t>> standard_sizes_to_test = {
    // 1D
    {7},
    {13},
    {24},
    // 2D
    {7, 11},
    {13, 6},
    // 3D
    {2, 9, 7},
    {9, 15, 19},
    {7, 11, 24},
    {13, 8, 11},
    {12, 11, 19},
    // 4D
    {2, 2, 3, 5},
    {9, 13, 11, 17},
    {17, 14, 18, 20},
    {7, 13, 12, 21},
    {3, 8, 13, 17},
};

//
// Compute API Tests
//

class VulkanComputeAPITest : public ::testing::Test {
 public:
  void SetUp() override {
    // Make sure we are starting with a clean slate
    EXPECT_TRUE(get_vma_allocation_count() == 0);
  }

  void TearDown() override {
    context()->flush();

    // Make sure we are ending with a clean slate
    EXPECT_TRUE(get_vma_allocation_count() == 0);
  }
};

TEST_F(VulkanComputeAPITest, print_adapter) {
  std::cout << *(context()->adapter_ptr()) << std::endl;
}

#if defined(VK_KHR_pipeline_executable_properties) && \
    defined(ETVK_INSPECT_PIPELINES)

TEST_F(VulkanComputeAPITest, print_shader_executable_properties) {
  context()->print_shader_executable_properties(
      VK_KERNEL(binary_add_nobroadcast__test_half), {0});
}

#endif // VK_KHR_pipeline_executable_properties && ETVK_INSPECT_PIPELINES

std::vector<int64_t> get_reference_dim_order(
    const size_t ndim,
    const int32_t packed_dim,
    const int32_t outer_packed_dim,
    const bool block_transposed) {
  // Special case for zero dim tensors
  if (ndim == 0) {
    return {0};
  }

  // Initialize dim_order as {0, 1, 2, ..., ndim-1}
  std::vector<int64_t> dim_order(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    dim_order[i] = static_cast<int64_t>(i);
  }

  int64_t ndim_signed = static_cast<int64_t>(ndim);

  // Convert WHCN indices to NCHW indices
  // packed_dim and outer_packed_dim are in WHCN order (0=W, 1=H, 2=C, 3=N)
  // NCHW index = ndim - 1 - WHCN index
  int64_t last_dim_nchw = ndim_signed - 1 - packed_dim;
  int64_t second_last_dim_nchw = ndim_signed - 1 - outer_packed_dim;
  if (block_transposed) {
    std::swap(last_dim_nchw, second_last_dim_nchw);
  }

  // Move last_dim_nchw to the back (if valid)
  bool last_dim_valid = last_dim_nchw >= 0 && last_dim_nchw < ndim_signed;
  if (last_dim_valid) {
    auto it = std::find(dim_order.begin(), dim_order.end(), last_dim_nchw);
    if (it != dim_order.end()) {
      dim_order.erase(it);
      dim_order.push_back(last_dim_nchw);
    }
  }

  // Move second_last_dim_nchw to:
  // a) second last position if last_dim_nchw is valid
  // b) last position if last_dim_nchw was not valid
  bool second_last_dim_valid =
      second_last_dim_nchw >= 0 && second_last_dim_nchw < ndim_signed;
  if (second_last_dim_valid) {
    auto it =
        std::find(dim_order.begin(), dim_order.end(), second_last_dim_nchw);
    if (it != dim_order.end()) {
      dim_order.erase(it);
      if (last_dim_valid && ndim >= 2) {
        // Insert at second last position
        dim_order.insert(dim_order.end() - 1, second_last_dim_nchw);
      } else {
        // Insert at last position
        dim_order.push_back(second_last_dim_nchw);
      }
    }
  }

  return dim_order;
}

std::vector<int64_t> get_reference_padded_sizes(
    const std::vector<int64_t>& sizes,
    const int32_t packed_dim,
    const int32_t packed_dim_block_size,
    const int32_t outer_packed_dim = -1,
    const int32_t outer_packed_dim_block_size = 1) {
  int64_t ndim = sizes.size();
  if (ndim == 0) {
    ndim = 1;
  }

  // Tensor sizes will be unsqueezed up to the next multiple of 4
  const int64_t ndim_up4 = utils::align_up_4(ndim);
  std::vector<int64_t> padded_sizes(ndim_up4);
  for (int64_t i = 0; i < ndim_up4; ++i) {
    padded_sizes.at(i) = utils::val_at(i - ndim_up4, sizes);
  }

  // Pad the packed dim to the next multiple of the block size if > 1
  if (packed_dim_block_size > 1) {
    const int64_t dim_offset = packed_dim + 1;
    const int64_t padded_dim_size = utils::val_at(-dim_offset, sizes);
    padded_sizes.at(ndim_up4 - dim_offset) = utils::align_up(
        padded_dim_size, static_cast<int64_t>(packed_dim_block_size));
  }

  // For block-packed layouts, also pad the outer packed dimension if > 1
  if (outer_packed_dim >= 0 && outer_packed_dim != packed_dim &&
      outer_packed_dim_block_size > 1) {
    const int64_t outer_dim_offset = outer_packed_dim + 1;
    const int64_t outer_padded_dim_size =
        utils::val_at(-outer_dim_offset, sizes);
    padded_sizes.at(ndim_up4 - outer_dim_offset) = utils::align_up(
        outer_padded_dim_size,
        static_cast<int64_t>(outer_packed_dim_block_size));
  }

  return padded_sizes;
}

std::vector<int64_t> get_reference_strides(
    const std::vector<int64_t>& sizes,
    const utils::GPUMemoryLayout layout,
    const bool flip_unsqueezed = false) {
  int64_t C = utils::val_at(-3, sizes);
  int64_t H = utils::val_at(-2, sizes);
  int64_t W = utils::val_at(-1, sizes);

  int64_t numel = utils::multiply_integers(sizes);

  switch (layout) {
    case utils::kWidthPacked:
      switch (sizes.size()) {
        case 1:
          if (flip_unsqueezed)
            return {1, numel, numel, numel};
          return {1};
        case 2:
          if (flip_unsqueezed)
            return {1, W, numel, numel};
          return {W, 1};
        case 3:
          if (flip_unsqueezed)
            return {1, W, H * W, numel};
          return {H * W, W, 1};
        case 4:
          if (flip_unsqueezed)
            return {1, W, H * W, C * H * W};
          return {C * H * W, H * W, W, 1};
        default:
          return {};
      }
      break;
    case utils::kHeightPacked:
      switch (sizes.size()) {
        case 1:
          if (flip_unsqueezed)
            return {1, numel, numel, numel};
          return {1};
        case 2:
          if (flip_unsqueezed)
            return {H, 1, numel, numel};
          return {1, H};
          return {1, H};
        case 3:
          if (flip_unsqueezed)
            return {H, 1, H * W, numel};
          return {W * H, 1, H};
        case 4:
          if (flip_unsqueezed)
            return {H, 1, W * H, C * W * H};
          return {C * W * H, W * H, 1, H};
        default:
          return {};
      }
    case utils::kChannelsPacked:
      switch (sizes.size()) {
        case 1:
          if (flip_unsqueezed)
            return {1, numel, numel, numel};
          return {1};
        case 2:
          if (flip_unsqueezed)
            return {1, W, numel, numel};
          return {W, 1};
        case 3:
          if (flip_unsqueezed)
            return {C, W * C, 1, numel};
          return {1, W * C, C};
        case 4:
          if (flip_unsqueezed)
            return {C, W * C, 1, H * W * C};
          return {H * W * C, 1, W * C, C};
        default:
          return {};
      }
    default:
      VK_THROW("Unsupported memory layout: ", layout);
  }
  return {};
}

int64_t get_reference_physical_numel(
    const vkapi::ScalarType dtype,
    const std::vector<int64_t>& padded_sizes) {
  size_t numel = utils::multiply_integers(padded_sizes);

  // For kInt8x4, the data buffer is interpreted as an array of int32, where
  // each int32 contains 4xint8 values. To account for this, the number of
  // elements needs to be divided by 4.
  if (dtype == vkapi::kInt8x4) {
    // Should already be a multiple of 4 due to padding
    if (numel % 4 != 0) {
      VK_THROW("Expected numel to be multiple of 4 for kInt8x4");
    }
    numel /= 4;
  }

  // For 8-bit types, align to the next multiple of 4. For devices that do not
  // support 8-bit storage buffers, the tensor data will be interpreted as an
  // array of int32 instead.
  if (vkapi::element_size(dtype) == 1) {
    numel = utils::align_up_4(numel);
  }
  return numel;
}

utils::uvec3 get_reference_image_extents(
    const vkapi::ScalarType dtype,
    const int32_t packed_dim,
    const int32_t outer_packed_dim,
    const bool is_block_packed,
    const std::vector<int64_t>& padded_sizes,
    const std::vector<int64_t>& axis_map) {
  utils::uvec3 extents({1, 1, 1});

  const int64_t packed_dim_axis = axis_map.at(packed_dim);
  const int64_t outer_packed_dim_axis = axis_map.at(outer_packed_dim);

  // If the packed dim is not padded to the next multiple of 4, then that means
  // this tensor is using buffer storage and does not require texture extents.
  const int64_t packed_dim_idx = padded_sizes.size() - 1 - packed_dim;
  if (padded_sizes.at(packed_dim_idx) % 4 != 0) {
    return extents;
  }

  // For high dimensional tensors, buffer storage must be used. No need to
  // compute image extents in this case.
  if (padded_sizes.size() > 4) {
    return extents;
  }

  // First three elements of axis_map indicate which (X,Y,Z) image axis the
  // width, height, and channels dim of the tensor maps to.
  for (int whcn_dim = 0; whcn_dim < 3; ++whcn_dim) {
    const int64_t axis = axis_map.at(whcn_dim);
    const int64_t dim = padded_sizes.size() - 1 - whcn_dim;
    extents[axis] = utils::safe_downcast<uint32_t>(padded_sizes.at(dim));
  }

  // For "regular" tensor dtypes, 4 elements along the packed dim are packed
  // into one texel (4-component vectorized type). However, for kInt8x4 dtype,
  // an additional level of packing is employed where 4 int8 elements are
  // packed into one int32, and then 4 int32 are packed into each ivec4 texel.
  if (dtype == vkapi::kInt8x4) {
    // For layouts with only one packed dimension, loading an ivec4 texel from
    // the texture loads 16 int8 values (4 int32 that each contain 4 int8).
    if (!is_block_packed) {
      extents[packed_dim_axis] = utils::div_up(extents[packed_dim_axis], 16u);
    }
    // Layouts with two packed dimension (e.g., 4W4C, 4H4W) load a 4x4 block of
    // data from two dimensions with each ivec4 texel load, as opposed to 16
    // adjacent values from a single dimension.
    else {
      if (extents[outer_packed_dim_axis] % 4 != 0) {
        VK_THROW("Expected outer_packed_dim_axis extent to be multiple of 4");
      }
      extents[outer_packed_dim_axis] /= 4;
      if (extents[packed_dim_axis] % 4 != 0) {
        VK_THROW("Expected packed_dim_axis extent to be multiple of 4");
      }
      extents[packed_dim_axis] /= 4;
    }
  } else {
    extents[packed_dim_axis] /= 4;
  }

  // axis_map[3] indicates the WHCN index of the dimension used for batch
  // concatenation. Thus a double lookup is required to determine the image axis
  // used for batch concatenation.
  const int64_t concatted_whcn_dim = axis_map.at(3);
  const int64_t batch_axis = axis_map.at(concatted_whcn_dim);
  // Multiply the extents of the batch axis by the batch size.
  extents[batch_axis] *= padded_sizes.at(0);

  return extents;
}

TEST_F(VulkanComputeAPITest, empty_init_shader_info_test) {
  vkapi::ShaderInfo empty_shader_info;
  EXPECT_FALSE(empty_shader_info);
  EXPECT_TRUE(empty_shader_info.src_code.bin == nullptr);
  EXPECT_TRUE(empty_shader_info.src_code.size == 0u);
}

bool compare_vectors(
    const std::vector<int32_t>& v32,
    const std::vector<int64_t>& v64) {
  if (v32.size() != v64.size()) {
    return false;
  }
  for (size_t i = 0; i < v32.size(); ++i) {
    if (static_cast<int64_t>(v32[i]) != v64[i]) {
      return false;
    }
  }
  return true;
}

TEST_F(VulkanComputeAPITest, tensor_layout_metadata_test) {
  // Test all combinations of tensor sizes, storage types, and memory layouts
  // to ensure that layout metadata is computed correctly

  // Define test configuration for each layout type
  struct LayoutTestConfig {
    utils::GPUMemoryLayout layout;
    vkapi::ScalarType dtype;
    int32_t packed_dim;
    int32_t outer_packed_dim;
    bool is_block_packed;
    bool block_transposed;
  };

  std::vector<LayoutTestConfig> layout_configs = {
      // Standard layouts with float dtype
      // For non-block-packed: outer_packed_dim = (packed_dim == 0) ? 1 : 0
      {utils::kWidthPacked,
       vkapi::kFloat,
       WHCN::kWidthDim,
       WHCN::kHeightDim,
       false,
       false},
      {utils::kHeightPacked,
       vkapi::kFloat,
       WHCN::kHeightDim,
       WHCN::kWidthDim,
       false,
       false},
      {utils::kChannelsPacked,
       vkapi::kFloat,
       WHCN::kChannelsDim,
       WHCN::kWidthDim,
       false,
       false},

      // Packed int8 vector layouts (single-dimension packed)
      // Use kChar, which should be converted to kInt8x4
      {utils::kPackedInt8_4W,
       vkapi::kChar,
       WHCN::kWidthDim,
       WHCN::kHeightDim,
       false,
       false},
      {utils::kPackedInt8_4C,
       vkapi::kChar,
       WHCN::kChannelsDim,
       WHCN::kWidthDim,
       false,
       false},

      // Packed int8 block layouts (two-dimension packed)
      // Use kChar, which should be converted to kInt8x4
      {utils::kPackedInt8_4W4C,
       vkapi::kChar,
       WHCN::kChannelsDim,
       WHCN::kWidthDim,
       true,
       false},
      {utils::kPackedInt8_4H4W,
       vkapi::kChar,
       WHCN::kWidthDim,
       WHCN::kHeightDim,
       true,
       false},
      {utils::kPackedInt8_4C1W,
       vkapi::kChar,
       WHCN::kChannelsDim,
       WHCN::kWidthDim,
       false,
       true},
  };

  std::vector<utils::StorageType> storage_types = {
      utils::kBuffer, utils::kTexture3D};

  for (const auto& sizes : standard_sizes_to_test) {
    if (sizes.size() < 2) {
      continue; // Skip 1D tensors
    }

    for (const auto& storage_type : storage_types) {
      for (const auto& config : layout_configs) {
        // Skip block-packed layouts for tensors with less than 3 dimensions
        if (config.is_block_packed && sizes.size() < 3) {
          continue;
        }

        // Create tensor
        vTensor tensor(
            context(),
            sizes,
            config.dtype,
            storage_type,
            config.layout,
            /*allocate_memory = */ false);

        // Verify sizes
        ASSERT_TRUE(tensor.sizes() == sizes)
            << "Sizes mismatch for layout=" << static_cast<int>(config.layout)
            << ", storage=" << static_cast<int>(storage_type);

        // Verify dtype
        // For packed int8 layouts, kChar should be converted to kInt8x4
        vkapi::ScalarType expected_dtype = config.dtype;
        if (config.dtype == vkapi::kChar) {
          expected_dtype = vkapi::kInt8x4;
        }
        ASSERT_EQ(tensor.dtype(), expected_dtype)
            << "Dtype mismatch for layout=" << static_cast<int>(config.layout)
            << ", expected=" << static_cast<int>(expected_dtype)
            << ", got=" << static_cast<int>(tensor.dtype());

        // Determine packed_dim_block_size based on layout and storage type
        // - kInt8 non-block-packed + texture: 16 (16 values per texel)
        // - kInt8 non-block-packed + buffer: 4 (alignment)
        // - kInt8 block-packed: 4 (4 values per dim per texel)
        // - Standard texture: 4 (4 values per texel)
        // - Contiguous buffer: 1 (no padding)
        const bool is_non_block_packed_int8 =
            config.dtype == vkapi::kChar && !config.is_block_packed;
        int32_t expected_packed_dim_block_size;
        if (is_non_block_packed_int8 && storage_type != utils::kBuffer) {
          expected_packed_dim_block_size = 16;
        } else if (config.dtype == vkapi::kChar) {
          expected_packed_dim_block_size = 4;
        } else if (storage_type != utils::kBuffer) {
          expected_packed_dim_block_size = 4;
        } else {
          expected_packed_dim_block_size = 1;
        }

        // For block-packed layouts, outer_packed_dim is also padded
        const int32_t expected_outer_packed_dim_block_size =
            config.is_block_packed ? 4 : 1;

        // Expected block_numel is the product of the two block sizes
        const int32_t expected_block_numel = expected_packed_dim_block_size *
            expected_outer_packed_dim_block_size;

        // Verify packed_dim_info
        const auto& packed_dim_info = tensor.packed_dim_info();
        ASSERT_EQ(packed_dim_info.packed_dim, config.packed_dim)
            << "packed_dim mismatch for layout="
            << static_cast<int>(config.layout);
        ASSERT_EQ(
            packed_dim_info.packed_dim_block_size,
            expected_packed_dim_block_size)
            << "packed_dim_block_size mismatch for layout="
            << static_cast<int>(config.layout);
        ASSERT_EQ(packed_dim_info.outer_packed_dim, config.outer_packed_dim)
            << "outer_packed_dim mismatch for layout="
            << static_cast<int>(config.layout);
        ASSERT_EQ(
            packed_dim_info.outer_packed_dim_block_size,
            expected_outer_packed_dim_block_size)
            << "outer_packed_dim_block_size mismatch for layout="
            << static_cast<int>(config.layout);
        ASSERT_EQ(packed_dim_info.block_numel, expected_block_numel)
            << "block_numel mismatch for layout="
            << static_cast<int>(config.layout);
        ASSERT_EQ(packed_dim_info.block_transposed, config.block_transposed)
            << "block_transposed mismatch for layout="
            << static_cast<int>(config.layout);

        // Verify dim_order
        std::vector<int64_t> ref_dim_order = get_reference_dim_order(
            sizes.size(),
            config.packed_dim,
            config.outer_packed_dim,
            config.block_transposed);
        ASSERT_TRUE(tensor.dim_order() == ref_dim_order)
            << "Dim order mismatch for layout="
            << static_cast<int>(config.layout);

        // Verify padded_sizes
        std::vector<int64_t> ref_padded_sizes = get_reference_padded_sizes(
            sizes,
            config.packed_dim,
            expected_packed_dim_block_size,
            config.outer_packed_dim,
            expected_outer_packed_dim_block_size);
        ASSERT_TRUE(tensor.padded_sizes() == ref_padded_sizes)
            << "Padded sizes mismatch for layout="
            << static_cast<int>(config.layout);

        if (storage_type == utils::kBuffer) {
          // For buffer tensors, verify strides (only for standard layouts)
          // For int8 layouts, we rely on padded_sizes and dim_order
          // verification
          if (config.dtype == vkapi::kFloat) {
            std::vector<int64_t> ref_strides =
                get_reference_strides(sizes, config.layout);
            ASSERT_TRUE(tensor.strides() == ref_strides)
                << "Strides mismatch for layout="
                << static_cast<int>(config.layout);

            // Also test flip_and_unsqueeze operations
            int64_t numel = utils::multiply_integers(sizes);
            std::vector<int64_t> unsqueezed_strides =
                flip_and_unsqueeze<int64_t>(
                    tensor.strides(), kTensorStrides, numel);
            std::vector<int64_t> ref_unsqueezed_strides =
                get_reference_strides(sizes, config.layout, true);
            ASSERT_TRUE(unsqueezed_strides == ref_unsqueezed_strides);
          }

          // Verify physical_numel for buffer storage
          int64_t ref_physical_numel =
              get_reference_physical_numel(expected_dtype, ref_padded_sizes);
          ASSERT_EQ(tensor.physical_numel(), ref_physical_numel)
              << "Physical numel mismatch for buffer storage with layout="
              << static_cast<int>(config.layout);
        } else {
          // For texture tensors, verify axis_map
          std::vector<int64_t> expected_axis_map = {0, 1, 2, 2};
          ASSERT_TRUE(tensor.axis_map() == expected_axis_map)
              << "Axis map mismatch for texture tensor with layout="
              << static_cast<int>(config.layout);
          ASSERT_TRUE(tensor.has_standard_axis_map());

          // Verify image_extents for texture storage
          utils::uvec3 ref_image_extents = get_reference_image_extents(
              expected_dtype,
              config.packed_dim,
              config.outer_packed_dim,
              config.is_block_packed,
              ref_padded_sizes,
              expected_axis_map);
          ASSERT_EQ(tensor.image_extents(), ref_image_extents)
              << "Image extents mismatch for texture storage with layout="
              << static_cast<int>(config.layout);
        }
      }
    }
  }
}

TEST_F(VulkanComputeAPITest, tensor_layout_metadata_test_against_golden) {
  // Test with hardcoded golden values for specific test cases.
  // This complements the reference implementation test by providing concrete
  // examples with known-good values.

  struct TestCase {
    std::vector<int64_t> sizes;
    vkapi::ScalarType dtype;
    utils::GPUMemoryLayout layout;
    // Expected values for both buffer and texture storage
    std::vector<int64_t> expected_dim_order;
    std::vector<int64_t> expected_padded_sizes_buffer;
    std::vector<int64_t> expected_padded_sizes_texture;
    std::vector<int64_t> expected_strides_buffer;
    int64_t expected_physical_numel_buffer;
    int64_t expected_physical_numel_texture;
    utils::uvec3 expected_image_extents;
  };

  std::vector<TestCase> test_cases = {
      // 1D tensor [7] with width packed, float dtype
      {/* sizes */ {7},
       /* dtype */ vkapi::kFloat,
       /* layout */ utils::kWidthPacked,
       /* expected_dim_order */ {0},
       /* expected_padded_sizes_buffer */ {1, 1, 1, 7},
       /* expected_padded_sizes_texture */ {1, 1, 1, 8},
       /* expected_strides_buffer */ {1},
       /* expected_physical_numel_buffer */ 7,
       /* expected_physical_numel_texture */ 8,
       /* expected_image_extents */ {2, 1, 1}},

      // 1D tensor [13] with width packed, float dtype
      {/* sizes */ {13},
       /* dtype */ vkapi::kFloat,
       /* layout */ utils::kWidthPacked,
       /* expected_dim_order */ {0},
       /* expected_padded_sizes_buffer */ {1, 1, 1, 13},
       /* expected_padded_sizes_texture */ {1, 1, 1, 16},
       /* expected_strides_buffer */ {1},
       /* expected_physical_numel_buffer */ 13,
       /* expected_physical_numel_texture */ 16,
       /* expected_image_extents */ {4, 1, 1}},

      // 1D tensor [7] with channels packed, float dtype
      // C dimension (implicit, size 1) is padded to 4
      {/* sizes */ {7},
       /* dtype */ vkapi::kFloat,
       /* layout */ utils::kChannelsPacked,
       /* expected_dim_order */ {0},
       /* expected_padded_sizes_buffer */ {1, 1, 1, 7},
       /* expected_padded_sizes_texture */ {1, 4, 1, 7},
       /* expected_strides_buffer */ {1},
       /* expected_physical_numel_buffer */ 7,
       /* expected_physical_numel_texture */ 28,
       /* expected_image_extents */ {7, 1, 1}},

      // 2D tensor [5, 7] with width packed, float dtype
      {/* sizes */ {5, 7},
       /* dtype */ vkapi::kFloat,
       /* layout */ utils::kWidthPacked,
       /* expected_dim_order */ {0, 1},
       /* expected_padded_sizes_buffer */ {1, 1, 5, 7},
       /* expected_padded_sizes_texture */ {1, 1, 5, 8},
       /* expected_strides_buffer */ {7, 1},
       /* expected_physical_numel_buffer */ 35,
       /* expected_physical_numel_texture */ 40,
       /* expected_image_extents */ {2, 5, 1}},

      // 3D tensor [3, 5, 7] with channels packed, float dtype
      {/* sizes */ {3, 5, 7},
       /* dtype */ vkapi::kFloat,
       /* layout */ utils::kChannelsPacked,
       /* expected_dim_order */ {1, 2, 0},
       /* expected_padded_sizes_buffer */ {1, 3, 5, 7},
       /* expected_padded_sizes_texture */ {1, 4, 5, 7},
       /* expected_strides_buffer */ {1, 7 * 3, 3},
       /* expected_physical_numel_buffer */ 105,
       /* expected_physical_numel_texture */ 140,
       /* expected_image_extents */ {7, 5, 1}},

      // 4D tensor [2, 3, 5, 7] with height packed, float dtype
      {/* sizes */ {2, 3, 5, 7},
       /* dtype */ vkapi::kFloat,
       /* layout */ utils::kHeightPacked,
       /* expected_dim_order */ {0, 1, 3, 2},
       /* expected_padded_sizes_buffer */ {2, 3, 5, 7},
       /* expected_padded_sizes_texture */ {2, 3, 8, 7},
       /* expected_strides_buffer */ {3 * 5 * 7, 5 * 7, 1, 5},
       /* expected_physical_numel_buffer */ 210,
       /* expected_physical_numel_texture */ 336,
       /* expected_image_extents */ {7, 2, 6}},

      // 3D tensor [8, 12, 16] with packed int8 4W layout
      {/* sizes */ {8, 12, 16},
       /* dtype */ vkapi::kChar,
       /* layout */ utils::kPackedInt8_4W,
       /* expected_dim_order */ {0, 1, 2},
       /* expected_padded_sizes_buffer */ {1, 8, 12, 16},
       /* expected_padded_sizes_texture */ {1, 8, 12, 16},
       /* expected_strides_buffer */ {},
       /* expected_physical_numel_buffer */ 384,
       /* expected_physical_numel_texture */ 384,
       /* expected_image_extents */ {1, 12, 8}},

      // 3D tensor [8, 12, 16] with packed int8 4W4C block layout
      {/* sizes */ {8, 12, 16},
       /* dtype */ vkapi::kChar,
       /* layout */ utils::kPackedInt8_4W4C,
       /* expected_dim_order */ {1, 2, 0},
       /* expected_padded_sizes_buffer */ {1, 8, 12, 16},
       /* expected_padded_sizes_texture */ {1, 8, 12, 16},
       /* expected_strides_buffer */ {},
       /* expected_physical_numel_buffer */ 384,
       /* expected_physical_numel_texture */ 384,
       /* expected_image_extents */ {4, 12, 2}},

      // 3D tensor [9, 13, 17] with packed int8 4C layout (odd sizes)
      // For texture, packed_dim (channels) is padded to multiple of 16
      {/* sizes */ {9, 13, 17},
       /* dtype */ vkapi::kChar,
       /* layout */ utils::kPackedInt8_4C,
       /* expected_dim_order */ {1, 2, 0},
       /* expected_padded_sizes_buffer */ {1, 12, 13, 17},
       /* expected_padded_sizes_texture */ {1, 16, 13, 17},
       /* expected_strides_buffer */ {},
       /* expected_physical_numel_buffer */ 663,
       /* expected_physical_numel_texture */ 884,
       /* expected_image_extents */ {17, 13, 1}},

      // 3D tensor [9, 13, 17] with packed int8 4H4W block layout (odd sizes)
      {/* sizes */ {9, 13, 17},
       /* dtype */ vkapi::kChar,
       /* layout */ utils::kPackedInt8_4H4W,
       /* expected_dim_order */ {0, 1, 2},
       /* expected_padded_sizes_buffer */ {1, 9, 16, 20},
       /* expected_padded_sizes_texture */ {1, 9, 16, 20},
       /* expected_strides_buffer */ {},
       /* expected_physical_numel_buffer */ 720,
       /* expected_physical_numel_texture */ 720,
       /* expected_image_extents */ {5, 4, 9}},

      // 3D tensor [9, 13, 17] with packed int8 4C1W block-transposed layout
      // packed_dim = channels (2), outer_packed_dim = width (0)
      // block_transposed = true, so dim_order swaps: [1, 0, 2] instead of
      // [1, 2, 0] Channels padded to 12 (multiple of 4), width padded to 20
      // (multiple of 4)
      {/* sizes */ {9, 13, 17},
       /* dtype */ vkapi::kChar,
       /* layout */ utils::kPackedInt8_4C1W,
       /* expected_dim_order */ {1, 0, 2},
       /* expected_padded_sizes_buffer */ {1, 12, 13, 17},
       /* expected_padded_sizes_texture */ {1, 16, 13, 17},
       /* expected_strides_buffer */ {},
       /* expected_physical_numel_buffer */ 663,
       /* expected_physical_numel_texture */ 884,
       /* expected_image_extents */ {17, 13, 1}},

      // 4D tensor [2, 8, 12, 16] with packed int8 4C1W block-transposed layout
      // Tests 4D case with block_transposed = true
      {/* sizes */ {2, 8, 12, 16},
       /* dtype */ vkapi::kChar,
       /* layout */ utils::kPackedInt8_4C1W,
       /* expected_dim_order */ {0, 2, 1, 3},
       /* expected_padded_sizes_buffer */ {2, 8, 12, 16},
       /* expected_padded_sizes_texture */ {2, 16, 12, 16},
       /* expected_strides_buffer */ {},
       /* expected_physical_numel_buffer */ 768,
       /* expected_physical_numel_texture */ 1536,
       /* expected_image_extents */ {16, 12, 2}},
  };

  for (size_t i = 0; i < test_cases.size(); ++i) {
    const auto& tc = test_cases[i];

    // Test with buffer storage
    {
      vTensor tensor_buffer(
          context(),
          tc.sizes,
          tc.dtype,
          utils::kBuffer,
          tc.layout,
          /*allocate_memory = */ false);

      // Verify dtype (kChar -> kInt8x4)
      vkapi::ScalarType expected_dtype = tc.dtype;
      if (tc.dtype == vkapi::kChar) {
        expected_dtype = vkapi::kInt8x4;
      }
      ASSERT_EQ(tensor_buffer.dtype(), expected_dtype)
          << "Test case " << i << ": Buffer dtype mismatch";

      // Verify dim_order
      ASSERT_TRUE(tensor_buffer.dim_order() == tc.expected_dim_order)
          << "Test case " << i << ": Buffer dim_order mismatch"
          << " (expected size: " << tc.expected_dim_order.size()
          << ", actual size: " << tensor_buffer.dim_order().size() << ")";

      // Verify padded_sizes
      ASSERT_TRUE(
          tensor_buffer.padded_sizes() == tc.expected_padded_sizes_buffer)
          << "Test case " << i << ": Buffer padded_sizes mismatch";

      // Verify strides (only for float dtype)
      if (tc.dtype == vkapi::kFloat && !tc.expected_strides_buffer.empty()) {
        ASSERT_TRUE(tensor_buffer.strides() == tc.expected_strides_buffer)
            << "Test case " << i << ": Buffer strides mismatch";
      }

      // Verify physical_numel
      ASSERT_EQ(
          tensor_buffer.physical_numel(), tc.expected_physical_numel_buffer)
          << "Test case " << i << ": Buffer physical_numel mismatch";
    }

    // Test with texture storage
    {
      vTensor tensor_texture(
          context(),
          tc.sizes,
          tc.dtype,
          utils::kTexture3D,
          tc.layout,
          /*allocate_memory = */ false);

      // Verify dtype (kChar -> kInt8x4)
      vkapi::ScalarType expected_dtype = tc.dtype;
      if (tc.dtype == vkapi::kChar) {
        expected_dtype = vkapi::kInt8x4;
      }
      ASSERT_EQ(tensor_texture.dtype(), expected_dtype)
          << "Test case " << i << ": Texture dtype mismatch";

      // Verify dim_order (texture doesn't use dim_order, but it's still
      // computed)
      ASSERT_TRUE(tensor_texture.dim_order() == tc.expected_dim_order)
          << "Test case " << i << ": Texture dim_order mismatch";

      // Verify padded_sizes
      ASSERT_TRUE(
          tensor_texture.padded_sizes() == tc.expected_padded_sizes_texture)
          << "Test case " << i << ": Texture padded_sizes mismatch";

      // Verify axis_map
      std::vector<int64_t> expected_axis_map = {0, 1, 2, 2};
      ASSERT_TRUE(tensor_texture.axis_map() == expected_axis_map)
          << "Test case " << i << ": Texture axis_map mismatch";

      // Verify physical_numel
      ASSERT_EQ(
          tensor_texture.physical_numel(), tc.expected_physical_numel_texture)
          << "Test case " << i << ": Texture physical_numel mismatch";

      // Verify image_extents
      ASSERT_EQ(tensor_texture.image_extents(), tc.expected_image_extents)
          << "Test case " << i << ": Texture image_extents mismatch"
          << " (expected: [" << tc.expected_image_extents[0] << ", "
          << tc.expected_image_extents[1] << ", "
          << tc.expected_image_extents[2] << "], got: ["
          << tensor_texture.image_extents()[0] << ", "
          << tensor_texture.image_extents()[1] << ", "
          << tensor_texture.image_extents()[2] << "])";
    }
  }
}

TEST_F(VulkanComputeAPITest, virtual_transpose_test) {
  std::vector<int64_t> sizes = {7, 9, 11, 13};
  // (dim0, dim1), new_sizes, new_dim_order, new_axis_map, new_packed_dim_idx
  std::vector<std::vector<std::vector<int64_t>>> test_cases = {
      {{2, 3}, {7, 9, 13, 11}, {0, 1, 3, 2}, {1, 0, 2, 2}, {1}},
      {{2, 1}, {7, 11, 9, 13}, {0, 2, 1, 3}, {0, 2, 1, 1}, {0}},
      {{1, 3}, {7, 13, 11, 9}, {0, 3, 2, 1}, {2, 1, 0, 0}, {2}},
  };

  for (const auto& test_case : test_cases) {
    const int dim0 = test_case.at(0).at(0);
    const int dim1 = test_case.at(0).at(1);

    const auto& expected_sizes = test_case.at(1);
    const auto& expected_dim_order = test_case.at(2);
    const auto& expected_axis_map = test_case.at(3);
    const int expected_packed_dim = test_case.at(4).at(0);

    {
      vTensor a_buffer = vTensor(
          context(), sizes, vkapi::kFloat, utils::kBuffer, utils::kWidthPacked);

      a_buffer.virtual_transpose(dim0, dim1);
      EXPECT_TRUE(a_buffer.sizes() == expected_sizes);
      EXPECT_TRUE(a_buffer.dim_order() == expected_dim_order);
    }

    {
      vTensor a_texture = vTensor(
          context(),
          sizes,
          vkapi::kFloat,
          utils::kTexture3D,
          utils::kWidthPacked);
      a_texture.virtual_transpose(dim0, dim1);
      EXPECT_TRUE(a_texture.sizes() == expected_sizes);
      EXPECT_TRUE(a_texture.axis_map() == expected_axis_map);
      EXPECT_TRUE(a_texture.packed_dim() == expected_packed_dim);
    }
  }
}

TEST_F(VulkanComputeAPITest, view_of_view_test) {
  constexpr int N = 3;
  constexpr int C = 5;
  constexpr int H = 17;
  constexpr int W = 19;

  std::vector<int64_t> sizes = {N, C, H, W};

  vTensor t1 = vTensor(
      context(), sizes, vkapi::kFloat, utils::kTexture3D, utils::kWidthPacked);

  vTensor t2 = vTensor(t1);
  EXPECT_TRUE(t2.sizes() == sizes);
  vTensor t3 = vTensor(t2);
  EXPECT_TRUE(t2.sizes() == sizes);

  t2.virtual_transpose(1, 2);
  std::vector<int64_t> expected_t2_sizes = {N, H, C, W};
  EXPECT_TRUE(t2.sizes() == expected_t2_sizes);

  // Because t3 was created before t2's metadata was updated, we need to first
  // update t3's metadata to match t2's metadata. Then the transpose will yield
  // the correct metadata.
  t3.virtual_clone(t2);
  t3.virtual_transpose(2, 3);
  std::vector<int64_t> expected_t3_sizes = {N, H, W, C};
  EXPECT_TRUE(t3.sizes() == expected_t3_sizes);
}

utils::ivec3 make_temp_ivec3(int x, int y, int z) {
  return utils::ivec3{x, y, z};
}

TEST_F(VulkanComputeAPITest, vec_test) {
  {
    utils::vec3 v3({1, 2, 3});
    ASSERT_TRUE(v3[0] == 1);
    ASSERT_TRUE(v3[1] == 2);
    ASSERT_TRUE(v3[2] == 3);
    v3 = {4, 5, 6};
    ASSERT_TRUE(v3[0] == 4);
    ASSERT_TRUE(v3[1] == 5);
    ASSERT_TRUE(v3[2] == 6);
  }

  {
    utils::uvec4 uv4({4, 3, 2, 1});
    ASSERT_TRUE(uv4[0] == 4);
    ASSERT_TRUE(uv4[1] == 3);
    ASSERT_TRUE(uv4[2] == 2);
    ASSERT_TRUE(uv4[3] == 1);
    uv4 = {11, 13, 12, 88};
    ASSERT_TRUE(uv4[0] == 11);
    ASSERT_TRUE(uv4[1] == 13);
    ASSERT_TRUE(uv4[2] == 12);
    ASSERT_TRUE(uv4[3] == 88);
  }

  // Test copy from same type
  {
    utils::ivec3 v{5, 6, 8};
    utils::ivec3 v2 = v;

    ASSERT_TRUE(v2[0] == 5);
    ASSERT_TRUE(v2[1] == 6);
    ASSERT_TRUE(v2[2] == 8);
  }

  // Test copy from different type
  {
    utils::uvec3 v{5, 6, 8};
    utils::ivec3 v2 = v;

    ASSERT_TRUE(v2[0] == 5);
    ASSERT_TRUE(v2[1] == 6);
    ASSERT_TRUE(v2[2] == 8);
  }

  // Test construction from temporary vec
  {
    utils::uvec3 v{make_temp_ivec3(4, 5, 10)};
    ASSERT_TRUE(v[0] == 4);
    ASSERT_TRUE(v[1] == 5);
    ASSERT_TRUE(v[2] == 10);
  }

  // Test initalization from temporary vec
  {
    utils::uvec3 v = make_temp_ivec3(4, 5, 10);
    ASSERT_TRUE(v[0] == 4);
    ASSERT_TRUE(v[1] == 5);
    ASSERT_TRUE(v[2] == 10);
  }
}

TEST_F(VulkanComputeAPITest, retrieve_custom_shader_test) {
  // Try to get shader from custom shader library
  const vkapi::ShaderInfo& kernel = VK_KERNEL(test_shader);

  ASSERT_TRUE(kernel.kernel_name == "test_shader");
}

TEST_F(VulkanComputeAPITest, spec_var_classes_test) {
  // Check equality operator
  ASSERT_TRUE(SV(1.5f) == SV(1.5f));
  ASSERT_FALSE(SV(15.0f) == SV(15));
  ASSERT_FALSE(SV(1u) == SV(true));

  size_t sv_size = sizeof(vkapi::SpecVar);

  vkapi::SpecVarList spec_vars = {};
  ASSERT_TRUE(spec_vars.size() == 0);
  spec_vars = {SV(1.1f), SV(32), SV(45)};
  ASSERT_TRUE(spec_vars.size() == 3);
  vkapi::SpecVarList spec_vars_other = {SV(2.6f), SV(true), SV(78u), SV(5.5f)};
  spec_vars.append(spec_vars_other);
  ASSERT_TRUE(spec_vars.size() == 7);

  // Check validity of the data
  const vkapi::SpecVar* data = spec_vars.data();
  ASSERT_TRUE(*(reinterpret_cast<const float*>(data + 3)) == 2.6f);
  ASSERT_TRUE(*(reinterpret_cast<const int32_t*>(data + 1)) == 32);
  ASSERT_TRUE(*(reinterpret_cast<const int32_t*>(data + 5)) == 78u);

  // Check validity of the map entries
  std::vector<VkSpecializationMapEntry> entries =
      spec_vars.generate_map_entries();

  for (size_t i = 0; i < spec_vars.size(); ++i) {
    ASSERT_TRUE(entries[i].constantID == i);
    ASSERT_TRUE(entries[i].offset == sv_size * i);
    if (i != 4) {
      ASSERT_TRUE(entries[i].size == 4);
    } else {
      ASSERT_TRUE(entries[i].size == 1);
    }
  }

  // Check copy
  vkapi::SpecVarList spec_vars_copy(spec_vars);
  ASSERT_TRUE(spec_vars_copy.size() == 7);

  // Check validity of the copied data
  const vkapi::SpecVar* copy_data = spec_vars_copy.data();
  ASSERT_TRUE(*(reinterpret_cast<const bool*>(copy_data + 4)) == true);
  ASSERT_TRUE(*(reinterpret_cast<const int32_t*>(copy_data + 2)) == 45);
  ASSERT_TRUE(*(reinterpret_cast<const float*>(copy_data + 6)) == 5.5f);
}

TEST_F(VulkanComputeAPITest, spec_var_shader_test) {
  size_t len = 16;
  StagingBuffer buffer(
      context(), vkapi::kFloat, len, vkapi::CopyDirection::DEVICE_TO_HOST);

  float scale = 3.0f;
  float offset = 1.5f;

  {
    ParamsBuffer params(context(), int32_t(len));
    uint32_t len_div4 = utils::div_up(uint32_t(len), uint32_t(4));
    vkapi::PipelineBarrier pipeline_barrier{};
    context()->submit_compute_job(
        VK_KERNEL(fill_buffer),
        pipeline_barrier,
        {64, 1, 1},
        {len_div4, 1, 1},
        {SV(scale), SV(offset)},
        VK_NULL_HANDLE,
        0,
        buffer.buffer(),
        params.buffer());
  }

  submit_to_gpu();

  std::vector<float> data(len);
  buffer.copy_to(data.data(), buffer.nbytes());

  for (size_t i = 0; i < len; ++i) {
    CHECK_VALUE(data, i, scale * i + offset);
  }
}

TEST_F(VulkanComputeAPITest, update_params_between_submit) {
  context()->set_cmd(/*reusable = */ true);
  std::vector<int64_t> sizes = {4, 4, 2};
  vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);

  std::string kernel_name("fill_texture__test");
  add_dtype_suffix(kernel_name, a.dtype());

  struct Params final {
    utils::ivec3 size;
    int32_t fill;
    utils::vec4 values;
  };

  Params block{
      {2, 4, 1},
      0,
      {5.0, 5.0, 5.0, 5.0},
  };

  ParamsBuffer params(context(), block);

  {
    vkapi::PipelineBarrier pipeline_barrier{};
    vkapi::SpecVarList specialization_constants = {};
    context()->submit_compute_job(
        VK_KERNEL_FROM_STR(kernel_name),
        pipeline_barrier,
        {4, 4, 4},
        {4, 4, 4},
        specialization_constants,
        VK_NULL_HANDLE,
        0,
        a.image(
            pipeline_barrier,
            vkapi::PipelineStage::COMPUTE,
            vkapi::MemoryAccessType::WRITE),
        params.buffer());
  }

  StagingBuffer staging_buffer(
      context(),
      vkapi::kFloat,
      a.staging_buffer_numel(),
      vkapi::CopyDirection::DEVICE_TO_HOST);
  record_image_to_nchw_op(context(), a, staging_buffer.buffer());

  submit_to_gpu();
  check_staging_buffer(staging_buffer, 5.0f);

  Params new_block{
      {2, 4, 1},
      0,
      {4.0, 4.0, 4.0, 4.0},
  };

  params.update(new_block);

  submit_to_gpu();
  check_staging_buffer(staging_buffer, 4.0f);
}

template <typename T, vkapi::ScalarType dtype>
void test_storage_buffer_type(const size_t len) {
  StagingBuffer buffer(
      context(), dtype, len, vkapi::CopyDirection::DEVICE_TO_HOST);

  std::string kernel_name("idx_fill_buffer");
  switch (buffer.dtype()) {
    case vkapi::kFloat:
      kernel_name += "_float";
      break;
    case vkapi::kHalf:
      kernel_name += "_half";
      break;
    case vkapi::kQInt8:
      kernel_name += "_int8";
      break;
    case vkapi::kQUInt8:
      kernel_name += "_uint8";
      break;
    default:
      throw std::runtime_error("Unsupported dtype");
      break;
  }

  ParamsBuffer params(context(), int32_t(len));

  {
    uint32_t len_div4 = utils::div_up(uint32_t(len), uint32_t(4));
    vkapi::PipelineBarrier pipeline_barrier{};
    vkapi::SpecVarList specialization_constants = {};
    context()->submit_compute_job(
        VK_KERNEL_FROM_STR(kernel_name),
        pipeline_barrier,
        {64, 1, 1},
        {len_div4, 1, 1},
        specialization_constants,
        VK_NULL_HANDLE,
        0,
        buffer.buffer(),
        params.buffer());
  }

  submit_to_gpu();

  std::vector<T> data(len);

  if (dtype == vkapi::kHalf && buffer.dtype() == vkapi::kFloat) {
    buffer.cast_float_to_half_and_copy_to(
        reinterpret_cast<uint16_t*>(data.data()), data.size());
  } else {
    VK_CHECK_COND(dtype == buffer.dtype());
    buffer.copy_to(data.data(), buffer.nbytes());
  }

  for (size_t i = 0; i < len; ++i) {
    CHECK_VALUE(data, i, T(i));
  }
}

TEST_F(VulkanComputeAPITest, test_buffer_float) {
  test_storage_buffer_type<float, vkapi::kFloat>(16);
}

TEST_F(VulkanComputeAPITest, test_buffer_float16) {
  test_storage_buffer_type<executorch::aten::Half, vkapi::kHalf>(16);
}

TEST_F(VulkanComputeAPITest, test_buffer_int8) {
  if (!context()->adapter_ptr()->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  test_storage_buffer_type<int8_t, vkapi::kQInt8>(16);
}

TEST_F(VulkanComputeAPITest, test_zero_size_tensor) {
  // Simple test that performs a + b -> c

  std::vector<int64_t> sizes = {0, 5, 7};
  vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);
  vTensor b = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);
  vTensor c = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);

  // Fill input tensors
  fill_vtensor(a, 2.5f);
  fill_vtensor(b, 1.5f);

  // a + b -> c
  record_binary_op(context(), "add", a, b, c);

  // Extract output tensor
  std::vector<float> data_out = extract_vtensor(c);

  // Assert all tensors are empty
  ASSERT_TRUE(a.numel() == 0);
  ASSERT_TRUE(b.numel() == 0);
  ASSERT_TRUE(c.numel() == 0);
  ASSERT_TRUE(a.nbytes() == 0);
  ASSERT_TRUE(b.nbytes() == 0);
  ASSERT_TRUE(c.nbytes() == 0);

  // Check output
  for (size_t i = 0; i < data_out.size(); ++i) {
    CHECK_VALUE(data_out, i, 4.0f);
  }
}

template <typename T>
void run_buffer_tensor_sanity_check(vTensor& tensor) {
  fill_vtensor(tensor, 0.0f, true);

  record_scalar_add_buffer(context(), tensor, 2.0f);
  std::vector<float> data_out = extract_vtensor(tensor);

  // Check output
  for (size_t i = 0; i < tensor.numel(); ++i) {
    CHECK_VALUE(data_out, i, i + 2.0f);
  }
}

TEST_F(VulkanComputeAPITest, buffer_tensor_sanity_check) {
  for (const auto& sizes : standard_sizes_to_test) {
    for (const auto& dtype : {vkapi::kFloat, vkapi::kHalf, vkapi::kChar}) {
      if (dtype == vkapi::kHalf &&
          !context()->adapter_ptr()->has_full_float16_buffers_support()) {
        continue;
      }
      if (dtype == vkapi::kHalf && utils::multiply_integers(sizes) >= 2048) {
        continue;
      }
      if (dtype == vkapi::kChar &&
          !context()->adapter_ptr()->has_full_int8_buffers_support()) {
        continue;
      }
      if (dtype == vkapi::kChar && utils::multiply_integers(sizes) >= 128) {
        continue;
      }
      for (const auto& layout :
           {utils::kWidthPacked,
            utils::kHeightPacked,
            utils::kChannelsPacked}) {
        vTensor a = vTensor(context(), sizes, dtype, utils::kBuffer, layout);
        switch (dtype) {
          case vkapi::kFloat:
            run_buffer_tensor_sanity_check<float>(a);
            break;
          case vkapi::kHalf:
            run_buffer_tensor_sanity_check<executorch::aten::Half>(a);
            break;
          case vkapi::kChar:
            run_buffer_tensor_sanity_check<int8_t>(a);
            break;
          default:
            VK_THROW("Unsupported dtype");
        }
      }
    }
  }
}

TEST_F(VulkanComputeAPITest, texture_add_sanity_check) {
  // Simple test that performs a + b -> c

  std::vector<int64_t> sizes = {4, 4, 1};
  vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);
  vTensor b = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);
  vTensor c = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);

  // Fill input tensors
  fill_vtensor(a, 2.5f);
  fill_vtensor(b, 1.5f);

  // a + b -> c
  record_binary_op(context(), "add", a, b, c);

  // Extract output tensor
  std::vector<float> data_out = extract_vtensor(c);

  // Check output
  for (size_t i = 0; i < data_out.size(); ++i) {
    CHECK_VALUE(data_out, i, 4.0f);
  }
}

TEST_F(VulkanComputeAPITest, tensor_alias_test) {
  for (utils::StorageType storage_type : {utils::kTexture3D, utils::kBuffer}) {
    std::vector<int64_t> sizes = {9, 9};

    const size_t alloc_count_before = get_vma_allocation_count();

    vTensor original = vTensor(context(), sizes, vkapi::kFloat, storage_type);

    vTensor copy = vTensor(original);

    // Two tensors but only one additional allocation.
    EXPECT_TRUE(get_vma_allocation_count() == alloc_count_before + 1);
    EXPECT_TRUE(copy.is_view_of(original));

    // Fill original tensor with some data
    fill_vtensor(original, 2.5f, true);

    std::vector<float> data_out(copy.staging_buffer_numel());
    // Extract the copy tensor; should contain the data of the original tensor
    extract_vtensor(copy, data_out);

    for (size_t i = 0; i < original.numel(); ++i) {
      CHECK_VALUE(data_out, i, 2.5f + i);
    }
  }
}

TEST_F(VulkanComputeAPITest, tensor_no_copy_transpose_test) {
  constexpr int M = 11;
  constexpr int K = 23;
  constexpr int N = 17;
  std::vector<int64_t> mat1_sizes = {M, K};
  std::vector<int64_t> mat2_sizes = {N, K};
  std::vector<int64_t> out_sizes = {M, N};

  for (const auto storage_type : {utils::kBuffer}) {
    vTensor mat1 = vTensor(
        context(),
        mat1_sizes,
        vkapi::kFloat,
        storage_type,
        utils::kWidthPacked);
    vTensor mat2 = vTensor(
        context(),
        mat2_sizes,
        vkapi::kFloat,
        storage_type,
        utils::kWidthPacked);
    vTensor out = vTensor(
        context(), out_sizes, vkapi::kFloat, storage_type, utils::kWidthPacked);

    // Generate data
    std::vector<float> mat1_data =
        create_random_float_buffer(mat1.staging_buffer_numel());
    std::vector<float> mat2_data =
        create_random_float_buffer(mat2.staging_buffer_numel());

    // Create direct view and modify sizes and strides later
    vTensor mat2_t = vTensor(mat2);
    // Update sizes and strides of mat2_t to be that of a transposed tensor
    mat2_t.virtual_transpose(0, 1);

    EXPECT_TRUE(mat2_t.packed_dim() == WHCN::kHeightDim);

    std::vector<float> mat2_t_data = transpose_matrix(mat2_data, N, K);
    std::vector<float> ref_out =
        compute_reference_matmul(mat1_data, mat2_t_data, M, K, N);

    // Fill original tensor with some data
    fill_vtensor(mat1, mat1_data);
    fill_vtensor(mat2, mat2_data);

    if (storage_type == utils::kTexture3D) {
      record_matmul_texture3d(
          context(), out, mat1, mat2_t, /*mat2_is_transposed=*/true);
    } else {
      record_reference_matmul(context(), out, mat1, mat2_t);
    }

    std::vector<float> data_out(out.staging_buffer_numel());
    // Extract the copy tensor; should contain the data of the original tensor
    extract_vtensor(out, data_out);

    for (size_t i = 0; i < ref_out.size(); ++i) {
      EXPECT_TRUE(check_close(data_out[i], ref_out[i]));
    }
  }
}

TEST_F(VulkanComputeAPITest, texture_deferred_allocation_test) {
  // This test is the same as texture_add_sanity_check, except that the tensor
  // memory is allocated in a deferred fashion

  std::vector<int64_t> sizes = {4, 4, 1};
  vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);
  vTensor b = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);
  vTensor c = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);

  // No allocations made so far
  EXPECT_TRUE(get_vma_allocation_count() == 0);

  std::vector<float> data_a(a.staging_buffer_numel());
  std::fill(data_a.begin(), data_a.end(), 2.5f);
  std::vector<float> data_b(b.staging_buffer_numel());
  std::fill(data_b.begin(), data_b.end(), 1.5f);

  // Allocate memory at the last possible opportunity
  vkapi::Allocation a_mem = allocate_memory_for(a);
  a.image().bind_allocation(a_mem);
  vkapi::Allocation b_mem = allocate_memory_for(b);
  b.image().bind_allocation(b_mem);
  vkapi::Allocation c_mem = allocate_memory_for(c);
  c.image().bind_allocation(c_mem);

  // One allocation for each tensor
  EXPECT_TRUE(get_vma_allocation_count() == 3);

  fill_vtensor(a, data_a);
  fill_vtensor(b, data_b);

  record_binary_op(context(), "add", a, b, c);

  std::vector<float> data_c(c.staging_buffer_numel());
  extract_vtensor(c, data_c);

  for (size_t i = 0; i < data_c.size(); ++i) {
    CHECK_VALUE(data_c, i, 4.0f);
  }
}

TEST_F(VulkanComputeAPITest, texture_resource_aliasing_test) {
  // This test performs the following operations:
  // 1. a + b -> c
  // 2. c + d -> e
  // and share memory between tensors whenever possible.

  std::vector<int64_t> sizes = {4, 4, 1};
  vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);
  vTensor b = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);
  vTensor c = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);
  vTensor d = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);
  vTensor e = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);

  // No allocations made so far
  EXPECT_TRUE(get_vma_allocation_count() == 0);

  // a and d can share the same memory allocation
  vkapi::Allocation a_d_mem = allocate_memory_for(a);
  a.image().bind_allocation(a_d_mem);
  d.image().bind_allocation(a_d_mem);
  // b and e can share the same memory allocation
  vkapi::Allocation b_e_mem = allocate_memory_for(b);
  b.image().bind_allocation(b_e_mem);
  e.image().bind_allocation(b_e_mem);
  // c must have its own memory allocation
  vkapi::Allocation c_mem = allocate_memory_for(c);
  c.image().bind_allocation(c_mem);

  // 3 allocations should be made
  EXPECT_TRUE(get_vma_allocation_count() == 3);

  // Specify input data
  std::vector<float> data_a(a.staging_buffer_numel());
  std::fill(data_a.begin(), data_a.end(), 2.5f);
  std::vector<float> data_b(b.staging_buffer_numel());
  std::fill(data_b.begin(), data_b.end(), 1.5f);
  std::vector<float> data_d(b.staging_buffer_numel());
  std::fill(data_d.begin(), data_d.end(), 1.0f);

  // First, fill a and b with data
  fill_vtensor(a, data_a);
  fill_vtensor(b, data_b);

  // a + b -> c
  record_binary_op(context(), "add", a, b, c);

  // Now d can be filled with data
  fill_vtensor(d, data_d);

  // c + d -> e
  record_binary_op(context(), "add", c, d, e);

  // Extract data from e
  std::vector<float> data_e(e.staging_buffer_numel());
  extract_vtensor(e, data_e);

  // Sanity check that the values are correct
  for (size_t i = 0; i < data_e.size(); ++i) {
    CHECK_VALUE(data_e, i, 5.0f);
  }
}

TEST_F(VulkanComputeAPITest, resource_bind_twice_fails) {
  // Check that binding a resource that already has memory associated with it
  // fails

  std::vector<int64_t> sizes = {4, 4, 1};
  vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);

  // Try to double bind a resource, which should fail
  vkapi::Allocation a_mem = allocate_memory_for(a);
  EXPECT_THROW(a.image().bind_allocation(a_mem), vkapi::Error);
}

TEST_F(VulkanComputeAPITest, resource_destructor_non_owning_memory) {
  // Check that the destructor of a vTensor that does not own its memory
  // does not free the memory

  vkapi::Allocation memory;

  // Default Allocation constructor should not allocate memory
  EXPECT_TRUE(get_vma_allocation_count() == 0);

  std::vector<int64_t> sizes = {4, 4, 1};
  {
    vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);

    memory = allocate_memory_for(a);
    EXPECT_TRUE(get_vma_allocation_count() == 1);
    a.image().bind_allocation(memory);
  }

  // Check that the memory is still allocated
  EXPECT_TRUE(get_vma_allocation_count() == 1);
}

TEST_F(VulkanComputeAPITest, use_non_bound_textures_fails) {
  // Try to encode a command buffer with a vTensor that does not have
  // memory

  std::vector<int64_t> sizes = {4, 4, 1};
  vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);

  // No allocations yet
  EXPECT_TRUE(get_vma_allocation_count() == 0);

  std::vector<float> data_a(a.staging_buffer_numel());
  std::fill(data_a.begin(), data_a.end(), 2.5f);

  // Encoding a command buffer with a vTensor without memory should throw
  EXPECT_THROW(fill_vtensor(a, data_a), vkapi::Error);
}

TEST_F(VulkanComputeAPITest, texture_virtual_resize) {
  context()->set_cmd(/*reusable = */ true);
  std::vector<int64_t> sizes = {8, 12, 12};
  vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);
  vTensor b = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);
  vTensor c = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);

  DEFINE_STAGING_BUFFER_AND_RECORD_TO_GPU_FOR(a)
  DEFINE_STAGING_BUFFER_AND_RECORD_TO_GPU_FOR(b)

  fill_staging(staging_buffer_a, 11.5f);
  fill_staging(staging_buffer_b, 12.5f);

  record_binary_op(context(), "add", a, b, c);

  DEFINE_STAGING_BUFFER_AND_RECORD_FROM_GPU_FOR(c)

  submit_to_gpu();
  check_staging_buffer(staging_buffer_c, 24.0f);

  std::vector<std::vector<int64_t>> new_sizes_list = {
      {4, 2, 4}, {4, 3, 6}, {8, 12, 12}, {8, 1, 1}, {8, 11, 10}};

  for (auto& new_sizes : new_sizes_list) {
    a.virtual_resize(new_sizes);
    b.virtual_resize(new_sizes);
    c.virtual_resize(new_sizes);

    fill_staging(staging_buffer_a, float(new_sizes[1] + 1.5f), a.numel());
    fill_staging(staging_buffer_b, float(new_sizes[2] + 55.0f), b.numel());

    submit_to_gpu();
    check_staging_buffer(
        staging_buffer_c,
        float(new_sizes[1] + new_sizes[2] + 56.5f),
        c.numel());
  }
}

//
// Compute Graph Tests
//

#define EXTRACT_TENSOR(name)                                                 \
  std::vector<float> data_##name(graph.staging_buffer_numel_of(name.value)); \
  graph.maybe_cast_and_copy_from_staging(                                    \
      name.staging, data_##name.data(), data_##name.size(), vkapi::kFloat);

// The purpose of this test is simply to track the size of various classes over
// time, in the interest of making sure that they doesn't grow too large.
TEST_F(VulkanComputeAPITest, print_object_sizes) {
#define PRINT_SIZE(name) \
  std::cout << #name << " size: " << sizeof(name) << " B" << std::endl
  PRINT_SIZE(vTensor);
  PRINT_SIZE(Value);
  PRINT_SIZE(StagingBuffer);
  PRINT_SIZE(ComputeGraph);
  PRINT_SIZE(DispatchNode);
#undef PRINT_SIZE

  // The actual sizes of each object is dependent on the platform. However, we
  // can alert ourselves to any significant changes in the sizes of these
  // objects by checking the `sizeof()` the class against some loose thresholds.

  // Current known size on 64 bit system: 1040 B
  EXPECT_TRUE(sizeof(vTensor) < 1200);
  // Current known size on 64 bit system: 80 B
  EXPECT_TRUE(sizeof(Value) < 100);
  // Current known size on 64 bit system: 120 B
  EXPECT_TRUE(sizeof(StagingBuffer) < 500);
  // Current known size on 64 bit system: 608 B
  EXPECT_TRUE(sizeof(ComputeGraph) < 700);
  // Current known size on 64 bit system: 248 B
  EXPECT_TRUE(sizeof(DispatchNode) < 500);
}

TEST_F(VulkanComputeAPITest, test_tensor_creation_from_vulkan_image) {
  const auto w = 16;
  const auto h = 12;
  const auto d = 1;
  const utils::uvec3 image_extents = {w, h, d};

  vkapi::Adapter* adapter_ptr = context()->adapter_ptr();

  vkapi::ImageSampler::Properties sampler_props{
      VK_FILTER_NEAREST,
      VK_SAMPLER_MIPMAP_MODE_NEAREST,
      VK_SAMPLER_ADDRESS_MODE_REPEAT,
      VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
  };

  VkFormat image_format = VK_FORMAT_R32G32B32A32_SFLOAT;
  VkImageType image_type = VK_IMAGE_TYPE_3D;
  VkImageViewType image_view_type = VK_IMAGE_VIEW_TYPE_3D;

  VkSampler sampler = adapter_ptr->sampler_cache().retrieve(sampler_props);

  auto image = adapter_ptr->vma().create_image(
      context()->device(),
      vkapi::create_extent3d(image_extents),
      image_format,
      image_type,
      context()->preferred_image_tiling(),
      image_view_type,
      sampler_props,
      sampler,
      /*allow_transfer = */ true,
      /*allocate_memory = */ true);

  auto tensor = vTensor(context(), image);

  const auto exp_sizes = std::vector<int64_t>{w, h, d * 4};
  EXPECT_TRUE(tensor.sizes() == exp_sizes);
  EXPECT_TRUE(tensor.packed_dim() == 2);

  const auto exp_numel = w * h * d * 4;
  EXPECT_TRUE(tensor.numel() == exp_numel);
}

TEST(VulkanComputeGraphTest, test_values_scalars) {
  GraphConfig config;
  ComputeGraph graph(config);

  ValueRef idx;

  idx = graph.add_scalar<int64_t>(4);
  EXPECT_TRUE(graph.get_int(idx) == 4);

  idx = graph.add_scalar<double>(5.5f);
  EXPECT_TRUE(graph.get_double(idx) == 5.5f);
}

TEST(VulkanComputeGraphTest, test_values_scalar_list_inplace_constructed) {
  GraphConfig config;
  ComputeGraph graph(config);

  ValueRef idx = graph.add_scalar_list<int64_t>({1, 2, 3, 4});
  const auto arr = graph.get_int_list(idx);
  EXPECT_TRUE(arr->size() == 4);
  for (int i = 0; i < 4; i++) {
    EXPECT_TRUE(arr->at(i) == i + 1);
  }
}

TEST(VulkanComputeGraphTest, test_values_scalar_list_outside_constructed) {
  GraphConfig config;
  ComputeGraph graph(config);

  ValueRef idx;
  {
    std::vector<double> data = {5.0, 4.0, 3.0, 2.0, 1.0};
    idx = graph.add_scalar_list(std::move(data));
  }
  const auto& arr = graph.get_double_list(idx);
  EXPECT_TRUE(arr->size() == 5);
  for (int i = 0; i < 5; i++) {
    EXPECT_TRUE(arr->at(i) == (5 - i));
  }
}

TEST(VulkanComputeGraphTest, test_values_string) {
  GraphConfig config;
  ComputeGraph graph(config);

  ValueRef idx;
  {
    std::string data = "hello, world";
    idx = graph.add_string(std::move(data));
  }
  std::string stored = graph.get_string(idx);
  EXPECT_TRUE(stored == "hello, world");
}

TEST(VulkanComputeGraphTest, empty_init_graphnode_test) {
  ExecuteNode node(nullptr, {});

  GraphConfig config;
  ComputeGraph graph(config);

  // Encode an empty ExecuteNode and check that command buffer encoding does not
  // crash.
  graph.execute_nodes().emplace_back(new ExecuteNode(nullptr, {}));
}

TEST(VulkanComputeGraphTest, test_zero_dim_tensor) {
  GraphConfig config;
  ComputeGraph graph(config);

  std::vector<int64_t> size_big = {7, 3, 5};
  std::vector<int64_t> size_small = {};

  // Build graph

  IOValueRef a = graph.add_input_tensor(size_big, vkapi::kFloat);
  IOValueRef b = graph.add_input_tensor(size_small, vkapi::kFloat);

  IOValueRef out = {};

  out.value = graph.add_tensor(size_big, vkapi::kFloat);

  auto addFn = VK_GET_OP_FN("aten.add.Tensor");
  addFn(graph, {a.value, b.value, kDummyValueRef, out.value});

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();
  graph.prepack();

  // Run graph

  for (float i = 5.0f; i < 30.0f; i += 10.0f) {
    float val_a = i + 2.0f;
    float val_b = i + 1.5f;
    float val_c = val_a + val_b;

    fill_vtensor(graph, a, val_a);
    fill_vtensor(graph, b, val_b);

    graph.execute();

    EXTRACT_TENSOR(out);

    // Sanity check that the values are correct
    for (size_t i = 0; i < graph.numel_of(out.value); ++i) {
      CHECK_VALUE(data_out, i, val_c);
    }
  }
}

TEST(VulkanComputeGraphTest, test_simple_graph_with_buffer) {
  GraphConfig config;
  ComputeGraph graph(config);

  std::vector<int64_t> sizes = {7, 13, 19};

  // Build graph

  IOValueRef a = graph.add_input_tensor(sizes, vkapi::kFloat, utils::kBuffer);

  IOValueRef out = {};

  out.value = graph.add_tensor(sizes, vkapi::kFloat, utils::kBuffer);

  auto addFn = VK_GET_OP_FN("aten.abs.default");
  addFn(graph, {a.value, out.value, kDummyValueRef, kDummyValueRef});

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();
  graph.prepack();

  // Run graph

  for (float i = 5.0f; i < 30.0f; i += 10.0f) {
    float val = -i + 2.0f;
    float expected_val = std::abs(val);

    fill_vtensor(graph, a, val);

    graph.execute();

    EXTRACT_TENSOR(out);

    // Sanity check that the values are correct
    for (size_t i = 0; i < graph.numel_of(out.value); ++i) {
      CHECK_VALUE(data_out, i, expected_val);
    }
  }
}

TEST(VulkanComputeGraphTest, test_graph_view_of_view) {
  GraphConfig config;
  config.set_storage_type_override(utils::kTexture3D);
  ComputeGraph graph(config);

  constexpr int N = 3;
  constexpr int C = 5;
  constexpr int H = 17;
  constexpr int W = 19;

  std::vector<int64_t> orig_sizes = {N, C, H, W};

  // Test a common view of view usage pattern. In delegate execution, the values
  // of the graph are created first; then operators are added. As a result,
  // creating views of views is a bit tricky because metadata updates to a view
  // does not update the metadata of the view's views. Nonetheless, view
  // operators have an implicit assumption that the metadata of the output is
  // equivalent to the metadata of the input. Therefore, view operators must
  // account for unseen updates to the input view by first calling
  // `virtual_clone()` to make the output equivalent to the input before.
  // modifying metadata.

  ValueRef t1 = graph.add_tensor(orig_sizes, vkapi::kFloat);
  ValueRef t2 = graph.add_tensor_view(t1);
  ValueRef t3 = graph.add_tensor_view(t2);

  ValueRef channels = graph.add_scalar<int64_t>(1);
  ValueRef height = graph.add_scalar<int64_t>(2);
  ValueRef width = graph.add_scalar<int64_t>(3);

  auto opFn = VK_GET_OP_FN("aten.transpose.int");

  opFn(graph, {t1, channels, height, t2});
  std::vector<int64_t> t2_sizes = graph.sizes_of(t2);
  std::vector<int64_t> expected_t2_sizes = {N, H, C, W};
  EXPECT_TRUE(t2_sizes == expected_t2_sizes);

  opFn(graph, {t2, height, width, t3});
  std::vector<int64_t> t3_sizes = graph.sizes_of(t3);
  std::vector<int64_t> expected_t3_sizes = {N, H, W, C};
  EXPECT_TRUE(t3_sizes == expected_t3_sizes);
}

TEST(VulkanComputeGraphTest, test_simple_graph) {
  GraphConfig config;
  ComputeGraph graph(config);

  std::vector<int64_t> size_big = {1, 8, 8};
  std::vector<int64_t> size_small = {1, 1, 8};

  // Build graph

  IOValueRef a = graph.add_input_tensor(size_big, vkapi::kFloat);
  IOValueRef b = graph.add_input_tensor(size_small, vkapi::kFloat);

  IOValueRef out = {};

  out.value = graph.add_tensor(size_big, vkapi::kFloat);

  auto addFn = VK_GET_OP_FN("aten.add.Tensor");
  addFn(graph, {a.value, b.value, kDummyValueRef, out.value});

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();
  graph.prepack();

  // Run graph

  for (float i = 5.0f; i < 30.0f; i += 10.0f) {
    float val_a = i + 2.0f;
    float val_b = i + 1.5f;
    float val_c = val_a + val_b;

    fill_vtensor(graph, a, val_a);
    fill_vtensor(graph, b, val_b);

    graph.execute();

    EXTRACT_TENSOR(out);

    // Sanity check that the values are correct
    for (size_t i = 0; i < graph.numel_of(out.value); ++i) {
      CHECK_VALUE(data_out, i, val_c);
    }
  }
}

TEST(VulkanComputeGraphTest, test_simple_graph_with_symint) {
  GraphConfig config;
  config.set_storage_type_override(utils::kTexture3D);
  ComputeGraph graph(config);

  std::vector<int64_t> sizes = {8, 64, 124};

  // Build graph

  ValueRef scalar = graph.add_symint(1);
  IOValueRef a = graph.add_input_tensor(sizes, vkapi::kFloat);

  IOValueRef out = {};
  out.value = a.value;

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR("scalar_add_texture"),
      graph.create_global_wg_size(a.value),
      graph.create_local_wg_size(a.value),
      // Inputs and Outputs
      {{out.value, vkapi::MemoryAccessType::WRITE}},
      // Shader params buffers
      {graph.logical_limits_ubo(a.value),
       graph.get_or_create_int_param_buffer(scalar)},
      // Push constants
      {},
      // Specialization Constants
      {},
      // Resizing Logic
      {},
      nullptr));

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();
  graph.prepack();

  // Run graph

  for (float i = 5.0f; i < 30.0f; i += 10.0f) {
    int scalar_val = i - 3.0f;
    graph.set_symint(scalar, scalar_val);

    int32_t scalar_val_read = graph.read_symint(scalar);
    EXPECT_TRUE(scalar_val_read == scalar_val);

    float val_a = i + 2.0f;
    float val_out = val_a + scalar_val;

    fill_vtensor(graph, a, val_a);

    graph.execute();

    EXTRACT_TENSOR(out);

    // Sanity check that the values are correct
    for (size_t i = 0; i < graph.numel_of(out.value); i++) {
      CHECK_VALUE(data_out, i, val_out);
    }
  }
}

#define CREATE_WEIGHT_TENSOR(name, sizes, dtype, val)              \
  std::vector<float> data_##name(utils::multiply_integers(sizes)); \
  std::fill(data_##name.begin(), data_##name.end(), val);          \
  ValueRef name = graph.add_tensorref(sizes, dtype, data_##name.data());

TEST(VulkanComputeGraphTest, test_simple_prepacked_graph) {
  GraphConfig config;
  config.enable_querypool = true;
  ComputeGraph graph(config);

  std::vector<int64_t> size_big = {8, 73, 62};
  std::vector<int64_t> size_small = {8, 73, 1};

  CREATE_WEIGHT_TENSOR(w1, size_small, vkapi::kFloat, 3.5f);
  CREATE_WEIGHT_TENSOR(w2, size_small, vkapi::kFloat, 3.0f);

  // Build graph

  IOValueRef a = graph.add_input_tensor(size_big, vkapi::kFloat);

  ValueRef c = graph.add_tensor(size_big, vkapi::kFloat);
  ValueRef e = graph.add_tensor(size_big, vkapi::kFloat);

  ValueRef w1_packed = graph.add_tensor(size_small, vkapi::kFloat);
  ValueRef w2_packed = graph.add_tensor(size_small, vkapi::kFloat);

  auto prepackFn = VK_GET_OP_FN("et_vk.prepack.default");
  prepackFn(graph, {w1, w1_packed});
  prepackFn(graph, {w2, w2_packed});

  auto addFn = VK_GET_OP_FN("aten.add.Tensor");
  addFn(graph, {a.value, w1_packed, kDummyValueRef, c});

  auto mulFn = VK_GET_OP_FN("aten.mul.Tensor");
  mulFn(graph, {c, w2_packed, e});

  IOValueRef out = {};
  out.value = e;
  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();

  graph.prepack();

  // Run graph

  for (float i = 5.0f; i < 30.0f; i += 10.0f) {
    float val_out = (i + 3.5f) * 3.0f;

    fill_vtensor(graph, a, i);

    // Execute graph
    graph.execute();

    EXTRACT_TENSOR(out);

    // Sanity check that the values are correct
    for (size_t i = 0; i < graph.numel_of(out.value); ++i) {
      CHECK_VALUE(data_out, i, val_out);
    }

    if (graph.context()->querypool()) {
      graph.context()->querypool().extract_results();
      graph.context()->querypool().print_results();
    }
  }
}

TEST(VulkanComputeGraphTest, test_simple_shared_objects_with_resize) {
  GraphConfig config;
  config.expect_dynamic_shapes = true;
  ComputeGraph graph(config);
  size_t expected_vma_allocation_count = 0;

  std::vector<int64_t> size_big = {12, 64, 64};
  std::vector<int64_t> size_small = {12, 64, 64};

  // Build graph and regularly check allocation counts

  IOValueRef a = graph.add_input_tensor(
      size_big,
      vkapi::kFloat,
      /*shared_object_idx = */ 2);
  IOValueRef b = graph.add_input_tensor(
      size_small,
      vkapi::kFloat,
      /*shared_object_idx = */ 4);

  // +2: t.sizes_ubo() for each staging shader
  expected_vma_allocation_count += 2;
  EXPECT_EQ(get_vma_allocation_count(), expected_vma_allocation_count);

  ValueRef c = graph.add_tensor(
      size_big,
      vkapi::kFloat,
      /*shared_object_idx = */ 6);

  auto addFn = VK_GET_OP_FN("aten.add.Tensor");
  addFn(graph, {a.value, b.value, kDummyValueRef, c});

  // no new allocations if binary op uses push constants
  EXPECT_EQ(get_vma_allocation_count(), expected_vma_allocation_count);

  IOValueRef d = graph.add_input_tensor(
      size_small,
      vkapi::kFloat,
      /*shared_object_idx = */ 2);

  // +1: t.sizes_ubo() uniform buffer for staging shader
  expected_vma_allocation_count += 1;
  EXPECT_EQ(get_vma_allocation_count(), expected_vma_allocation_count);

  ValueRef e = graph.add_tensor(
      size_big,
      vkapi::kFloat,
      /*shared_object_idx = */ 4);

  auto mulFn = VK_GET_OP_FN("aten.mul.Tensor");
  mulFn(graph, {c, d.value, e});

  // no new allocations if binary op uses push constants
  EXPECT_EQ(get_vma_allocation_count(), expected_vma_allocation_count);

  IOValueRef out = {};
  out.value = e;
  out.staging = graph.set_output_tensor(out.value);

  // +1: staging buffer input tensor
  expected_vma_allocation_count += 1;
  EXPECT_EQ(get_vma_allocation_count(), expected_vma_allocation_count);

  graph.prepare();
  graph.prepack();

  // +3: shared memory allocations for tensors
  expected_vma_allocation_count += 3;
  EXPECT_EQ(get_vma_allocation_count(), expected_vma_allocation_count);

  // Run graph

  std::vector<std::vector<int64_t>> new_sizes_list = {
      {8, 44, 34}, {4, 13, 56}, {8, 12, 64}, {12, 55, 33}, {4, 54, 10}};

  for (auto& new_sizes : new_sizes_list) {
    graph.virtual_resize(a.value, new_sizes);
    graph.virtual_resize(b.value, new_sizes);
    graph.virtual_resize(d.value, new_sizes);
    graph.propagate_resize();

    float val_a = new_sizes[1] + 4.0f;
    float val_b = new_sizes[2] + 1.5f;
    float val_d = new_sizes[0] + 2.0f;
    float val_out = (val_a + val_b) * val_d;

    fill_vtensor(graph, a, val_a);
    fill_vtensor(graph, b, val_b);
    fill_vtensor(graph, d, val_d);

    // Execute graph
    graph.execute();

    EXTRACT_TENSOR(out);

    // Sanity check that the values are correct
    for (size_t i = 0; i < graph.numel_of(out.value); i++) {
      CHECK_VALUE(data_out, i, val_out);
    }
  }

  std::vector<std::vector<int64_t>> new_sizes_list_2 = {
      {8, 44, 34}, {4, 13, 56}, {8, 12, 64}, {12, 55, 33}, {4, 54, 10}};

  for (auto& new_sizes : new_sizes_list_2) {
    graph.resize_input(0, new_sizes);
    graph.resize_input(1, new_sizes);
    graph.resize_input(2, new_sizes);
    graph.propagate_resize();

    // Check output shape
    EXPECT_TRUE(graph.sizes_of(out.value) == new_sizes);

    float val_a = new_sizes[1] + 6.0f;
    float val_b = new_sizes[2] + 2.5f;
    float val_d = new_sizes[0] + 4.0f;
    float val_out = (val_a + val_b) * val_d;

    fill_vtensor(graph, a, val_a);
    fill_vtensor(graph, b, val_b);
    fill_vtensor(graph, d, val_d);

    // Execute graph
    graph.execute();

    EXTRACT_TENSOR(out);

    // Sanity check that the values are correct
    for (size_t i = 0; i < graph.numel_of(out.value); i++) {
      CHECK_VALUE(data_out, i, val_out);
    }
  }
}

TEST(VulkanComputeGraphTest, test_simple_graph_with_tmp_tensors) {
  GraphConfig config;
  ComputeGraph graph(config);

  std::vector<int64_t> size_big = {8, 64, 124};
  std::vector<int64_t> size_small = {8, 1, 124};

  // Build graph

  IOValueRef a = graph.add_input_tensor(
      size_big, vkapi::kFloat, /*shared_object_idx = */ 0);
  IOValueRef b = graph.add_input_tensor(
      size_small, vkapi::kFloat, /*shared_object_idx = */ 1);

  IOValueRef out = {};

  out.value =
      graph.add_tensor(size_big, vkapi::kFloat, /*shared_object_idx = */ 2);

  // Perform the following compute
  //
  // a, b, out;
  // {
  //   inter;
  //   {
  //     tmp = a + b
  //     tmp2 = tmp + a
  //     inter = tmp2 + b
  //   }
  //   {
  //     tmp = inter + b;
  //     tmp2 = tmp + a
  //     out = tmp2 + b;
  //   }
  // }
  {
    TmpTensor inter(&graph, size_big, vkapi::kFloat);
    EXPECT_TRUE(inter.sobj_idx == 3);
    {
      TmpTensor tmp(&graph, size_big, vkapi::kFloat);
      EXPECT_TRUE(tmp.sobj_idx == 4);
      VK_GET_OP_FN("aten.add.Tensor")
      (graph, {a, b, kDummyValueRef, tmp});

      TmpTensor tmp2(&graph, size_big, vkapi::kFloat);
      EXPECT_TRUE(tmp2.sobj_idx == 5);
      VK_GET_OP_FN("aten.add.Tensor")
      (graph, {tmp, a, kDummyValueRef, tmp2});

      VK_GET_OP_FN("aten.add.Tensor")
      (graph, {tmp2, b, kDummyValueRef, inter});
    }
    {
      TmpTensor tmp(&graph, size_big, vkapi::kFloat);
      EXPECT_TRUE(tmp.sobj_idx == 4);
      VK_GET_OP_FN("aten.add.Tensor")
      (graph, {inter, b, kDummyValueRef, tmp});

      TmpTensor tmp2(&graph, size_big, vkapi::kFloat);
      EXPECT_TRUE(tmp2.sobj_idx == 5);
      VK_GET_OP_FN("aten.add.Tensor")
      (graph, {tmp, a, kDummyValueRef, tmp2});

      VK_GET_OP_FN("aten.add.Tensor")
      (graph, {tmp2, b, kDummyValueRef, out});
    }
  }

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();
  graph.prepack();

  // Run graph

  for (float i = 5.0f; i < 30.0f; i += 10.0f) {
    float val_a = i + 2.0f;
    float val_b = i + 1.5f;
    float val_tmp = val_a + val_b;
    float val_tmp2 = val_tmp + val_a;
    float val_inter = val_tmp2 + val_b;
    float val_tmp_2 = val_inter + val_b;
    float val_tmp2_2 = val_tmp_2 + val_a;
    float val_out = val_tmp2_2 + val_b;

    fill_vtensor(graph, a, val_a);
    fill_vtensor(graph, b, val_b);

    graph.execute();

    EXTRACT_TENSOR(out);

    // Sanity check that the values are correct
    for (size_t i = 0; i < graph.numel_of(out.value); ++i) {
      CHECK_VALUE(data_out, i, val_out);
    }
  }
}

TEST(VulkanComputeGraphTest, test_large_graph) {
  auto build_start_time = std::chrono::system_clock::now();
  GraphConfig config;
  config.expect_dynamic_shapes = true;
  ComputeGraph graph(config);

  int64_t input_w = 256;
  int64_t input_h = 256;
  int64_t input_c = 8;

  std::vector<int64_t> size_big = {input_c, input_h, input_w};
  std::vector<int64_t> size_small = {input_c, input_h, 1};

  std::vector<int64_t> size_big_alt = {input_c / 2, input_h / 2, input_w / 2};
  std::vector<int64_t> size_small_alt = {input_c / 2, input_h / 2, 1};

  // Build graph

  IOValueRef a = graph.add_input_tensor(size_big, vkapi::kFloat, 2);
  IOValueRef b = graph.add_input_tensor(size_small, vkapi::kFloat, 4);

  ValueRef c = graph.add_tensor(size_big, vkapi::kFloat, 6);

  auto addFn = VK_GET_OP_FN("aten.add.Tensor");
  addFn(graph, {a.value, b.value, kDummyValueRef, c});

  int n = 100;

  for (int i = 0; i < n; i++) {
    addFn(graph, {c, b.value, kDummyValueRef, a.value});

    addFn(graph, {a.value, b.value, kDummyValueRef, c});
  }

  IOValueRef out = {};
  out.value = c;
  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();
  graph.prepack();

  auto build_end_time = std::chrono::system_clock::now();

  auto build_time = std::chrono::duration_cast<std::chrono::microseconds>(
      build_end_time - build_start_time);

  std::stringstream ss;
  for (int i = 0; i < 10; i++) {
    auto resize_start_time = std::chrono::system_clock::now();
    if (i % 2 == 0) {
      graph.resize_input(0, size_big_alt);
      graph.resize_input(1, size_small_alt);
    } else {
      graph.resize_input(0, size_big);
      graph.resize_input(1, size_small);
    }
    graph.propagate_resize();
    auto resize_end_time = std::chrono::system_clock::now();

    auto resize_time = std::chrono::duration_cast<std::chrono::microseconds>(
        resize_end_time - resize_start_time);

    float val_a = 1.0f;
    float val_b = 2.0f;

    float val_e = val_a + val_b * (2 * n + 1);

    auto inference_start_time = std::chrono::system_clock::now();

    fill_vtensor(graph, a, val_a);
    fill_vtensor(graph, b, val_b);

    graph.execute();

    EXTRACT_TENSOR(out);

    auto inference_end_time = std::chrono::system_clock::now();

    auto inference_time = std::chrono::duration_cast<std::chrono::microseconds>(
        inference_end_time - inference_start_time);

    for (int i = 0; i < graph.numel_of(out.value); i++) {
      CHECK_VALUE(data_out, i, val_e);
    }

    ss << "[          ] Resize:    " << std::setw(10) << std::right
       << resize_time.count() << " us" << std::endl;
    ss << "[          ] Inference: " << std::setw(10) << std::right
       << inference_time.count() << " us" << std::endl;
  }
  ss << "[          ] Model Load:" << std::setw(10) << std::right
     << build_time.count() << " us" << std::endl;
  std::cout << ss.str();
}

void test_clone(
    std::vector<int64_t> sizes,
    utils::StorageType src_storage,
    utils::GPUMemoryLayout src_layout,
    utils::StorageType dst_storage,
    utils::GPUMemoryLayout dst_layout) {
  GraphConfig config;
  ComputeGraph graph(config);

  IOValueRef a =
      graph.add_input_tensor(sizes, vkapi::kFloat, src_storage, src_layout);

  IOValueRef out = {};
  out.value = graph.add_tensor(sizes, vkapi::kFloat, dst_storage, dst_layout);

  auto copyFn = VK_GET_OP_FN("aten.clone.default");
  copyFn(graph, {a.value, kDummyValueRef, out.value});

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();
  graph.prepack();

  fill_vtensor(graph, a, 0.0f, /*iota = */ true);

  graph.propagate_resize();
  graph.execute();

  EXTRACT_TENSOR(out);
  EXTRACT_TENSOR(a);

  for (int i = 0; i < graph.numel_of(a.value); ++i) {
    EXPECT_TRUE(data_out[i] == data_a[i]);
  }
}

TEST(VulkanComputeGraphTest, test_clone) {
  std::vector<std::pair<utils::GPUMemoryLayout, utils::GPUMemoryLayout>> cases{
      {utils::kWidthPacked, utils::kWidthPacked},
      {utils::kWidthPacked, utils::kChannelsPacked},
      {utils::kChannelsPacked, utils::kChannelsPacked},
  };

  for (std::vector<int64_t> sizes : standard_sizes_to_test) {
    for (auto& [src_layout, dst_layout] : cases) {
      test_clone(
          sizes, utils::kTexture3D, src_layout, utils::kBuffer, dst_layout);
      test_clone(
          sizes, utils::kBuffer, src_layout, utils::kTexture3D, dst_layout);
      test_clone(
          sizes, utils::kTexture3D, src_layout, utils::kTexture3D, dst_layout);
    }
  }
}

TEST(VulkanComputeGraphTest, test_view_change_packing) {
  std::vector<std::pair<utils::GPUMemoryLayout, utils::GPUMemoryLayout>>
      layout_pairs = {
          {utils::kWidthPacked, utils::kChannelsPacked},
          {utils::kWidthPacked, utils::kHeightPacked},
          {utils::kWidthPacked, utils::kWidthPacked},
          {utils::kHeightPacked, utils::kChannelsPacked},
          {utils::kHeightPacked, utils::kHeightPacked},
          {utils::kHeightPacked, utils::kHeightPacked},
          {utils::kChannelsPacked, utils::kChannelsPacked},
          {utils::kChannelsPacked, utils::kHeightPacked},
          {utils::kChannelsPacked, utils::kHeightPacked},
      };

  int64_t n = 3;
  int64_t c = 2;
  int64_t h = 2;
  int64_t w = 5;
  std::vector<int64_t> size = {n, c, h, w};

  for (auto layout_pair : layout_pairs) {
    GraphConfig config;
    ComputeGraph graph(config);

    IOValueRef in =
        graph.add_input_tensor(size, vkapi::kFloat, layout_pair.first);

    IOValueRef out = {};
    out.value = graph.add_tensor(size, vkapi::kFloat, layout_pair.second);

    auto viewFn = VK_GET_OP_FN("aten.view_copy.default");
    viewFn(graph, {in.value, graph.add_none(), out.value});

    out.staging = graph.set_output_tensor(out.value);

    graph.prepare();
    graph.prepack();

    fill_vtensor(graph, in, 0.0, true);

    graph.execute();

    EXTRACT_TENSOR(out);

    // The extracted data is a flattened nchw buffer. Hence, should expect the
    // all elements inside the out array to match the index.
    for (int i = 0; i < graph.numel_of(out.value); i++) {
      CHECK_VALUE(data_out, i, i);
    }
  }
}

class VulkanToFromGPUShaderTest : public ::testing::Test {
 public:
  void SetUp() override {
    // Make sure we are starting with a clean slate
    EXPECT_TRUE(get_vma_allocation_count() == 0);
  }

  void TearDown() override {
    context()->flush();

    // Make sure we are ending with a clean slate
    EXPECT_TRUE(get_vma_allocation_count() == 0);
  }
};

template <typename T>
void run_from_gpu_test(
    std::vector<int64_t>& sizes,
    utils::GPUMemoryLayout memory_layout =
        utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
    vkapi::ScalarType dtype = vkapi::kFloat,
    utils::StorageType storage_type = utils::StorageType::TEXTURE_3D) {
  vTensor vten = vTensor(context(), sizes, dtype, storage_type, memory_layout);

  std::string kernel_name("idx_fill_texture");
  add_dtype_suffix(kernel_name, vten.dtype());

  int32_t offset = -50;

  {
    vkapi::PipelineBarrier pipeline_barrier{};
    context()->submit_compute_job(
        VK_KERNEL_FROM_STR(kernel_name),
        pipeline_barrier,
        vten.logical_limits(),
        {4, 4, 4},
        {vten.packed_dim(), offset},
        VK_NULL_HANDLE,
        0,
        vten.image(
            pipeline_barrier,
            vkapi::PipelineStage::COMPUTE,
            vkapi::MemoryAccessType::WRITE),
        vten.sizes_ubo());
  }

  StagingBuffer staging_buffer(
      context(),
      dtype,
      vten.staging_buffer_numel(),
      vkapi::CopyDirection::DEVICE_TO_HOST);

  if (dtype == vkapi::kChar &&
      !context()->adapter_ptr()->has_full_int8_buffers_support()) {
    record_bitw8_image_to_nchw_nobitw8buffer_op(
        context(), vten, staging_buffer);
  } else {
    record_image_to_nchw_op(context(), vten, staging_buffer.buffer());
  }

  submit_to_gpu();

  std::vector<T> data_out(vten.staging_buffer_numel());

  if (dtype == vkapi::kHalf && staging_buffer.dtype() == vkapi::kFloat) {
    staging_buffer.cast_float_to_half_and_copy_to(
        reinterpret_cast<uint16_t*>(data_out.data()), data_out.size());
  } else {
    VK_CHECK_COND(dtype == staging_buffer.dtype());
    staging_buffer.copy_to(data_out.data(), staging_buffer.nbytes());
  }

  for (int i = 0; i < vten.numel(); i++) {
    CHECK_VALUE(data_out, i, i + offset);
  }
}

template <typename T>
void round_trip_test(
    std::vector<int64_t>& sizes,
    utils::GPUMemoryLayout memory_layout =
        utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
    vkapi::ScalarType dtype = vkapi::kFloat,
    utils::StorageType storage_type = utils::StorageType::TEXTURE_3D) {
  vTensor vten = vTensor(context(), sizes, dtype, storage_type, memory_layout);

  // Create and fill input staging buffer
  StagingBuffer staging_buffer_in(
      context(),
      dtype,
      vten.staging_buffer_numel(),
      vkapi::CopyDirection::HOST_TO_DEVICE);

  std::vector<T> data_in(vten.staging_buffer_numel());
  for (int i = 0; i < data_in.size(); i++) {
    data_in[i] = T(i * -1);
  }
  // When float16 buffers are not supported, staging buffer uses float32
  // storage. Use conversion methods to properly copy half data.
  if (dtype == vkapi::kHalf && staging_buffer_in.dtype() == vkapi::kFloat) {
    staging_buffer_in.cast_half_to_float_and_copy_from(
        reinterpret_cast<const uint16_t*>(data_in.data()), data_in.size());
  } else {
    staging_buffer_in.copy_from(data_in.data(), vten.staging_buffer_nbytes());
  }

  // Output staging buffer
  StagingBuffer staging_buffer_out(
      context(),
      dtype,
      vten.staging_buffer_numel(),
      vkapi::CopyDirection::DEVICE_TO_HOST);

  record_nchw_to_image_op(context(), staging_buffer_in.buffer(), vten);

  // Copy data in and out of the tensor
  if (dtype == vkapi::kChar &&
      !context()->adapter_ptr()->has_full_int8_buffers_support()) {
    record_bitw8_image_to_nchw_nobitw8buffer_op(
        context(), vten, staging_buffer_out);
  } else {
    record_image_to_nchw_op(context(), vten, staging_buffer_out.buffer());
  }

  // Execute command buffer
  submit_to_gpu();

  // Extract data from output staging buffer
  std::vector<T> data_out(vten.staging_buffer_numel());
  // When float16 buffers are not supported, staging buffer uses float32
  // storage. Use conversion methods to properly copy out to half data.
  if (dtype == vkapi::kHalf && staging_buffer_out.dtype() == vkapi::kFloat) {
    staging_buffer_out.cast_float_to_half_and_copy_to(
        reinterpret_cast<uint16_t*>(data_out.data()), data_out.size());
  } else {
    staging_buffer_out.copy_to(data_out.data(), staging_buffer_out.nbytes());
  }

  // All indices should be equal to the input data
  for (int i = 0; i < vten.numel(); i++) {
    CHECK_VALUE(data_out, i, data_in[i]);
  }
}

template <typename T>
void compute_graph_round_trip_test(
    std::vector<int64_t>& sizes,
    utils::GPUMemoryLayout memory_layout =
        utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
    vkapi::ScalarType dtype = vkapi::kFloat,
    utils::StorageType storage_type = utils::StorageType::TEXTURE_3D) {
  GraphConfig config;
  ComputeGraph graph(config);

  ValueRef r_tensor =
      graph.add_tensor(sizes, dtype, storage_type, memory_layout);
  ValueRef r_staging_in = graph.set_input_tensor(r_tensor);
  ValueRef r_staging_out = graph.set_output_tensor(r_tensor);

  graph.prepare();
  graph.prepack();

  std::vector<T> data_in(graph.numel_of(r_tensor));
  for (int i = 0; i < data_in.size(); i++) {
    data_in[i] = T(i * -1);
  }
  graph.maybe_cast_and_copy_into_staging(
      r_staging_in, data_in.data(), data_in.size(), dtype);

  graph.execute();

  std::vector<T> data_out(graph.staging_buffer_numel_of(r_tensor));
  graph.maybe_cast_and_copy_from_staging(
      r_staging_out, data_out.data(), data_out.size(), dtype);

  for (int i = 0; i < data_in.size(); i++) {
    CHECK_VALUE(data_out, i, data_in[i]);
  }
}

TEST(VulkanToFromGPUShaderTest, round_trip_tests) {
  // The below tests will fill each texel element with the value of the linear
  // buffer index that corresponds to it. The texel at position (0, 0, 0) will
  // be filled with the values [0, 1, 2, 3], the texel at position (1, 0, 0)
  // will be filled with the values [4, 5, 6, 7], and so forth. The contents of
  // the texture are then written back to the CPU, and to check that the
  // transfer has ben performed correctly the value at each index of the CPU
  // data buffer should be equal to the index.
  //
  // The below test cases should ensure that the total number of elements does
  // not exceed 2048, or else the tests will fail for FP16 textures due to
  // precision issues. Half precision floating point formats can only represent
  // integers from 2048 to 4096 using intervals of 2.
  std::vector<std::vector<int64_t>> to_test = {
      // 2D sizes
      {17, 21},
      {67, 23},
      {55, 33},
      // 3D sizes
      {7, 9, 13},
      {21, 2, 19},
      {17, 17, 5},
      // 4D sizes
      {7, 3, 13, 7},
      {11, 9, 9, 1},
      {3, 3, 3, 3},
      {3, 1, 7, 13},
  };

  // These sizes are set such that the total number of elements is less than
  // 128 which is the maximum representable value for int8.
  std::vector<std::vector<int64_t>> to_test_int8 = {
      // 2D sizes
      {14, 7},
      // 3D sizes
      {3, 7, 5},
      {4, 2, 11},
      // 4D sizes
      {3, 3, 3, 3},
      {7, 1, 6, 3},
  };

#define RUN_TESTS(ctype, dtype)                                      \
  round_trip_test<ctype>(                                            \
      sizes, utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED, dtype); \
  round_trip_test<ctype>(                                            \
      sizes, utils::GPUMemoryLayout::TENSOR_WIDTH_PACKED, dtype);    \
  round_trip_test<ctype>(                                            \
      sizes, utils::GPUMemoryLayout::TENSOR_HEIGHT_PACKED, dtype);   \
  compute_graph_round_trip_test<ctype>(                              \
      sizes, utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED, dtype); \
  compute_graph_round_trip_test<ctype>(                              \
      sizes, utils::GPUMemoryLayout::TENSOR_WIDTH_PACKED, dtype);    \
  compute_graph_round_trip_test<ctype>(                              \
      sizes, utils::GPUMemoryLayout::TENSOR_HEIGHT_PACKED, dtype);

  for (auto& sizes : to_test) {
    RUN_TESTS(float, vkapi::kFloat)
    RUN_TESTS(executorch::aten::Half, vkapi::kHalf)
  }

  for (auto& sizes : to_test_int8) {
    RUN_TESTS(int8_t, vkapi::kChar);
  }

#undef RUN_TESTS
}

//
// Operator Smoke Tests
//

void test_binary_op(
    std::string op_name,
    std::vector<int64_t> sizes_big,
    std::vector<int64_t> sizes_small,
    vkapi::ScalarType dtype,
    utils::GPUMemoryLayout memory_layout) {
  GraphConfig config;
  ComputeGraph graph(config);

  IOValueRef arg2{};

  // Build graph

  IOValueRef arg1 = graph.add_input_tensor(sizes_big, dtype, memory_layout);
  arg2 = graph.add_input_tensor(sizes_small, dtype, memory_layout);

  IOValueRef out;
  out.value = graph.add_tensor(sizes_big, dtype, memory_layout);

  std::stringstream ss;
  ss << "aten.";
  ss << op_name;
  ss << ".Tensor";
  VK_GET_OP_FN(ss.str())
  (graph, {arg1.value, arg2.value, kDummyValueRef, out.value});

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();

  graph.prepack();

  for (int i = 1; i < 4; i++) {
    float val_arg1 = i + 1.5;
    float val_arg2 = i - 3.5;

    float val_out = val_arg1 + val_arg2;
    if (op_name == "sub") {
      val_out = val_arg1 - val_arg2;
    }
    if (op_name == "mul") {
      val_out = val_arg1 * val_arg2;
    }
    if (op_name == "div") {
      val_out = val_arg1 / val_arg2;
    }

    execute_graph_and_check_output(graph, {val_arg1, val_arg2}, {val_out});
  }
}

#define CALL_TEST_FN_FORALL_CONDITIONS(_)                   \
  _(vkapi::kFloat, utils::kTexture3D, utils::kWidthPacked)  \
  _(vkapi::kFloat, utils::kTexture3D, utils::kHeightPacked) \
  _(vkapi::kFloat, utils::kTexture3D, utils::kChannelsPacked)

#define CALL_TEST_FN_FOR_W_PACKED(_)                              \
  _(vkapi::kFloat, utils::kTexture3D, utils::kWidthPacked, false) \
  _(vkapi::kFloat, utils::kTexture3D, utils::kWidthPacked, true)  \
  _(vkapi::kFloat, utils::kBuffer, utils::kWidthPacked, false)    \
  _(vkapi::kFloat, utils::kBuffer, utils::kWidthPacked, true)

#define CALL_TEST_FN_FOR_C_PACKED(_)                                 \
  _(vkapi::kFloat, utils::kTexture3D, utils::kChannelsPacked, false) \
  _(vkapi::kFloat, utils::kTexture3D, utils::kChannelsPacked, true)  \
  _(vkapi::kFloat, utils::kBuffer, utils::kChannelsPacked, false)    \
  _(vkapi::kFloat, utils::kBuffer, utils::kChannelsPacked, true)

TEST(VulkanComputeGraphOpsTest, add_smoke_test) {
#define RUN_TESTS(dtype, storage, layout)                         \
  test_binary_op("add", {17, 21}, {17, 21}, dtype, layout);       \
  test_binary_op("add", {17, 21}, {1, 1}, dtype, layout);         \
  test_binary_op("sub", {11, 22}, {11, 22}, dtype, layout);       \
  test_binary_op("sub", {11, 22}, {11, 1}, dtype, layout);        \
  test_binary_op("add", {7, 17, 17}, {7, 17, 17}, dtype, layout); \
  test_binary_op("add", {7, 17, 17}, {7, 1, 17}, dtype, layout);  \
  test_binary_op("sub", {9, 9, 7}, {9, 9, 7}, dtype, layout);     \
  test_binary_op("sub", {9, 9, 7}, {9, 1, 1}, dtype, layout);

  CALL_TEST_FN_FORALL_CONDITIONS(RUN_TESTS);

#undef RUN_TESTS
}

void test_mm(
    int B,
    int M,
    int K,
    int N,
    vkapi::ScalarType dtype,
    utils::StorageType storage_type,
    utils::GPUMemoryLayout memory_layout,
    bool prepack = true) {
  std::vector<int64_t> mat2_size = {B, K, N};

  std::vector<float> mat2_data(utils::multiply_integers(mat2_size));
  std::fill(mat2_data.begin(), mat2_data.end(), 2.0f);
  ComputeGraph graph = build_mm_graph(
      B, M, K, N, dtype, storage_type, memory_layout, mat2_data, prepack);

  graph.prepare();
  graph.prepack();

  for (int i = 1; i < 4; i++) {
    if (prepack) {
      float val_mat1 = i;
      float val_out = K * (val_mat1 * 2.0f);
      execute_graph_and_check_output(graph, {val_mat1}, {val_out});
    } else {
      float val_mat1 = i;
      float val_mat2 = i + 1;
      float val_out = K * (val_mat1 * val_mat2);
      execute_graph_and_check_output(graph, {val_mat1, val_mat2}, {val_out});
    }
  }
}

TEST(VulkanComputeGraphOpsTest, mm_smoke_test) {
#define RUN_TESTS(dtype, storage_type, layout, prepack) \
  test_mm(/*B = */ 1,                                   \
          /*M = */ 31,                                  \
          /*K = */ 127,                                 \
          /*N = */ 23,                                  \
          dtype,                                        \
          storage_type,                                 \
          layout,                                       \
          prepack);                                     \
  test_mm(/*B = */ 5,                                   \
          /*M = */ 31,                                  \
          /*K = */ 127,                                 \
          /*N = */ 23,                                  \
          dtype,                                        \
          storage_type,                                 \
          layout,                                       \
          prepack);                                     \
  test_mm(/*B = */ 7,                                   \
          /*M = */ 13,                                  \
          /*K = */ 89,                                  \
          /*N = */ 17,                                  \
          dtype,                                        \
          storage_type,                                 \
          layout,                                       \
          prepack);                                     \
  test_mm(/*B = */ 1,                                   \
          /*M = */ 13,                                  \
          /*K = */ 89,                                  \
          /*N = */ 17,                                  \
          dtype,                                        \
          storage_type,                                 \
          layout,                                       \
          prepack);

  CALL_TEST_FN_FOR_W_PACKED(RUN_TESTS);
  CALL_TEST_FN_FOR_C_PACKED(RUN_TESTS);

#undef RUN_TESTS
}

void test_mm_with_resize_reencode(
    int B,
    int M,
    int K,
    int N,
    vkapi::ScalarType dtype,
    utils::StorageType storage_type,
    utils::GPUMemoryLayout memory_layout) {
  ASSERT_TRUE(M > 1);

  std::vector<int64_t> mat2_size = {B, K, N};
  std::vector<float> mat2_data(utils::multiply_integers(mat2_size));
  std::fill(mat2_data.begin(), mat2_data.end(), 2.0f);

  ComputeGraph graph = build_mm_graph(
      B, M, K, N, dtype, storage_type, memory_layout, mat2_data, false);

  graph.prepare();
  graph.prepack();

  for (int i = 1; i < 4; i++) {
    float val_mat1 = i;
    float val_mat2 = i + 1;
    float val_out = K * (val_mat1 * val_mat2);
    execute_graph_and_check_output(graph, {val_mat1, val_mat2}, {val_out});
  }

  // Switch to GEMV mode
  int new_K = K / 2;
  std::vector<int64_t> new_mat1_size = {1, new_K};
  std::vector<int64_t> new_mat2_size = {new_K, N};
  graph.resize_input(0, new_mat1_size);
  graph.resize_input(1, new_mat2_size);
  graph.propagate_resize();

  for (int i = 1; i < 4; i++) {
    float val_mat1 = i;
    float val_mat2 = i + 1;
    float val_out = new_K * (val_mat1 * val_mat2);
    execute_graph_and_check_output(graph, {val_mat1, val_mat2}, {val_out});
  }
}

TEST(VulkanComputeGraphOpsTest, test_graph_resize_reencode) {
  test_mm_with_resize_reencode(
      /*B = */ 1,
      /*M = */ 31,
      /*K = */ 127,
      /*N = */ 23,
      vkapi::kFloat,
      utils::kTexture3D,
      utils::kWidthPacked);
}

void test_grid_priors(
    std::vector<int64_t> input_sizes,
    std::vector<int64_t> output_sizes,
    int stride,
    double offset,
    const std::vector<float>& data_out_expected) {
  GraphConfig config;
  ComputeGraph graph(config);

  // Build graph
  IOValueRef in = graph.add_input_tensor(
      input_sizes,
      vkapi::kFloat,
      utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);
  IOValueRef out;
  out.value = graph.add_tensor(
      output_sizes,
      vkapi::kFloat,
      utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);

  VK_GET_OP_FN("et_vk.grid_priors.default")
  (graph,
   {in.value,
    graph.add_scalar<int64_t>(stride),
    graph.add_scalar<double>(offset),
    out.value});

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();

  graph.prepack();

  // Resize input
  graph.propagate_resize();

  // run graph
  graph.execute();

  std::vector<float> output_data(graph.staging_buffer_numel_of(out.value));
  graph.maybe_cast_and_copy_from_staging(
      out.staging, output_data.data(), output_data.size(), vkapi::kFloat);

  // check results
  std::vector<int64_t> out_sizes = graph.sizes_of(out.value);
  int h_out = utils::val_at(-2, out_sizes);
  int w_out = utils::val_at(-1, out_sizes);
  for (size_t i = 0; i < h_out; ++i) {
    for (size_t j = 0; j < w_out; ++j) {
      size_t idx_out = i * w_out + j;
      CHECK_VALUE(output_data, idx_out, data_out_expected[idx_out]);
    }
  }
}

TEST(VulkanComputeGraphOpsTest, grid_priors_test) {
  test_grid_priors(
      /*input size = */ {1, 5, 2, 3},
      /*output size = */ {6, 2},
      /*stride = */ 1,
      /*offset = */ 0.0,
      /*data_out_expected = */ {0, 0, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1});

  test_grid_priors(
      /*input size = */ {1, 5, 2, 3},
      /*output size = */ {6, 2},
      /*stride = */ 8,
      /*offset = */ 0.5,
      /*data_out_expected = */ {4, 4, 12, 4, 20, 4, 4, 12, 12, 12, 20, 12});
}

void test_transpose_view_mm(
    const int B,
    const int M,
    const int K,
    const int N,
    utils::StorageType storage_type) {
  GraphConfig config;
  config.expect_dynamic_shapes = true;
  config.set_storage_type_override(storage_type);
  ComputeGraph graph(config);

  std::vector<int64_t> mat1_size = {M, K};
  std::vector<int64_t> mat2_t_size = {N, K};
  std::vector<int64_t> out_size = {M, N};

  std::vector<int64_t> mat1_small_size = {M - 4, K - 3};
  std::vector<int64_t> mat2_t_small_size = {N - 1, K - 3};

  if (B > 1) {
    mat1_size.resize(3);
    mat1_size = {B, M, K};
    mat2_t_size.resize(3);
    mat2_t_size = {B, N, K};
    out_size.resize(3);
    out_size = {B, M, N};

    mat1_small_size.resize(3);
    mat1_small_size = {B, M - 4, K - 3};
    mat2_t_small_size.resize(3);
    mat2_t_small_size = {B, N - 1, K - 3};
  }

  // Build graph; use shared objects to test views of shared objects

  IOValueRef mat1 =
      graph.add_input_tensor(mat1_size, vkapi::kFloat, utils::kWidthPacked, 0);
  IOValueRef mat2_transpose = graph.add_input_tensor(
      mat2_t_size, vkapi::kFloat, utils::kWidthPacked, 1);

  ValueRef mat2 = graph.add_tensor_view(mat2_transpose.value);

  ValueRef dim0;
  ValueRef dim1;

  if (B > 1) {
    dim0 = graph.add_scalar<int64_t>(1);
    dim1 = graph.add_scalar<int64_t>(2);
  } else {
    dim0 = graph.add_scalar<int64_t>(0);
    dim1 = graph.add_scalar<int64_t>(1);
  }

  IOValueRef out;
  out.value = graph.add_tensor(out_size, vkapi::kFloat, utils::kWidthPacked, 2);

  VK_GET_OP_FN("aten.transpose.int")
  (graph, {mat2_transpose.value, dim0, dim1, mat2});
  VK_GET_OP_FN("aten.mm.default")(graph, {mat1.value, mat2, out.value});

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();

  graph.prepack();

  for (int i = 1; i < 4; i++) {
    float val_mat1 = i;
    float val_mat2 = i + 1;
    float val_out = K * (val_mat1 * val_mat2);

    // Try at full size
    graph.resize_input(0, mat1_size);
    graph.resize_input(1, mat2_t_size);
    graph.propagate_resize();
    execute_graph_and_check_output(graph, {val_mat1, val_mat2}, {val_out});

    // Try at reduced sizes
    val_out = (K - 3) * (val_mat1 * val_mat2);
    graph.resize_input(0, mat1_small_size);
    graph.resize_input(1, mat2_t_small_size);
    graph.propagate_resize();
    execute_graph_and_check_output(graph, {val_mat1, val_mat2}, {val_out});
  }
}

TEST(VulkanComputeGraphOpsTest, test_transpose_with_mm) {
  for (auto storage_type : {utils::kBuffer, utils::kTexture3D}) {
    test_transpose_view_mm(2, 7, 17, 5, storage_type);
  }
}

void test_to_copy() {
  GraphConfig config;
  config.set_storage_type_override(utils::kTexture3D);
  ComputeGraph graph(config);
  int M = 8;
  int N = 8;
  int K = 8;
  // Build graph
  IOValueRef in = graph.add_input_tensor(
      {1, M, N, K},
      vkapi::kFloat,
      utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);

  std::vector<float> data_in =
      create_random_float_buffer(M * N * K, -1024, 1024);
  graph.maybe_cast_and_copy_into_staging(
      in.staging, data_in.data(), data_in.size(), vkapi::kFloat);

  IOValueRef out;
  out.value = graph.add_tensor(
      {1, M, N, K},
      vkapi::kHalf,
      utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);

  auto op = VK_GET_OP_FN("aten._to_copy.default");
  op(graph,
     {in.value,
      graph.add_none(),
      graph.add_none(),
      graph.add_none(),
      graph.add_none(),
      graph.add_none(),
      graph.add_none(),
      out.value});

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();

  graph.prepack();
  graph.propagate_resize();
  graph.execute();

  std::vector<torch::executor::Half> output_data(graph.numel_of(out.value));
  graph.maybe_cast_and_copy_from_staging(
      out.staging, output_data.data(), output_data.size(), vkapi::kHalf);

  EXPECT_EQ(data_in.size(), output_data.size());

#ifdef VULKAN_DEBUG
  float mse_ex = 0.0f;
  float mse_vk = 0.0f;
#endif

  // check results
  for (size_t i = 0; i < output_data.size(); ++i) {
    float input = data_in[i];
    torch::executor::Half expected_output =
        static_cast<torch::executor::Half>(input);
    uint16_t* expected_bits = reinterpret_cast<uint16_t*>(&expected_output);
    torch::executor::Half output = output_data[i];
    uint16_t* output_bits = reinterpret_cast<uint16_t*>(&output);

#ifdef VULKAN_DEBUG
    std::string msg;
    msg.reserve(64);
    msg = "input = " + std::to_string(input) + "(0b" +
        std::bitset<32>(*reinterpret_cast<uint32_t*>(&input)).to_string() +
        "), expected output = " + std::to_string(expected_output) + "(0b" +
        std::bitset<16>(*expected_bits).to_string() +
        "), recieved output = " + std::to_string(output) + "(0b" +
        std::bitset<16>(*output_bits).to_string() + ")";

    std::cout << msg << std::endl;

    mse_ex += std::pow(expected_output - input, 2);
    mse_vk += std::pow(output - input, 2);
#endif

    // Note: Torch executor half "rounds up" when converting to fp16 whereas
    // most driver implementations of Vulkan's opFConvert() just truncates the
    // extra bits for performance (rounding introduces conditional).
    // Example:
    // INPUT F32 = 25.248 (sign{0b0}, exp{0b10000011},
    // mantissa{0b10010011111101111100111}),
    // TORCH HALF OUTPUT F16 = 25.25 (sign{0b0}, exp{0b10011},
    // mantissa{0b1001010000}),
    // VULKAN OUTPUT F16 = 25.2344 (sign{0b0}, exp{0b10011},
    // mantissa{0b1001001111})
    // Note:
    // The vulkan mantissa exactly matches the first 10
    // bits of the input 23 bit mantissa. But since the 11th bit is 1, the
    // torch half output is rounded up (essentially adding a 1).
    // Vulkan mantissa{0b1001001111} + 1 = Torch half mantissa{0b1001010000}

    EXPECT_TRUE(
        (*output_bits == *expected_bits) ||
        /*rounding error*/ ((*output_bits + 1u) == *expected_bits));
  }

#ifdef VULKAN_DEBUG
  mse_ex /= output_data.size();
  mse_vk /= output_data.size();

  std::cout << "========================================================="
            << std::endl;
  std::cout << "mse_ex = " << mse_ex << ", mse_vk = " << mse_vk << std::endl;
#endif
}

TEST(VulkanComputeGraphOpsTest, test_to_copy) {
  test_to_copy();
}

vkapi::ShaderInfo pick_dynamic_dispatch_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& additional_args) {
  const ValueRef mat1 = args[1].refs[0];

  std::string kernel_name = "dynamic_dispatch_test";
  if (graph->size_at<int32_t>(-2, mat1) == 1) {
    kernel_name += "_var1";
  } else {
    kernel_name += "_var2";
  }
  return VK_KERNEL_FROM_STR(kernel_name);
}

utils::uvec3 pick_dynamic_dispatch_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  const ValueRef out = args[0].refs[0];
  return graph->logical_limits_of(out);
}

utils::uvec3 pick_dynamic_dispatch_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)shader;
  (void)global_workgroup_size;
  return {64, 1, 1};
}

void resize_dynamic_dispatch_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& additional_args) {
  const ValueRef out = args[0].refs[0];
  const ValueRef mat1 = args[1].refs[0];

  std::vector<int64_t> out_sizes = graph->sizes_of(mat1);
  out_sizes.at(out_sizes.size() - 2) = 1;

  graph->virtual_resize(out, out_sizes);
}

void add_dynamic_dispatch_test_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2,
    const ValueRef out) {
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_dynamic_dispatch_shader,
      pick_dynamic_dispatch_global_wg_size,
      pick_dynamic_dispatch_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{mat1, mat2}, vkapi::kRead}},
      // Shader params buffers
      {},
      // Push Constants
      {graph.sizes_pc_of(out),
       graph.sizes_pc_of(mat1),
       graph.sizes_pc_of(mat2)},
      // Specialization constants
      {},
      // Resize Logic
      {},
      resize_dynamic_dispatch_node));
}

vkcompute::ComputeGraph build_dynamic_dispatch_test_graph(int M, int N) {
  using namespace vkcompute;
  GraphConfig config;
  config.expect_dynamic_shapes = true;
  ComputeGraph graph(config);

  vkapi::ScalarType dtype = vkapi::kFloat;
  utils::StorageType in_out_stype = utils::kTexture3D;
  utils::GPUMemoryLayout memory_layout = utils::kWidthPacked;

  std::vector<int64_t> mat1_size = {M, N};
  std::vector<int64_t> mat2_size = {M, N};
  std::vector<int64_t> out_size = {1, N};

  IOValueRef mat1 =
      graph.add_input_tensor(mat1_size, dtype, in_out_stype, memory_layout);
  IOValueRef mat2{};

  mat2.value = graph.add_tensor(mat2_size, dtype, in_out_stype, memory_layout);
  mat2.staging = graph.set_input_tensor(mat2.value);

  IOValueRef out;
  out.value = graph.add_tensor(out_size, dtype, in_out_stype, memory_layout);

  add_dynamic_dispatch_test_node(graph, mat1, mat2, out);

  out.staging = graph.set_output_tensor(out.value);

  return graph;
}

void test_dynamic_dispatch(int M, int N) {
  ComputeGraph graph = build_dynamic_dispatch_test_graph(M, N);

  graph.prepare();
  graph.prepack();

  for (int i = 1; i < 4; i++) {
    float val_mat1 = i;
    float val_mat2 = i + 1;
    // 5.3 is a hardcoded offset in the compute shader
    float val_out = M * (val_mat1 * val_mat2) + 5.5;
    execute_graph_and_check_output(graph, {val_mat1, val_mat2}, {val_out});
  }

  // Switch to GEMV mode
  int new_N = N / 2;
  std::vector<int64_t> new_mat1_size = {1, new_N};
  std::vector<int64_t> new_mat2_size = {1, new_N};
  graph.resize_input(0, new_mat1_size);
  graph.resize_input(1, new_mat2_size);
  graph.propagate_resize();

  for (int i = 1; i < 4; i++) {
    float val_mat1 = i;
    float val_mat2 = i + 1;
    float val_out = (val_mat1 * val_mat2) + 2.25;
    execute_graph_and_check_output(graph, {val_mat1, val_mat2}, {val_out});
  }
}

TEST(VulkanComputeGraphOpsTest, test_dynamic_dispatch_graph) {
  test_dynamic_dispatch(128, 128);
}
