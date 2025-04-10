//
//  Copyright (c) 2025 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <gtest/gtest.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpGridSampler2dTest : public OperatorTest {
protected:

    // Test grid_sampler_2d with bilinear interpolation
    void test_grid_sampler_2d_bilinear() {
        TensorFactory<ScalarType::Float> tf;

        // Create a 1x1x2x2 input tensor
        const std::vector<int32_t> input_sizes = {1, 1, 2, 2};
        std::vector<float> input_data = {
            1.0, 2.0,
            3.0, 4.0
        };
        Tensor input = tf.make(input_sizes, input_data);
        
        // Create a 1x1x1x2 grid tensor
        const std::vector<int32_t> grid_sizes = {1, 1, 1, 2};
        std::vector<float> grid_data = {
            0.0, 0.0
        };
        Tensor grid = tf.make(grid_sizes, grid_data);
        
        // Create output tensor with expected shape 1x1x1x1
        const std::vector<int32_t> output_sizes = {1, 1, 1, 1};
        Tensor out = tf.zeros(output_sizes);

        // Set testing modes
        const int64_t interpolation_mode = 0; // bilinear
        const int64_t padding_mode = 0;
        const bool align_corners = false;
        
        torch::executor::aten::grid_sampler_2d_outf(
            context_, input, grid, interpolation_mode, padding_mode, align_corners, out);

        // Expected output
        const std::vector<float> expected_output = {
            2.5 // weighted sum of 1.0, 2.0, 3.0, 4.0 with unitary distances
        };

        // Test
        EXPECT_TENSOR_CLOSE(out, tf.make(output_sizes, expected_output));
    }

    // Test grid_sampler_2d with nearest neighbor interpolation
    void test_grid_sampler_2d_nearest() {
        TensorFactory<ScalarType::Float> tf;

        // Create a 1x1x4x4 input tensor
        const std::vector<int32_t> input_sizes = {1, 1, 4, 4};
        std::vector<float> input_data = {
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0
        };
        Tensor input = tf.make(input_sizes, input_data);
        
        // Create a 1x2x2x2 grid tensor
        const std::vector<int32_t> grid_sizes = {1, 2, 2, 2};
        std::vector<float> grid_data = {
            -0.5, -0.5,
            0.5, -0.5,
            -0.5, 0.5,
            0.5, 0.5
        };
        Tensor grid = tf.make(grid_sizes, grid_data);
        
        // Create output tensor with expected shape 1x1x2x2
        const std::vector<int32_t> output_sizes = {1, 1, 2, 2};
        Tensor out = tf.zeros(output_sizes);

        // Set testing modes
        const int64_t interpolation_mode = 1; // nearest
        const int64_t padding_mode = 0;
        const bool align_corners = false;
        
        torch::executor::aten::grid_sampler_2d_outf(
            context_, input, grid, interpolation_mode, padding_mode, align_corners, out);

        // Expected output
        const std::vector<float> expected_output = {
            6.0, 8.0,  // Nearest pixels for top row
            14.0, 16.0 // Nearest pixels for bottom row
        };

        // Test
        EXPECT_TENSOR_CLOSE(out, tf.make(output_sizes, expected_output));
    }

    // Test grid_sampler_2d with border padding mode
    void test_grid_sampler_2d_border_padding() {
        TensorFactory<ScalarType::Float> tf;

        // Create a 1x1x4x4 input tensor
        const std::vector<int32_t> input_sizes = {1, 1, 4, 4};
        std::vector<float> input_data = {
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0
        };
        Tensor input = tf.make(input_sizes, input_data);
        
        // Create a 1x2x2x2 grid tensor with points outside the input boundaries
        const std::vector<int32_t> grid_sizes = {1, 2, 2, 2};
        std::vector<float> grid_data = {
            -2.0, -2.0, // Way outside top-left
            2.0, -2.0,  // Way outside top-right
            -2.0, 2.0,  // Way outside bottom-left
            2.0, 2.0    // Way outside bottom-right
        };
        Tensor grid = tf.make(grid_sizes, grid_data);
        
        // Create output tensor with expected shape 1x1x2x2
        const std::vector<int32_t> output_sizes = {1, 1, 2, 2};
        Tensor out = tf.zeros(output_sizes);

        // Set testing modes
        const int64_t interpolation_mode = 0;
        const int64_t padding_mode = 1; // border
        const bool align_corners = false;
        
        torch::executor::aten::grid_sampler_2d_outf(
            context_, input, grid, interpolation_mode, padding_mode, align_corners, out);

        // Expected output
        const std::vector<float> expected_output = {
            1.0, 4.0,   // Top corners
            13.0, 16.0  // Bottom corners
        };

        // Test
        EXPECT_TENSOR_CLOSE(out, tf.make(output_sizes, expected_output));
    }

    // Test grid_sampler_2d with reflection padding mode
    void test_grid_sampler_2d_reflection_padding() {
        TensorFactory<ScalarType::Float> tf;

        // Create a 1x1x4x4 input tensor
        const std::vector<int32_t> input_sizes = {1, 1, 4, 4};
        std::vector<float> input_data = {
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0
        };
        Tensor input = tf.make(input_sizes, input_data);
        
        // Create a 1x2x2x2 grid with points outside the input boundaries
        const std::vector<int32_t> grid_sizes = {1, 2, 2, 2};
        std::vector<float> grid_data = {
            -2.0, -2.0, // Way outside top-left
            2.0, -2.0,  // Way outside top-right
            -2.0, 2.0,  // Way outside bottom-left
            2.0, 2.0    // Way outside bottom-right
        };
        Tensor grid = tf.make(grid_sizes, grid_data);
        
        // Create output tensor with expected shape 1x1x2x2
        const std::vector<int32_t> output_sizes = {1, 1, 2, 2};
        Tensor out = tf.zeros(output_sizes);

        // Set testing modes
        const int64_t interpolation_mode = 0;
        const int64_t padding_mode = 2; // reflection
        const bool align_corners = false;
        
        torch::executor::aten::grid_sampler_2d_outf(
            context_, input, grid, interpolation_mode, padding_mode, align_corners, out);

        // Check non-zero
        EXPECT_NE(out.data_ptr<float>()[0], 0.0f);
        EXPECT_NE(out.data_ptr<float>()[1], 0.0f);
        EXPECT_NE(out.data_ptr<float>()[2], 0.0f);
        EXPECT_NE(out.data_ptr<float>()[3], 0.0f);
    }

    // Test grid_sampler_2d with align_corners
    void test_grid_sampler_2d_align_corners() {
        TensorFactory<ScalarType::Float> tf;

        // Create a 1x1x4x4 input tensor
        const std::vector<int32_t> input_sizes = {1, 1, 4, 4};
        std::vector<float> input_data = {
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0
        };
        Tensor input = tf.make(input_sizes, input_data);
        
        // Create a 1x2x2x2 grid tensor
        const std::vector<int32_t> grid_sizes = {1, 2, 2, 2};
        std::vector<float> grid_data = {
            -0.5, -0.5,
            0.5, -0.5,
            -0.5, 0.5,
            0.5, 0.5
        };
        Tensor grid = tf.make(grid_sizes, grid_data);
        
        // Create output tensor with expected shape 1x1x2x2
        const std::vector<int32_t> output_sizes = {1, 1, 2, 2};
        Tensor out = tf.zeros(output_sizes);

        // Set testing modes
        const int64_t interpolation_mode = 0;
        const int64_t padding_mode = 0;
        const bool align_corners = true; // align corners
        
        torch::executor::aten::grid_sampler_2d_outf(
            context_, input, grid, interpolation_mode, padding_mode, align_corners, out);

        // Check dimensions
        EXPECT_EQ(out.dim(), 4);
        EXPECT_EQ(out.size(0), 1);
        EXPECT_EQ(out.size(1), 1);
        EXPECT_EQ(out.size(2), 2);
        EXPECT_EQ(out.size(3), 2);
    }

    // Test grid_sampler_2d with bilinear interpolation, border padding mode, and align corners
    void test_grid_sampler_2d_bilinear_border_align() {
        TensorFactory<ScalarType::Float> tf;

        // Create a 1x1x3x3 input tensor
        const std::vector<int32_t> input_sizes = {1, 1, 3, 3};
        std::vector<float> input_data = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        };
        Tensor input = tf.make(input_sizes, input_data);
        
        // Create a 1x2x2x2 grid tensor
        const std::vector<int32_t> grid_sizes = {1, 2, 2, 2};
        std::vector<float> grid_data = {
            -1.0, -1.0,
            1.0, -1.0,
            -1.0, 1.0,
            1.0, 1.0
        };
        Tensor grid = tf.make(grid_sizes, grid_data);
        
        // Create output tensor with expected shape 1x1x2x2
        const std::vector<int32_t> output_sizes = {1, 1, 2, 2};
        Tensor out = tf.zeros(output_sizes);

        // Set testing modes
        const int64_t interpolation_mode = 0; // bilinear
        const int64_t padding_mode = 1; // border
        const bool align_corners = true; // align corners
        
        torch::executor::aten::grid_sampler_2d_outf(
            context_, input, grid, interpolation_mode, padding_mode, align_corners, out);

        // Expected output
        const std::vector<float> expected_output = {
            1.0, 3.0,
            7.0, 9.0
        };

        // Test
        EXPECT_TENSOR_CLOSE(out, tf.make(output_sizes, expected_output));
    }

};

TEST_F(OpGridSampler2dTest, BilinearInterpolation) {
    test_grid_sampler_2d_bilinear();
}

TEST_F(OpGridSampler2dTest, NearestInterpolation) {
    test_grid_sampler_2d_nearest();
}

TEST_F(OpGridSampler2dTest, BorderPadding) {
    test_grid_sampler_2d_border_padding();
}

TEST_F(OpGridSampler2dTest, ReflectionPadding) {
    test_grid_sampler_2d_reflection_padding();
}

TEST_F(OpGridSampler2dTest, AlignCorners) {
    test_grid_sampler_2d_align_corners();
}

TEST_F(OpGridSampler2dTest, BilinearBorderAlign) {
    test_grid_sampler_2d_bilinear_border_align();
}
