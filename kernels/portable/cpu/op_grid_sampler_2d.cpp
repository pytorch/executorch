//
//  Copyright (c) 2025 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>

#include <cmath>
#include <cstdint>
#include <algorithm>

#include <c10/util/irange.h>
 
namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = executorch::aten::ScalarType;
using SizesType = executorch::aten::SizesType;

// Transform normalized coordinates to pixel space
inline float unnormalize_coord(float coord, int64_t size, bool align_corners) {
    if (align_corners) {
        // -1 and 1 correspond to the centers of the first and last pixels
        return ((coord + 1.0f) / 2.0f) * (size - 1);
    } else {
        // -1 and 1 correspond to the boundary of the image
        return ((coord + 1.0f) * size - 1.0f) / 2.0f;
    }
}

// Compute source index and interpolation weight
inline void compute_source_index_and_weight(
    float coord, int64_t size, bool align_corners,
    int64_t& index, float& weight) {
    
    float real_coord = unnormalize_coord(coord, size, align_corners);
    index = std::floor(real_coord);
    weight = real_coord - index;
}

// Apply reflective padding to handle out-of-bounds coordinates
inline int64_t reflect_coord(int64_t coord, int64_t size) {
    if (size <= 1) return 0;
    
    int64_t double_size = 2 * size - 2;
    if (double_size <= 0) return 0;
    
    // Handle negative coordinates
    int64_t abs_coord = std::abs(coord);
    abs_coord = abs_coord % double_size;
    if (abs_coord >= size) {
        abs_coord = double_size - abs_coord;
    }
    
    return abs_coord;
}

// Get pixel value with proper boundary handling based on padding mode
template <typename T>
T get_pixel_value(
    const T* input_data,
    int64_t n, int64_t c, int64_t h, int64_t w,
    int64_t height, int64_t width,
    int64_t padding_mode) {
    
    // Handle out-of-bounds coordinates
    if (h < 0 || h >= height || w < 0 || w >= width) {
        if (padding_mode == 0) { // Zeros
            return 0;
        } else if (padding_mode == 1) { // Border
            h = std::min(std::max(h, static_cast<int64_t>(0)), height - 1);
            w = std::min(std::max(w, static_cast<int64_t>(0)), width - 1);
        } else if (padding_mode == 2) { // Reflection
            h = reflect_coord(h, height);
            w = reflect_coord(w, width);
        }
    }
    
    // Calculate offset using strides for memory access
    const int64_t batch_stride = c * height * width;
    const int64_t channel_stride = height * width;
    const int64_t height_stride = width;
    
    return input_data[n * batch_stride + c * channel_stride + h * height_stride + w];
}

// Process grid sampling with specified input, grid, and modes
template <typename T>
void grid_sampler_2d_impl(
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    Tensor& out) {

    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t inp_H = input.size(2);
    const int64_t inp_W = input.size(3);
    const int64_t out_H = grid.size(1);
    const int64_t out_W = grid.size(2);

    const T* input_data = input.data_ptr<T>();
    T* out_data = out.data_ptr<T>();
    const float* grid_data = grid.data_ptr<float>();

    // Calculate output tensor strides for indexing
    const int64_t out_batch_stride = C * out_H * out_W;
    const int64_t out_channel_stride = out_H * out_W;
    const int64_t out_height_stride = out_W;
    
    // Calculate grid tensor strides based on its actual dimensions
    const int64_t grid_batch_stride = out_H * out_W * 2;
    const int64_t grid_height_stride = out_W * 2;
    const int64_t grid_width_stride = 2;

    // Process each output pixel
    for (const auto n : c10::irange(N)) {
        for (const auto c : c10::irange(C)) {
            for (const auto h : c10::irange(out_H)) {
                for (const auto w : c10::irange(out_W)) {

                    // Get grid coordinates (x, y) with stride calculation
                    const int64_t grid_offset = n * grid_batch_stride + h * grid_height_stride + w * grid_width_stride;
                    const float x = grid_data[grid_offset];
                    const float y = grid_data[grid_offset + 1];
                    
                    // Calculate output index
                    const int64_t out_idx = n * out_batch_stride + c * out_channel_stride + 
                                           h * out_height_stride + w;
                    
                    // Apply interpolation method
                    if (interpolation_mode == 0) { // Bilinear
                        // Calculate corner indices and weights
                        int64_t ix_nw;
                        float lambda_x;
                        compute_source_index_and_weight(x, inp_W, align_corners, ix_nw, lambda_x);
                        int64_t iy_nw;
                        float lambda_y;
                        compute_source_index_and_weight(y, inp_H, align_corners, iy_nw, lambda_y);
                        
                        // Calculate bilinear weights
                        float w_nw = (1 - lambda_x) * (1 - lambda_y);
                        float w_ne = lambda_x * (1 - lambda_y);
                        float w_sw = (1 - lambda_x) * lambda_y;
                        float w_se = lambda_x * lambda_y;
                        
                        // Get corner pixel values with boundary checking
                        T nw = get_pixel_value(input_data, n, c, iy_nw, ix_nw, inp_H, inp_W, padding_mode);
                        T ne = get_pixel_value(input_data, n, c, iy_nw, ix_nw + 1, inp_H, inp_W, padding_mode);
                        T sw = get_pixel_value(input_data, n, c, iy_nw + 1, ix_nw, inp_H, inp_W, padding_mode);
                        T se = get_pixel_value(input_data, n, c, iy_nw + 1, ix_nw + 1, inp_H, inp_W, padding_mode);
                        
                        // Perform bilinear interpolation (weighted sum)
                        out_data[out_idx] = static_cast<T>(nw * w_nw + ne * w_ne + sw * w_sw + se * w_se);
                    } 
                    else if (interpolation_mode == 1) { // Nearest
                        // Convert to pixel space and round to nearest pixel
                        float ix = unnormalize_coord(x, inp_W, align_corners);
                        float iy = unnormalize_coord(y, inp_H, align_corners);
                        
                        int64_t ix_nearest = static_cast<int64_t>(std::round(ix));
                        int64_t iy_nearest = static_cast<int64_t>(std::round(iy));
                        
                        // Get nearest pixel value
                        out_data[out_idx] = get_pixel_value(
                            input_data, n, c, iy_nearest, ix_nearest, inp_H, inp_W, padding_mode);
                    }
                    else if (interpolation_mode == 2) { // Bicubic (not implemented)
                        out_data[out_idx] = 0;
                    }
                }
            }
        }
    }
}

// Main grid_sampler_2d function that validates inputs and dispatches to implementation
Tensor& grid_sampler_2d_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    Tensor& out) {

    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t out_H = grid.size(1);
    const int64_t out_W = grid.size(2);

    // Check for 4D input and grid
    ET_KERNEL_CHECK(ctx, (input.dim() == 4), InvalidArgument, out);
    ET_KERNEL_CHECK(ctx, (grid.dim() == 4), InvalidArgument, out);
    ET_KERNEL_CHECK(ctx, (grid.size(3) == 2), InvalidArgument, out);
    
    // Check that grid is float type
    ET_KERNEL_CHECK(ctx, (grid.scalar_type() == ScalarType::Float), InvalidArgument, out);

    // Check interpolation mode is valid (0=bilinear, 1=nearest, 2=bicubic)
    ET_KERNEL_CHECK(ctx, (interpolation_mode >= 0 && interpolation_mode <= 2), InvalidArgument, out);
    
    // Check padding mode is valid (0=zeros, 1=border, 2=reflection)
    ET_KERNEL_CHECK(ctx, (padding_mode >= 0 && padding_mode <= 2), InvalidArgument, out);

    // Check for output shape
    ET_KERNEL_CHECK(ctx, (out.size(0) == N && out.size(1) == C && out.size(2) == out_H && out.size(3) == out_W), 
                   InvalidArgument, out);

    // Dispatch based on input scalar type
    ET_SWITCH_REAL_TYPES(input.scalar_type(), ctx, "grid_sampler_2d.out", T, [&]() {
        grid_sampler_2d_impl<T>(input, grid, interpolation_mode, padding_mode, align_corners, out);
    });

    return out;
}
 
} // namespace native
} // namespace executor
} // namespace torch
