#include "kernels.h"

#include <executorch/runtime/kernel/kernel_includes.h>
#include <algorithm>
#include <cmath>
#include <tuple>

namespace impl {
namespace HiFi {
namespace native {

using Tensor = exec_aten::Tensor;
using exec_aten::IntArrayRef;
using RuntimeContext = torch::executor::RuntimeContext;

namespace layer_norm_util {
// This function compute the product of dim[0:dim] where dim is not inclusive
size_t getLeadingDims(const Tensor& tensor, int64_t dim) {
  size_t dims = 1;
  for (size_t i = 0; i < dim; ++i) {
    dims *= tensor.size(i);
  }
  return dims;
}
} // namespace layer_norm_util

// Dequantize an int8_t/uint8_t value to a fp32 value
template <typename T>
__attribute__((always_inline)) float
dequantize(const T x, float scale, int32_t zero_point) {
  return scale * (x - zero_point);
}

void quantized_layer_norm_pt2_out(
    RuntimeContext& ctx,
    const Tensor& input,
    double in_scale,
    int64_t in_zero_point,
    const IntArrayRef normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    double eps,
    double output_scale,
    int64_t output_zero_point,
    Tensor& out) {
  // Get the raw pointers to input, output, weight, and bias
  const uint8_t* __restrict__ in_data = input.const_data_ptr<uint8_t>();
  uint8_t* __restrict__ out_data = out.mutable_data_ptr<uint8_t>();
  const float* __restrict__ weight_data = weight.const_data_ptr<float>();
  const float* __restrict__ bias_data = bias.const_data_ptr<float>();

  float output_inv_scale = XT_RECIP_S(output_scale);

  size_t last_dim = input.size(input.dim() - 1);
  size_t leading_dims = layer_norm_util::getLeadingDims(input, input.dim() - 1);

  // Visualize the input tensor as a set of 1d vectors, and compute the
  // layer_norm for each vector.
  for (size_t i = 0; i < leading_dims; ++i) {
    const uint8_t* __restrict__ x = in_data + i * last_dim;
    uint8_t* __restrict__ y = out_data + i * last_dim;

    // compute sum and squared sum. The fp32 sum can be approximated as:
    // (X_1 - in_zero_point) * in_scale + (X_2 - in_zero_point) * in_scale + ...
    // (X_N - in_zero_point) * in_scale.
    int32_t sum = 0;
    int32_t sq_sum = last_dim * in_zero_point * in_zero_point;
#pragma simd
    for (size_t j = 0; j < last_dim; ++j) {
      int32_t val = x[j];
      sum += val;
      sq_sum += val * val;
    }
    sq_sum -= (2 * sum * in_zero_point);
    sum -= (last_dim * in_zero_point);

    float mean = XT_DIV_S(XT_MUL_S(in_scale, sum), last_dim);
    float variance =
        XT_DIV_S(XT_MUL_S(sq_sum, XT_MUL_S(in_scale, in_scale)), last_dim) -
        XT_MUL_S(mean, mean);
    float inv_std = XT_RECIP_S(XT_SQRT_S(XT_ADD_S(variance, (float)eps)));

    // y = (x - mean) / std * kGamma + kBeta
#pragma simd
    for (size_t j = 0; j < last_dim; ++j) {
      // Since X is quantized, we dequantize it, compute fp32 result, and
      // quantize the result to an int8/uint8 value.
      float val = dequantize<uint8_t>(x[j], in_scale, in_zero_point);
      val = (val - mean) * inv_std * weight_data[j] + bias_data[j];
      y[j] = dequantize<uint8_t>(val, output_inv_scale, output_zero_point);
    }
  }
}

}; // namespace native
}; // namespace HiFi
}; // namespace impl