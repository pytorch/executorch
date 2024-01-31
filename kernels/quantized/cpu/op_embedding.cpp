/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>
#include <algorithm>
#include <cinttypes>
#include <cmath>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using Scalar = exec_aten::Scalar;
using ScalarType = exec_aten::ScalarType;

namespace {

/**
 * Asserts that the parameters are valid.
 */
void check_embedding_byte_args(
    const Tensor& weight,
    const Tensor& weight_scales,
    const optional<Tensor>& opt_weight_zero_points,
    const int64_t weight_quant_min,
    const int64_t weight_quant_max,
    const Tensor& indices,
    Tensor& out) {
  ET_CHECK_MSG(
      weight.scalar_type() == ScalarType::Byte ||
          weight.scalar_type() == ScalarType::Char,
      "weight.scalar_type() %" PRId8 " is not supported:",
      static_cast<int8_t>(weight.scalar_type()));

  ET_CHECK_MSG(
      out.scalar_type() == ScalarType::Float ||
          out.scalar_type() == ScalarType::Half,
      "out.scalar_type() %" PRId8 " is not supported:",
      static_cast<int8_t>(out.scalar_type()));

  ET_CHECK_MSG(
      weight_scales.scalar_type() == out.scalar_type(),
      "weight scales scalar type %" PRId8 " does not match out.scalar_type()",
      static_cast<int8_t>(weight_scales.scalar_type()));

  if (opt_weight_zero_points.has_value()) {
    ET_CHECK_MSG(
        opt_weight_zero_points.value().scalar_type() == out.scalar_type(),
        "weight zero points scalar type %" PRId8
        " does not match out.scalar_type()",
        static_cast<int8_t>(opt_weight_zero_points.value().scalar_type()));
  }

  ET_CHECK_MSG(
      indices.scalar_type() == ScalarType::Long,
      "indices.scalar_type() %" PRId8 " is not Long only Long is supported:",
      static_cast<int8_t>(indices.scalar_type()));

  ET_CHECK_MSG(
      weight_quant_min <= weight_quant_max,
      "weight quant min: %" PRId64
      " is greater than weight quant max: %" PRId64,
      weight_quant_min,
      weight_quant_max);
}

/**
 * Retrieves the embeddings specified by indices, dequantizes them, and stores
 * them in out
 */
template <class CTYPE_WEIGHT, class CTYPE_OUT>
void embedding_byte_per_channel(
    const Tensor& weight,
    const Tensor& weight_scales,
    const optional<Tensor>& opt_weight_zero_points,
    const Tensor& indices,
    Tensor& out) {
  // An embedding layer nn.Embedding(num_embeddings, embedding_dim) has a weight
  // of shape (num_embeddings, embedding_dim).
  auto embedding_dim = weight.size(1);

  CTYPE_OUT* out_data = out.mutable_data_ptr<CTYPE_OUT>();
  const int64_t* indices_ptr = indices.const_data_ptr<int64_t>();

  const CTYPE_OUT* scales = weight_scales.const_data_ptr<CTYPE_OUT>();
  const CTYPE_OUT* zero_points = nullptr;
  if (opt_weight_zero_points.has_value()) {
    zero_points = opt_weight_zero_points.value().const_data_ptr<CTYPE_OUT>();
  }

  for (int i = 0; i < indices.numel(); i++) {
    int64_t index = indices_ptr[i];
    CTYPE_OUT zp = 0.0;
    if (opt_weight_zero_points.has_value()) {
      zp = zero_points[index];
    }
    CTYPE_OUT scale = scales[index];

    const CTYPE_WEIGHT* w_data =
        weight.data_ptr<CTYPE_WEIGHT>() + embedding_dim * index;

    for (int j = 0; j < embedding_dim; ++j) {
      out_data[j] = static_cast<CTYPE_OUT>(
          (static_cast<float>(w_data[j]) - static_cast<float>(zp)) *
          static_cast<float>(scale));
    }
    out_data += embedding_dim;
  }
}

void resize_out_tensor(
    const Tensor& weight,
    const Tensor& indices,
    Tensor& out) {
  exec_aten::SizesType expected_output_size[kTensorDimensionLimit];
  for (size_t i = 0; i < indices.dim(); i++) {
    expected_output_size[i] = indices.size(i);
  }
  const size_t embedding_dim = weight.size(1);
  expected_output_size[out.dim() - 1] = embedding_dim;

  exec_aten::ArrayRef<exec_aten::SizesType> output_size{
      expected_output_size, static_cast<size_t>(out.dim())};

  torch::executor::Error err = resize_tensor(out, output_size);
  ET_CHECK_MSG(
      err == torch::executor::Error::Ok,
      "Failed to resize out Tensor in quantized_embedding_byte_out");
}

} // namespace

/**
 * Retrieves the embeddings specified by indices, dequantizes them, and stores
 * them in out. The weight is quantized per channel, with a scale and zero_point
 * for each embedding.
 *
 * Corresponds as the out variant to torch.ops.quantized.embedding_byte
 *
 * NOTE: quant_min, quant_max, and Dtype are not used in computation, but rather
 * metadata that is passed around which can be useful for pattern matching. See
 * https://github.com/pytorch/pytorch/pull/87093#discussion_r1000841181 for more
 * info.
 */
Tensor& quantized_embedding_byte_out(
    // TODO Evaluate whether this name is appropriate for an operator that takes
    // non quant input and returns fp output
    const Tensor& weight,
    const Tensor& weight_scales,
    const optional<Tensor>& opt_weight_zero_points,
    const int64_t weight_quant_min,
    const int64_t weight_quant_max,
    const Tensor& indices,
    Tensor& out) {
  // TODO (jakeszwe): improve these to account for the size of out in relation
  // to weight and indices accounting for a possible batch dimension
  check_embedding_byte_args(
      weight,
      weight_scales,
      opt_weight_zero_points,
      weight_quant_min,
      weight_quant_max,
      indices,
      out);

  ScalarType w_type = weight.scalar_type();
  ScalarType out_type = out.scalar_type();

  ET_SWITCH_TWO_TYPES(Byte, Char, w_type, ctx, "q_embedding", CTYPE_W, [&]() {
    ET_SWITCH_TWO_TYPES(
        Float, Half, out_type, ctx, "q_embedding", CTYPE_OUT, [&]() {
          embedding_byte_per_channel<CTYPE_W, CTYPE_OUT>(
              weight, weight_scales, opt_weight_zero_points, indices, out);
        });
  });

  return out;
}

Tensor& quantized_embedding_byte_out(
    RuntimeContext& context,
    const Tensor& weight,
    const Tensor& weight_scales,
    const optional<Tensor>& opt_weight_zero_points,
    int64_t weight_quant_min,
    int64_t weight_quant_max,
    const Tensor& indices,
    Tensor& out) {
  // TODO(larryliu): Add a context arg to the real op function and remove this
  // wrapper
  (void)context;
  resize_out_tensor(weight, indices, out);
  return quantized_embedding_byte_out(
      weight,
      weight_scales,
      opt_weight_zero_points,
      weight_quant_min,
      weight_quant_max,
      indices,
      out);
}

} // namespace native
} // namespace executor
} // namespace torch
