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
    exec_aten::optional<ScalarType> out_dtype,
    Tensor& out) {
  ET_CHECK_MSG(
      weight.dim() == 2, "weight must be 2D but got() %zd dims", weight.dim());

  ET_CHECK_MSG(
      weight_scales.dim() == 1 || weight_scales.dim() == 2,
      "weight_scales must be 1D or 2D but got() %zd dims",
      weight_scales.dim());

  ET_CHECK_MSG(
      weight_scales.size(0) == weight.size(0),
      "Number of scales must be == weight.size(0)=%zd"
      ", but got %zd",
      weight_scales.size(0),
      weight.size(0));

  if (weight_scales.dim() == 2) {
    auto num_groups = weight_scales.size(1);
    ET_CHECK_MSG(
        weight.size(1) % num_groups == 0,
        "Number of groups must divide weight.size(1)=%zd"
        ", but got # of groups = %zd",
        weight.size(1),
        num_groups);
  }

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
      weight_scales.scalar_type() == ScalarType::Float ||
          weight_scales.scalar_type() == ScalarType::Half,
      "weight_scales.scalar_type() %" PRId8 " is not supported:",
      static_cast<int8_t>(weight_scales.scalar_type()));

  if (opt_weight_zero_points.has_value()) {
    ET_CHECK_MSG(
        opt_weight_zero_points.value().dim() == weight_scales.dim(),
        "weight_zero_points's rank match that of weight_scales. "
        "weight_zero_points rank: %" PRId8 ", weight_scales rank: %" PRId8,
        static_cast<int8_t>(opt_weight_zero_points.value().dim()),
        static_cast<int8_t>(weight_scales.dim()));

    ET_CHECK_MSG(
        opt_weight_zero_points.value().scalar_type() == out.scalar_type(),
        "weight zero points scalar type %" PRId8
        " does not match out.scalar_type()",
        static_cast<int8_t>(opt_weight_zero_points.value().scalar_type()));

    for (int32_t i = 0; i < weight_scales.dim(); ++i) {
      ET_CHECK_MSG(
          opt_weight_zero_points.value().size(i) == weight_scales.size(i),
          "Dimension size misatch at dim %" PRId8
          "Weight_zero_point size = %zd"
          ", weight_scales size = %zd.",
          i,
          opt_weight_zero_points.value().size(i),
          weight_scales.size(i));
    }
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

  if (out_dtype.has_value()) {
    ET_CHECK_MSG(
        out.scalar_type() == out_dtype.value(),
        "output_dtype must match the dtype of the out tensor");
  }
}

/**
 * Retrieves the embeddings specified by indices, dequantizes them, and stores
 * them in out
 */
template <typename CTYPE_WEIGHT, typename CTYPE_PARAMS, typename CTYPE_OUT>
void embedding_byte_per_channel(
    const Tensor& weight,
    const Tensor& weight_scales,
    const optional<Tensor>& opt_weight_zero_points,
    const Tensor& indices,
    Tensor& out) {
  // An embedding layer nn.Embedding(num_embeddings, embedding_dim) has a
  // weight of shape (num_embeddings, embedding_dim).
  auto embedding_dim = weight.size(1);

  int32_t num_groups_per_channel = 1;
  if (weight_scales.dim() == 2) {
    num_groups_per_channel = weight_scales.size(1);
  }
  int32_t group_size = weight.size(1) / num_groups_per_channel;

  CTYPE_OUT* out_data = out.mutable_data_ptr<CTYPE_OUT>();
  const int64_t* indices_ptr = indices.const_data_ptr<int64_t>();

  const CTYPE_PARAMS* scales = weight_scales.const_data_ptr<CTYPE_PARAMS>();
  const CTYPE_PARAMS* zero_points = nullptr;
  if (opt_weight_zero_points.has_value()) {
    zero_points = opt_weight_zero_points.value().const_data_ptr<CTYPE_PARAMS>();
  }

  for (int i = 0; i < indices.numel(); i++) {
    int64_t index = indices_ptr[i];
    // If using groupwise embedding
    int32_t qparams_index = index * num_groups_per_channel;
    CTYPE_PARAMS zp = 0.0;
    const CTYPE_PARAMS* scale_ptr = scales + qparams_index;
    const CTYPE_PARAMS* zero_points_ptr = nullptr;
    if (opt_weight_zero_points.has_value()) {
      zero_points_ptr = zero_points + qparams_index;
    }

    const CTYPE_WEIGHT* w_data =
        weight.const_data_ptr<CTYPE_WEIGHT>() + embedding_dim * index;

    for (int j = 0; j < embedding_dim; ++j) {
      int32_t group_id = j / group_size;
      const CTYPE_PARAMS scale = scale_ptr[group_id];
      if (opt_weight_zero_points.has_value()) {
        zp = zero_points_ptr[group_id];
      }
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
  ScalarType w_type = weight.scalar_type();
  ScalarType out_type = out.scalar_type();

  // TODO (jakeszwe): improve these to account for the size of out in relation
  // to weight and indices accounting for a possible batch dimension
  check_embedding_byte_args(
      weight,
      weight_scales,
      opt_weight_zero_points,
      weight_quant_min,
      weight_quant_max,
      indices,
      out_type,
      out);

  constexpr auto name = "quantized_decomposed::embedding_byte.out";
  ET_SWITCH_TWO_TYPES(Byte, Char, w_type, ctx, name, CTYPE_W, [&]() {
    ET_SWITCH_TWO_TYPES(Float, Half, out_type, ctx, name, CTYPE_OUT, [&]() {
      embedding_byte_per_channel<CTYPE_W, CTYPE_OUT, CTYPE_OUT>(
          weight, weight_scales, opt_weight_zero_points, indices, out);
    });
  });

  return out;
}

Tensor& quantized_embedding_byte_out(
    KernelRuntimeContext& context,
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

Tensor& quantized_embedding_byte_dtype_out(
    // TODO Evaluate whether this name is appropriate for an operator that takes
    // non quant input and returns fp output
    const Tensor& weight,
    const Tensor& weight_scales,
    const optional<Tensor>& opt_weight_zero_points,
    const int64_t weight_quant_min,
    const int64_t weight_quant_max,
    const Tensor& indices,
    exec_aten::optional<ScalarType> out_dtype,
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
      out_dtype,
      out);

  ScalarType weight_type = weight.scalar_type();
  ScalarType params_type = weight_scales.scalar_type();
  ScalarType out_type = out.scalar_type();

  constexpr auto name = "quantized_decomposed::embedding_byte.dtype_out";
  ET_SWITCH_TWO_TYPES(Byte, Char, weight_type, ctx, name, CTYPE_W, [&]() {
    ET_SWITCH_TWO_TYPES(Float, Half, params_type, ctx, name, CTYPE_P, [&]() {
      ET_SWITCH_TWO_TYPES(Float, Half, out_type, ctx, name, CTYPE_OUT, [&]() {
        embedding_byte_per_channel<CTYPE_W, CTYPE_P, CTYPE_OUT>(
            weight, weight_scales, opt_weight_zero_points, indices, out);
      });
    });
  });

  return out;
}

Tensor& quantized_embedding_byte_dtype_out(
    KernelRuntimeContext& context,
    const Tensor& weight,
    const Tensor& weight_scales,
    const optional<Tensor>& opt_weight_zero_points,
    int64_t weight_quant_min,
    int64_t weight_quant_max,
    const Tensor& indices,
    exec_aten::optional<ScalarType> out_dtype,
    Tensor& out) {
  // TODO(larryliu): Add a context arg to the real op function and remove this
  // wrapper
  (void)context;
  resize_out_tensor(weight, indices, out);
  return quantized_embedding_byte_dtype_out(
      weight,
      weight_scales,
      opt_weight_zero_points,
      weight_quant_min,
      weight_quant_max,
      indices,
      out_dtype,
      out);
}

} // namespace native
} // namespace executor
} // namespace torch
