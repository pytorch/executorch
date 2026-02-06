/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/quantized/cpu/embeddingxb.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <cinttypes>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using Scalar = executorch::aten::Scalar;
using ScalarType = executorch::aten::ScalarType;

namespace {

static inline int32_t
weight_value(const unsigned char* w_data, int32_t index, int32_t weight_nbit) {
  if (weight_nbit == 2) {
    int32_t subbyte = index % 4;
    index >>= 2;
    switch (subbyte) {
      case 0:
        return (int32_t)(w_data[index] & 3) - 2;
      case 1:
        return (int32_t)((w_data[index] & 12) >> 2) - 2;
      case 2:
        return (int32_t)((w_data[index] & 48) >> 4) - 2;
      case 3:
        return (int32_t)((w_data[index] & 192) >> 6) - 2;
    }
  } else if (weight_nbit == 4) {
    int32_t odd = index & 1;
    index >>= 1;
    if (odd) {
      return (int32_t)(w_data[index] & 0x0F) - 8;
    } else {
      return (int32_t)((w_data[index] >> 4) & 0x0F) - 8;
    }
  }

  ET_CHECK_MSG(false, "invalid weight_nbit");
}

static inline int32_t get_embedding_dim(
    int32_t packed_dim,
    int32_t weight_nbit) {
  ET_CHECK_MSG(8 % weight_nbit == 0, "invalid embedding dim");
  int packed_values_per_byte = 8 / weight_nbit;
  return packed_dim * packed_values_per_byte;
}

/**
 * Asserts that the parameters are valid.
 */
void check_embedding_xbit_args(
    const Tensor& weight,
    const Tensor& weight_scales,
    const std::optional<Tensor>& opt_weight_zero_points,
    const int64_t weight_quant_min,
    const int64_t weight_quant_max,
    const Tensor& indices,
    std::optional<ScalarType> out_dtype,
    Tensor& out,
    int weight_nbit) {
  ET_CHECK_MSG(8 % weight_nbit == 0, "nbit must divide 8");

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
        // each 8b uint8 column is packed_values_per_byte columns
        get_embedding_dim(weight.size(1), weight_nbit) % num_groups == 0,
        "Number of groups must divide weight.size(1)=%zd"
        ", but got # of groups = %zd",
        weight.size(1),
        num_groups);
  }

  ET_CHECK_MSG(
      weight.scalar_type() == ScalarType::Byte,
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
          "Dimension size misatch at dim %" PRIi32
          "Weight_zero_point size = %zd"
          ", weight_scales size = %zd.",
          i,
          opt_weight_zero_points.value().size(i),
          weight_scales.size(i));
    }
  }

  ET_CHECK_MSG(
      indices.scalar_type() == ScalarType::Long ||
          indices.scalar_type() == ScalarType::Int,
      "indices.scalar_type() %" PRId8 " is not Long or Int",
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
 * them in out. Weight will always be uint8
 */
template <typename CTYPE_PARAMS, typename CTYPE_OUT, typename CTYPE_INDICES>
void embedding_xbit_per_channel(
    const Tensor& weight,
    const Tensor& weight_scales,
    const std::optional<Tensor>& opt_weight_zero_points,
    const Tensor& indices,
    Tensor& out,
    int weight_nbit) {
  auto embedding_dim = get_embedding_dim(weight.size(1), weight_nbit);

  int32_t num_groups_per_channel = 1;
  if (weight_scales.dim() == 2) {
    num_groups_per_channel = weight_scales.size(1);
  }
  int32_t group_size = embedding_dim / num_groups_per_channel;

  CTYPE_OUT* out_data = out.mutable_data_ptr<CTYPE_OUT>();
  const CTYPE_INDICES* indices_ptr = indices.const_data_ptr<CTYPE_INDICES>();

  const CTYPE_PARAMS* scales = weight_scales.const_data_ptr<CTYPE_PARAMS>();
  const CTYPE_PARAMS* zero_points = nullptr;
  if (opt_weight_zero_points.has_value()) {
    zero_points = opt_weight_zero_points.value().const_data_ptr<CTYPE_PARAMS>();
  }

  for (int i = 0; i < indices.numel(); i++) {
    CTYPE_INDICES index = indices_ptr[i];
    // If using groupwise embedding
    int32_t qparams_index = index * num_groups_per_channel;
    CTYPE_PARAMS zp = 0.0;
    const CTYPE_PARAMS* scale_ptr = scales + qparams_index;
    const CTYPE_PARAMS* zero_points_ptr = nullptr;
    if (opt_weight_zero_points.has_value()) {
      zero_points_ptr = zero_points + qparams_index;
    }

    const uint8_t* w_data =
        weight.const_data_ptr<uint8_t>() + weight.size(1) * index;

    for (int j = 0; j < embedding_dim; ++j) {
      int32_t group_id = j / group_size;
      const CTYPE_PARAMS scale = scale_ptr[group_id];
      if (opt_weight_zero_points.has_value()) {
        zp = zero_points_ptr[group_id];
      }
      out_data[j] = static_cast<CTYPE_OUT>(
          (static_cast<float>(weight_value(w_data, j, weight_nbit)) -
           static_cast<float>(zp)) *
          static_cast<float>(scale));
    }
    out_data += embedding_dim;
  }
}

void resize_out_tensor(
    const Tensor& weight,
    const Tensor& indices,
    Tensor& out,
    int weight_nbit) {
  executorch::aten::SizesType expected_output_size[kTensorDimensionLimit];
  for (size_t i = 0; i < indices.dim(); i++) {
    expected_output_size[i] = indices.size(i);
  }
  const size_t embedding_dim = get_embedding_dim(weight.size(1), weight_nbit);
  expected_output_size[out.dim() - 1] = embedding_dim;

  executorch::aten::ArrayRef<executorch::aten::SizesType> output_size{
      expected_output_size, static_cast<size_t>(out.dim())};

  torch::executor::Error err = resize_tensor(out, output_size);
  ET_CHECK_MSG(
      err == torch::executor::Error::Ok,
      "Failed to resize out Tensor in quantized_embedding_xbit_out");
}

} // namespace

/**
 * Retrieves the embeddings specified by indices, dequantizes them, and stores
 * them in out. The weight is quantized per channel, with a scale and zero_point
 * for each embedding.
 *
 * Corresponds as the out variant to torch.ops.quantized.embedding_xbit
 *
 * NOTE: quant_min, quant_max, and Dtype are not used in computation, but rather
 * metadata that is passed around which can be useful for pattern matching. See
 * https://github.com/pytorch/pytorch/pull/87093#discussion_r1000841181 for more
 * info.
 */
Tensor& quantized_embedding_xbit_out(
    // TODO Evaluate whether this name is appropriate for an operator that takes
    // non quant input and returns fp output
    KernelRuntimeContext& ctx,
    const Tensor& weight,
    const Tensor& weight_scales,
    const std::optional<Tensor>& opt_weight_zero_points,
    const int64_t weight_quant_min,
    const int64_t weight_quant_max,
    const Tensor& indices,
    Tensor& out,
    int weight_nbit) {
  ScalarType out_type = out.scalar_type();

  resize_out_tensor(weight, indices, out, weight_nbit);

  // TODO (jakeszwe): improve these to account for the size of out in relation
  // to weight and indices accounting for a possible batch dimension
  check_embedding_xbit_args(
      weight,
      weight_scales,
      opt_weight_zero_points,
      weight_quant_min,
      weight_quant_max,
      indices,
      out_type,
      out,
      weight_nbit);

  constexpr auto name = "quantized_decomposed::embedding_xbit.out";
  ScalarType indices_type = indices.scalar_type();
  ET_SWITCH_TWO_TYPES(Float, Half, out_type, ctx, name, CTYPE_OUT, [&]() {
    ET_SWITCH_TWO_TYPES(Int, Long, indices_type, ctx, name, CTYPE_IDX, [&]() {
      embedding_xbit_per_channel<CTYPE_OUT, CTYPE_OUT, CTYPE_IDX>(
          weight,
          weight_scales,
          opt_weight_zero_points,
          indices,
          out,
          weight_nbit);
    });
  });

  return out;
}

Tensor& quantized_embedding_xbit_out(
    const Tensor& weight,
    const Tensor& weight_scales,
    const std::optional<Tensor>& opt_weight_zero_points,
    int64_t weight_quant_min,
    int64_t weight_quant_max,
    const Tensor& indices,
    Tensor& out,
    int weight_nbit) {
  // TODO(larryliu): Add a context arg to the real op function and remove this
  // wrapper
  KernelRuntimeContext context;
  auto& res = quantized_embedding_xbit_out(
      context,
      weight,
      weight_scales,
      opt_weight_zero_points,
      weight_quant_min,
      weight_quant_max,
      indices,
      out,
      weight_nbit);
  ET_CHECK(context.failure_state() == Error::Ok);
  return res;
}

Tensor& quantized_embedding_xbit_dtype_out(
    // TODO Evaluate whether this name is appropriate for an operator that takes
    // non quant input and returns fp output
    KernelRuntimeContext& ctx,
    const Tensor& weight,
    const Tensor& weight_scales,
    const std::optional<Tensor>& opt_weight_zero_points,
    const int64_t weight_quant_min,
    const int64_t weight_quant_max,
    const Tensor& indices,
    std::optional<ScalarType> out_dtype,
    Tensor& out,
    int weight_nbit) {
  resize_out_tensor(weight, indices, out, weight_nbit);

  // TODO (jakeszwe): improve these to account for the size of out in relation
  // to weight and indices accounting for a possible batch dimension
  check_embedding_xbit_args(
      weight,
      weight_scales,
      opt_weight_zero_points,
      weight_quant_min,
      weight_quant_max,
      indices,
      out_dtype,
      out,
      weight_nbit);

  ScalarType params_type = weight_scales.scalar_type();
  ScalarType out_type = out.scalar_type();

  constexpr auto name = "quantized_decomposed::embedding_xbit.dtype_out";
  ScalarType indices_type = indices.scalar_type();
  ET_SWITCH_TWO_TYPES(Float, Half, params_type, ctx, name, CTYPE_P, [&]() {
    ET_SWITCH_TWO_TYPES(Float, Half, out_type, ctx, name, CTYPE_OUT, [&]() {
      ET_SWITCH_TWO_TYPES(Int, Long, indices_type, ctx, name, CTYPE_IDX, [&]() {
        embedding_xbit_per_channel<CTYPE_P, CTYPE_OUT, CTYPE_IDX>(
            weight,
            weight_scales,
            opt_weight_zero_points,
            indices,
            out,
            weight_nbit);
      });
    });
  });

  return out;
}

Tensor& quantized_embedding_xbit_dtype_out(
    const Tensor& weight,
    const Tensor& weight_scales,
    const std::optional<Tensor>& opt_weight_zero_points,
    int64_t weight_quant_min,
    int64_t weight_quant_max,
    const Tensor& indices,
    std::optional<ScalarType> out_dtype,
    Tensor& out,
    int weight_nbit) {
  // TODO(larryliu): Add a context arg to the real op function and remove this
  // wrapper
  KernelRuntimeContext context;
  auto& res = quantized_embedding_xbit_dtype_out(
      context,
      weight,
      weight_scales,
      opt_weight_zero_points,
      weight_quant_min,
      weight_quant_max,
      indices,
      out_dtype,
      out,
      weight_nbit);
  ET_CHECK(context.failure_state() == Error::Ok);
  return res;
}

} // namespace native
} // namespace executor
} // namespace torch
