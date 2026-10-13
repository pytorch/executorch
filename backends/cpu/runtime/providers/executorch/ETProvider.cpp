// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/cpu/runtime/providers/executorch/ETProvider.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <stdexcept>

namespace executorch::backends::cpu {
namespace {
using namespace executorch::runtime;
struct KernelSpec {
  const char* target = nullptr;
  const char* kernel = nullptr;
  size_t inputs = 0;
  size_t outputs = 0;
  // Digit string selecting which node inputs reach the kernel, in order.
  // Null passes every input. Factory kernels (full, empty, ...) ignore
  // dtype/device kwargs, so their rows list only the stacked indices.
  const char* arg_map = nullptr;
  bool tensor_list_output = false;
  // Schema positions declared float or float?, rather than Scalar.
  std::string_view float_args{};
};

aten::ScalarType to_aten(ptn::ScalarType dtype) {
  switch (dtype) {
    case ptn::ScalarType::Byte:
      return aten::ScalarType::Byte;
    case ptn::ScalarType::Char:
      return aten::ScalarType::Char;
    case ptn::ScalarType::Short:
      return aten::ScalarType::Short;
    case ptn::ScalarType::Int:
      return aten::ScalarType::Int;
    case ptn::ScalarType::Long:
      return aten::ScalarType::Long;
    case ptn::ScalarType::Half:
      return aten::ScalarType::Half;
    case ptn::ScalarType::Float:
      return aten::ScalarType::Float;
    case ptn::ScalarType::Double:
      return aten::ScalarType::Double;
    case ptn::ScalarType::Bool:
      return aten::ScalarType::Bool;
    case ptn::ScalarType::BFloat16:
      return aten::ScalarType::BFloat16;
    case ptn::ScalarType::UInt16:
      return aten::ScalarType::UInt16;
    case ptn::ScalarType::UInt32:
      return aten::ScalarType::UInt32;
    case ptn::ScalarType::UInt64:
      return aten::ScalarType::UInt64;
  }
  throw std::invalid_argument("Unsupported CPU scalar type");
}

// One route per linked portable kernel. Targets pair functional overloads
// with their out-variant kernels; arities come from the ATen schemas and
// factory arg maps from the kernel implementations.
constexpr auto kKernels = std::to_array<KernelSpec>({
    {"torch.ops.aten._adaptive_avg_pool2d.default",
     "aten::_adaptive_avg_pool2d.out",
     2,
     1,
     nullptr},
    {"torch.ops.aten._cdist_forward.default",
     "aten::_cdist_forward.out",
     4,
     1,
     nullptr,
     false,
     "2"},
    {"torch.ops.aten._conj_physical.default",
     "aten::_conj_physical.out",
     1,
     1,
     nullptr},
    {"torch.ops.aten._log_softmax.default",
     "aten::_log_softmax.out",
     3,
     1,
     nullptr},
    {"torch.ops.aten._native_batch_norm_legit.no_stats",
     "aten::_native_batch_norm_legit.no_stats_out",
     6,
     3,
     nullptr,
     false,
     "45"},
    {"torch.ops.aten._native_batch_norm_legit_no_training.default",
     "aten::_native_batch_norm_legit_no_training.out",
     7,
     3,
     nullptr,
     false,
     "56"},
    {"torch.ops.aten._pdist_forward.default",
     "aten::_pdist_forward.out",
     2,
     1,
     nullptr,
     false,
     "1"},
    {"torch.ops.aten._softmax.default", "aten::_softmax.out", 3, 1, nullptr},
    {"torch.ops.aten._to_copy.default", "aten::_to_copy.out", 7, 1, "056"},
    {"torch.ops.aten._upsample_bilinear2d_aa.default",
     "aten::_upsample_bilinear2d_aa.out",
     5,
     1,
     nullptr,
     false,
     "34"},
    {"torch.ops.aten.abs.default", "aten::abs.out", 1, 1, nullptr},
    {"torch.ops.aten.acos.default", "aten::acos.out", 1, 1, nullptr},
    {"torch.ops.aten.acosh.default", "aten::acosh.out", 1, 1, nullptr},
    {"torch.ops.aten.add.Scalar", "aten::add.Scalar_out", 3, 1, nullptr},
    {"torch.ops.aten.add.Tensor", "aten::add.out", 3, 1, nullptr},
    {"torch.ops.aten.addmm.default", "aten::addmm.out", 5, 1, nullptr},
    {"torch.ops.aten.alias_copy.default",
     "aten::alias_copy.out",
     1,
     1,
     nullptr},
    {"torch.ops.aten.amax.default", "aten::amax.out", 3, 1, nullptr},
    {"torch.ops.aten.amin.default", "aten::amin.out", 3, 1, nullptr},
    {"torch.ops.aten.any.default", "aten::any.all_out", 1, 1, nullptr},
    {"torch.ops.aten.any.dim", "aten::any.out", 3, 1, nullptr},
    {"torch.ops.aten.any.dims", "aten::any.dims_out", 3, 1, nullptr},
    {"torch.ops.aten.arange.default", "aten::arange.out", 5, 1, "0"},
    {"torch.ops.aten.arange.start_step", "aten::arange.start_out", 7, 1, "012"},
    {"torch.ops.aten.argmax.default", "aten::argmax.out", 3, 1, nullptr},
    {"torch.ops.aten.argmin.default", "aten::argmin.out", 3, 1, nullptr},
    {"torch.ops.aten.as_strided_copy.default",
     "aten::as_strided_copy.out",
     4,
     1,
     nullptr},
    {"torch.ops.aten.asin.default", "aten::asin.out", 1, 1, nullptr},
    {"torch.ops.aten.asinh.default", "aten::asinh.out", 1, 1, nullptr},
    {"torch.ops.aten.atan.default", "aten::atan.out", 1, 1, nullptr},
    {"torch.ops.aten.atan2.default", "aten::atan2.out", 2, 1, nullptr},
    {"torch.ops.aten.atanh.default", "aten::atanh.out", 1, 1, nullptr},
    {"torch.ops.aten.avg_pool2d.default",
     "aten::avg_pool2d.out",
     7,
     1,
     nullptr},
    {"torch.ops.aten.bitwise_and.Scalar",
     "aten::bitwise_and.Scalar_out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.bitwise_and.Tensor",
     "aten::bitwise_and.Tensor_out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.bitwise_left_shift.Tensor",
     "aten::bitwise_left_shift.Tensor_out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.bitwise_left_shift.Tensor_Scalar",
     "aten::bitwise_left_shift.Tensor_Scalar_out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.bitwise_not.default",
     "aten::bitwise_not.out",
     1,
     1,
     nullptr},
    {"torch.ops.aten.bitwise_or.Scalar",
     "aten::bitwise_or.Scalar_out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.bitwise_or.Tensor",
     "aten::bitwise_or.Tensor_out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.bitwise_right_shift.Tensor",
     "aten::bitwise_right_shift.Tensor_out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.bitwise_right_shift.Tensor_Scalar",
     "aten::bitwise_right_shift.Tensor_Scalar_out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.bitwise_xor.Scalar",
     "aten::bitwise_xor.Scalar_out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.bitwise_xor.Tensor",
     "aten::bitwise_xor.Tensor_out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.bmm.default", "aten::bmm.out", 2, 1, nullptr},
    {"torch.ops.aten.bucketize.Scalar",
     "aten::bucketize.Scalar_out",
     4,
     1,
     nullptr},
    {"torch.ops.aten.bucketize.Tensor",
     "aten::bucketize.Tensor_out",
     4,
     1,
     nullptr},
    {"torch.ops.aten.cat.default", "aten::cat.out", 2, 1, nullptr},
    {"torch.ops.aten.ceil.default", "aten::ceil.out", 1, 1, nullptr},
    {"torch.ops.aten.clamp.Tensor", "aten::clamp.Tensor_out", 3, 1, nullptr},
    {"torch.ops.aten.clamp.default", "aten::clamp.out", 3, 1, nullptr},
    {"torch.ops.aten.clone.default", "aten::clone.out", 2, 1, nullptr},
    {"torch.ops.aten.constant_pad_nd.default",
     "aten::constant_pad_nd.out",
     3,
     1,
     nullptr},
    {"torch.ops.aten.convolution.default",
     "aten::convolution.out",
     9,
     1,
     nullptr},
    {"torch.ops.aten.copy.default", "aten::copy.out", 3, 1, nullptr},
    {"torch.ops.aten.cos.default", "aten::cos.out", 1, 1, nullptr},
    {"torch.ops.aten.cosh.default", "aten::cosh.out", 1, 1, nullptr},
    {"torch.ops.aten.cumsum.default", "aten::cumsum.out", 3, 1, nullptr},
    {"torch.ops.aten.detach_copy.default",
     "aten::detach_copy.out",
     1,
     1,
     nullptr},
    {"torch.ops.aten.diagonal_copy.default",
     "aten::diagonal_copy.out",
     4,
     1,
     nullptr},
    {"torch.ops.aten.div.Scalar", "aten::div.Scalar_out", 2, 1, nullptr},
    {"torch.ops.aten.div.Scalar_mode",
     "aten::div.Scalar_mode_out",
     3,
     1,
     nullptr},
    {"torch.ops.aten.div.Tensor", "aten::div.out", 2, 1, nullptr},
    {"torch.ops.aten.div.Tensor_mode", "aten::div.out_mode", 3, 1, nullptr},
    {"torch.ops.aten.elu.default", "aten::elu.out", 4, 1, nullptr},
    {"torch.ops.aten.embedding.default", "aten::embedding.out", 5, 1, nullptr},
    {"torch.ops.aten.empty.memory_format", "aten::empty.out", 6, 1, "05"},
    {"torch.ops.aten.eq.Scalar", "aten::eq.Scalar_out", 2, 1, nullptr},
    {"torch.ops.aten.eq.Tensor", "aten::eq.Tensor_out", 2, 1, nullptr},
    {"torch.ops.aten.erf.default", "aten::erf.out", 1, 1, nullptr},
    {"torch.ops.aten.exp.default", "aten::exp.out", 1, 1, nullptr},
    {"torch.ops.aten.expand_copy.default",
     "aten::expand_copy.out",
     3,
     1,
     nullptr},
    {"torch.ops.aten.expm1.default", "aten::expm1.out", 1, 1, nullptr},
    {"torch.ops.aten.fill.Scalar", "aten::fill.Scalar_out", 2, 1, nullptr},
    {"torch.ops.aten.fill.Tensor", "aten::fill.Tensor_out", 2, 1, nullptr},
    {"torch.ops.aten.flip.default", "aten::flip.out", 2, 1, nullptr},
    {"torch.ops.aten.floor.default", "aten::floor.out", 1, 1, nullptr},
    {"torch.ops.aten.floor_divide.default",
     "aten::floor_divide.out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.fmod.Scalar", "aten::fmod.Scalar_out", 2, 1, nullptr},
    {"torch.ops.aten.fmod.Tensor", "aten::fmod.Tensor_out", 2, 1, nullptr},
    {"torch.ops.aten.full.default", "aten::full.out", 6, 1, "01"},
    {"torch.ops.aten.full_like.default", "aten::full_like.out", 7, 1, "016"},
    {"torch.ops.aten.gather.default", "aten::gather.out", 4, 1, nullptr},
    {"torch.ops.aten.ge.Scalar", "aten::ge.Scalar_out", 2, 1, nullptr},
    {"torch.ops.aten.ge.Tensor", "aten::ge.Tensor_out", 2, 1, nullptr},
    {"torch.ops.aten.gelu.default", "aten::gelu.out", 2, 1, nullptr},
    {"torch.ops.aten.glu.default", "aten::glu.out", 2, 1, nullptr},
    {"torch.ops.aten.grid_sampler_2d.default",
     "aten::grid_sampler_2d.out",
     5,
     1,
     nullptr},
    {"torch.ops.aten.gt.Scalar", "aten::gt.Scalar_out", 2, 1, nullptr},
    {"torch.ops.aten.gt.Tensor", "aten::gt.Tensor_out", 2, 1, nullptr},
    {"torch.ops.aten.hardtanh.default", "aten::hardtanh.out", 3, 1, nullptr},
    {"torch.ops.aten.index.Tensor", "aten::index.Tensor_out", 2, 1, nullptr},
    {"torch.ops.aten.index_put.default", "aten::index_put.out", 4, 1, nullptr},
    {"torch.ops.aten.index_select.default",
     "aten::index_select.out",
     3,
     1,
     nullptr},
    {"torch.ops.aten.isinf.default", "aten::isinf.out", 1, 1, nullptr},
    {"torch.ops.aten.isnan.default", "aten::isnan.out", 1, 1, nullptr},
    {"torch.ops.aten.le.Scalar", "aten::le.Scalar_out", 2, 1, nullptr},
    {"torch.ops.aten.le.Tensor", "aten::le.Tensor_out", 2, 1, nullptr},
    {"torch.ops.aten.leaky_relu.default",
     "aten::leaky_relu.out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.lift_fresh_copy.default",
     "aten::lift_fresh_copy.out",
     1,
     1,
     nullptr},
    {"torch.ops.aten.linear.default", "aten::linear.out", 3, 1, nullptr},
    {"torch.ops.aten.log.default", "aten::log.out", 1, 1, nullptr},
    {"torch.ops.aten.log10.default", "aten::log10.out", 1, 1, nullptr},
    {"torch.ops.aten.log1p.default", "aten::log1p.out", 1, 1, nullptr},
    {"torch.ops.aten.log2.default", "aten::log2.out", 1, 1, nullptr},
    {"torch.ops.aten.logical_and.default",
     "aten::logical_and.out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.logical_not.default",
     "aten::logical_not.out",
     1,
     1,
     nullptr},
    {"torch.ops.aten.logical_or.default",
     "aten::logical_or.out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.logical_xor.default",
     "aten::logical_xor.out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.logit.default",
     "aten::logit.out",
     2,
     1,
     nullptr,
     false,
     "1"},
    {"torch.ops.aten.lt.Scalar", "aten::lt.Scalar_out", 2, 1, nullptr},
    {"torch.ops.aten.lt.Tensor", "aten::lt.Tensor_out", 2, 1, nullptr},
    {"torch.ops.aten.masked_fill.Scalar",
     "aten::masked_fill.Scalar_out",
     3,
     1,
     nullptr},
    {"torch.ops.aten.masked_scatter.default",
     "aten::masked_scatter.out",
     3,
     1,
     nullptr},
    {"torch.ops.aten.masked_select.default",
     "aten::masked_select.out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.max.default", "aten::max.unary_out", 1, 1, nullptr},
    {"torch.ops.aten.max.dim", "aten::max.dim_max", 3, 2, nullptr},
    {"torch.ops.aten.max_pool2d_with_indices.default",
     "aten::max_pool2d_with_indices.out",
     6,
     2,
     nullptr},
    {"torch.ops.aten.maximum.default", "aten::maximum.out", 2, 1, nullptr},
    {"torch.ops.aten.mean.default", "aten::mean.dtype_out", 2, 1, nullptr},
    {"torch.ops.aten.mean.dim", "aten::mean.out", 4, 1, nullptr},
    {"torch.ops.aten.min.default", "aten::min.unary_out", 1, 1, nullptr},
    {"torch.ops.aten.min.dim", "aten::min.dim_min", 3, 2, nullptr},
    {"torch.ops.aten.minimum.default", "aten::minimum.out", 2, 1, nullptr},
    {"torch.ops.aten.mm.default", "aten::mm.out", 2, 1, nullptr},
    {"torch.ops.aten.mul.Scalar", "aten::mul.Scalar_out", 2, 1, nullptr},
    {"torch.ops.aten.mul.Tensor", "aten::mul.out", 2, 1, nullptr},
    {"torch.ops.aten.narrow_copy.default",
     "aten::narrow_copy.out",
     4,
     1,
     nullptr},
    {"torch.ops.aten.native_dropout.default",
     "aten::native_dropout.out",
     3,
     2,
     nullptr,
     false,
     "1"},
    {"torch.ops.aten.native_group_norm.default",
     "aten::native_group_norm.out",
     8,
     3,
     nullptr,
     false,
     "7"},
    {"torch.ops.aten.native_layer_norm.default",
     "aten::native_layer_norm.out",
     5,
     3,
     nullptr,
     false,
     "4"},
    {"torch.ops.aten.ne.Scalar", "aten::ne.Scalar_out", 2, 1, nullptr},
    {"torch.ops.aten.ne.Tensor", "aten::ne.Tensor_out", 2, 1, nullptr},
    {"torch.ops.aten.neg.default", "aten::neg.out", 1, 1, nullptr},
    {"torch.ops.aten.nonzero.default", "aten::nonzero.out", 1, 1, nullptr},
    {"torch.ops.aten.ones.default", "aten::ones.out", 5, 1, "0"},
    {"torch.ops.aten.permute_copy.default",
     "aten::permute_copy.out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.pixel_shuffle.default",
     "aten::pixel_shuffle.out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.pixel_unshuffle.default",
     "aten::pixel_unshuffle.out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.pow.Scalar", "aten::pow.Scalar_out", 2, 1, nullptr},
    {"torch.ops.aten.pow.Tensor_Scalar",
     "aten::pow.Tensor_Scalar_out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.pow.Tensor_Tensor",
     "aten::pow.Tensor_Tensor_out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.prod.default", "aten::prod.out", 2, 1, nullptr},
    {"torch.ops.aten.prod.dim_int", "aten::prod.int_out", 4, 1, nullptr},
    {"torch.ops.aten.reciprocal.default",
     "aten::reciprocal.out",
     1,
     1,
     nullptr},
    {"torch.ops.aten.reflection_pad1d.default",
     "aten::reflection_pad1d.out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.reflection_pad2d.default",
     "aten::reflection_pad2d.out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.reflection_pad3d.default",
     "aten::reflection_pad3d.out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.relu.default", "aten::relu.out", 1, 1, nullptr},
    {"torch.ops.aten.remainder.Scalar",
     "aten::remainder.Scalar_out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.remainder.Tensor",
     "aten::remainder.Tensor_out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.repeat.default", "aten::repeat.out", 2, 1, nullptr},
    {"torch.ops.aten.repeat_interleave.Tensor",
     "aten::repeat_interleave.Tensor_out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.replication_pad1d.default",
     "aten::replication_pad1d.out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.replication_pad2d.default",
     "aten::replication_pad2d.out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.replication_pad3d.default",
     "aten::replication_pad3d.out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.roll.default", "aten::roll.out", 3, 1, nullptr},
    {"torch.ops.aten.round.default", "aten::round.out", 1, 1, nullptr},
    {"torch.ops.aten.rsqrt.default", "aten::rsqrt.out", 1, 1, nullptr},
    {"torch.ops.aten.rsub.Scalar", "aten::rsub.Scalar_out", 3, 1, nullptr},
    {"torch.ops.aten.scalar_tensor.default",
     "aten::scalar_tensor.out",
     5,
     1,
     "0"},
    {"torch.ops.aten.scatter.src", "aten::scatter.src_out", 4, 1, nullptr},
    {"torch.ops.aten.scatter.value", "aten::scatter.value_out", 4, 1, nullptr},
    {"torch.ops.aten.scatter_add.default",
     "aten::scatter_add.out",
     4,
     1,
     nullptr},
    {"torch.ops.aten.select_copy.int",
     "aten::select_copy.int_out",
     3,
     1,
     nullptr},
    {"torch.ops.aten.select_scatter.default",
     "aten::select_scatter.out",
     4,
     1,
     nullptr},
    {"torch.ops.aten.sigmoid.default", "aten::sigmoid.out", 1, 1, nullptr},
    {"torch.ops.aten.sign.default", "aten::sign.out", 1, 1, nullptr},
    {"torch.ops.aten.sin.default", "aten::sin.out", 1, 1, nullptr},
    {"torch.ops.aten.sinh.default", "aten::sinh.out", 1, 1, nullptr},
    {"torch.ops.aten.slice_copy.Tensor",
     "aten::slice_copy.Tensor_out",
     5,
     1,
     nullptr},
    {"torch.ops.aten.slice_scatter.default",
     "aten::slice_scatter.out",
     6,
     1,
     nullptr},
    {"torch.ops.aten.split_copy.Tensor",
     "aten::split_copy.Tensor_out",
     3,
     1,
     nullptr,
     true},
    {"torch.ops.aten.split_with_sizes_copy.default",
     "aten::split_with_sizes_copy.out",
     3,
     1,
     nullptr,
     true},
    {"torch.ops.aten.sqrt.default", "aten::sqrt.out", 1, 1, nullptr},
    {"torch.ops.aten.squeeze_copy.dim",
     "aten::squeeze_copy.dim_out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.squeeze_copy.dims",
     "aten::squeeze_copy.dims_out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.stack.default", "aten::stack.out", 2, 1, nullptr},
    {"torch.ops.aten.sub.Scalar", "aten::sub.Scalar_out", 3, 1, nullptr},
    {"torch.ops.aten.sub.Tensor", "aten::sub.out", 3, 1, nullptr},
    {"torch.ops.aten.sum.dim_IntList", "aten::sum.IntList_out", 4, 1, nullptr},
    {"torch.ops.aten.t_copy.default", "aten::t_copy.out", 1, 1, nullptr},
    {"torch.ops.aten.tan.default", "aten::tan.out", 1, 1, nullptr},
    {"torch.ops.aten.tanh.default", "aten::tanh.out", 1, 1, nullptr},
    {"torch.ops.aten.topk.default", "aten::topk.values", 5, 2, nullptr},
    {"torch.ops.aten.transpose_copy.int",
     "aten::transpose_copy.int_out",
     3,
     1,
     nullptr},
    {"torch.ops.aten.tril.default", "aten::tril.out", 2, 1, nullptr},
    {"torch.ops.aten.trunc.default", "aten::trunc.out", 1, 1, nullptr},
    {"torch.ops.aten.unbind_copy.int",
     "aten::unbind_copy.int_out",
     2,
     1,
     nullptr,
     true},
    {"torch.ops.aten.unfold_copy.default",
     "aten::unfold_copy.out",
     4,
     1,
     nullptr},
    {"torch.ops.aten.unsqueeze_copy.default",
     "aten::unsqueeze_copy.out",
     2,
     1,
     nullptr},
    {"torch.ops.aten.upsample_bilinear2d.vec",
     "aten::upsample_bilinear2d.vec_out",
     4,
     1,
     nullptr},
    {"torch.ops.aten.upsample_nearest2d.vec",
     "aten::upsample_nearest2d.vec_out",
     3,
     1,
     nullptr},
    {"torch.ops.aten.var.correction",
     "aten::var.correction_out",
     4,
     1,
     nullptr},
    {"torch.ops.aten.var.dim", "aten::var.out", 4, 1, nullptr},
    {"torch.ops.aten.var_mean.correction",
     "aten::var_mean.correction_out",
     4,
     2,
     nullptr},
    {"torch.ops.aten.view_as_real_copy.default",
     "aten::view_as_real_copy.out",
     1,
     1,
     nullptr},
    {"torch.ops.aten.view_copy.default", "aten::view_copy.out", 2, 1, nullptr},
    {"torch.ops.aten.where.self", "aten::where.self_out", 3, 1, nullptr},
    {"torch.ops.aten.zeros.default", "aten::zeros.out", 5, 1, "0"},
    {"torch.ops.dim_order_ops._clone_dim_order.default",
     "dim_order_ops::_clone_dim_order.out",
     3,
     1,
     nullptr},
    {"torch.ops.dim_order_ops._empty_dim_order.default",
     "dim_order_ops::_empty_dim_order.out",
     6,
     1,
     "05"},
    {"torch.ops.dim_order_ops._to_dim_order_copy.default",
     "dim_order_ops::_to_dim_order_copy.out",
     7,
     1,
     "056"},
});

struct ArgumentStorage {
  std::vector<EValue> elements;
  std::vector<EValue*> pointers;
  std::vector<int64_t> integers;
  BoxedEvalueList<int64_t> list;
  std::vector<double> doubles;
  std::vector<aten::Tensor> tensor_unwrapped;
  BoxedEvalueList<aten::Tensor> tensor_list;
  std::vector<std::optional<aten::Tensor>> opt_tensor_unwrapped;
  BoxedEvalueList<std::optional<aten::Tensor>> opt_tensor_list;
  executorch::aten::ArrayRef<char> string;
  executorch::aten::ArrayRef<double> floating_list;
  EValue value;
};

struct KernelRoute {
  const KernelSpec* spec;
  OpFunction kernel;
};

class KernelRoutes {
 public:
  KernelRoutes() {
    for (const auto& spec : kKernels) {
      auto kernel = get_op_function_from_registry(spec.kernel);
      if (kernel.ok()) {
        routes_.push_back({&spec, kernel.get()});
      }
    }
    std::sort(routes_.begin(), routes_.end(), [](const auto& a, const auto& b) {
      return std::string_view(a.spec->target) < b.spec->target;
    });
  }

  const KernelRoute* find(std::string_view target) const {
    const auto route = std::lower_bound(
        routes_.begin(),
        routes_.end(),
        target,
        [](const auto& candidate, auto key) {
          return std::string_view(candidate.spec->target) < key;
        });
    return route != routes_.end() && target == route->spec->target ? &*route
                                                                   : nullptr;
  }

 private:
  std::vector<KernelRoute> routes_;
};

// How a serializer string arg converts to a boxed kernel value. Genuine
// strings (gelu "approximate") pass through; device/layout/memory_format
// spellings convert to the int codes the kernels expect.
struct StringConversion {
  bool supported;
  bool is_int;
  int64_t int_value;
};

StringConversion convert_string_arg(std::string_view text) {
  // Only memory-format spellings convert; every other string reaching a
  // kernel is genuine (gelu "approximate", div "rounding_mode"). No linked
  // kernel takes device or layout arguments: factory rows skip those kwargs,
  // and skipped target strings are validated separately.
  if (text == "torch.contiguous_format") {
    return {true, true, static_cast<int64_t>(aten::MemoryFormat::Contiguous)};
  }
  if (text == "torch.preserve_format") {
    return {true, true, static_cast<int64_t>(aten::MemoryFormat::Preserve)};
  }
  if (text == "torch.channels_last" || text == "torch.channels_last_3d") {
    return {false, false, 0};
  }
  return {true, false, 0};
}

// Empty when every string arg converts and every skipped factory kwarg is
// satisfiable; otherwise a static reason. Passed strings must convert to a
// kernel value; skipped strings must be satisfiable target spellings, since
// factory kwargs are never genuine strings.
std::string_view check_string_args(const KernelSpec& spec, const Kernel& node) {
  std::vector<bool> passed(node.inputs.size(), spec.arg_map == nullptr);
  for (const char* digits = spec.arg_map; digits != nullptr && *digits != '\0';
       ++digits) {
    if (*digits >= '0' &&
        static_cast<size_t>(*digits - '0') < node.inputs.size()) {
      passed.at(static_cast<size_t>(*digits - '0')) = true;
    }
  }
  for (size_t index = 0; index < node.inputs.size(); ++index) {
    const auto& arg = node.inputs[index].arg;
    if (arg.kind() != ptn::ArgKind::String) {
      if (!passed.at(index)) {
        switch (arg.kind()) {
          case ptn::ArgKind::None:
          case ptn::ArgKind::Int:
          case ptn::ArgKind::Float:
          case ptn::ArgKind::Bool:
          case ptn::ArgKind::ScalarType:
            break;
          case ptn::ArgKind::Tensor:
          case ptn::ArgKind::String:
          case ptn::ArgKind::IntList:
          case ptn::ArgKind::FloatList:
          case ptn::ArgKind::BoolList:
          case ptn::ArgKind::TensorList:
          case ptn::ArgKind::OptionalTensorList:
          case ptn::ArgKind::Graph:
            return "unsupported skipped factory argument";
        }
      }
      continue;
    }
    const auto& text = arg.as_string().value;
    if (passed.at(index)) {
      if (!convert_string_arg(text).supported) {
        return "unsupported target string";
      }
      continue;
    }
    if (text != "cpu" && text != "torch.strided") {
      return "unsupported factory target string";
    }
  }
  return {};
}

std::string_view check_tensor(ptn::ValueId id, const ptn::Graph& graph) {
  if (id == ptn::kInvalid || id >= graph.values.size()) {
    return "ET tensor reference is missing or out of range";
  }
  const auto& value = graph.value(id);
  if (!tensor_bytes(value).ok()) {
    return "ET tensor metadata is incompatible with the CPU delegate";
  }
  if (value.tensor_meta().quant.has_value()) {
    return "ET routes do not interpret quantization metadata";
  }
  return {};
}

std::string_view check_tensor_list(
    const std::vector<ptn::ValueId>& ids,
    bool optional,
    const ptn::Graph& graph) {
  if (ids.empty()) {
    return "ET provider tensor list is empty";
  }
  for (auto id : ids) {
    if (optional && id == ptn::kInvalid) {
      continue;
    }
    if (const auto reason = check_tensor(id, graph); !reason.empty()) {
      return reason;
    }
  }
  return {};
}

std::string_view check_argument(
    const ptn::Argument& arg,
    const ptn::Graph& graph) {
  switch (arg.kind()) {
    case ptn::ArgKind::Tensor:
      return check_tensor(arg.as_tensor().id, graph);
    case ptn::ArgKind::TensorList:
      return check_tensor_list(arg.as_tensor_list().ids, false, graph);
    case ptn::ArgKind::OptionalTensorList:
      return check_tensor_list(arg.as_optional_tensor_list().ids, true, graph);
    case ptn::ArgKind::Int:
      return arg.as_int().id == ptn::kInvalid ? std::string_view{}
                                              : "Dynamic CPU scalar";
    case ptn::ArgKind::Float:
      return arg.as_float().id == ptn::kInvalid ? std::string_view{}
                                                : "Dynamic CPU scalar";
    case ptn::ArgKind::Bool:
      return arg.as_bool().id == ptn::kInvalid ? std::string_view{}
                                               : "Dynamic CPU scalar";
    case ptn::ArgKind::IntList:
      return std::all_of(
                 arg.as_int_list().ids.begin(),
                 arg.as_int_list().ids.end(),
                 [](auto id) { return id == ptn::kInvalid; })
          ? std::string_view{}
          : "Dynamic CPU integer list";
    case ptn::ArgKind::None:
    case ptn::ArgKind::String:
    case ptn::ArgKind::ScalarType:
    case ptn::ArgKind::FloatList:
      return {};
    case ptn::ArgKind::BoolList:
    case ptn::ArgKind::Graph:
      return "Unsupported CPU argument kind";
  }
  return "Unsupported CPU argument kind";
}

std::string_view check_node(
    const KernelSpec& spec,
    const Kernel& node,
    const ptn::Graph& graph) {
  if (node.inputs.size() != spec.inputs ||
      node.outputs.size() != spec.outputs) {
    return "ET provider schema mismatch";
  }
  if (const auto reason = check_string_args(spec, node); !reason.empty()) {
    return reason;
  }
  for (const auto& input : node.inputs) {
    if (input.mutated) {
      return "ET routes reject input mutation";
    }
    if (const auto reason = check_argument(input.arg, graph); !reason.empty()) {
      return reason;
    }
  }
  for (const auto& output : node.outputs) {
    const auto expected_kind = spec.tensor_list_output
        ? ptn::OutputValueKind::TensorList
        : ptn::OutputValueKind::Tensor;
    if (output.kind != expected_kind) {
      return "ET output kind does not match the kernel schema";
    }
    const auto reason = spec.tensor_list_output
        ? check_tensor_list(output.elem_ids, false, graph)
        : check_tensor(output.value_id, graph);
    if (!reason.empty()) {
      return reason;
    }
  }
  return {};
}

struct Instruction {
  const ptn::Node* node = nullptr;
  OpFunction kernel = nullptr;
  std::vector<std::unique_ptr<ArgumentStorage>> arguments;
  std::vector<EValue*> stack;
  EValue result;
};

class ETExecutable final : public Executable {
 public:
  explicit ETExecutable(PreparationContext context) : context_(context) {}

  struct TensorStorage {
    ptn::ValueId id = ptn::kInvalid;
    bool writable = false;
    std::vector<aten::SizesType> sizes;
    std::vector<aten::DimOrderType> order;
    std::vector<aten::StridesType> strides;
    std::unique_ptr<aten::TensorImpl> tensor;
    EValue value;
  };

  Result<EValue*> tensor_value(ptn::ValueId id, bool writable) {
    const auto existing = std::find_if(
        tensors_.begin(), tensors_.end(), [id](const auto& storage) {
          return storage->id == id;
        });
    if (existing != tensors_.end()) {
      (*existing)->writable |= writable;
      return &(*existing)->value;
    }
    auto bytes = tensor_bytes(context_.graph.value(id));
    if (!bytes.ok()) {
      return bytes.error();
    }
    const auto& meta = context_.graph.value(id).tensor_meta();
    const auto& shape = meta.sizes;
    auto storage = std::make_unique<TensorStorage>();
    storage->id = id;
    storage->writable = writable;
    storage->sizes.assign(shape.begin(), shape.end());
    storage->order.resize(shape.size());
    storage->strides.resize(shape.size());
    int64_t stride = 1;
    for (size_t reverse = shape.size(); reverse > 0; --reverse) {
      const auto dim = reverse - 1;
      storage->order[dim] = dim;
      storage->strides[dim] = static_cast<aten::StridesType>(stride);
      stride *= shape[dim];
    }
    storage->tensor = std::make_unique<aten::TensorImpl>(
        to_aten(meta.dtype),
        shape.size(),
        storage->sizes.data(),
        nullptr,
        storage->order.data(),
        storage->strides.data());
    storage->value = aten::Tensor(storage->tensor.get());
    auto* value = &storage->value;
    tensors_.push_back(std::move(storage));
    return value;
  }

  Error bind_tensor_list(
      const std::vector<ptn::ValueId>& ids,
      bool optional,
      bool writable,
      ArgumentStorage& storage) {
    ET_CHECK_OR_RETURN_ERROR(
        !ids.empty(), InvalidProgram, "ET provider tensor list is empty");
    storage.pointers.reserve(ids.size());
    if (optional) {
      storage.opt_tensor_unwrapped.reserve(ids.size());
      for (const auto id : ids) {
        if (id == ptn::kInvalid) {
          storage.pointers.push_back(nullptr);
          storage.opt_tensor_unwrapped.emplace_back(std::nullopt);
          continue;
        }
        auto value = tensor_value(id, writable);
        if (!value.ok()) {
          return value.error();
        }
        storage.pointers.push_back(value.get());
        storage.opt_tensor_unwrapped.emplace_back(value.get()->toTensor());
      }
      storage.opt_tensor_list = BoxedEvalueList<std::optional<aten::Tensor>>(
          storage.pointers.data(),
          storage.opt_tensor_unwrapped.data(),
          static_cast<int>(ids.size()));
      storage.value = EValue(&storage.opt_tensor_list);
      return Error::Ok;
    }
    storage.tensor_unwrapped.reserve(ids.size());
    for (const auto id : ids) {
      ET_CHECK_OR_RETURN_ERROR(
          id != ptn::kInvalid,
          InvalidProgram,
          "ET provider tensor list has a null entry");
      auto value = tensor_value(id, writable);
      if (!value.ok()) {
        return value.error();
      }
      storage.pointers.push_back(value.get());
      storage.tensor_unwrapped.push_back(value.get()->toTensor());
    }
    storage.tensor_list = BoxedEvalueList<aten::Tensor>(
        storage.pointers.data(),
        storage.tensor_unwrapped.data(),
        static_cast<int>(ids.size()));
    storage.value = EValue(&storage.tensor_list);
    return Error::Ok;
  }

  Error initialize(const KernelRegion& region, const KernelRoutes& routes) {
    instructions_.reserve(region.nodes.size());
    tensors_.reserve(region.inputs.size() + region.outputs.size());
    for (auto id : region.nodes) {
      const auto& node = context_.graph.node(id);
      const auto* route = routes.find(node.target);
      ET_CHECK_OR_RETURN_ERROR(
          route != nullptr,
          NotSupported,
          "ET provider has no linked route for %s",
          node.target.c_str());
      const auto* spec = route->spec;
      if (const auto reason = check_node(*spec, node, context_.graph);
          !reason.empty()) {
        ET_LOG(Error, "ET provider %s: %s", node.target.c_str(), reason.data());
        return Error::NotSupported;
      }
      auto instruction = std::make_unique<Instruction>();
      instruction->node = &node;
      instruction->kernel = route->kernel;
      instruction->arguments.reserve(node.inputs.size());
      instruction->stack.reserve(node.inputs.size() + node.outputs.size() + 1);
      const size_t arg_count = spec->arg_map != nullptr
          ? std::strlen(spec->arg_map)
          : node.inputs.size();
      for (size_t position = 0; position < arg_count; ++position) {
        size_t index = position;
        if (spec->arg_map != nullptr) {
          ET_CHECK_OR_RETURN_ERROR(
              spec->arg_map[position] >= '0' &&
                  static_cast<size_t>(spec->arg_map[position] - '0') <
                      node.inputs.size(),
              InvalidProgram,
              "ET provider route has a bad argument map: %s",
              node.target.c_str());
          index = static_cast<size_t>(spec->arg_map[position] - '0');
        }
        const auto& input = node.inputs[index];
        if (input.arg.kind() == ptn::ArgKind::Tensor) {
          const auto value_id = input.arg.as_tensor().id;
          auto value = tensor_value(value_id, false);
          if (!value.ok()) {
            return value.error();
          }
          instruction->stack.push_back(value.get());
          continue;
        }
        if (input.arg.kind() == ptn::ArgKind::TensorList ||
            input.arg.kind() == ptn::ArgKind::OptionalTensorList) {
          auto argument = std::make_unique<ArgumentStorage>();
          const bool optional =
              input.arg.kind() == ptn::ArgKind::OptionalTensorList;
          const auto& ids = optional ? input.arg.as_optional_tensor_list().ids
                                     : input.arg.as_tensor_list().ids;
          const auto error = bind_tensor_list(ids, optional, false, *argument);
          if (error != Error::Ok) {
            return error;
          }
          instruction->stack.push_back(&argument->value);
          instruction->arguments.push_back(std::move(argument));
          continue;
        }
        auto argument = std::make_unique<ArgumentStorage>();
        const bool floating =
            spec->float_args.find(static_cast<char>('0' + index)) !=
            std::string_view::npos;
        const auto error = bind_argument(input.arg, *argument, floating);
        if (error != Error::Ok) {
          return error;
        }
        instruction->stack.push_back(&argument->value);
        instruction->arguments.push_back(std::move(argument));
      }
      for (const auto& output : node.outputs) {
        if (output.kind == ptn::OutputValueKind::TensorList) {
          auto argument = std::make_unique<ArgumentStorage>();
          const auto error =
              bind_tensor_list(output.elem_ids, false, true, *argument);
          if (error != Error::Ok) {
            return error;
          }
          instruction->stack.push_back(&argument->value);
          instruction->arguments.push_back(std::move(argument));
          continue;
        }
        ET_CHECK_OR_RETURN_ERROR(
            output.kind == ptn::OutputValueKind::Tensor,
            InvalidProgram,
            "ET provider requires tensor outputs: %s",
            node.target.c_str());
        auto value = tensor_value(output.value_id, true);
        if (!value.ok()) {
          return value.error();
        }
        instruction->stack.push_back(value.get());
      }
      // ExecuTorch's boxed ABI reserves one return slot even for out kernels.
      instruction->stack.push_back(&instruction->result);
      instructions_.push_back(std::move(instruction));
    }
    return Error::Ok;
  }

  Error reshape() override {
    return Error::Ok;
  }

  Error bind(const Buffer&) override {
    bound_ = false;
    for (const auto& storage : tensors_) {
      ET_CHECK_OR_RETURN_ERROR(
          context_.buffers.at(storage->id)
              .accepts(storage->tensor->nbytes(), storage->writable),
          InvalidArgument,
          "ET tensor lacks required alignment/storage: %s",
          context_.graph.value(storage->id).name.c_str());
    }
    for (const auto& storage : tensors_) {
      storage->tensor->set_data(context_.buffers.at(storage->id).data);
    }
    bound_ = true;
    return Error::Ok;
  }

  Error run(const ExecutionContext& context) override {
    ET_CHECK_OR_RETURN_ERROR(
        bound_, InvalidState, "ET requires a successful bind");
    for (const auto& instruction : instructions_) {
      // These routes declare no temporary storage. Any undeclared request fails
      // within this instruction instead of consuming the outer ET allocator.
      MemoryAllocator temporary(0, nullptr);
      KernelRuntimeContext kernel_context(context.event_tracer, &temporary);
      instruction->kernel(
          kernel_context,
          {instruction->stack.data(), instruction->stack.size()});
      if (kernel_context.failure_state() != Error::Ok) {
        ET_LOG(
            Error,
            "ET provider execution failed: %s",
            instruction->node->target.c_str());
        return kernel_context.failure_state();
      }
    }
    return Error::Ok;
  }

 private:
  static Error bind_argument(
      const ptn::Argument& arg,
      ArgumentStorage& storage,
      bool floating) {
    switch (arg.kind()) {
      case ptn::ArgKind::None:
        break;
      case ptn::ArgKind::Int:
        ET_CHECK_OR_RETURN_ERROR(
            arg.as_int().id == ptn::kInvalid,
            NotSupported,
            "Dynamic CPU scalar");
        storage.value = floating
            ? EValue(static_cast<double>(arg.as_int().value))
            : EValue(arg.as_int().value);
        break;
      case ptn::ArgKind::Float:
        ET_CHECK_OR_RETURN_ERROR(
            arg.as_float().id == ptn::kInvalid,
            NotSupported,
            "Dynamic CPU scalar");
        storage.value = arg.as_float().value;
        break;
      case ptn::ArgKind::Bool:
        ET_CHECK_OR_RETURN_ERROR(
            arg.as_bool().id == ptn::kInvalid,
            NotSupported,
            "Dynamic CPU scalar");
        storage.value = arg.as_bool().value;
        break;
      case ptn::ArgKind::String: {
        const auto& text = arg.as_string().value;
        const auto conversion = convert_string_arg(text);
        ET_CHECK_OR_RETURN_ERROR(
            conversion.supported,
            NotSupported,
            "Unsupported CPU target string");
        if (conversion.is_int) {
          storage.value = conversion.int_value;
        } else {
          storage.string = {text.data(), text.size()};
          storage.value = EValue(&storage.string);
        }
        break;
      }
      case ptn::ArgKind::ScalarType:
        storage.value =
            static_cast<int64_t>(to_aten(arg.as_scalar_type().value));
        break;
      case ptn::ArgKind::FloatList: {
        storage.doubles = arg.as_float_list().values;
        storage.floating_list = {
            storage.doubles.data(), storage.doubles.size()};
        storage.value = EValue(&storage.floating_list);
        break;
      }
      case ptn::ArgKind::IntList: {
        const auto& list = arg.as_int_list();
        ET_CHECK_OR_RETURN_ERROR(
            std::all_of(
                list.ids.begin(),
                list.ids.end(),
                [](auto id) { return id == ptn::kInvalid; }),
            NotSupported,
            "Dynamic CPU integer list");
        const size_t count = list.values.size();
        storage.elements.resize(std::max(count, size_t{1}));
        storage.pointers.resize(std::max(count, size_t{1}));
        storage.integers.resize(std::max(count, size_t{1}));
        for (size_t index = 0; index < count; ++index) {
          storage.elements.at(index) = list.values.at(index);
          storage.pointers.at(index) = &storage.elements.at(index);
        }
        storage.list = {
            storage.pointers.data(),
            storage.integers.data(),
            static_cast<int>(count)};
        storage.value = EValue(&storage.list);
        break;
      }
      case ptn::ArgKind::Tensor:
      case ptn::ArgKind::BoolList:
      case ptn::ArgKind::TensorList:
      case ptn::ArgKind::OptionalTensorList:
      case ptn::ArgKind::Graph:
        ET_LOG(
            Error,
            "Unsupported CPU argument kind: %d",
            static_cast<int>(arg.kind()));
        return Error::NotSupported;
    }
    return Error::Ok;
  }

  PreparationContext context_;
  std::vector<std::unique_ptr<TensorStorage>> tensors_;
  bool bound_ = false;
  std::vector<std::unique_ptr<Instruction>> instructions_;
};

class ETImplementation final : public KernelImplementation {
 public:
  std::string_view name() const override {
    return "registry";
  }
  int baseline_priority() const override {
    return 0;
  }
  Support supports(
      const Kernel& kernel,
      const ptn::Graph& graph,
      const ExecutionContext&) const override {
    const auto* route = routes_.find(kernel.target);
    if (route == nullptr) {
      return {false, "no linked ET route"};
    }
    if (const auto reason = check_node(*route->spec, kernel, graph);
        !reason.empty()) {
      return {false, reason};
    }
    return {true, "linked ET route"};
  }
  Result<std::unique_ptr<Executable>> compile(
      const KernelRegion& region,
      PreparationContext& context) override {
    auto executable = std::make_unique<ETExecutable>(context);
    const auto error = executable->initialize(region, routes_);
    if (error != Error::Ok) {
      return error;
    }
    return std::unique_ptr<Executable>(std::move(executable));
  }

 private:
  KernelRoutes routes_;
};

class ETProvider final : public KernelProvider {
 public:
  std::string_view name() const override {
    return "ET";
  }
  std::vector<KernelImplementation*> implementations() override {
    return {&implementation_};
  }

 private:
  ETImplementation implementation_;
};
} // namespace
// Referenced by generated provider registries and provider tests.
// cppcheck-suppress unusedFunction
std::unique_ptr<KernelProvider> create_et_provider() {
  return std::make_unique<ETProvider>();
}
} // namespace executorch::backends::cpu
