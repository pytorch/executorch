#include <executorch/backends/xnnpack/runtime/operators/layer_norm.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/log.h>

#include <variant>

namespace executorch::backends::xnnpack::operators {

runtime::Error LayerNorm::setup(
    runtime::Span<const graph::ConstantArg> constant_args) {
  ET_CHECK_OR_RETURN_ERROR(
      constant_args.size() == 2,
      InvalidArgument,
      "LayerNorm expects 2 constant args (normalized dims, eps), got %zu",
      constant_args.size());
  const auto* num_dims = std::get_if<int64_t>(&constant_args[0]);
  const auto* eps = std::get_if<double>(&constant_args[1]);
  ET_CHECK_OR_RETURN_ERROR(
      num_dims != nullptr && eps != nullptr,
      InvalidArgument,
      "LayerNorm constant args have unexpected types");
  kernel_ = kernels::select_layer_norm_f32_kernel();
  num_normalized_dims_ = static_cast<uint32_t>(*num_dims);
  eps_ = static_cast<float>(*eps);
  return runtime::Error::Ok;
}

runtime::Error LayerNorm::execute(
    runtime::Span<core::Tensor*> inputs,
    runtime::Span<core::Tensor*> outputs) {
  ET_CHECK_OR_RETURN_ERROR(
      inputs.size() >= 1 && inputs.size() <= 3 && outputs.size() == 1,
      InvalidArgument,
      "LayerNorm expects 1-3 inputs and 1 output, got %zu inputs / %zu outputs",
      inputs.size(),
      outputs.size());

  auto* input = inputs[0];
  auto* output = outputs[0];

  ET_CHECK_OR_RETURN_ERROR(
      input->dtype == core::DType::Float32 &&
          output->dtype == core::DType::Float32,
      NotSupported,
      "LayerNorm in-tree kernel only supports float32");
  ET_CHECK_OR_RETURN_ERROR(
      num_normalized_dims_ <= input->sizes.size(),
      InvalidArgument,
      "LayerNorm normalized dims %u exceeds input rank %zu",
      num_normalized_dims_,
      input->sizes.size());

  size_t split = input->sizes.size() - num_normalized_dims_;
  size_t inner_size = 1;
  for (size_t i = split; i < input->sizes.size(); i++) {
    inner_size *= input->sizes[i];
  }
  size_t outer_size = 1;
  for (size_t i = 0; i < split; i++) {
    outer_size *= input->sizes[i];
  }

  const float* weight =
      (inputs.size() > 1) ? inputs[1]->data_const<float>() : nullptr;
  const float* bias =
      (inputs.size() > 2) ? inputs[2]->data_const<float>() : nullptr;

  kernel_(
      input->data_const<float>(),
      output->data_mut<float>(),
      weight,
      bias,
      outer_size,
      inner_size,
      eps_);
  return runtime::Error::Ok;
}

} // namespace executorch::backends::xnnpack::operators
