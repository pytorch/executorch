/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/runtime/XNNExecutor.h>

namespace executorch {
namespace backends {
namespace xnnpack {
namespace delegate {

using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::is_contiguous_dim_order;
using executorch::runtime::kTensorDimensionLimit;

/**
 * Initializes the XNNExecutor with the runtime and given number of
 * inputs/outputs externals_ is resized to the total number of inputs and
 * outputs
 */
ET_NODISCARD Error XNNExecutor::initialize(
    xnn_runtime_t runtime,
    std::vector<uint32_t>&& input_ids,
    std::vector<uint32_t>&& output_ids) {
  runtime_ = std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)>(
      runtime, xnn_delete_runtime);

  auto error = profiler_.initialize(runtime);
  if (error != Error::Ok) {
    ET_LOG(
        Error,
        "Failed to start profiling: %u.",
        static_cast<unsigned int>(error));
  }

  // Initialize the external values for inputs and outputs
  // mapping the executorch arg idx to external IDs
  input_ids_ = std::move(input_ids);
  std::sort(input_ids_.begin(), input_ids_.end());

  output_ids_ = std::move(output_ids);
  std::sort(output_ids_.begin(), output_ids_.end());

  externals_.resize(input_ids_.size() + output_ids_.size());

  return Error::Ok;
}

/**
 * Prepares the args for XNNPACK Runtime.
 *
 * Creates an array of xnn_externals_values from the EValues passed in.
 * Reshapes all the external input tensors, in case any input shapes have
 * changed. The reshapes the entire runtime, propagating shape information
 * through the runtime.
 *
 * Note: the external ids given to the external tensors in the XNNPACK
 * runtime correspond to their index in the list of arg passed into
 * delegate->execute()
 */
ET_NODISCARD Error XNNExecutor::prepare_args(EValue** args) {
  // Create xnn_externals_value from evalue args
  xnn_status status;
  for (uint32_t i = 0; i < externals_.size(); ++i) {
    if (i < input_ids_.size()) {
      externals_[i].id = input_ids_[i];
    } else {
      externals_[i].id = output_ids_[i - input_ids_.size()];
    }
    uint32_t ext_id = externals_[i].id;

    ET_CHECK_OR_RETURN_ERROR(
        args[ext_id]->isTensor(),
        InvalidArgument,
        "Expected argument to delegate at index %u to be a Tensor, but got %" PRIu32,
        i,
        static_cast<uint32_t>(args[ext_id]->tag));

    Tensor* tensor = &args[ext_id]->toTensor();
    externals_[i].data = tensor->mutable_data_ptr<float>();

    // Reshape runtime inputs
    if (i < input_ids_.size()) {
      size_t num_dims = tensor->dim();
      ET_CHECK_OR_RETURN_ERROR(
          is_contiguous_dim_order(tensor->dim_order().data(), tensor->dim()),
          Internal,
          "Expecting default dim_order but got a non default dim_order tensor for external input %u",
          i);
      size_t dims[XNN_MAX_TENSOR_DIMS];
      ET_CHECK_OR_RETURN_ERROR(
          num_dims <= XNN_MAX_TENSOR_DIMS,
          InvalidArgument,
          "XNNPACK backend accepts tensors with at most %d dims, but got %zu",
          XNN_MAX_TENSOR_DIMS,
          num_dims);
      for (int d = 0; d < num_dims; ++d) {
        dims[d] = tensor->size(d);
      }
      status =
          xnn_reshape_external_value(runtime_.get(), ext_id, num_dims, dims);
      ET_CHECK_OR_RETURN_ERROR(
          status == xnn_status_success,
          Internal,
          "Internal Error: Reshape Input Tensor Failed with code: %s",
          xnn_status_to_string(status));
    }
  }
  // // Propagate Input Shape and Memory Plan for increased allocation
  status = xnn_reshape_runtime(runtime_.get());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Internal Error: Propagating input shapes failed with code: %s",
      xnn_status_to_string(status));

  return Error::Ok;
}

/**
 * Runs the XNNPACK Runtime.
 *
 * We first setup the runtime by feeding the externals_ to runtime setup.
 * After which we then execute the runtime through invoke_runtime.
 */
ET_NODISCARD Error XNNExecutor::forward(BackendExecutionContext& context) {
  ET_CHECK_OR_RETURN_ERROR(
      runtime_ != nullptr,
      Internal,
      "XNNPACK Delegate did not compile correctly");

  xnn_status status = xnn_setup_runtime_v2(
      runtime_.get(), externals_.size(), externals_.data());

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "Internal Error: Setting up the runtime failed with code: %s",
      xnn_status_to_string(status));

  auto error = profiler_.start(context.event_tracer());
  if (error != Error::Ok) {
    ET_LOG(
        Error,
        "Failed to start profiling: %u.",
        static_cast<unsigned int>(error));
  }

  status = xnn_invoke_runtime(runtime_.get());

  error = profiler_.end();
  if (error != Error::Ok) {
    ET_LOG(
        Error,
        "Failed to end profiling: %u.",
        static_cast<unsigned int>(error));
  }

  ET_CHECK_OR_RETURN_ERROR(
      status == xnn_status_success,
      Internal,
      "XNN Runtime invoke failed with code: %s",
      xnn_status_to_string(status));

  return Error::Ok;
}

/**
 * Prepares the outputs for ExecuTorch
 *
 * Resizes the output tensors based on the output shapes returned by
 * the xnnpack runtime.
 *
 * Note: For arg_max pooling, we recast the output index tensor. Since
 * XNNPACK gives the index tensor to us as int32, we need to convert it
 * back to int64 for ExecuTorch.
 */
ET_NODISCARD Error XNNExecutor::resize_outputs(EValue** args) const {
  size_t output_idx_start = input_ids_.size();
  for (size_t i = output_idx_start; i < externals_.size(); ++i) {
    uint32_t ext_id = externals_[i].id;
    Tensor* out_tensor = &args[ext_id]->toTensor();

    size_t num_dim;
    size_t dims[XNN_MAX_TENSOR_DIMS];

    // Fetch the updated output shapes from xnnpack runtime
    xnn_status status =
        xnn_get_external_value_shape(runtime_.get(), ext_id, &num_dim, dims);

    ET_CHECK_OR_RETURN_ERROR(
        status == xnn_status_success,
        Internal,
        "Internal Error: Failed to retrieve graph output shapes");

    // Convert new output shape into SizesType
    SizesType expected_output_size[kTensorDimensionLimit];
    for (size_t d = 0; d < num_dim; ++d) {
      expected_output_size[d] = static_cast<SizesType>(dims[d]);
    }

    executorch::aten::ArrayRef<SizesType> output_size{
        expected_output_size, static_cast<size_t>(num_dim)};

    ET_LOG(Debug, "Resizing output tensor to a new shape");
    Error err = resize_tensor(*out_tensor, output_size);
    if (err != Error::Ok) {
      ET_LOG(Error, "Failed to resize output tensor for XNNExecutor");
      return err;
    }

    // Output datatype is int64. However, XNNPACK doesn't support
    // int64. This means that the data was put into this tensor
    // by XNNPACK as int32 and needs to be copied to int64 form
    if (out_tensor->scalar_type() == ScalarType::Long) {
      int64_t* data_64 = out_tensor->mutable_data_ptr<int64_t>();
      const int32_t* data_32 = out_tensor->const_data_ptr<int32_t>();
      for (size_t j = out_tensor->numel() - 1; j >= 0; --j) {
        data_64[j] = data_32[j];
      }
    }
  }

  return Error::Ok;
}

} // namespace delegate
} // namespace xnnpack
} // namespace backends
} // namespace executorch
