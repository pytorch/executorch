/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/xnnpack/runtime/XNNStatus.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include <xnnpack.h>
#include <map>
#include <memory>
#include <vector>

namespace torch {
namespace executor {
namespace xnnpack {
namespace delegate {

struct XNNShape {
  size_t num_dims;
  size_t dim[XNN_MAX_TENSOR_DIMS];
};

class XNNExecutor {
 private:
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> runtime_{
      nullptr,
      &xnn_delete_runtime};
  std::vector<uint32_t> input_ids_;
  std::vector<uint32_t> output_ids_;
  std::vector<uint32_t> external_id_args_;
  bool is_sorted_args_list_ = false;
  std::vector<xnn_external_value> externals_;
  std::vector<uint32_t> qinputs_;

  Error set_external_input(uint32_t id, Tensor* input, struct XNNShape* shape);

  // XNNPACK Profiling
  // Used to hold profiling data
  //  * To hold op names and duration (in usec) for each operator execution
  //  * Both indexed with xnn_node_idx (0.. node_id)
  using microsecond_t = uint64_t;
  size_t num_ops_;
  std::vector<char> op_names_;
  // op_timings[i][j] represents the runtime of operator j on the ith run
  std::vector<std::vector<microsecond_t>> op_timings_;

  void get_runtime_operator_names(std::vector<char>& operator_names);
  void get_runtime_num_operators(size_t& num_operators);
  void get_runtime_operator_timings(std::vector<uint64_t>& timing_stats);

 public:
  XNNExecutor() = default;

  // XNNPACK Profiling public fn
  void init_profiler();
  void log_op_timings();
  void print_avg_op_timings();

  inline void append_arg(uint32_t id) {
    external_id_args_.push_back(id);
    // Insertion order is not guaranteed here.
    is_sorted_args_list_ = false;
  }

  inline size_t get_args_size() {
    return external_id_args_.size();
  }

  inline uint32_t get_arg_index(size_t i) {
    if (!is_sorted_args_list_) {
      // Could have been inserted out of order.
      sort(external_id_args_.begin(), external_id_args_.end());
      is_sorted_args_list_ = true;
    }

    size_t ret = external_id_args_.size();
    ET_CHECK_MSG(
        i < ret,
        "Invalid arg index, requested: %zu, total args consumed by xnnpack: %zu\n",
        i,
        ret);
    return external_id_args_[i];
  }

  inline size_t getNumInputs() {
    return input_ids_.size();
  }

  inline size_t getNumOutputs() {
    return output_ids_.size();
  }

  inline void addDynamicQinput(uint32_t id) {
    qinputs_.emplace_back(id);
  }

  __ET_NODISCARD Error set_inputs(
      std::vector<Tensor*>& inputs,
      std::vector<Tensor*>& outputs,
      std::vector<struct XNNShape>& input_shapes,
      std::vector<struct XNNShape>& output_shapes) {
    externals_.clear();

    ET_CHECK_OR_RETURN_ERROR(
        inputs.size() == input_ids_.size(),
        InvalidArgument,
        "Expected %zu inputs but given %zu",
        input_ids_.size(),
        inputs.size());

    for (int i = 0; i < inputs.size(); i++) {
      auto err = set_external_input(input_ids_[i], inputs[i], &input_shapes[i]);
      ET_CHECK_OR_RETURN_ERROR(
          err == Error::Ok, Internal, "Failed to set_external_input");
    }
    ET_CHECK_OR_RETURN_ERROR(
        outputs.size() == output_ids_.size(),
        InvalidArgument,
        "Expected %zu outputs gut given %zu",
        output_ids_.size(),
        outputs.size());

    for (int i = 0; i < outputs.size(); i++) {
#ifdef ENABLE_DYNAMIC_QUANTIZATION
      externals_.emplace_back(xnn_external_value{
          output_ids_[i],
          outputs[i]->mutable_data_ptr<float>(),
          static_cast<size_t>(output_shapes[i].num_dims),
          output_shapes[i].dim});
#else
      externals_.emplace_back(xnn_external_value{
          output_ids_[i], outputs[i]->mutable_data_ptr<float>()});
#endif
    }

    return Error::Ok;
  }

  __ET_NODISCARD Error forward() {
    ET_CHECK_OR_RETURN_ERROR(
        runtime_ != nullptr,
        Internal,
        "XNNPACK Delegate did not compile correctly");
    xnn_status status =
        xnn_setup_runtime(runtime_.get(), externals_.size(), externals_.data());

    ET_CHECK_OR_RETURN_ERROR(
        status == xnn_status_success,
        Internal,
        "XNN Runtime setup failed with code: %s",
        xnn_status_to_string(status));

    status = xnn_invoke_runtime(runtime_.get());

    ET_CHECK_OR_RETURN_ERROR(
        status == xnn_status_success,
        Internal,
        "XNN Runtime invoke failed with code: %s",
        xnn_status_to_string(status));

    return Error::Ok;
  }

  /** Resize output tensor to support dynamic input shapes */
  __ET_NODISCARD Error resizeOutput(
      exec_aten::Tensor* output_tensor,
      struct XNNShape* output_shape) const {
    const size_t n_dim = output_tensor->dim();

    // Rank can't change
    if (n_dim != output_shape->num_dims) {
      ET_LOG(
          Error,
          "Found output shape with a different number of dimensions than the output tensor. Expected: %zu, Actual: %zu",
          n_dim,
          output_shape->num_dims);
      return Error::NotSupported;
    }

    // Early exit?
    bool same_shape = true;
    for (size_t i = 0; (i < n_dim) && same_shape; i++) {
      same_shape = (output_tensor->size(i) == output_shape->dim[i]);
    }
    if (same_shape) {
      return Error::Ok;
    }

    exec_aten::SizesType expected_output_size[kTensorDimensionLimit];
    for (size_t i = 0; i < n_dim; i++) {
      expected_output_size[i] =
          static_cast<exec_aten::SizesType>(output_shape->dim[i]);
    }

    exec_aten::ArrayRef<exec_aten::SizesType> output_size{
        expected_output_size, static_cast<size_t>(output_tensor->dim())};

    // Ok to dereference pointer here because resize_tensor takes in a tensor
    // and not a tensor&
    ET_LOG(Debug, "Resizing output tensor to a new shape");
    Error err = resize_tensor(*output_tensor, output_size);
    if (err != Error::Ok) {
      ET_LOG(Error, "Failed to resize output tensor for XNNExecutor");
    }
    return err;
  }

  friend class XNNCompiler;
};

} // namespace delegate
} // namespace xnnpack
} // namespace executor
} // namespace torch
