/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/runtime/XNNExecutor.h>
#ifdef ENABLE_DYNAMIC_QUANTIZATION
#include <executorch/backends/xnnpack/runtime/utils/utils.h>
#endif

namespace torch {
namespace executor {
namespace xnnpack {
namespace delegate {

Error XNNExecutor::set_external_input(uint32_t id, Tensor* input) {
  auto qinput_pair = qinputs_.find(id);
  if (qinput_pair != qinputs_.end()) {
#ifdef ENABLE_DYNAMIC_QUANTIZATION
    auto qinput = qinput_pair->second;
    // dq the input and copy it in to qinput
    float input_min, input_max;
    std::tie(input_min, input_max) = qnnpack_utils::GetMinMax(*input);

    qnnpack_utils::QuantizationParams input_qparam;

    int8_t qmin = std::numeric_limits<int8_t>::min();
    int8_t qmax = std::numeric_limits<int8_t>::max();
    Error e = qnnpack_utils::ChooseQuantizationParams(
        input_min,
        input_max,
        qmin,
        qmax,
        input_qparam,
        false, /* preserve_sparsity */
        false, /* force_scale_power_of_two */
        false /* reduce_range */
    );
    ET_CHECK_OR_RETURN_ERROR(
        e == Error::Ok, Internal, "ChooseQuantizationParams() failed");

    ET_CHECK_OR_RETURN_ERROR(
        input_qparam.zero_point <= qmax && input_qparam.zero_point >= qmin,
        Internal,
        "ChooseQuantizationParams() selected invalid input_zero_point: %d",
        input_qparam.zero_point);

    e = qnnpack_utils::QuantizePerTensor<int8_t>(
        *input, qinput, input_qparam.scale, input_qparam.zero_point);

    size_t batch_size = 1;
    for (int i = 0; i < input->dim() - 1; i++) {
      batch_size *= input->size(i);
    }
    ET_CHECK_OR_RETURN_ERROR(
        e == Error::Ok, Internal, "QuantizePerTensor() failed");
    externals_.emplace_back(xnn_external_value{
        id,
        qinput.mutable_data_ptr(),
        {static_cast<float>(input_qparam.scale),
         static_cast<int8_t>(input_qparam.zero_point)},
        batch_size});
#else
    ET_LOG(Error, "Dynamic Quantization is not supported");
    return Error::NotSupported;
#endif
  } else {
    // TODO(T165403530): Test insure accuracy for int64 --> float32 conversion
    if (input->scalar_type() == ScalarType::Long) {
      // Input data type is int64. However, XNNPACK doesn't support
      // int64. This means that the data needs to be casted to float
      // In order for XNNPACK to properly use it.
      const int64_t* data_64 = input->const_data_ptr<int64_t>();
      float* data_f32 = input->mutable_data_ptr<float>();
      for (int j = 0; j < input->numel(); j++) {
        data_f32[j] = data_64[j];
      }
    }
    externals_.emplace_back(xnn_external_value{id, input->mutable_data_ptr()});
  }
  return Error::Ok;
}

inline void XNNExecutor::get_runtime_operator_names(
    std::vector<char>& operator_names) {
  size_t required_size = 0;
  // First call returns xnn_status_out_of_memory, but sets required_size to
  // the correct size of the buffer to store the result
  xnn_status status = xnn_get_runtime_profiling_info(
      runtime_.get(), // runtime
      xnn_profile_info_operator_name, // param_name
      0, // param_value_size
      nullptr, // param_value
      &required_size // param_value_size_ret
  );

  if (status == xnn_status_out_of_memory) {
    operator_names.resize(required_size);
    status = xnn_get_runtime_profiling_info(
        runtime_.get(),
        xnn_profile_info_operator_name,
        operator_names.size(),
        operator_names.data(),
        &required_size);
  }
  if (status != xnn_status_success) {
    ET_LOG(Error, "Failed to get XNNPACK Operator Timings");
  }
}

inline void XNNExecutor::get_runtime_num_operators(size_t& num_operators) {
  size_t required_size = 0;
  xnn_status status = xnn_get_runtime_profiling_info(
      runtime_.get(),
      xnn_profile_info_num_operators,
      sizeof(num_operators),
      &num_operators,
      &required_size);
  if (status != xnn_status_success) {
    ET_LOG(Error, "Failed to get XNNPACK Operator Timings");
  }
}

inline void XNNExecutor::get_runtime_operator_timings(
    std::vector<uint64_t>& timing_stats) {
  size_t required_size;
  // Get number of runtime operators for timing_stats.size
  timing_stats.resize(num_ops_);
  xnn_status status = xnn_get_runtime_profiling_info(
      runtime_.get(),
      xnn_profile_info_operator_timing,
      timing_stats.size() * sizeof(uint64_t),
      timing_stats.data(),
      &required_size);
  if (status != xnn_status_success) {
    ET_LOG(Error, "Failed to get XNNPACK Operator Timings");
  }
}

void XNNExecutor::init_profiler() {
  get_runtime_operator_names(op_names_);
  get_runtime_num_operators(num_ops_);
}

void XNNExecutor::log_op_timings() {
  std::vector<uint64_t> op_stats;
  get_runtime_operator_timings(op_stats);
  op_timings_.emplace_back(op_stats);
}

void XNNExecutor::print_avg_op_timings() {
  size_t num_iterations = op_timings_.size();
  size_t name_len = 0;
  const char* op_name = nullptr;
  float avg_total = 0;
  for (size_t xnn_node_idx = 0; xnn_node_idx < num_ops_; xnn_node_idx++) {
    op_name = &op_names_[name_len];
    name_len += strlen(op_name) + 1;
    float total_op_time = 0;
    for (size_t it = 0; it < num_iterations; it++) {
      total_op_time += op_timings_[it][xnn_node_idx];
    }
    float avg_op_time = total_op_time / static_cast<float>(num_iterations);
    ET_LOG(Info, ">>, %s, %f", op_name, avg_op_time);
    avg_total += avg_op_time;
  }
  ET_LOG(Info, ">>, Total Time, %f", avg_total);
}
} // namespace delegate
} // namespace xnnpack
} // namespace executor
} // namespace torch
