/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/runtime/XNNExecutor.h>
#include <executorch/backends/xnnpack/runtime/utils/utils.h>

namespace torch {
namespace executor {
namespace xnnpack {
namespace delegate {

Error XNNExecutor::set_external_input(uint32_t id, Tensor* input) {
  auto qinput_pair = qinputs_.find(id);
  if (qinput_pair != qinputs_.end()) {
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
  } else {
    externals_.emplace_back(xnn_external_value{id, input->mutable_data_ptr()});
  }
  return Error::Ok;
}

} // namespace delegate
} // namespace xnnpack
} // namespace executor
} // namespace torch
