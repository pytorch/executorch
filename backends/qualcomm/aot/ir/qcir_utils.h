/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/qualcomm/aot/ir/qcir_generated.h>
#include "QnnTypes.h"

namespace torch {
namespace executor {
namespace qnn {

typedef flatbuffers::Vector<::flatbuffers::Offset<qcir::Tensor>>::return_type
    tensor_type;
typedef flatbuffers::Vector<
    ::flatbuffers::Offset<qcir::QuantizeParam>>::return_type qparam_type;

qcir::TensorType ToTensorType(Qnn_TensorType_t type);
Qnn_TensorType_t ToTensorType(qcir::TensorType type);
qcir::DataType ToDataType(Qnn_DataType_t type);
Qnn_DataType_t ToDataType(qcir::DataType type);

flatbuffers::Offset<qcir::QuantizeParam> ToQuantizeParam(
    const Qnn_Tensor_t& tensor,
    flatbuffers::FlatBufferBuilder* builder);
Qnn_QuantizeParams_t ToQuantizeParam(const tensor_type& tensor);

flatbuffers::Offset<qcir::Tensor> ToTensor(
    const Qnn_Tensor_t& tensor,
    flatbuffers::FlatBufferBuilder* builder);
Qnn_Tensor_t ToTensor(const tensor_type& tensor);

} // namespace qnn
} // namespace executor
} // namespace torch
