#pragma once

#include <executorch/runtime/core/error.h>
#include <pytorch_qnnpack.h>
#include <qnnpack_func.h>
#include <memory>

#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace torch {
namespace executor {

struct QNNExecutor {
  std::unique_ptr<qnnpack::PackBMatrix> packed_weight_;
  Tensor bias_;
  Tensor qinput_;
  Tensor weight_scales_;
  Tensor weight_zero_points_;

  QNNExecutor(
      std::unique_ptr<qnnpack::PackBMatrix> packed_weight,
      TensorImpl* bias,
      TensorImpl* qinput,
      TensorImpl* weight_scales,
      TensorImpl* weight_zero_points)
      : packed_weight_(std::move(packed_weight)),
        bias_(bias),
        qinput_(qinput),
        weight_scales_(weight_scales),
        weight_zero_points_(weight_zero_points){};
};

} // namespace executor
} // namespace torch
