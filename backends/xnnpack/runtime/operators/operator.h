#pragma once

#include <executorch/backends/xnnpack/runtime/core/tensor.h>
#include <executorch/backends/xnnpack/runtime/graph/node.h>
#include <executorch/backends/xnnpack/runtime/graph/operator.h>
#include <executorch/backends/xnnpack/runtime/graph/tensor_spec.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/span.h>

#include <memory>

namespace executorch::backends::xnnpack::operators {

class Operator {
 public:
  virtual runtime::Error setup(
      runtime::Span<const graph::ConstantArg> constant_args) {
    return runtime::Error::Ok;
  }
  virtual runtime::Error prepare(
      runtime::Span<core::Tensor*> inputs,
      runtime::Span<core::Tensor*> outputs) {
    return runtime::Error::Ok;
  }
  virtual runtime::Error reshape(
      runtime::Span<const graph::TensorSpec> input_specs) {
    return runtime::Error::Ok;
  }
  virtual runtime::Error execute(
      runtime::Span<core::Tensor*> inputs,
      runtime::Span<core::Tensor*> outputs) = 0;
  virtual ~Operator() = default;
};

std::unique_ptr<Operator> create_operator(graph::Operator op);

} // namespace executorch::backends::xnnpack::operators
