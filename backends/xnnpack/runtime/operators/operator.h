#pragma once

#include <executorch/backends/xnnpack/runtime/core/span.h>
#include <executorch/backends/xnnpack/runtime/core/tensor.h>
#include <executorch/backends/xnnpack/runtime/graph/node.h>
#include <executorch/backends/xnnpack/runtime/graph/operator.h>
#include <executorch/backends/xnnpack/runtime/graph/tensor_spec.h>

#include <memory>

namespace executorch::backends::xnnpack::operators {

class Operator {
public:
    virtual void setup(core::Span<const graph::ConstantArg> constant_args) {};
    virtual void reshape(core::Span<const graph::TensorSpec> input_specs) {};
    virtual void execute(core::Span<core::Tensor*> inputs, core::Span<core::Tensor*> outputs) = 0;
    virtual ~Operator() = default;
};

std::unique_ptr<Operator> create_operator(graph::Operator op);

}
