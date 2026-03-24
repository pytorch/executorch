#pragma once

#include <executorch/backends/xnnpack/runtime/kernels/layer_norm/layer_norm.h>
#include <executorch/backends/xnnpack/runtime/operators/operator.h>

namespace executorch::backends::xnnpack::operators {

class LayerNorm : public Operator {
public:
    void setup(core::Span<const graph::ConstantArg> constant_args) override;
    void execute(core::Span<core::Tensor*> inputs, core::Span<core::Tensor*> outputs) override;

private:
    kernels::LayerNormF32Fn kernel_ = nullptr;
    uint32_t num_normalized_dims_ = 0;
    float eps_ = 1e-5f;
};

}
