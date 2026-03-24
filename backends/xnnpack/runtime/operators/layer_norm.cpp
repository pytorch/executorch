#include <executorch/backends/xnnpack/runtime/operators/layer_norm.h>

#include <cassert>

namespace executorch::backends::xnnpack::operators {

void LayerNorm::setup(core::Span<const graph::ConstantArg> constant_args) {
    assert(constant_args.size() == 2);
    kernel_ = kernels::select_layer_norm_f32_kernel();
    num_normalized_dims_ = static_cast<uint32_t>(std::get<int64_t>(constant_args[0]));
    eps_ = static_cast<float>(std::get<double>(constant_args[1]));
}

void LayerNorm::execute(
    core::Span<core::Tensor*> inputs,
    core::Span<core::Tensor*> outputs) {
    assert(inputs.size() >= 1 && inputs.size() <= 3);
    assert(outputs.size() == 1);

    auto* input = inputs[0];
    auto* output = outputs[0];

    size_t inner_size = 1;
    for (size_t i = input->sizes.size() - num_normalized_dims_; i < input->sizes.size(); i++) {
        inner_size *= input->sizes[i];
    }
    size_t outer_size = input->numel() / inner_size;

    const float* weight = (inputs.size() > 1) ? inputs[1]->data_const<float>() : nullptr;
    const float* bias = (inputs.size() > 2) ? inputs[2]->data_const<float>() : nullptr;

    kernel_(
        input->data_const<float>(),
        output->data_mut<float>(),
        weight, bias,
        outer_size, inner_size, eps_);
}

}
