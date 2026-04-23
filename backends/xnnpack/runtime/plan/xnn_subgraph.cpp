#include <executorch/backends/xnnpack/runtime/plan/xnn_subgraph.h>

#include <executorch/backends/xnnpack/runtime/core/quant_params.h>
#include <executorch/backends/xnnpack/runtime/core/span.h>
#include <executorch/backends/xnnpack/runtime/core/tensor.h>

#include <executorch/backends/xnnpack/runtime/graph/graph.h>
#include <executorch/backends/xnnpack/runtime/graph/tensor_spec.h>

#include <cassert>
#include <cmath>
#include <optional>

namespace executorch::backends::xnnpack::plan {

using core::DType;
using graph::CallOperatorNode;
using graph::ConstantNode;
using graph::InputNode;
using graph::NodeHandle;
using graph::Operator;

namespace {

xnn_datatype map_xnn_datatype(const graph::TensorSpec& spec) {
    if (!spec.quant_params) {
        switch (spec.dtype) {
            case DType::Float32: return xnn_datatype_fp32;
            default: abort();
        }
    }
    switch (spec.dtype) {
        case DType::QUInt8Asym: return xnn_datatype_quint8;
        case DType::QInt8Sym:
            if (std::holds_alternative<core::PerAxisQuant>(*spec.quant_params)) {
                return xnn_datatype_qcint8;
            }
            return xnn_datatype_qint8;
        case DType::QInt32Sym:  return xnn_datatype_qint32;
        default: abort();
    }
}

std::optional<xnn_binary_operator> map_binary_op(Operator op) {
    switch (op) {
        case Operator::Add:               return xnn_binary_add;
        case Operator::Subtract:          return xnn_binary_subtract;
        case Operator::Multiply:          return xnn_binary_multiply;
        case Operator::Divide:            return xnn_binary_divide;
        case Operator::Maximum:           return xnn_binary_maximum;
        case Operator::Minimum:           return xnn_binary_minimum;
        case Operator::CopySign:          return xnn_binary_copysign;
        case Operator::SquaredDifference: return xnn_binary_squared_difference;
        case Operator::PReLU:             return xnn_binary_prelu;
        case Operator::Modulus:           return xnn_binary_modulus;
        case Operator::Atan2:            return xnn_binary_atan2;
        case Operator::Pow:              return xnn_binary_pow;
        default:                          return std::nullopt;
    }
}

std::optional<xnn_unary_operator> map_unary_op(Operator op) {
    switch (op) {
        case Operator::Abs:                    return xnn_unary_abs;
        case Operator::Negate:                 return xnn_unary_negate;
        case Operator::Clamp:                  return xnn_unary_clamp;
        case Operator::Ceiling:                return xnn_unary_ceiling;
        case Operator::Floor:                  return xnn_unary_floor;
        case Operator::Round:                  return xnn_unary_bankers_rounding;
        case Operator::Square:                 return xnn_unary_square;
        case Operator::SquareRoot:             return xnn_unary_square_root;
        case Operator::ReciprocalSquareRoot:   return xnn_unary_reciprocal_square_root;
        case Operator::Exp:                    return xnn_unary_exp;
        case Operator::Log:                    return xnn_unary_log;
        case Operator::Sigmoid:                return xnn_unary_sigmoid;
        case Operator::Tanh:                   return xnn_unary_tanh;
        case Operator::ELU:                    return xnn_unary_elu;
        case Operator::GELU:                   return xnn_unary_gelu;
        case Operator::HardSwish:              return xnn_unary_hardswish;
        case Operator::LeakyReLU:              return xnn_unary_leaky_relu;
        case Operator::Sine:                   return xnn_unary_sine;
        case Operator::Cosine:                 return xnn_unary_cosine;
        case Operator::Sign:                   return xnn_unary_sign;
        case Operator::ReLU:                   return xnn_unary_clamp;
        default:                               return std::nullopt;
    }
}

template <typename T>
T get_const(const graph::ConstantArg& arg) {
    return std::get<T>(arg);
}

template <typename T>
std::vector<size_t> to_size_vec(const std::vector<T>& v) {
    return {v.begin(), v.end()};
}

void define_node(
    const CallOperatorNode& op,
    uint32_t output_id,
    core::Span<const uint32_t> xnn_ids,
    xnn_subgraph_t subgraph) {

    if (auto bin_op = map_binary_op(op.op)) {
        xnn_binary_params params = {
            .output_min = -INFINITY,
            .output_max = INFINITY,
        };
        auto status = xnn_define_binary(
            subgraph,
            *bin_op,
            &params,
            xnn_ids[op.args[0].node],
            xnn_ids[op.args[1].node],
            output_id,
            /*flags=*/0);
        if (status != xnn_status_success) { abort(); }
        return;
    }

    if (auto unary_op = map_unary_op(op.op)) {
        xnn_unary_params params = {};

        if (op.op == Operator::Clamp) {
            params.clamp.min = static_cast<float>(
                get_const<double>(op.constant_args[0]));
            params.clamp.max = static_cast<float>(
                get_const<double>(op.constant_args[1]));
        } else if (op.op == Operator::ReLU) {
            params.clamp.min = 0.0f;
            params.clamp.max = INFINITY;
        } else if (op.op == Operator::ELU) {
            params.elu.alpha = op.constant_args.empty()
                ? 1.0f
                : static_cast<float>(get_const<double>(op.constant_args[0]));
        } else if (op.op == Operator::LeakyReLU) {
            params.leaky_relu.negative_slope = static_cast<float>(
                get_const<double>(op.constant_args[0]));
        }

        auto status = xnn_define_unary(
            subgraph, *unary_op, &params,
            xnn_ids[op.args[0].node],
            output_id, /*flags=*/0);
        if (status != xnn_status_success) { abort(); }
        return;
    }

    if (op.op == Operator::Linear) {
        auto bias_id = op.args[2].is_null()
            ? XNN_INVALID_VALUE_ID
            : xnn_ids[op.args[2].node];
        auto status = xnn_define_fully_connected(
            subgraph, -INFINITY, INFINITY,
            xnn_ids[op.args[0].node],
            xnn_ids[op.args[1].node],
            bias_id,
            output_id, /*flags=*/0);
        if (status != xnn_status_success) { abort(); }
        return;
    }

    if (op.op == Operator::BatchMatrixMultiply) {
        auto status = xnn_define_batch_matrix_multiply(
            subgraph,
            xnn_ids[op.args[0].node],
            xnn_ids[op.args[1].node],
            output_id, /*flags=*/0);
        if (status != xnn_status_success) { abort(); }
        return;
    }

    if (op.op == Operator::Conv2d) {
        auto stride = get_const<std::vector<int64_t>>(op.constant_args[0]);
        auto padding = get_const<std::vector<int64_t>>(op.constant_args[1]);
        auto dilation = get_const<std::vector<int64_t>>(op.constant_args[2]);
        auto groups = get_const<int64_t>(op.constant_args[3]);
        auto kernel = get_const<std::vector<int64_t>>(op.constant_args[4]);
        auto group_input_channels = get_const<int64_t>(op.constant_args[5]);
        auto group_output_channels = get_const<int64_t>(op.constant_args[6]);
        auto bias_id = op.args[2].is_null()
            ? XNN_INVALID_VALUE_ID
            : xnn_ids[op.args[2].node];

        auto status = xnn_define_convolution_2d(
            subgraph,
            padding[0], padding[1], padding[0], padding[1],
            kernel[0], kernel[1],
            stride[0], stride[1],
            dilation[0], dilation[1],
            groups,
            group_input_channels, group_output_channels,
            -INFINITY, INFINITY,
            xnn_ids[op.args[0].node],
            xnn_ids[op.args[1].node],
            bias_id,
            output_id, /*flags=*/0);
        if (status != xnn_status_success) { abort(); }
        return;
    }

    if (op.op == Operator::ConvTranspose2d) {
        auto stride = get_const<std::vector<int64_t>>(op.constant_args[0]);
        auto padding = get_const<std::vector<int64_t>>(op.constant_args[1]);
        auto output_padding = get_const<std::vector<int64_t>>(op.constant_args[2]);
        auto groups = get_const<int64_t>(op.constant_args[3]);
        auto dilation = get_const<std::vector<int64_t>>(op.constant_args[4]);
        auto kernel = get_const<std::vector<int64_t>>(op.constant_args[5]);
        auto group_input_channels = get_const<int64_t>(op.constant_args[6]);
        auto group_output_channels = get_const<int64_t>(op.constant_args[7]);
        auto bias_id = op.args[2].is_null()
            ? XNN_INVALID_VALUE_ID
            : xnn_ids[op.args[2].node];

        auto status = xnn_define_deconvolution_2d(
            subgraph,
            padding[0], padding[1], padding[0], padding[1],
            output_padding[0], output_padding[1],
            kernel[0], kernel[1],
            stride[0], stride[1],
            dilation[0], dilation[1],
            groups,
            group_input_channels, group_output_channels,
            -INFINITY, INFINITY,
            xnn_ids[op.args[0].node],
            xnn_ids[op.args[1].node],
            bias_id,
            output_id, /*flags=*/0);
        if (status != xnn_status_success) { abort(); }
        return;
    }

    if (op.op == Operator::AvgPool2d) {
        auto kernel = get_const<std::vector<int64_t>>(op.constant_args[0]);
        auto stride = get_const<std::vector<int64_t>>(op.constant_args[1]);
        auto padding = get_const<std::vector<int64_t>>(op.constant_args[2]);
        uint32_t flags = 0;
        auto status = xnn_define_average_pooling_2d(
            subgraph,
            padding[0], padding[0], padding[1], padding[1],
            kernel[0], kernel[1],
            stride[0], stride[1],
            -INFINITY, INFINITY,
            xnn_ids[op.args[0].node],
            output_id, flags);
        if (status != xnn_status_success) { abort(); }
        return;
    }

    if (op.op == Operator::AdaptiveAvgPool2d) {
        auto status = xnn_define_global_average_pooling_2d(
            subgraph, -INFINITY, INFINITY,
            xnn_ids[op.args[0].node],
            output_id, XNN_FLAG_KEEP_DIMS);
        if (status != xnn_status_success) { abort(); }
        return;
    }

    if (op.op == Operator::MaxPool2d) {
        auto kernel = get_const<std::vector<int64_t>>(op.constant_args[0]);
        auto stride = get_const<std::vector<int64_t>>(op.constant_args[1]);
        auto padding = get_const<std::vector<int64_t>>(op.constant_args[2]);
        auto dilation = get_const<std::vector<int64_t>>(op.constant_args[3]);
        auto status = xnn_define_max_pooling_2d(
            subgraph,
            padding[0], padding[0], padding[1], padding[1],
            kernel[0], kernel[1],
            stride[0], stride[1],
            dilation[0], dilation[1],
            -INFINITY, INFINITY,
            xnn_ids[op.args[0].node],
            output_id, /*flags=*/0);
        if (status != xnn_status_success) { abort(); }
        return;
    }

    if (op.op == Operator::Softmax) {
        auto status = xnn_define_softmax(
            subgraph,
            xnn_ids[op.args[0].node],
            output_id, /*flags=*/0);
        if (status != xnn_status_success) { abort(); }
        return;
    }

    if (op.op == Operator::Mean || op.op == Operator::Sum) {
        auto dims = get_const<std::vector<int64_t>>(op.constant_args[0]);
        auto keepdim = get_const<int64_t>(op.constant_args[1]);
        std::vector<size_t> reduction_axes(dims.begin(), dims.end());
        uint32_t flags = keepdim ? XNN_FLAG_KEEP_DIMS : 0;

        auto reduce_op = (op.op == Operator::Mean)
            ? xnn_reduce_mean : xnn_reduce_sum;

        auto status = xnn_define_static_reduce(
            subgraph, reduce_op,
            reduction_axes.size(), reduction_axes.data(),
            xnn_ids[op.args[0].node],
            output_id, flags);
        if (status != xnn_status_success) { abort(); }
        return;
    }

    if (op.op == Operator::Reshape || op.op == Operator::View) {
        auto shape = get_const<std::vector<int64_t>>(op.constant_args[0]);
        auto new_shape = to_size_vec(shape);
        auto status = xnn_define_static_reshape(
            subgraph,
            new_shape.size(), new_shape.data(),
            xnn_ids[op.args[0].node],
            output_id, /*flags=*/0);
        if (status != xnn_status_success) { abort(); }
        return;
    }

    if (op.op == Operator::Transpose || op.op == Operator::Permute) {
        auto perm = get_const<std::vector<int64_t>>(op.constant_args[0]);
        auto perm_sz = to_size_vec(perm);
        auto status = xnn_define_static_transpose(
            subgraph,
            perm_sz.size(), perm_sz.data(),
            xnn_ids[op.args[0].node],
            output_id, /*flags=*/0);
        if (status != xnn_status_success) { abort(); }
        return;
    }

    if (op.op == Operator::Slice) {
        auto dim = get_const<int64_t>(op.constant_args[0]);
        auto start = get_const<int64_t>(op.constant_args[1]);
        auto end = get_const<int64_t>(op.constant_args[2]);
        (void)end;

        auto& out_spec = std::get<graph::TensorSpec>(op.output_specs);
        auto ndims = out_spec.sizes.size();
        std::vector<size_t> offsets(ndims, 0);
        std::vector<size_t> sizes(ndims, 0);
        for (size_t i = 0; i < ndims; i++) {
            sizes[i] = static_cast<size_t>(out_spec.sizes[i].offset);
        }
        offsets[dim] = static_cast<size_t>(start);

        auto status = xnn_define_static_slice(
            subgraph,
            ndims, offsets.data(), sizes.data(),
            xnn_ids[op.args[0].node],
            output_id, /*flags=*/0);
        if (status != xnn_status_success) { abort(); }
        return;
    }

    if (op.op == Operator::Cat) {
        auto axis = get_const<int64_t>(
            op.constant_args[0]);
        std::vector<uint32_t> input_ids;
        for (auto& arg : op.args) {
            input_ids.push_back(xnn_ids[arg.node]);
        }
        auto status = xnn_define_concatenate(
            subgraph,
            static_cast<size_t>(axis),
            input_ids.size(), input_ids.data(),
            output_id, /*flags=*/0);
        if (status != xnn_status_success) { abort(); }
        return;
    }

    if (op.op == Operator::Unsqueeze) {
        auto dim = get_const<int64_t>(op.constant_args[0]);
        size_t axis = static_cast<size_t>(dim);
        auto status = xnn_define_static_expand_dims(
            subgraph,
            1, &axis,
            xnn_ids[op.args[0].node],
            output_id, /*flags=*/0);
        if (status != xnn_status_success) { abort(); }
        return;
    }

    if (op.op == Operator::Expand) {
        auto shape = get_const<std::vector<int64_t>>(op.constant_args[0]);
        auto new_shape = to_size_vec(shape);
        auto status = xnn_define_static_broadcast(
            subgraph,
            new_shape.size(), new_shape.data(),
            xnn_ids[op.args[0].node],
            output_id, /*flags=*/0);
        if (status != xnn_status_success) { abort(); }
        return;
    }

    if (op.op == Operator::Clone) {
        auto status = xnn_define_copy(
            subgraph,
            xnn_ids[op.args[0].node],
            output_id, /*flags=*/0);
        if (status != xnn_status_success) { abort(); }
        return;
    }

    if (op.op == Operator::Pad) {
        auto pad = get_const<std::vector<int64_t>>(op.constant_args[0]);
        auto& out_spec = std::get<graph::TensorSpec>(op.output_specs);
        auto ndims = out_spec.sizes.size();
        std::vector<size_t> pre_paddings(ndims, 0);
        std::vector<size_t> post_paddings(ndims, 0);
        for (size_t i = 0; i < pad.size() / 2; i++) {
            auto dim_idx = ndims - 1 - i;
            pre_paddings[dim_idx] = static_cast<size_t>(pad[2 * i]);
            post_paddings[dim_idx] = static_cast<size_t>(pad[2 * i + 1]);
        }

        auto status = xnn_define_static_constant_pad(
            subgraph, pre_paddings.data(), post_paddings.data(),
            0.0f,
            xnn_ids[op.args[0].node],
            output_id, /*flags=*/0);
        if (status != xnn_status_success) { abort(); }
        return;
    }

    if (op.op == Operator::Quantize || op.op == Operator::Dequantize) {
        auto status = xnn_define_unary(
            subgraph, xnn_unary_convert, nullptr,
            xnn_ids[op.args[0].node],
            output_id, /*flags=*/0);
        if (status != xnn_status_success) { abort(); }
        return;
    }

    abort();
}


uint32_t define_tensor(
    const graph::TensorSpec& spec,
    xnn_subgraph_t subgraph,
    bool is_input = false,
    bool is_output = false,
    uint32_t external_id = XNN_INVALID_VALUE_ID,
    const core::Tensor* constant_tensor = nullptr) {
    std::vector<size_t> dims(spec.sizes.size());
    for (auto i = 0u; i < spec.sizes.size(); i++) {
        auto& s = spec.sizes[i];
        dims[i] = s.is_constant() ? static_cast<size_t>(s.offset) : 1;
    }

    uint32_t flags = 0;
    if (is_input) { flags |= XNN_VALUE_FLAG_EXTERNAL_INPUT; }
    if (is_output) { flags |= XNN_VALUE_FLAG_EXTERNAL_OUTPUT; }

    const void* data = constant_tensor ? constant_tensor->storage.data : nullptr;
    auto xnn_dtype = map_xnn_datatype(spec);

    uint32_t id = 0;
    xnn_status status;

    if (!spec.quant_params) {
        status = xnn_define_tensor_value(
            subgraph, xnn_dtype,
            spec.sizes.size(), dims.data(),
            data, external_id, flags, &id);
    } else if (auto* pt = std::get_if<core::PerTensorQuant>(&*spec.quant_params)) {
        status = xnn_define_quantized_tensor_value(
            subgraph, xnn_dtype,
            pt->zero_point, pt->scale,
            spec.sizes.size(), dims.data(),
            data, external_id, flags, &id);
    } else if (auto* pa = std::get_if<core::PerAxisQuant>(&*spec.quant_params)) {
        assert(constant_tensor && !constant_tensor->aux_storage.empty());
        auto* scales = static_cast<const float*>(
            constant_tensor->aux_storage[0].data);
        status = xnn_define_channelwise_quantized_tensor_value(
            subgraph, xnn_dtype,
            scales,
            spec.sizes.size(), static_cast<size_t>(pa->axis),
            dims.data(),
            data, external_id, flags, &id);
    } else {
        abort();
    }

    if (status != xnn_status_success) { abort(); }
    return id;
}

} // namespace

XnnSubgraph build_xnn_subgraph(const graph::Graph& graph) {
    auto num_external_values = static_cast<uint32_t>(
        graph.input_specs.size() + graph.outputs.size());

    xnn_subgraph_t raw_subgraph = nullptr;
    auto status = xnn_create_subgraph(
        num_external_values,
        /*flags=*/0,
        &raw_subgraph
    );

    if (status != xnn_status_success) {
        abort();
    }

    auto subgraph = XnnSubgraph(raw_subgraph);

    std::vector<uint32_t> xnn_input_ids(graph.input_specs.size());
    for (auto i = 0u; i < graph.input_specs.size(); i++) {
        xnn_input_ids[i] = define_tensor(
            graph.input_specs[i],
            subgraph.get(),
            /*is_input=*/true,
            /*is_output=*/false,
            /*external_id=*/i);
    }

    std::vector<uint32_t> xnn_output_ids(graph.outputs.size());
    for (auto i = 0u; i < graph.outputs.size(); i++) {
        auto external_id = static_cast<uint32_t>(i + graph.input_specs.size());
        xnn_output_ids[i] = define_tensor(
            graph.get_tensor_spec(graph.outputs[i]),
            subgraph.get(),
            /*is_input=*/false,
            /*is_output=*/true,
            external_id);
    }

    std::vector<uint32_t> xnn_ids(graph.nodes.size(), XNN_INVALID_VALUE_ID);

    for (NodeHandle n = 0; n < graph.nodes.size(); n++) {
        if (auto* inp = std::get_if<InputNode>(&graph.nodes[n].value)) {
            xnn_ids[n] = xnn_input_ids[inp->input];
        }
    }

    for (NodeHandle n = 0; n < graph.nodes.size(); n++) {
        if (auto* cn = std::get_if<ConstantNode>(&graph.nodes[n].value)) {
            auto spec = std::get<graph::TensorSpec>(graph.get_output_spec_for_node(n));
            xnn_ids[n] = define_tensor(
                spec, subgraph.get(),
                /*is_input=*/false, /*is_output=*/false,
                XNN_INVALID_VALUE_ID, cn->tensor.get());
        }
    }

    for (auto i = 0u; i < graph.outputs.size(); i++) {
        xnn_ids[graph.outputs[i].node] = xnn_output_ids[i];
    }

    for (NodeHandle n = 0; n < graph.nodes.size(); n++) {
        auto* op = std::get_if<CallOperatorNode>(&graph.nodes[n].value);
        if (!op) continue;

        if (xnn_ids[n] == XNN_INVALID_VALUE_ID) {
            auto spec = std::get<graph::TensorSpec>(op->output_specs);
            xnn_ids[n] = define_tensor(spec, subgraph.get());
        }
    }

    for (NodeHandle n = 0; n < graph.nodes.size(); n++) {
        auto* op = std::get_if<CallOperatorNode>(&graph.nodes[n].value);
        if (!op) continue;

        define_node(*op, xnn_ids[n], xnn_ids, subgraph.get());
    }

    return subgraph;
}

XnnRuntime compile_xnn_subgraph(const graph::Graph& graph, xnn_workspace_t workspace) {
    auto subgraph = build_xnn_subgraph(graph);

    xnn_runtime_t runtime = nullptr;
    auto status = xnn_create_runtime_v4(
        subgraph.get(),
        /*weights_cache=*/nullptr,
        workspace,
        /*threadpool=*/nullptr,
        /*flags=*/0,
        &runtime
    );

    if (status != xnn_status_success) {
        abort();
    }

    return XnnRuntime(runtime);
}

}
