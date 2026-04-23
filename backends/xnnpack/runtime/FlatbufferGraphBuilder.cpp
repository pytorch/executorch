#include <executorch/backends/xnnpack/runtime/FlatbufferGraphBuilder.h>

#include <executorch/backends/xnnpack/runtime/XNNHeader.h>
#include <executorch/backends/xnnpack/runtime/core/dtype.h>
#include <executorch/backends/xnnpack/runtime/core/quant_params.h>
#include <executorch/backends/xnnpack/runtime/core/tensor.h>
#include <executorch/backends/xnnpack/runtime/graph/graph_builder.h>
#include <executorch/backends/xnnpack/runtime/graph/handles.h>
#include <executorch/backends/xnnpack/runtime/graph/operator.h>
#include <executorch/backends/xnnpack/runtime/graph/tensor_spec.h>
#include <executorch/backends/xnnpack/serialization/schema_generated.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

namespace executorch::backends::xnnpack {

using namespace core;
using namespace graph;

namespace fb = fb_xnnpack;

static DType map_dtype(fb::XNNDatatype dt) {
    switch (dt) {
        case fb::XNNDatatype::xnn_datatype_fp32:
            return DType::Float32;
        case fb::XNNDatatype::xnn_datatype_qint8:
            return DType::QInt8Sym;
        case fb::XNNDatatype::xnn_datatype_quint8:
            return DType::QUInt8Asym;
        case fb::XNNDatatype::xnn_datatype_qcint8:
            return DType::QInt8Sym;
        case fb::XNNDatatype::xnn_datatype_qcint4:
            return DType::QInt4Sym;
        case fb::XNNDatatype::xnn_datatype_qint32:
            return DType::QInt32Sym;
        case fb::XNNDatatype::xnn_datatype_qcint32:
            return DType::QInt32Sym;
        case fb::XNNDatatype::xnn_datatype_qbint4:
            return DType::QInt4Sym;
        default:
            return DType::Float32;
    }
}

static std::vector<DimSizeSpec> dims_to_sizes(
    const flatbuffers::Vector<uint32_t>* dims) {
    std::vector<DimSizeSpec> sizes;
    sizes.reserve(dims->size());
    for (auto d : *dims) {
        sizes.push_back(DimSizeSpec::constant(static_cast<int64_t>(d)));
    }
    return sizes;
}

static QuantParams map_quant_params(
    const fb::XNNQuantizedTensorValue* qtv) {
    switch (qtv->quant_params_type()) {
        case fb::XNNQuantParams::PerTensorQuant: {
            auto qp = qtv->quant_params_as_PerTensorQuant();
            return PerTensorQuant{
                .scale = qp->scale(),
                .zero_point = qp->zero_point(),
            };
        }
        case fb::XNNQuantParams::PerChannelQuant: {
            auto qp = qtv->quant_params_as_PerChannelQuant();
            return PerAxisQuant{
                .axis = static_cast<int8_t>(qp->channel_dim()),
                .scale_dtype = DType::Float32,
            };
        }
        case fb::XNNQuantParams::PerChannelGroupQuant: {
            auto qp = qtv->quant_params_as_PerChannelGroupQuant();
            return BlockwiseQuant{
                .axis = static_cast<int8_t>(qp->channel_dim()),
                .block_size = static_cast<int32_t>(qp->group_size()),
                .scale_dtype = DType::Float32,
            };
        }
        case fb::XNNQuantParams::PerTokenDynamicQuant: {
            return PerTensorQuant{.scale = 1.0f, .zero_point = 0};
        }
        default:
            return PerTensorQuant{.scale = 1.0f, .zero_point = 0};
    }
}

static Operator map_binary_op(fb::XNodeUnion type) {
    switch (type) {
        case fb::XNodeUnion::XNNAdd: return Operator::Add;
        case fb::XNodeUnion::XNNSubtract: return Operator::Subtract;
        case fb::XNodeUnion::XNNMultiply: return Operator::Multiply;
        case fb::XNodeUnion::XNNDiv: return Operator::Divide;
        case fb::XNodeUnion::XNNMinimum: return Operator::Minimum;
        case fb::XNodeUnion::XNNMaximum: return Operator::Maximum;
        default: return Operator::Add;
    }
}

static bool is_binary_op(fb::XNodeUnion type) {
    switch (type) {
        case fb::XNodeUnion::XNNAdd:
        case fb::XNodeUnion::XNNSubtract:
        case fb::XNodeUnion::XNNMultiply:
        case fb::XNodeUnion::XNNDiv:
        case fb::XNodeUnion::XNNMinimum:
        case fb::XNodeUnion::XNNMaximum:
            return true;
        default:
            return false;
    }
}

static Operator map_unary_op(fb::XNodeUnion type) {
    switch (type) {
        case fb::XNodeUnion::XNNSigmoid: return Operator::Sigmoid;
        case fb::XNodeUnion::XNNFloor: return Operator::Floor;
        case fb::XNodeUnion::XNNSquareRoot: return Operator::SquareRoot;
        case fb::XNodeUnion::XNNReciprocalSquareRoot: return Operator::ReciprocalSquareRoot;
        case fb::XNodeUnion::XNNCeiling: return Operator::Ceiling;
        case fb::XNodeUnion::XNNGelu: return Operator::GELU;
        case fb::XNodeUnion::XNNHardswish: return Operator::HardSwish;
        case fb::XNodeUnion::XNNLog: return Operator::Log;
        case fb::XNodeUnion::XNNNegate: return Operator::Negate;
        case fb::XNodeUnion::XNNSquare: return Operator::Square;
        case fb::XNodeUnion::XNNAbs: return Operator::Abs;
        case fb::XNodeUnion::XNNSin: return Operator::Sine;
        case fb::XNodeUnion::XNNCos: return Operator::Cosine;
        case fb::XNodeUnion::XNNClamp: return Operator::Clamp;
        case fb::XNodeUnion::XNNLeakyReLU: return Operator::LeakyReLU;
        case fb::XNodeUnion::XNNELU: return Operator::ELU;
        default: return Operator::Abs;
    }
}

static bool is_unary_op(fb::XNodeUnion type) {
    switch (type) {
        case fb::XNodeUnion::XNNSigmoid:
        case fb::XNodeUnion::XNNFloor:
        case fb::XNodeUnion::XNNSquareRoot:
        case fb::XNodeUnion::XNNReciprocalSquareRoot:
        case fb::XNodeUnion::XNNCeiling:
        case fb::XNodeUnion::XNNGelu:
        case fb::XNodeUnion::XNNHardswish:
        case fb::XNodeUnion::XNNLog:
        case fb::XNodeUnion::XNNNegate:
        case fb::XNodeUnion::XNNSquare:
        case fb::XNodeUnion::XNNAbs:
        case fb::XNodeUnion::XNNSin:
        case fb::XNodeUnion::XNNCos:
        case fb::XNodeUnion::XNNClamp:
        case fb::XNodeUnion::XNNLeakyReLU:
        case fb::XNodeUnion::XNNELU:
            return true;
        default:
            return false;
    }
}

struct BuildContext {
    GraphBuilder builder;
    std::unordered_map<uint32_t, ValueHandle> id_map;
    std::unordered_map<uint32_t, TensorSpec> spec_map;
    const fb::XNNGraph* graph;
    const uint8_t* constant_data;

    ValueHandle lookup(uint32_t id) const {
        if (id == UINT32_MAX) return ValueHandle::null();
        auto it = id_map.find(id);
        assert(it != id_map.end());
        return it->second;
    }

    TensorSpec lookup_spec(uint32_t id) const {
        auto it = spec_map.find(id);
        assert(it != spec_map.end());
        return it->second;
    }
};

static TensorSpec make_spec(
    const fb::XNNTensorValue* tv,
    const fb::XNNQuantizedTensorValue* qtv) {
    TensorSpec spec;
    spec.dtype = map_dtype(tv->datatype());
    spec.sizes = dims_to_sizes(tv->dims());
    if (qtv) {
        spec.quant_params = map_quant_params(qtv);
    }
    return spec;
}

static void define_value(
    BuildContext& ctx,
    const fb::XValue* value) {
    const fb::XNNTensorValue* tv = nullptr;
    const fb::XNNQuantizedTensorValue* qtv = nullptr;

    switch (value->xvalue_union_type()) {
        case fb::XValueUnion::XNNTensorValue:
            tv = value->xvalue_union_as_XNNTensorValue();
            break;
        case fb::XValueUnion::XNNQuantizedTensorValue:
            qtv = value->xvalue_union_as_XNNQuantizedTensorValue();
            tv = qtv->tensor_value();
            break;
        default:
            return;
    }

    auto spec = make_spec(tv, qtv);
    uint32_t serial_id = tv->id_out();
    ctx.spec_map[serial_id] = spec;

    bool is_external_input = (tv->flags() & 1) != 0;  // XNN_VALUE_FLAG_EXTERNAL_INPUT
    bool is_external_output = (tv->flags() & 2) != 0;  // XNN_VALUE_FLAG_EXTERNAL_OUTPUT
    bool has_constant_data = tv->constant_buffer_idx() != 0;

    if (is_external_input) {
        auto handle = ctx.builder.createInput(spec);
        ctx.id_map[serial_id] = handle;
    } else if (has_constant_data) {
        auto tensor = std::make_shared<Tensor>();
        tensor->dtype = spec.dtype;
        for (auto& dim : spec.sizes) {
            assert(dim.coeffs.empty());
            tensor->sizes.push_back(static_cast<uint64_t>(dim.offset));
        }

        uint32_t buf_idx = tv->constant_buffer_idx();
        auto offsets = ctx.graph->constant_data();
        if (offsets && buf_idx < offsets->size()) {
            auto offset_entry = offsets->Get(buf_idx);
            size_t data_offset = offset_entry->offset();
            size_t data_size = offset_entry->size();

            tensor->storage = Storage::create_owned(data_size);
            if (ctx.constant_data) {
                std::memcpy(tensor->storage.data,
                    ctx.constant_data + data_offset, data_size);
            }
        }

        std::optional<QuantParams> qp;
        if (qtv) qp = map_quant_params(qtv);
        auto handle = ctx.builder.createConstant(tensor, qp);
        ctx.id_map[serial_id] = handle;
    } else if (is_external_output) {
        // Outputs that are not inputs or constants will be mapped
        // when a node produces them.
    }
    // Internal intermediates will be created as operator outputs.
}

static const fb::_XNNNode2x1* as_binary(const fb::XNode* node) {
    switch (node->xnode_union_type()) {
        case fb::XNodeUnion::XNNAdd: return node->xnode_union_as_XNNAdd();
        case fb::XNodeUnion::XNNSubtract: return node->xnode_union_as_XNNSubtract();
        case fb::XNodeUnion::XNNMultiply: return node->xnode_union_as_XNNMultiply();
        case fb::XNodeUnion::XNNDiv: return node->xnode_union_as_XNNDiv();
        case fb::XNodeUnion::XNNMinimum: return node->xnode_union_as_XNNMinimum();
        case fb::XNodeUnion::XNNMaximum: return node->xnode_union_as_XNNMaximum();
        case fb::XNodeUnion::XNNPReLU: return node->xnode_union_as_XNNPReLU();
        case fb::XNodeUnion::XNNBatchMatrixMultiply: return node->xnode_union_as_XNNBatchMatrixMultiply();
        default: return nullptr;
    }
}

static const fb::_XNNNode1x1* as_unary(const fb::XNode* node) {
    switch (node->xnode_union_type()) {
        case fb::XNodeUnion::XNNSigmoid: return node->xnode_union_as_XNNSigmoid();
        case fb::XNodeUnion::XNNFloor: return node->xnode_union_as_XNNFloor();
        case fb::XNodeUnion::XNNSquareRoot: return node->xnode_union_as_XNNSquareRoot();
        case fb::XNodeUnion::XNNReciprocalSquareRoot: return node->xnode_union_as_XNNReciprocalSquareRoot();
        case fb::XNodeUnion::XNNCeiling: return node->xnode_union_as_XNNCeiling();
        case fb::XNodeUnion::XNNGelu: return node->xnode_union_as_XNNGelu();
        case fb::XNodeUnion::XNNHardswish: return node->xnode_union_as_XNNHardswish();
        case fb::XNodeUnion::XNNLog: return node->xnode_union_as_XNNLog();
        case fb::XNodeUnion::XNNNegate: return node->xnode_union_as_XNNNegate();
        case fb::XNodeUnion::XNNSquare: return node->xnode_union_as_XNNSquare();
        case fb::XNodeUnion::XNNAbs: return node->xnode_union_as_XNNAbs();
        case fb::XNodeUnion::XNNSin: return node->xnode_union_as_XNNSin();
        case fb::XNodeUnion::XNNCos: return node->xnode_union_as_XNNCos();
        case fb::XNodeUnion::XNNClamp: return node->xnode_union_as_XNNClamp();
        case fb::XNodeUnion::XNNSoftmax: return node->xnode_union_as_XNNSoftmax();
        case fb::XNodeUnion::XNNGlobalAvgPooling2d: return node->xnode_union_as_XNNGlobalAvgPooling2d();
        case fb::XNodeUnion::XNNTanh: return node->xnode_union_as_XNNTanh();
        case fb::XNodeUnion::XNNExp: return node->xnode_union_as_XNNExp();
        case fb::XNodeUnion::XNNCopy: return node->xnode_union_as_XNNCopy();
        default: return nullptr;
    }
}

static void define_binary_node(
    BuildContext& ctx,
    const fb::XNode* node) {
    auto n = as_binary(node);
    auto output_spec = ctx.lookup_spec(n->output_id());
    auto op = map_binary_op(node->xnode_union_type());
    auto handle = ctx.builder.createOperator(op, output_spec,
        {ctx.lookup(n->input1_id()), ctx.lookup(n->input2_id())});
    ctx.id_map[n->output_id()] = handle;
}

static void define_unary_node(
    BuildContext& ctx,
    const fb::XNode* node) {
    uint32_t input_id = 0;
    uint32_t output_id = 0;

    // Most unary ops use _XNNNode1x1, but ELU and LeakyReLU have custom types.
    switch (node->xnode_union_type()) {
        case fb::XNodeUnion::XNNELU: {
            auto n = node->xnode_union_as_XNNELU();
            input_id = n->input_id();
            output_id = n->output_id();
            break;
        }
        case fb::XNodeUnion::XNNLeakyReLU: {
            auto n = node->xnode_union_as_XNNLeakyReLU();
            input_id = n->input_id();
            output_id = n->output_id();
            break;
        }
        default: {
            auto n = as_unary(node);
            input_id = n->input_id();
            output_id = n->output_id();
            break;
        }
    }

    auto output_spec = ctx.lookup_spec(output_id);
    auto op = map_unary_op(node->xnode_union_type());
    auto handle = ctx.builder.createOperator(op, output_spec,
        {ctx.lookup(input_id)});
    ctx.id_map[output_id] = handle;
}

static void define_fully_connected_node(
    BuildContext& ctx,
    const fb::XNode* node) {
    auto n = node->xnode_union_as_XNNFullyConnected();
    auto output_spec = ctx.lookup_spec(n->output_id());
    auto input = ctx.lookup(n->input1_id());
    auto filter = ctx.lookup(n->filter_id());
    auto bias = ctx.lookup(n->bias_id());
    auto handle = ctx.builder.createOperator(Operator::Linear, output_spec,
        {input, filter, bias});
    ctx.id_map[n->output_id()] = handle;
}

static void define_conv2d_node(
    BuildContext& ctx,
    const fb::XNode* node) {
    auto n = node->xnode_union_as_XNNConv2d();
    auto output_spec = ctx.lookup_spec(n->output_id());
    auto input = ctx.lookup(n->input1_id());
    auto filter = ctx.lookup(n->filter_id());
    auto bias = ctx.lookup(n->bias_id());
    auto handle = ctx.builder.createOperator(Operator::Conv2d, output_spec,
        {input, filter, bias});
    ctx.id_map[n->output_id()] = handle;
}

static void define_conv_transpose2d_node(
    BuildContext& ctx,
    const fb::XNode* node) {
    auto n = node->xnode_union_as_XNNConvTranspose2d();
    auto output_spec = ctx.lookup_spec(n->output_id());
    auto input = ctx.lookup(n->input1_id());
    auto filter = ctx.lookup(n->filter_id());
    auto bias = ctx.lookup(n->bias_id());
    auto handle = ctx.builder.createOperator(Operator::ConvTranspose2d, output_spec,
        {input, filter, bias});
    ctx.id_map[n->output_id()] = handle;
}

static void define_depthwise_conv2d_node(
    BuildContext& ctx,
    const fb::XNode* node) {
    auto n = node->xnode_union_as_XNNDepthwiseConv2d();
    auto output_spec = ctx.lookup_spec(n->output_id());
    auto input = ctx.lookup(n->input1_id());
    auto filter = ctx.lookup(n->filter_id());
    auto bias = ctx.lookup(n->bias_id());
    auto handle = ctx.builder.createOperator(Operator::Conv2d, output_spec,
        {input, filter, bias});
    ctx.id_map[n->output_id()] = handle;
}

static void define_softmax_node(
    BuildContext& ctx,
    const fb::XNode* node) {
    auto n = node->xnode_union_as_XNNSoftmax();
    auto output_spec = ctx.lookup_spec(n->output_id());
    auto handle = ctx.builder.createOperator(Operator::Softmax, output_spec,
        {ctx.lookup(n->input_id())});
    ctx.id_map[n->output_id()] = handle;
}

static void define_convert_node(
    BuildContext& ctx,
    const fb::XNode* node) {
    auto n = node->xnode_union_as_XNNConvert();
    auto input_spec = ctx.lookup_spec(n->input_id());
    auto output_spec = ctx.lookup_spec(n->output_id());

    // Determine if this is a quantize or dequantize based on dtypes.
    bool input_quantized = is_quantized(input_spec.dtype);
    bool output_quantized = is_quantized(output_spec.dtype);

    Operator op;
    if (input_quantized && !output_quantized) {
        op = Operator::Dequantize;
    } else if (!input_quantized && output_quantized) {
        op = Operator::Quantize;
    } else {
        op = Operator::Clone;
    }

    auto handle = ctx.builder.createOperator(op, output_spec,
        {ctx.lookup(n->input_id())});
    ctx.id_map[n->output_id()] = handle;
}

static void define_static_transpose_node(
    BuildContext& ctx,
    const fb::XNode* node) {
    auto n = node->xnode_union_as_XNNStaticTranspose();
    auto output_spec = ctx.lookup_spec(n->output_id());
    auto handle = ctx.builder.createOperator(Operator::Permute, output_spec,
        {ctx.lookup(n->input_id())});
    ctx.id_map[n->output_id()] = handle;
}

static void define_static_reshape_node(
    BuildContext& ctx,
    const fb::XNode* node) {
    auto n = node->xnode_union_as_XNNStaticReshape();
    auto output_spec = ctx.lookup_spec(n->output_id());
    auto handle = ctx.builder.createOperator(Operator::Reshape, output_spec,
        {ctx.lookup(n->input_id())});
    ctx.id_map[n->output_id()] = handle;
}

static void define_static_slice_node(
    BuildContext& ctx,
    const fb::XNode* node) {
    auto n = node->xnode_union_as_XNNStaticSlice();
    auto output_spec = ctx.lookup_spec(n->output_id());
    auto handle = ctx.builder.createOperator(Operator::Slice, output_spec,
        {ctx.lookup(n->input_id())});
    ctx.id_map[n->output_id()] = handle;
}

static void define_static_constant_pad_node(
    BuildContext& ctx,
    const fb::XNode* node) {
    auto n = node->xnode_union_as_XNNStaticConstantPad();
    auto output_spec = ctx.lookup_spec(n->output_id());
    auto handle = ctx.builder.createOperator(Operator::Pad, output_spec,
        {ctx.lookup(n->input_id())});
    ctx.id_map[n->output_id()] = handle;
}

static void define_avg_pooling_node(
    BuildContext& ctx,
    const fb::XNode* node) {
    auto n = node->xnode_union_as_XNNAvgPooling2d();
    auto output_spec = ctx.lookup_spec(n->output_id());
    auto handle = ctx.builder.createOperator(Operator::AvgPool2d, output_spec,
        {ctx.lookup(n->input_id())});
    ctx.id_map[n->output_id()] = handle;
}

static void define_max_pooling_node(
    BuildContext& ctx,
    const fb::XNode* node) {
    auto n = node->xnode_union_as_XNNMaxPooling2d();
    auto output_spec = ctx.lookup_spec(n->output_id());
    auto handle = ctx.builder.createOperator(Operator::MaxPool2d, output_spec,
        {ctx.lookup(n->input_id())});
    ctx.id_map[n->output_id()] = handle;
}

static void define_global_avg_pooling_node(
    BuildContext& ctx,
    const fb::XNode* node) {
    auto n = node->xnode_union_as_XNNGlobalAvgPooling2d();
    auto output_spec = ctx.lookup_spec(n->output_id());
    auto handle = ctx.builder.createOperator(Operator::AdaptiveAvgPool2d, output_spec,
        {ctx.lookup(n->input_id())});
    ctx.id_map[n->output_id()] = handle;
}

static void define_batch_matrix_multiply_node(
    BuildContext& ctx,
    const fb::XNode* node) {
    auto n = node->xnode_union_as_XNNBatchMatrixMultiply();
    auto output_spec = ctx.lookup_spec(n->output_id());
    auto handle = ctx.builder.createOperator(Operator::BatchMatrixMultiply, output_spec,
        {ctx.lookup(n->input1_id()), ctx.lookup(n->input2_id())});
    ctx.id_map[n->output_id()] = handle;
}

static void define_prelu_node(
    BuildContext& ctx,
    const fb::XNode* node) {
    auto n = node->xnode_union_as_XNNPReLU();
    auto output_spec = ctx.lookup_spec(n->output_id());
    auto handle = ctx.builder.createOperator(Operator::PReLU, output_spec,
        {ctx.lookup(n->input1_id()), ctx.lookup(n->input2_id())});
    ctx.id_map[n->output_id()] = handle;
}

static void define_copy_node(
    BuildContext& ctx,
    const fb::XNode* node) {
    auto n = node->xnode_union_as_XNNCopy();
    auto output_spec = ctx.lookup_spec(n->output_id());
    auto handle = ctx.builder.createOperator(Operator::Clone, output_spec,
        {ctx.lookup(n->input_id())});
    ctx.id_map[n->output_id()] = handle;
}

static void define_concatenate_node(
    BuildContext& ctx,
    const fb::XNode* node) {
    const fb::_XNNCat* n = nullptr;
    switch (node->xnode_union_type()) {
        case fb::XNodeUnion::XNNConcatenate2: n = node->xnode_union_as_XNNConcatenate2(); break;
        case fb::XNodeUnion::XNNConcatenate3: n = node->xnode_union_as_XNNConcatenate3(); break;
        case fb::XNodeUnion::XNNConcatenate4: n = node->xnode_union_as_XNNConcatenate4(); break;
        case fb::XNodeUnion::XNNConcatenate5: n = node->xnode_union_as_XNNConcatenate5(); break;
        default: return;
    }
    auto output_spec = ctx.lookup_spec(n->output_id());

    ValueHandles inputs;
    inputs.push_back(ctx.lookup(n->input1_id()));
    inputs.push_back(ctx.lookup(n->input2_id()));
    if (node->xnode_union_type() == fb::XNodeUnion::XNNConcatenate3 ||
        node->xnode_union_type() == fb::XNodeUnion::XNNConcatenate4 ||
        node->xnode_union_type() == fb::XNodeUnion::XNNConcatenate5) {
        inputs.push_back(ctx.lookup(n->input3_id()));
    }
    if (node->xnode_union_type() == fb::XNodeUnion::XNNConcatenate4 ||
        node->xnode_union_type() == fb::XNodeUnion::XNNConcatenate5) {
        inputs.push_back(ctx.lookup(n->input4_id()));
    }
    if (node->xnode_union_type() == fb::XNodeUnion::XNNConcatenate5) {
        inputs.push_back(ctx.lookup(n->input5_id()));
    }

    auto handle = ctx.builder.createOperator(Operator::Cat, output_spec, inputs);
    ctx.id_map[n->output_id()] = handle;
}

static void define_tanh_node(
    BuildContext& ctx,
    const fb::XNode* node) {
    auto n = node->xnode_union_as_XNNTanh();
    auto output_spec = ctx.lookup_spec(n->output_id());
    auto handle = ctx.builder.createOperator(Operator::Tanh, output_spec,
        {ctx.lookup(n->input_id())});
    ctx.id_map[n->output_id()] = handle;
}

static void define_exp_node(
    BuildContext& ctx,
    const fb::XNode* node) {
    auto n = node->xnode_union_as_XNNExp();
    auto output_spec = ctx.lookup_spec(n->output_id());
    auto handle = ctx.builder.createOperator(Operator::Exp, output_spec,
        {ctx.lookup(n->input_id())});
    ctx.id_map[n->output_id()] = handle;
}

static void define_node(BuildContext& ctx, const fb::XNode* node) {
    auto type = node->xnode_union_type();

    if (is_binary_op(type)) {
        define_binary_node(ctx, node);
    } else if (is_unary_op(type)) {
        define_unary_node(ctx, node);
    } else {
        switch (type) {
            case fb::XNodeUnion::XNNFullyConnected:
                define_fully_connected_node(ctx, node);
                break;
            case fb::XNodeUnion::XNNConv2d:
                define_conv2d_node(ctx, node);
                break;
            case fb::XNodeUnion::XNNConvTranspose2d:
                define_conv_transpose2d_node(ctx, node);
                break;
            case fb::XNodeUnion::XNNDepthwiseConv2d:
                define_depthwise_conv2d_node(ctx, node);
                break;
            case fb::XNodeUnion::XNNSoftmax:
                define_softmax_node(ctx, node);
                break;
            case fb::XNodeUnion::XNNConvert:
                define_convert_node(ctx, node);
                break;
            case fb::XNodeUnion::XNNStaticTranspose:
                define_static_transpose_node(ctx, node);
                break;
            case fb::XNodeUnion::XNNStaticReshape:
                define_static_reshape_node(ctx, node);
                break;
            case fb::XNodeUnion::XNNStaticSlice:
                define_static_slice_node(ctx, node);
                break;
            case fb::XNodeUnion::XNNStaticConstantPad:
                define_static_constant_pad_node(ctx, node);
                break;
            case fb::XNodeUnion::XNNAvgPooling2d:
                define_avg_pooling_node(ctx, node);
                break;
            case fb::XNodeUnion::XNNMaxPooling2d:
                define_max_pooling_node(ctx, node);
                break;
            case fb::XNodeUnion::XNNGlobalAvgPooling2d:
                define_global_avg_pooling_node(ctx, node);
                break;
            case fb::XNodeUnion::XNNBatchMatrixMultiply:
                define_batch_matrix_multiply_node(ctx, node);
                break;
            case fb::XNodeUnion::XNNPReLU:
                define_prelu_node(ctx, node);
                break;
            case fb::XNodeUnion::XNNCopy:
                define_copy_node(ctx, node);
                break;
            case fb::XNodeUnion::XNNConcatenate2:
            case fb::XNodeUnion::XNNConcatenate3:
            case fb::XNodeUnion::XNNConcatenate4:
            case fb::XNodeUnion::XNNConcatenate5:
                define_concatenate_node(ctx, node);
                break;
            case fb::XNodeUnion::XNNTanh:
                define_tanh_node(ctx, node);
                break;
            case fb::XNodeUnion::XNNExp:
                define_exp_node(ctx, node);
                break;
            case fb::XNodeUnion::XNNStaticResizeBilinear2D:
                // Resize maps to a reshape-like op; skip for now.
                break;
            default:
                break;
        }
    }
}

FlatbufferBuildResult FlatbufferGraphBuilder::build(
    const void* buffer, size_t size) {
    using delegate::XNNHeader;

    auto header = XNNHeader::Parse(buffer, size);
    const uint8_t* flatbuffer_data = nullptr;
    const uint8_t* constant_data = nullptr;

    if (header.ok()) {
        flatbuffer_data = reinterpret_cast<const uint8_t*>(buffer) +
            header->flatbuffer_offset;
        constant_data = reinterpret_cast<const uint8_t*>(buffer) +
            header->constant_data_offset;
    } else {
        flatbuffer_data = reinterpret_cast<const uint8_t*>(buffer);
    }

    auto fb_graph = fb::GetXNNGraph(flatbuffer_data);

    BuildContext ctx;
    ctx.graph = fb_graph;
    ctx.constant_data = constant_data;

    // Collect input/output external_id mappings.
    struct ExternalEntry {
        uint32_t external_id;
        uint32_t serial_id;
    };
    std::vector<ExternalEntry> input_entries;
    std::vector<ExternalEntry> output_entries;

    for (auto value : *fb_graph->xvalues()) {
        const fb::XNNTensorValue* tv = nullptr;
        switch (value->xvalue_union_type()) {
            case fb::XValueUnion::XNNTensorValue:
                tv = value->xvalue_union_as_XNNTensorValue();
                break;
            case fb::XValueUnion::XNNQuantizedTensorValue:
                tv = value->xvalue_union_as_XNNQuantizedTensorValue()->tensor_value();
                break;
            default:
                continue;
        }
        if (tv->flags() & 1) {  // XNN_VALUE_FLAG_EXTERNAL_INPUT
            input_entries.push_back({tv->external_id(), tv->id_out()});
        }
        if (tv->flags() & 2) {  // XNN_VALUE_FLAG_EXTERNAL_OUTPUT
            output_entries.push_back({tv->external_id(), tv->id_out()});
        }
    }

    // Sort by external_id so graph input/output order matches.
    auto by_external_id = [](const ExternalEntry& a, const ExternalEntry& b) {
        return a.external_id < b.external_id;
    };
    std::sort(input_entries.begin(), input_entries.end(), by_external_id);
    std::sort(output_entries.begin(), output_entries.end(), by_external_id);

    // Define values (inputs, constants, intermediates).
    // Create inputs in sorted external_id order so graph input indices match.
    for (auto& entry : input_entries) {
        // define_value will be called for all values below, but we need
        // inputs created in the right order. Pre-create them here.
        auto value_it = std::find_if(
            fb_graph->xvalues()->begin(), fb_graph->xvalues()->end(),
            [&](const fb::XValue* v) {
                const fb::XNNTensorValue* tv = nullptr;
                switch (v->xvalue_union_type()) {
                    case fb::XValueUnion::XNNTensorValue:
                        tv = v->xvalue_union_as_XNNTensorValue();
                        break;
                    case fb::XValueUnion::XNNQuantizedTensorValue:
                        tv = v->xvalue_union_as_XNNQuantizedTensorValue()->tensor_value();
                        break;
                    default:
                        return false;
                }
                return tv->id_out() == entry.serial_id;
            });
        if (value_it != fb_graph->xvalues()->end()) {
            define_value(ctx, *value_it);
        }
    }

    // Define remaining values (constants, intermediates, outputs-only).
    for (auto value : *fb_graph->xvalues()) {
        const fb::XNNTensorValue* tv = nullptr;
        switch (value->xvalue_union_type()) {
            case fb::XValueUnion::XNNTensorValue:
                tv = value->xvalue_union_as_XNNTensorValue();
                break;
            case fb::XValueUnion::XNNQuantizedTensorValue:
                tv = value->xvalue_union_as_XNNQuantizedTensorValue()->tensor_value();
                break;
            default:
                continue;
        }
        if (ctx.id_map.count(tv->id_out()) == 0) {
            define_value(ctx, value);
        }
    }

    // Define nodes (operators).
    for (auto node : *fb_graph->xnodes()) {
        define_node(ctx, node);
    }

    // Wire up outputs in external_id order.
    for (auto& entry : output_entries) {
        auto it = ctx.id_map.find(entry.serial_id);
        if (it != ctx.id_map.end()) {
            ctx.builder.createOutput(it->second);
        }
    }

    FlatbufferBuildResult result;
    result.graph = ctx.builder.build();
    result.input_external_ids.reserve(input_entries.size());
    for (auto& e : input_entries) {
        result.input_external_ids.push_back(e.external_id);
    }
    result.output_external_ids.reserve(output_entries.size());
    for (auto& e : output_entries) {
        result.output_external_ids.push_back(e.external_id);
    }
    return result;
}

}
