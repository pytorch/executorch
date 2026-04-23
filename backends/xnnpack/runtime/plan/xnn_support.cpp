#include <executorch/backends/xnnpack/runtime/plan/xnn_support.h>

#include <executorch/backends/xnnpack/runtime/core/dtype.h>
#include <executorch/backends/xnnpack/runtime/core/variant_util.h>

namespace executorch::backends::xnnpack::plan {

namespace {

using namespace graph;

bool check_xnn_dtype_support(core::DType dtype) {
    switch (dtype) {
        case core::DType::Float32:
        case core::DType::QUInt8Asym:
        case core::DType::QInt8Sym:
        case core::DType::QInt32Sym:
            return true;
        default:
            return false;
    }
}

bool check_xnn_op_support(Operator op) {
    switch (op) {
        case Operator::Add:
        case Operator::Subtract:
        case Operator::Multiply:
        case Operator::Divide:
        case Operator::Maximum:
        case Operator::Minimum:
        case Operator::CopySign:
        case Operator::SquaredDifference:
        case Operator::PReLU:
        case Operator::Modulus:
        case Operator::Atan2:
        case Operator::Pow:
        case Operator::Abs:
        case Operator::Negate:
        case Operator::Clamp:
        case Operator::Ceiling:
        case Operator::Floor:
        case Operator::Round:
        case Operator::Square:
        case Operator::SquareRoot:
        case Operator::ReciprocalSquareRoot:
        case Operator::Exp:
        case Operator::Log:
        case Operator::Sigmoid:
        case Operator::Tanh:
        case Operator::ELU:
        case Operator::GELU:
        case Operator::HardSwish:
        case Operator::LeakyReLU:
        case Operator::Sine:
        case Operator::Cosine:
        case Operator::Sign:
        case Operator::ReLU:
        case Operator::Linear:
        case Operator::BatchMatrixMultiply:
        case Operator::Conv2d:
        case Operator::ConvTranspose2d:
        case Operator::AvgPool2d:
        case Operator::AdaptiveAvgPool2d:
        case Operator::MaxPool2d:
        case Operator::Softmax:
        case Operator::Mean:
        case Operator::Sum:
        case Operator::Reshape:
        case Operator::View:
        case Operator::Transpose:
        case Operator::Permute:
        case Operator::Slice:
        case Operator::Cat:
        case Operator::Unsqueeze:
        case Operator::Expand:
        case Operator::Clone:
        case Operator::Pad:
        case Operator::Quantize:
        case Operator::Dequantize:
            return true;
        default:
            return false;
    }
}

}

bool check_xnn_node_support(CallOperatorNode& node, Graph& graph) {
    if (!check_xnn_op_support(node.op)) {
        return false;
    }

    for (auto& arg : node.args) {
        if (arg.is_null()) continue;
        const auto& tensor_spec = graph.get_tensor_spec(arg);

        if (!check_xnn_dtype_support(tensor_spec.dtype)) {
            return false;
        }
    }

    return true;
}

}
