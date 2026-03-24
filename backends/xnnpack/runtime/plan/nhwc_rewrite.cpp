#include <executorch/backends/xnnpack/runtime/plan/nhwc_rewrite.h>

#include <executorch/backends/xnnpack/runtime/core/quant_params.h>
#include <executorch/backends/xnnpack/runtime/core/variant_util.h>
#include <executorch/backends/xnnpack/runtime/graph/node.h>
#include <executorch/backends/xnnpack/runtime/graph/operator.h>
#include <executorch/backends/xnnpack/runtime/graph/tensor_spec.h>

#include <cassert>
#include <cstring>
#include <vector>

namespace executorch::backends::xnnpack::plan {

using namespace graph;
namespace {

bool is_nhwc_op(Operator op) {
    switch (op) {
        case Operator::Conv2d:
        case Operator::ConvTranspose2d:
        case Operator::AvgPool2d:
        case Operator::AdaptiveAvgPool2d:
        case Operator::MaxPool2d:
            return true;
        default:
            return false;
    }
}

bool is_layout_agnostic(Operator op) {
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
        case Operator::Clone:
        case Operator::Pad:
        case Operator::Quantize:
        case Operator::Dequantize:
            return true;
        default:
            return false;
    }
}

int ndims(const TensorSpec& spec) {
    return static_cast<int>(spec.sizes.size());
}

TensorSpec permute_spec(const TensorSpec& spec, const std::vector<int>& perm) {
    TensorSpec out = spec;
    out.sizes.clear();
    out.sizes.reserve(perm.size());
    for (int p : perm) {
        out.sizes.push_back(spec.sizes[p]);
    }
    return out;
}

ValueHandle insert_permute(
    Graph& graph,
    ValueHandle input,
    const TensorSpec& output_spec,
    std::vector<int64_t> perm
) {
    auto node_idx = static_cast<uint16_t>(graph.nodes.size());
    CallOperatorNode con;
    con.args = { input };
    con.op = Operator::Permute;
    con.output_specs = output_spec;
    con.constant_args = { std::move(perm) };
    Node node;
    node.value = std::move(con);
    graph.nodes.push_back(std::move(node));
    return ValueHandle { node_idx };
}

void transpose_weight(Graph& graph, NodeHandle node_handle,
                      const std::vector<int>& perm) {
    auto& cnode = std::get<ConstantNode>(graph.nodes[node_handle].value);
    auto& old_tensor = *cnode.tensor;

    assert(old_tensor.sizes.size() == 4);
    assert(perm.size() == 4);

    std::vector<uint64_t> new_sizes(4);
    for (int i = 0; i < 4; i++) {
        new_sizes[i] = old_tensor.sizes[perm[i]];
    }

    uint64_t old_strides[4];
    old_strides[3] = 1;
    for (int i = 2; i >= 0; i--) {
        old_strides[i] = old_strides[i + 1] * old_tensor.sizes[i + 1];
    }

    uint64_t new_strides[4];
    new_strides[3] = 1;
    for (int i = 2; i >= 0; i--) {
        new_strides[i] = new_strides[i + 1] * new_sizes[i + 1];
    }

    size_t elem_size = core::element_size(old_tensor.dtype);
    size_t total_bytes = old_tensor.storage.size_in_bytes;

    auto new_storage = core::Storage::create_owned(total_bytes);
    auto* src = static_cast<const uint8_t*>(old_tensor.storage.data);
    auto* dst = static_cast<uint8_t*>(new_storage.data);

    for (uint64_t j0 = 0; j0 < new_sizes[0]; j0++) {
        for (uint64_t j1 = 0; j1 < new_sizes[1]; j1++) {
            for (uint64_t j2 = 0; j2 < new_sizes[2]; j2++) {
                for (uint64_t j3 = 0; j3 < new_sizes[3]; j3++) {
                    uint64_t old_idx[4] = {};
                    old_idx[perm[0]] = j0;
                    old_idx[perm[1]] = j1;
                    old_idx[perm[2]] = j2;
                    old_idx[perm[3]] = j3;

                    size_t src_offset = old_idx[0] * old_strides[0]
                                      + old_idx[1] * old_strides[1]
                                      + old_idx[2] * old_strides[2]
                                      + old_idx[3] * old_strides[3];
                    size_t dst_offset = j0 * new_strides[0]
                                      + j1 * new_strides[1]
                                      + j2 * new_strides[2]
                                      + j3 * new_strides[3];

                    std::memcpy(dst + dst_offset * elem_size,
                                src + src_offset * elem_size,
                                elem_size);
                }
            }
        }
    }

    auto new_tensor = std::make_shared<core::Tensor>();
    new_tensor->dtype = old_tensor.dtype;
    new_tensor->sizes = std::move(new_sizes);
    new_tensor->storage = std::move(new_storage);
    cnode.tensor = std::move(new_tensor);
}

TensorSpec get_spec(const Graph& graph, ValueHandle vh) {
    return graph.get_tensor_spec(vh);
}

} // namespace

void rewrite_nhwc(Graph& graph) {
    enum class Layout { NCHW, NHWC };

    std::vector<Layout> layout(graph.nodes.size(), Layout::NCHW);

    size_t original_count = graph.nodes.size();
    graph.nodes.reserve(original_count + original_count);

    auto ensure_nchw = [&](ValueHandle& vh) {
        if (layout[vh.node] == Layout::NHWC) {
            auto spec = get_spec(graph, vh);
            auto nchw_spec = permute_spec(spec, {0, 3, 1, 2});
            auto perm_vh = insert_permute(graph, vh, nchw_spec,
                                          {0, 3, 1, 2});
            layout.push_back(Layout::NCHW);
            vh = perm_vh;
        }
    };

    auto ensure_nhwc = [&](ValueHandle& vh) {
        if (layout[vh.node] == Layout::NCHW) {
            auto spec = get_spec(graph, vh);
            auto nhwc_spec = permute_spec(spec, {0, 2, 3, 1});
            auto perm_vh = insert_permute(graph, vh, nhwc_spec,
                                          {0, 2, 3, 1});
            layout.push_back(Layout::NHWC);
            vh = perm_vh;
        }
    };

    for (size_t i = 0; i < original_count; i++) {
        auto* call = std::get_if<CallOperatorNode>(&graph.nodes[i].value);
        if (!call) continue;

        auto& op = *call;

        if (is_nhwc_op(op.op)) {
            auto input_spec = get_spec(graph, op.args[0]);
            if (ndims(input_spec) == 4) {
                ensure_nhwc(op.args[0]);
            }

            if (op.op == Operator::Conv2d) {
                transpose_weight(graph, op.args[1].node, {0, 2, 3, 1});
            } else if (op.op == Operator::ConvTranspose2d) {
                transpose_weight(graph, op.args[1].node, {1, 2, 3, 0});
            }

            auto& out_spec = std::get<TensorSpec>(op.output_specs);
            if (ndims(out_spec) == 4) {
                out_spec = permute_spec(out_spec, {0, 2, 3, 1});
            }

            layout[i] = Layout::NHWC;

        } else if (is_layout_agnostic(op.op)) {
            bool all_nhwc = true;
            bool any_4d = false;
            for (auto& arg : op.args) {
                if (arg.is_null()) continue;
                auto spec = get_spec(graph, arg);
                if (ndims(spec) == 4) {
                    any_4d = true;
                    if (layout[arg.node] != Layout::NHWC) {
                        all_nhwc = false;
                    }
                }
            }

            if (any_4d && all_nhwc) {
                layout[i] = Layout::NHWC;
                std::visit(overloaded {
                    [](TensorSpec& s) {
                        if (static_cast<int>(s.sizes.size()) == 4) {
                            s = permute_spec(s, {0, 2, 3, 1});
                        }
                    },
                    [](std::vector<TensorSpec>& v) {
                        for (auto& s : v) {
                            if (static_cast<int>(s.sizes.size()) == 4) {
                                s = permute_spec(s, {0, 2, 3, 1});
                            }
                        }
                    },
                }, op.output_specs);
            } else {
                for (auto& arg : op.args) {
                    if (arg.is_null()) continue;
                    ensure_nchw(arg);
                }
                layout[i] = Layout::NCHW;
            }

        } else {
            for (auto& arg : op.args) {
                if (arg.is_null()) continue;
                auto spec = get_spec(graph, arg);
                if (ndims(spec) == 4) {
                    ensure_nchw(arg);
                }
            }
            layout[i] = Layout::NCHW;
        }
    }

    for (auto& out : graph.outputs) {
        if (out.is_null()) continue;
        auto spec = get_spec(graph, out);
        if (ndims(spec) == 4) {
            ensure_nchw(out);
        }
    }

    graph.update_users();
}

}
