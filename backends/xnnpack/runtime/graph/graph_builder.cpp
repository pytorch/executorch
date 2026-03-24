#include <executorch/backends/xnnpack/runtime/graph/graph_builder.h>

#include <utility>

namespace executorch::backends::xnnpack::graph {

Graph GraphBuilder::build() {
    Graph g;
    g.input_specs = std::move(input_specs_);
    g.nodes = std::move(nodes_);
    g.outputs = std::move(outputs_);
    return g;
}

ValueHandle GraphBuilder::createInput(TensorSpec spec) {
    input_specs_.push_back(std::move(spec));

    InputHandle input = next_input_;
    next_input_++;

    ValueHandle handle{static_cast<uint16_t>(nodes_.size())};
    Node node;
    node.value = InputNode { input };
    nodes_.push_back(std::move(node));
    return handle;
}

ValueHandle GraphBuilder::createConstant(
    std::shared_ptr<const core::Tensor> tensor,
    std::optional<core::QuantParams> quant_params) {
    ValueHandle handle{static_cast<uint16_t>(nodes_.size())};
    ConstantNode cn;
    cn.tensor = std::move(tensor);
    cn.quant_params = std::move(quant_params);
    Node node;
    node.value = std::move(cn);
    nodes_.push_back(std::move(node));
    return handle;
}

ValueHandle GraphBuilder::createOperator(Operator op, TensorSpec output_spec, ValueHandles args) {
    ValueHandle handle{static_cast<uint16_t>(nodes_.size())};
    CallOperatorNode con;
    con.args = std::move(args);
    con.op = op;
    con.output_specs = std::move(output_spec);
    Node node;
    node.value = std::move(con);
    nodes_.push_back(std::move(node));
    return handle;
}

ValueHandle GraphBuilder::createOperator(
    Operator op, TensorSpec output_spec, ValueHandles args,
    std::vector<ConstantArg> constant_args) {
    ValueHandle handle{static_cast<uint16_t>(nodes_.size())};
    CallOperatorNode con;
    con.args = std::move(args);
    con.op = op;
    con.output_specs = std::move(output_spec);
    con.constant_args = std::move(constant_args);
    Node node;
    node.value = std::move(con);
    nodes_.push_back(std::move(node));
    return handle;
}

ValueHandle GraphBuilder::createOperatorM(Operator op, std::vector<TensorSpec> output_specs, ValueHandles args) {
    ValueHandle handle{static_cast<uint16_t>(nodes_.size())};
    CallOperatorNode con;
    con.args = std::move(args);
    con.op = op;
    con.output_specs = std::move(output_specs);
    Node node;
    node.value = std::move(con);
    nodes_.push_back(std::move(node));
    return handle;
}

OutputHandle GraphBuilder::createOutput(ValueHandle handle) {
    OutputHandle output = static_cast<OutputHandle>(outputs_.size());
    outputs_.push_back(handle);
    return output;
}

ValueHandle GraphBuilder::createSymInt() {
    return ValueHandle{static_cast<uint16_t>(next_sym_int_++)};
}

}
