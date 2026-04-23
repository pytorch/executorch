#include <executorch/backends/xnnpack/runtime/plan/partition.h>
#include <executorch/backends/xnnpack/runtime/plan/xnn_support.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <deque>
#include <limits>
#include <optional>

namespace executorch::backends::xnnpack::plan {

using namespace graph;

namespace {

std::optional<NodeHandle> take_from_queue(
    Graph& graph,
    std::deque<NodeHandle>& queue,
    std::vector<uint16_t>& block_in_partition,
    std::vector<uint32_t>& in_edges,
    std::vector<NodeHandle>& deferred,
    uint16_t current_partition
) {
    while (!queue.empty()) {
        auto node_handle = queue.front();
        queue.pop_front();

        auto& node = graph.nodes[node_handle];

        assert((node.flags & NodeFlags::UseXnnpack) != NodeFlags::None);

        if (node.tag != 0) { continue; }
        if (in_edges[node_handle] != 0) { continue; }
        if (block_in_partition[node_handle] == current_partition) {
            deferred.push_back(node_handle);
            continue;
        }

        return node_handle;
    }

    return {};
}

void update_block_frontier(
    Graph& graph,
    Node& node,
    std::vector<uint16_t>& block_in_partition,
    std::vector<uint16_t>& seen_in_partition,
    uint16_t current_partition
) {
    std::deque<NodeHandle> queue;
    for (auto user_handle : node.users) {
        auto& user = graph.nodes[user_handle];
        if ((user.flags & NodeFlags::UseXnnpack) == NodeFlags::None) {
            queue.push_back(user_handle);
        }
    }

    while (!queue.empty()) {
        auto handle = queue.front();
        queue.pop_front();

        if (seen_in_partition[handle] == current_partition) { continue; }
        seen_in_partition[handle] = current_partition;

        for (auto user_handle : graph.nodes[handle].users) {
            auto& user = graph.nodes[user_handle];
            if ((user.flags & NodeFlags::UseXnnpack) != NodeFlags::None) {
                block_in_partition[user_handle] = current_partition;
            } else {
                queue.push_back(user_handle);
            }
        }
    }
}

} // namespace

uint16_t assign_partitions(Graph& graph) {
    uint16_t current_partition_id = 0;

    std::deque<NodeHandle> delegated_escape_queue;
    std::deque<NodeHandle> delegated_noescape_queue;
    std::deque<NodeHandle> non_delegated_queue;

    uint32_t remaining_delegated_node_count = 0;

    std::vector<uint16_t> block_in_partition(graph.nodes.size(), 0);
    std::vector<uint16_t> seen_in_partition(graph.nodes.size(), 0);
    std::vector<uint32_t> in_edges(graph.nodes.size(), 0);

    for (NodeHandle n = 0u; n < graph.nodes.size(); n++) {
        auto& node = graph.nodes[n];
        auto& args = node.get_args();

        auto has_nondelegated_user = std::any_of(
            node.users.begin(),
            node.users.end(),
            [&](NodeHandle u) { return (graph.nodes[u].flags & NodeFlags::UseXnnpack) == NodeFlags::None; }
        );
        if (has_nondelegated_user) {
            node.flags |= NodeFlags::PassInternal1;
        } else {
            node.flags &= ~NodeFlags::PassInternal1;
        }

        in_edges[n] = std::count_if(
            args.begin(),
            args.end(),
            [&](const ValueHandle& a) {
                if (a.is_null()) return false;
                return !std::holds_alternative<InputNode>(graph.nodes[a.node].value)
                    && !std::holds_alternative<ConstantNode>(graph.nodes[a.node].value);
            });

        if ((node.flags & NodeFlags::UseXnnpack) != NodeFlags::None) {
            remaining_delegated_node_count++;
        }

        if (in_edges[n] == 0) {
            if ((node.flags & NodeFlags::UseXnnpack) != NodeFlags::None) {
                if ((node.flags & NodeFlags::PassInternal1) != NodeFlags::None) {
                    delegated_escape_queue.push_back(n);
                } else {
                    delegated_noescape_queue.push_back(n);
                }
            } else {
                non_delegated_queue.push_back(n);
            }
        }
    }

    while (remaining_delegated_node_count > 0) {
        std::vector<NodeHandle> deferred;

        if (current_partition_id == std::numeric_limits<uint16_t>::max()) {
            abort();
        }
        current_partition_id++;

        while (true) {
            while (!non_delegated_queue.empty()) {
                auto ndh = non_delegated_queue.front();
                non_delegated_queue.pop_front();

                bool is_input = std::holds_alternative<InputNode>(graph.nodes[ndh].value)
                    || std::holds_alternative<ConstantNode>(graph.nodes[ndh].value);

                for (auto user : graph.nodes[ndh].users) {
                    if (is_input) { continue; }
                    assert(in_edges[user] > 0);
                    in_edges[user]--;
                    if (in_edges[user] == 0) {
                        if ((graph.nodes[user].flags & NodeFlags::UseXnnpack) != NodeFlags::None) {
                            if ((graph.nodes[user].flags & NodeFlags::PassInternal1) != NodeFlags::None) {
                                delegated_escape_queue.push_back(user);
                            } else {
                                delegated_noescape_queue.push_back(user);
                            }
                        } else {
                            non_delegated_queue.push_back(user);
                        }
                    }
                }
            }

            std::optional<NodeHandle> nh = take_from_queue(
                graph,
                delegated_noescape_queue,
                block_in_partition,
                in_edges,
                deferred,
                current_partition_id
            );

            if (!nh) {
                nh = take_from_queue(
                    graph,
                    delegated_escape_queue,
                    block_in_partition,
                    in_edges,
                    deferred,
                    current_partition_id
                );
            }

            if (!nh) {
                break;
            }

            auto& node = graph.nodes[*nh];
            node.tag = current_partition_id;
            remaining_delegated_node_count--;

            if ((node.flags & NodeFlags::PassInternal1) != NodeFlags::None) {
                update_block_frontier(
                    graph, node, block_in_partition, seen_in_partition,
                    current_partition_id);
            }

            for (auto user : node.users) {
                assert(in_edges[user] > 0);

                in_edges[user]--;
                if (in_edges[user] == 0) {
                    if ((graph.nodes[user].flags & NodeFlags::UseXnnpack) != NodeFlags::None) {
                        if ((graph.nodes[user].flags & NodeFlags::PassInternal1) != NodeFlags::None) {
                            delegated_escape_queue.push_back(user);
                        } else {
                            delegated_noescape_queue.push_back(user);
                        }
                    } else {
                        non_delegated_queue.push_back(user);
                    }
                }
            }
        }

        for (auto handle : deferred) {
            if ((graph.nodes[handle].flags & NodeFlags::PassInternal1) != NodeFlags::None) {
                delegated_escape_queue.push_back(handle);
            } else {
                delegated_noescape_queue.push_back(handle);
            }
        }
    }

    return current_partition_id;
}

namespace {

void tag_xnn_nodes(Graph& graph) {
    for (auto& node : graph.nodes) {
        auto* op_node = std::get_if<CallOperatorNode>(&node.value);
        if (op_node && check_xnn_node_support(*op_node, graph)) {
            node.flags |= NodeFlags::UseXnnpack;
        }
    }
}

} // namespace

void fuse_partitions(Graph& graph, uint16_t partition_count) {
    const auto sentinel = std::numeric_limits<NodeHandle>::max();

    for (uint16_t p = 1; p <= partition_count; p++) {
        std::vector<NodeHandle> members;
        for (NodeHandle n = 0; n < graph.nodes.size(); n++) {
            if (graph.nodes[n].tag == p) {
                members.push_back(n);
            }
        }
        if (members.empty()) { continue; }

        auto is_member = [&](NodeHandle h) {
            return std::find(members.begin(), members.end(), h) != members.end();
        };

        std::vector<ValueHandle> ext_inputs;
        for (auto m : members) {
            for (auto arg : graph.nodes[m].get_args()) {
                if (arg.is_null()) continue;
                if (std::holds_alternative<ConstantNode>(graph.nodes[arg.node].value)) continue;
                if (!is_member(arg.node) &&
                    std::find(ext_inputs.begin(), ext_inputs.end(), arg) == ext_inputs.end()) {
                    ext_inputs.push_back(arg);
                }
            }
        }

        std::vector<NodeHandle> output_members;
        for (auto m : members) {
            bool external = std::any_of(
                graph.nodes[m].users.begin(), graph.nodes[m].users.end(),
                [&](NodeHandle u) { return !is_member(u); });
            if (!external) {
                external = std::any_of(
                    graph.outputs.begin(), graph.outputs.end(),
                    [&](const ValueHandle& vh) { return vh.node == m; });
            }
            if (external) {
                output_members.push_back(m);
            }
        }
        if (output_members.empty()) { continue; }

        NodeHandle anchor = members.back();

        auto subgraph = std::make_unique<Graph>();

        std::vector<NodeHandle> handle_map(graph.nodes.size(), sentinel);

        for (size_t i = 0; i < ext_inputs.size(); i++) {
            auto ext_node = ext_inputs[i].node;
            if (handle_map[ext_node] == sentinel) {
                handle_map[ext_node] = static_cast<NodeHandle>(subgraph->nodes.size());

                auto spec = graph.get_tensor_spec(ext_inputs[i]);
                subgraph->input_specs.push_back(spec);

                Node node;
                node.value = InputNode{static_cast<InputHandle>(i)};
                subgraph->nodes.push_back(std::move(node));
            }
        }

        for (auto m : members) {
            for (auto arg : graph.nodes[m].get_args()) {
                if (arg.is_null()) continue;
                if (handle_map[arg.node] != sentinel) continue;
                auto* cn = std::get_if<ConstantNode>(&graph.nodes[arg.node].value);
                if (!cn) continue;
                handle_map[arg.node] = static_cast<NodeHandle>(subgraph->nodes.size());
                ConstantNode cloned;
                cloned.tensor = cn->tensor;
                cloned.quant_params = cn->quant_params;
                Node node;
                node.value = std::move(cloned);
                subgraph->nodes.push_back(std::move(node));
            }
        }

        {
            auto next_pos = static_cast<NodeHandle>(subgraph->nodes.size());
            for (auto m : members) {
                handle_map[m] = next_pos++;
            }
        }

        for (auto m : members) {
            auto* op = std::get_if<CallOperatorNode>(&graph.nodes[m].value);
            assert(op);

            CallOperatorNode remapped;
            remapped.op = op->op;
            remapped.output_specs = op->output_specs;
            remapped.constant_args = op->constant_args;
            for (auto arg : op->args) {
                if (arg.is_null()) {
                    remapped.args.push_back(ValueHandle::null());
                    continue;
                }
                assert(handle_map[arg.node] != sentinel);
                remapped.args.push_back(ValueHandle{
                    static_cast<uint16_t>(handle_map[arg.node]),
                    arg.output,
                });
            }

            Node node;
            node.value = std::move(remapped);
            subgraph->nodes.push_back(std::move(node));
        }

        std::vector<uint32_t> output_index(graph.nodes.size(), sentinel);
        std::vector<TensorSpec> output_specs;
        for (auto m : output_members) {
            output_index[m] = static_cast<uint32_t>(subgraph->outputs.size());
            subgraph->outputs.push_back(ValueHandle{
                static_cast<uint16_t>(handle_map[m]),
            });
            output_specs.push_back(
                std::get<TensorSpec>(graph.get_output_spec_for_node(m)));
        }

        CallSubgraphNode fused;
        fused.args = std::move(ext_inputs);
        if (output_members.size() == 1) {
            fused.output_specs = output_specs[0];
        } else {
            fused.output_specs = std::move(output_specs);
        }
        fused.subgraph = std::move(subgraph);

        graph.nodes[anchor].value = std::move(fused);
        graph.nodes[anchor].tag = 0;
        graph.nodes[anchor].flags = NodeFlags::None;

        for (NodeHandle n = 0; n < graph.nodes.size(); n++) {
            if (is_member(n)) { continue; }
            auto* op = std::get_if<CallOperatorNode>(&graph.nodes[n].value);
            if (!op) { continue; }
            for (auto& arg : op->args) {
                if (arg.is_null()) continue;
                if (output_index[arg.node] != sentinel) {
                    arg = ValueHandle{
                        static_cast<uint16_t>(anchor),
                        static_cast<uint16_t>(output_index[arg.node])};
                }
            }
        }
        for (auto& out : graph.outputs) {
            if (output_index[out.node] != sentinel) {
                out = ValueHandle{
                    static_cast<uint16_t>(anchor),
                    static_cast<uint16_t>(output_index[out.node])};
            }
        }

        for (auto m : members) {
            if (m == anchor) { continue; }
            graph.nodes[m].value = InputNode{0};
            graph.nodes[m].tag = 0;
            graph.nodes[m].flags = NodeFlags::Dead;
        }
    }

    graph.update_users();
}

void partition_xnn_subgraphs(Graph& graph) {
    graph.update_users();
    tag_xnn_nodes(graph);
    auto partition_count = assign_partitions(graph);
    fuse_partitions(graph, partition_count);
    graph.compact_nodes();
}

}
