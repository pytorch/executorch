#include <executorch/backends/xnnpack/runtime/plan/schedule.h>

#include <algorithm>
#include <cassert>
#include <deque>
#include <variant>

namespace executorch::backends::xnnpack::plan {

using namespace graph;

std::vector<NodeHandle> schedule(const graph::Graph& graph) {
    const auto& nodes = graph.nodes;

    std::vector<uint32_t> in_edges(nodes.size(), 0);
    std::deque<NodeHandle> queue;

    for (NodeHandle n = 0; n < nodes.size(); n++) {
        auto& args = nodes[n].get_args();
        in_edges[n] = std::count_if(
            args.begin(), args.end(),
            [&](const ValueHandle& a) {
                if (a.is_null()) return false;
                return !std::holds_alternative<InputNode>(nodes[a.node].value)
                    && !std::holds_alternative<ConstantNode>(nodes[a.node].value);
            });
        if (in_edges[n] == 0) {
            queue.push_back(n);
        }
    }

    std::vector<NodeHandle> order;
    order.reserve(nodes.size());

    while (!queue.empty()) {
        auto nh = queue.front();
        queue.pop_front();
        order.push_back(nh);

        if (std::holds_alternative<InputNode>(nodes[nh].value)
            || std::holds_alternative<ConstantNode>(nodes[nh].value)) {
            continue;
        }

        for (auto user : nodes[nh].users) {
            assert(in_edges[user] > 0);
            in_edges[user]--;
            if (in_edges[user] == 0) {
                queue.push_back(user);
            }
        }
    }

    assert(order.size() == nodes.size());
    return order;
}

}
