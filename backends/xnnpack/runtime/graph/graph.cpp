#include <executorch/backends/xnnpack/runtime/graph/graph.h>

#include <executorch/backends/xnnpack/runtime/core/variant_util.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/log.h>

#include <cassert>

namespace executorch::backends::xnnpack::graph {

namespace {

void scan_spec(const TensorSpec& spec, uint32_t& max_id) {
  for (auto& dim : spec.sizes) {
    for (auto& term : dim.coeffs) {
      if (term.sym >= max_id) {
        max_id = term.sym + 1;
      }
    }
  }
}

void scan_output_spec(const OutputSpec& os, uint32_t& max_id) {
  std::visit(
      overloaded{
          [&](const TensorSpec& s) { scan_spec(s, max_id); },
          [&](const std::vector<TensorSpec>& v) {
            for (auto& s : v)
              scan_spec(s, max_id);
          },
      },
      os);
}

} // namespace

uint32_t Graph::symint_count() const {
  uint32_t count = 0;
  for (auto& spec : input_specs) {
    scan_spec(spec, count);
  }
  for (auto& node : nodes) {
    std::visit(
        overloaded{
            [](const InputNode&) {},
            [](const ConstantNode&) {},
            [&](const CallOperatorNode& n) {
              scan_output_spec(n.output_specs, count);
            },
            [&](const CallSubgraphNode& n) {
              scan_output_spec(n.output_specs, count);
            },
        },
        node.value);
  }
  return count;
}

void Graph::update_users() {
  for (auto& node : nodes) {
    node.users.clear();
  }

  for (NodeHandle i = 0; i < nodes.size(); ++i) {
    std::visit(
        overloaded{
            [](const InputNode&) {},
            [](const ConstantNode&) {},
            [&](const CallOperatorNode& n) {
              for (auto arg : n.args) {
                if (!arg.is_null()) {
                  nodes[arg.node].users.push_back(i);
                }
              }
            },
            [&](const CallSubgraphNode& n) {
              for (auto arg : n.args) {
                if (!arg.is_null()) {
                  nodes[arg.node].users.push_back(i);
                }
              }
            },
        },
        nodes[i].value);
  }
}

runtime::Error Graph::compact_nodes() {
  std::vector<uint32_t> remap(nodes.size(), UINT32_MAX);
  uint32_t new_idx = 0;
  for (NodeHandle i = 0; i < nodes.size(); i++) {
    if ((nodes[i].flags & NodeFlags::Dead) != NodeFlags::None) {
      continue;
    }
    remap[i] = new_idx++;
  }

  // Validate that no live node or output references a dead/invalid node before
  // mutating any handles, so a failure leaves the graph untouched.
  bool valid = true;
  auto check_vh = [&](const ValueHandle& vh) {
    if (vh.is_null()) {
      return;
    }
    if (vh.node >= remap.size() || remap[vh.node] == UINT32_MAX) {
      valid = false;
    }
  };
  for (NodeHandle i = 0; i < nodes.size(); i++) {
    if ((nodes[i].flags & NodeFlags::Dead) != NodeFlags::None) {
      continue;
    }
    for (const auto& a : nodes[i].get_args()) {
      check_vh(a);
    }
  }
  for (const auto& out : outputs) {
    check_vh(out);
  }
  ET_CHECK_OR_RETURN_ERROR(
      valid,
      Internal,
      "compact_nodes: a live node or output references a dead node");

  auto rewrite_vh = [&](ValueHandle& vh) {
    if (!vh.is_null()) {
      vh.node = remap[vh.node];
    }
  };

  for (NodeHandle i = 0; i < nodes.size(); i++) {
    if ((nodes[i].flags & NodeFlags::Dead) != NodeFlags::None) {
      continue;
    }
    std::visit(
        overloaded{
            [](InputNode&) {},
            [](ConstantNode&) {},
            [&](CallOperatorNode& n) {
              for (auto& a : n.args)
                rewrite_vh(a);
            },
            [&](CallSubgraphNode& n) {
              for (auto& a : n.args)
                rewrite_vh(a);
            },
        },
        nodes[i].value);
  }

  for (auto& out : outputs) {
    rewrite_vh(out);
  }

  std::vector<Node> compacted;
  compacted.reserve(new_idx);
  for (NodeHandle i = 0; i < nodes.size(); i++) {
    if ((nodes[i].flags & NodeFlags::Dead) != NodeFlags::None) {
      continue;
    }
    compacted.push_back(std::move(nodes[i]));
  }
  nodes = std::move(compacted);

  update_users();
  return runtime::Error::Ok;
}

} // namespace executorch::backends::xnnpack::graph
