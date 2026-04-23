#pragma once

#include <memory>

#include <xnnpack.h>

namespace executorch::backends::xnnpack::graph { struct Graph; }

namespace executorch::backends::xnnpack::plan {

struct XnnSubgraphDeleter {
    void operator()(xnn_subgraph_t subgraph) const {
        xnn_delete_subgraph(subgraph);
    }
};

struct XnnRuntimeDeleter {
    void operator()(xnn_runtime_t runtime) const {
        xnn_delete_runtime(runtime);
    }
};

struct XnnWorkspaceDeleter {
    void operator()(xnn_workspace_t workspace) const {
        xnn_release_workspace(workspace);
    }
};

using XnnSubgraph = std::unique_ptr<xnn_subgraph, XnnSubgraphDeleter>;
using XnnRuntime = std::unique_ptr<xnn_runtime, XnnRuntimeDeleter>;
using XnnWorkspace = std::unique_ptr<xnn_workspace, XnnWorkspaceDeleter>;

XnnRuntime compile_xnn_subgraph(const graph::Graph& graph, xnn_workspace_t workspace);

}
