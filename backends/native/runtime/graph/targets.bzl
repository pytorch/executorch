load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    # Debug-dump formatting shared by every renderer of the IR (header-only).
    runtime.cxx_library(
        name = "format",
        exported_headers = [
            "Format.h",
        ],
        visibility = ["//executorch/backends/native/..."],
    )

    # Scalar element type + C++-type mapping (header-only; macro-driven, standalone).
    runtime.cxx_library(
        name = "scalar_type",
        exported_headers = [
            "ScalarType.h",
        ],
        visibility = ["//executorch/backends/native/..."],
    )

    # A concrete scalar value (int / double / bool), tagged. c10::Scalar analog.
    runtime.cxx_library(
        name = "scalar",
        srcs = ["Scalar.cpp"],
        exported_headers = [
            "Scalar.h",
        ],
        deps = [":format"],
        visibility = ["//executorch/backends/native/..."],
    )

    # Concrete in-memory IR value types (pure std; no ExecuTorch, no flatbuffers).
    runtime.cxx_library(
        name = "tensor_meta",
        srcs = ["TensorMeta.cpp"],
        exported_headers = [
            "TensorMeta.h",
        ],
        exported_deps = [
            ":scalar_type",
        ],
        visibility = ["//executorch/backends/native/..."],
    )

    # Index-arena handles (NodeRef / ValueRef); header-only.
    runtime.cxx_library(
        name = "ids",
        exported_headers = [
            "Ids.h",
        ],
        visibility = ["//executorch/backends/native/..."],
    )

    # A single SSA value: a std::variant (tensor / scalar / list / none) plus
    # def-use wiring, storage alias, and an attrs scratch map.
    runtime.cxx_library(
        name = "value",
        srcs = ["Value.cpp"],
        exported_headers = [
            "Value.h",
        ],
        exported_deps = [
            ":ids",
            ":scalar",
            ":tensor_meta",
        ],
        visibility = ["//executorch/backends/native/..."],
    )

    # One fx argument value: a std::variant over the schema ArgumentValue
    # payload kinds (in-graph refs resolved to ValueRefs), plus NamedArgument.
    runtime.cxx_library(
        name = "argument",
        srcs = ["Argument.cpp"],
        exported_headers = [
            "Argument.h",
        ],
        exported_deps = [
            ":ids",
            ":scalar_type",
        ],
        visibility = ["//executorch/backends/native/..."],
    )

    # One fx node: an op invocation or graph-boundary marker, with typed Outputs
    # preserving the op return ABI (single / tuple / tensor-list / scalar).
    runtime.cxx_library(
        name = "node",
        srcs = ["Node.cpp"],
        exported_headers = [
            "Node.h",
        ],
        exported_deps = [
            ":argument",
            ":ids",
        ],
        deps = [":format"],
        visibility = ["//executorch/backends/native/..."],
    )

    # The index arena: a pure function body owning the Nodes and Values that Refs
    # index into, plus ordered graph I/O and the per-graph subgraph arena.
    runtime.cxx_library(
        name = "graph",
        srcs = ["Graph.cpp"],
        exported_headers = [
            "Graph.h",
        ],
        exported_deps = [
            ":ids",
            ":node",
            ":value",
        ],
        visibility = ["//executorch/backends/native/..."],
    )
