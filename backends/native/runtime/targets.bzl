load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "is_xplat", "runtime")

def define_common_targets():
    # The wrapper rewrites "//executorch/..." references for xplat in deps /
    # exported_deps / visibility only, not in srcs, so spell the schema target
    # out per cell. is_xplat() reads the package context, so it can only be
    # called from inside this function, not at module scope.
    native_graph_fbs = (
        "//xplat/executorch/backends/native:native_graph.fbs" if is_xplat() else "//executorch/backends/native:native_graph.fbs"
    )

    # Compile the native graph FlatBuffer schema to a C++ header. flatc takes an
    # output directory (not a file), so use `outs` to expand ${OUT} to the dir.
    #
    # Note that --cpp-std only picks the feature level of the flatbuffer
    # *generated* accessors. flatc takes c++0x / c++11 / c++17 only -- no c++20.
    runtime.genrule(
        name = "generate_native_graph",
        srcs = [native_graph_fbs],
        outs = {"native_graph_generated.h": ["native_graph_generated.h"]},
        default_outs = ["native_graph_generated.h"],
        cmd = " ".join([
            "$(exe {})".format(runtime.external_dep_location("flatc")),
            "--cpp",
            "--cpp-std c++11",
            "--gen-mutable",
            "--scoped-enums",
            "-o ${OUT}",
            "${SRCS}",
        ]),
    )

    # Header-only library exposing the generated FlatBuffer accessors. Kept internal
    # so flatbuffers stays an implementation detail of the reader.
    runtime.cxx_library(
        name = "native_graph_schema",
        srcs = [],
        exported_headers = {
            "native_graph_generated.h": ":generate_native_graph[native_graph_generated.h]",
        },
        exported_external_deps = ["flatbuffers-api"],
        visibility = ["//executorch/backends/native/..."],
    )

    # The native runtime program reader (standalone; no ExecuTorch dependency).
    runtime.cxx_library(
        name = "runtime",
        srcs = [
            "Program.cpp",
        ],
        exported_headers = [
            "Program.h",
        ],
        deps = [
            ":native_graph_schema",
        ],
        visibility = ["PUBLIC"],
    )

    # utils/ has no BUCK of its own, so the DOT renderer's target lives here.
    runtime.cxx_library(
        name = "to_dot",
        srcs = [
            "utils/ToDot.cpp",
        ],
        exported_headers = [
            "utils/ToDot.h",
        ],
        deps = [
            ":native_graph_schema",
            ":runtime",
            "//executorch/backends/native/runtime/graph:string_format",
        ],
        visibility = ["PUBLIC"],
    )
