load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "method_meta_bridge",
        srcs = ["MethodMetaBridge.cpp"],
        exported_headers = ["MethodMetaBridge.h"],
        exported_deps = [
            "//executorch/backends/native/runtime:method_meta",
            "//executorch/runtime/executor:program",
        ],
        deps = [
            "//executorch/schema:program",
        ],
        visibility = ["//executorch/backends/native/extension/module/..."],
    )

    runtime.cxx_test(
        name = "method_meta_bridge_test",
        srcs = ["test/MethodMetaBridgeTest.cpp"],
        deps = [
            ":method_meta_bridge",
            "//executorch/backends/native/runtime:method_meta",
        ],
    )
