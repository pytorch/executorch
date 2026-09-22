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

    runtime.cxx_library(
        name = "module_ptn",
        srcs = ["NativeModule.cpp"],
        exported_headers = ["NativeModule.h"],
        compiler_flags = [
            "-fexceptions",
            "-Wno-global-constructors",
        ],
        # Registration must survive static archive extraction and linker GC.
        # @lint-ignore BUCKLINT: Avoid `link_whole=True`
        link_whole = True,
        visibility = ["PUBLIC"],
        deps = [
            ":method_meta_bridge",
            "//executorch/backends/native/runtime:method_meta",
            "//executorch/backends/native/runtime:runtime",
            "//executorch/backends/native/runtime:validation",
            "//executorch/backends/native/runtime/deserialize:deserialize_error",
            "//executorch/backends/native/runtime/deserialize:limits",
            "//executorch/backends/native/runtime/deserialize:owned_bytes",
            "//executorch/backends/native/runtime/deserialize:package",
            "//executorch/extension/module:ptn_module_internal",
            "//executorch/runtime/platform:platform",
        ],
    )

    runtime.cxx_test(
        name = "native_module_load_test",
        srcs = ["test/NativeModuleLoadTest.cpp"],
        headers = ["test/TestData.h"],
        deps = [
            ":module_ptn",
            "//executorch/backends/native/runtime:native_graph_schema",
            "//executorch/backends/native/runtime/deserialize:package",
            "//executorch/backends/native/runtime/deserialize:package_test_data",
            "//executorch/extension/data_loader:buffer_data_loader",
            "//executorch/extension/module:module",
        ],
    )
