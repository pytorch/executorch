load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "is_xplat", "runtime")

def define_common_targets():
    runtime.cxx_test(
        name = "vulkan_constant_materialization_tracker_test",
        srcs = ["test_vulkan_constant_materialization_tracker.cpp"],
        deps = [
            "//executorch/backends/native/runtime/vulkan:constant_materialization_tracker",
        ],
    )

    if is_xplat():
        return

    runtime.genrule(
        name = "vulkan_test_driver",
        out = "vulkan_test_driver",
        cmd = "mkdir -p $OUT && cp $(location fbsource//third-party/swiftshader/lib/linux-x64:libvk_swiftshader_so)/libvulkan.so.1 $OUT/libvk_swiftshader.so",
    )

    runtime.cxx_test(
        name = "vulkan_engine_test",
        srcs = ["test_vulkan_engine.cpp"],
        env = {
            "ASAN_OPTIONS": "detect_leaks=0",
            "LD_LIBRARY_PATH": "$(location :vulkan_test_driver)",
        },
        deps = [
            "//executorch/backends/native/runtime:native_graph_schema",
            "//executorch/backends/native/runtime:runtime",
            "//executorch/backends/native/runtime/deserialize:owned_bytes",
            "//executorch/backends/native/runtime/deserialize:package",
            "//executorch/backends/native/runtime/deserialize:package_test_data",
            "//executorch/backends/native/runtime/engine:engine",
            "//executorch/backends/native/runtime/vulkan:vulkan_engine",
            "fbsource//third-party/swiftshader/lib/linux-x64:libvk_swiftshader_fbcode",
        ],
    )
