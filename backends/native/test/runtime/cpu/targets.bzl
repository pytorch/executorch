load("@fbsource//tools/target_determinator/macros:ci.bzl", "ci")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "is_xplat", "runtime")

def define_common_targets():
    runtime.cxx_test(
        name = "cpu_engine_test",
        srcs = ["test_cpu_engine.cpp"],
        deps = [
            "//executorch/backends/cpu:cpu_reference",
            "//executorch/backends/native/runtime:native_graph_schema",
            "//executorch/backends/native/runtime:runtime",
            "//executorch/backends/native/runtime/api:api",
            "//executorch/backends/native/runtime/cpu:cpu_engine",
            "//executorch/backends/native/runtime/deserialize:package",
            "//executorch/backends/native/runtime/deserialize:package_test_data",
            "//executorch/extension/threadpool:threadpool",
            "//executorch/runtime/core/exec_aten:lib",
        ],
    )

    if is_xplat():
        return

    runtime.cxx_library(
        name = "cpu_engine_host",
        srcs = ["cpu_ptn_runner.cpp"],
        link_whole = True,
        visibility = ["//executorch/backends/native/test/..."],
        exported_deps = ["//executorch/backends/native/test/runtime:engine_host"],
        deps = ["//executorch/backends/native/runtime/cpu:cpu_engine"],
    )

    # cpu_sample_test_static_float judges the methods the CPU engine runs:
    # unquantized ones with static shapes.
    # TODO: run cpu_sample_test in CI once the engine runs quantized and
    # dynamic-shape methods.
    # Float samples the CPU engine cannot run yet, by gap. Each entry is a known
    # CPU-engine gap; remove it once the engine runs it. test/CMakeLists.txt
    # keeps the same list.
    cpu_engine_gaps = [
        # CPU engine: topk runtime error 33.
        "topk_float.all_bf16",
        "topk_float.all_f16",
        "topk_float.all_f32",
        "topk_float.nan_and_inf_largest_bf16",
        "topk_float.nan_and_inf_largest_f16",
        "topk_float.nan_and_inf_largest_f32",
        "topk_float.nan_and_inf_smallest_bf16",
        "topk_float.nan_and_inf_smallest_f16",
        "topk_float.nan_and_inf_smallest_f32",
        "topk_float.quantized_router_bf16",
        "topk_float.quantized_router_f16",
        "topk_float.quantized_router_f32",
        "topk_float.router_bf16",
        "topk_float.router_f16",
        "topk_float.router_f32",
        "topk_float.smallest_bf16",
        "topk_float.smallest_f16",
        "topk_float.smallest_f32",
    ]
    static_float = ":".join(["*_float.*:-*.dynamic_*"] + cpu_engine_gaps)
    for name, flags, labels in [
        ("", [], ci.labels(ci.map(ci.skip_test()))),
        ("_static_float", ["--gtest_filter=" + static_float], []),
    ]:
        runtime.cxx_test(
            name = "cpu_sample_test" + name,
            args = ["--packages=$(location //executorch/backends/native/test/samples:op_samples)"] + flags,
            labels = labels,
            test_main = "//executorch/backends/native/test/samples:sample_test_main",
            deps = [":cpu_engine_host"],
        )
