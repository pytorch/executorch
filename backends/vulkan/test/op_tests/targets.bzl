load("@fbsource//tools/build_defs:platform_defs.bzl", "ANDROID")
load("@fbsource//xplat/caffe2:pt_defs.bzl", "get_pt_ops_deps")
load("@fbsource//xplat/caffe2:pt_ops.bzl", "pt_operator_library")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode = False):
    if is_fbcode:
        return

    runtime.python_library(
        name = "generate_op_correctness_tests_lib",
        srcs = native.glob(["utils/*.py"]) + [
            "generate_op_correctness_tests.py",
            "cases.py",
        ],
        base_module = "executorch.backends.vulkan.test.op_tests",
        deps = [
            "fbsource//third-party/pypi/expecttest:expecttest",
        ],
        external_deps = ["torchgen"],
    )

    runtime.python_library(
        name = "generate_op_benchmarks_lib",
        srcs = native.glob(["utils/*.py"]) + [
            "generate_op_benchmarks.py",
            "cases.py",
        ],
        base_module = "executorch.backends.vulkan.test.op_tests",
        deps = [
            "fbsource//third-party/pypi/expecttest:expecttest",
        ],
        external_deps = ["torchgen"],
    )

    runtime.python_binary(
        name = "generate_op_correctness_tests",
        main_module = "executorch.backends.vulkan.test.op_tests.generate_op_correctness_tests",
        deps = [
            ":generate_op_correctness_tests_lib",
        ],
    )

    runtime.python_binary(
        name = "generate_op_benchmarks",
        main_module = "executorch.backends.vulkan.test.op_tests.generate_op_benchmarks",
        deps = [
            ":generate_op_benchmarks_lib",
        ],
    )

    aten_src_path = runtime.external_dep_location("aten-src-path")
    genrule_cmd = [
        "$(exe :generate_op_correctness_tests)",
        "--tags-path $(location {})/aten/src/ATen/native/tags.yaml".format(aten_src_path),
        "--aten-yaml-path $(location {})/aten/src/ATen/native/native_functions.yaml".format(aten_src_path),
        "-o $OUT",
    ]

    runtime.genrule(
        name = "generated_op_correctness_tests_cpp",
        outs = {
            "op_tests.cpp": ["op_tests.cpp"],
        },
        cmd = " ".join(genrule_cmd),
        default_outs = ["."],
    )

    benchmarks_genrule_cmd = [
        "$(exe :generate_op_benchmarks)",
        "--tags-path $(location {})/aten/src/ATen/native/tags.yaml".format(aten_src_path),
        "--aten-yaml-path $(location {})/aten/src/ATen/native/native_functions.yaml".format(aten_src_path),
        "-o $OUT",
    ]

    runtime.genrule(
        name = "generated_op_benchmarks_cpp",
        outs = {
            "op_benchmarks.cpp": ["op_benchmarks.cpp"],
        },
        cmd = " ".join(benchmarks_genrule_cmd),
        default_outs = ["."],
    )

    runtime.cxx_binary(
        name = "compute_graph_op_tests_bin",
        srcs = [
            ":generated_op_correctness_tests_cpp[op_tests.cpp]",
        ],
        define_static_target = False,
        deps = [
            "//third-party/googletest:gtest_main",
            "//executorch/backends/vulkan:vulkan_graph_runtime",
            runtime.external_dep_location("libtorch"),
        ],
    )

    runtime.cxx_binary(
        name = "compute_graph_op_benchmarks_bin",
        srcs = [
            ":generated_op_benchmarks_cpp[op_benchmarks.cpp]",
        ],
        compiler_flags = [
            "-Wno-unused-variable",
        ],
        define_static_target = False,
        deps = [
            "//third-party/benchmark:benchmark",
            "//executorch/backends/vulkan:vulkan_graph_runtime",
            runtime.external_dep_location("libtorch"),
        ],
    )

    runtime.cxx_test(
        name = "compute_graph_op_tests",
        srcs = [
            ":generated_op_correctness_tests_cpp[op_tests.cpp]",
        ],
        contacts = ["oncall+ai_infra_mobile_platform@xmail.facebook.com"],
        fbandroid_additional_loaded_sonames = [
            "torch-code-gen",
            "vulkan_graph_runtime",
            "vulkan_graph_runtime_shaderlib",
        ],
        platforms = [ANDROID],
        use_instrumentation_test = True,
        deps = [
            "//third-party/googletest:gtest_main",
            "//executorch/backends/vulkan:vulkan_graph_runtime",
            runtime.external_dep_location("libtorch"),
        ],
    )

    runtime.cxx_binary(
        name = "sdpa_test_bin",
        srcs = [
            "sdpa_test.cpp",
        ],
        compiler_flags = [
            "-Wno-unused-variable",
        ],
        define_static_target = False,
        deps = [
            "//third-party/googletest:gtest_main",
            "//executorch/backends/vulkan:vulkan_graph_runtime",
            "//executorch/extension/llm/custom_ops:custom_ops_aot_lib",
        ],
    )

    runtime.cxx_test(
        name = "sdpa_test",
        srcs = [
            "sdpa_test.cpp",
        ],
        contacts = ["oncall+ai_infra_mobile_platform@xmail.facebook.com"],
        fbandroid_additional_loaded_sonames = [
            "torch-code-gen",
            "vulkan_graph_runtime",
            "vulkan_graph_runtime_shaderlib",
        ],
        platforms = [ANDROID],
        use_instrumentation_test = True,
        deps = [
            "//third-party/googletest:gtest_main",
            "//executorch/backends/vulkan:vulkan_graph_runtime",
            "//executorch/extension/llm/custom_ops:custom_ops_aot_lib",
            "//executorch/extension/tensor:tensor",
            runtime.external_dep_location("libtorch"),
        ],
    )

    runtime.cxx_binary(
        name = "linear_weight_int4_test_bin",
        srcs = [
            "linear_weight_int4_test.cpp",
        ],
        compiler_flags = [
            "-Wno-unused-variable",
        ],
        define_static_target = False,
        deps = [
            "//third-party/googletest:gtest_main",
            "//executorch/backends/vulkan:vulkan_graph_runtime",
            runtime.external_dep_location("libtorch"),
        ],
    )

    runtime.cxx_test(
        name = "linear_weight_int4_test",
        srcs = [
            "linear_weight_int4_test.cpp",
        ],
        contacts = ["oncall+ai_infra_mobile_platform@xmail.facebook.com"],
        fbandroid_additional_loaded_sonames = [
            "torch-code-gen",
            "vulkan_graph_runtime",
            "vulkan_graph_runtime_shaderlib",
        ],
        platforms = [ANDROID],
        use_instrumentation_test = True,
        deps = [
            "//third-party/googletest:gtest_main",
            "//executorch/backends/vulkan:vulkan_graph_runtime",
            "//executorch/extension/llm/custom_ops:custom_ops_aot_lib",
            "//executorch/extension/tensor:tensor",
            runtime.external_dep_location("libtorch"),
        ],
    )

    runtime.cxx_binary(
        name = "rotary_embedding_test_bin",
        srcs = [
            "rotary_embedding_test.cpp",
        ],
        compiler_flags = [
            "-Wno-unused-variable",
        ],
        define_static_target = False,
        deps = [
            "//third-party/googletest:gtest_main",
            "//executorch/backends/vulkan:vulkan_graph_runtime",
            runtime.external_dep_location("libtorch"),
        ],
    )

    runtime.cxx_test(
        name = "rotary_embedding_test",
        srcs = [
            "rotary_embedding_test.cpp",
        ],
        contacts = ["oncall+ai_infra_mobile_platform@xmail.facebook.com"],
        fbandroid_additional_loaded_sonames = [
            "torch-code-gen",
            "vulkan_graph_runtime",
            "vulkan_graph_runtime_shaderlib",
        ],
        platforms = [ANDROID],
        use_instrumentation_test = True,
        deps = [
            "//third-party/googletest:gtest_main",
            "//executorch/backends/vulkan:vulkan_graph_runtime",
            "//executorch/extension/tensor:tensor",
            runtime.external_dep_location("libtorch"),
        ],
    )
