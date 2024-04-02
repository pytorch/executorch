load("@fbsource//tools/build_defs:platform_defs.bzl", "ANDROID")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode = False):
    if is_fbcode:
        return

    runtime.python_library(
        name = "generate_op_tests_lib",
        srcs = native.glob(["utils/*.py"]) + [
            "generate_op_tests.py",
            "cases.py",
        ],
        base_module = "executorch.backends.vulkan.test.op_tests",
        deps = [
            "//caffe2/torchgen:torchgen",
            "fbsource//third-party/pypi/expecttest:expecttest",
        ],
    )

    runtime.python_binary(
        name = "generate_op_tests",
        main_module = "executorch.backends.vulkan.test.op_tests.generate_op_tests",
        deps = [
            ":generate_op_tests_lib",
        ],
    )

    aten_src_path = runtime.external_dep_location("aten-src-path")
    genrule_cmd = [
        "$(exe :generate_op_tests)",
        "--tags-path $(location {})/aten/src/ATen/native/tags.yaml".format(aten_src_path),
        "--aten-yaml-path $(location {})/aten/src/ATen/native/native_functions.yaml".format(aten_src_path),
        "-o $OUT",
    ]

    runtime.genrule(
        name = "generated_op_tests_cpp",
        outs = {
            "op_tests.cpp": ["op_tests.cpp"],
        },
        cmd = " ".join(genrule_cmd),
        default_outs = ["."],
    )

    runtime.cxx_binary(
        name = "compute_graph_op_tests_bin",
        srcs = [
            ":generated_op_tests_cpp[op_tests.cpp]",
        ],
        define_static_target = False,
        deps = [
            "//third-party/googletest:gtest_main",
            "//executorch/backends/vulkan:vulkan_graph_runtime",
            runtime.external_dep_location("libtorch"),
        ],
    )

    runtime.cxx_test(
        name = "compute_graph_op_tests",
        srcs = [
            ":generated_op_tests_cpp[op_tests.cpp]",
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
