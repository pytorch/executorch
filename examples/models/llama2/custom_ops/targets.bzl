load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library", "executorch_generated_lib")
load("@fbsource//xplat/executorch/kernels/test:util.bzl", "codegen_function_header_wrapper")

def define_tests():
    codegen_function_header_wrapper("executorch/examples/models/llama2/custom_ops", "custom_ops")

    # In the long run we should really have aten variant available as well
    deps = [":function_header_wrapper_custom_ops"]
    generated_lib_and_op_deps = [
        ":custom_ops",
        ":sdpa",
        ":custom_ops_headers",
    ]
    runtime.cxx_test(
        name = "op_sdpa_test",
        srcs = [
            "op_sdpa_test.cpp",
        ],
        visibility = ["//executorch/..."],
        deps = [
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
            "//executorch/kernels/test:test_util",
        ] + generated_lib_and_op_deps + deps,
    )
    runtime.cxx_test(
        name = "op_sdpa_with_kv_cache_test",
        srcs = [
            "op_sdpa_with_kv_cache_test.cpp",
        ],
        visibility = ["//executorch/..."],
        deps = [
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
            "//executorch/kernels/test:test_util",
        ] + generated_lib_and_op_deps + deps,
    )

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.python_library(
        name = "llama_custom_ops_aot_lib",
        srcs = [
            "sdpa_with_kv_cache.py",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            "//caffe2:torch",
        ],
    )

    runtime.export_file(
        name = "custom_ops.yaml",
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    # ~~~ START of custom ops 1 `my_ops::mul3` library definitions ~~~
    et_operator_library(
        name = "sdpa_op",
        ops = [
            "llama::sdpa.out",
        ],
        define_static_targets = True,
        visibility = [
            "//executorch/codegen/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    et_operator_library(
        name = "sdpa_with_kv_cache",
        ops = [
            "llama::sdpa_with_kv_cache.out",
        ],
        define_static_targets = True,
        visibility = [
            "//executorch/codegen/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "sdpa",
        srcs = ["op_sdpa.cpp"],
        deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/kernels/portable/cpu:scalar_utils",
            "//executorch/kernels/optimized:libblas",
            "//executorch/kernels/optimized:libvec",
        ],
        compiler_flags = ["-Wno-missing-prototypes"],
        visibility = [
            "//executorch/...",
            "//executorch/examples/models/llama2/custom_ops/...",
            "@EXECUTORCH_CLIENTS",
        ],
        force_static = True,
    )

    executorch_generated_lib(
        name = "custom_ops",
        deps = [
            ":sdpa_op",
            ":sdpa_with_kv_cache",
            ":sdpa",
        ],
        custom_ops_yaml_target = ":custom_ops.yaml",
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        define_static_targets = True,
    )
    define_tests()
