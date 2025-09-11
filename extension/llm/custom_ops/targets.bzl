load("@fbsource//xplat/executorch/build:build_variables.bzl", "EXTENSION_LLM_CUSTOM_OPS_BUCK_SRCS")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load(
    "@fbsource//xplat/executorch/kernels/optimized:lib_defs.bzl",
    "get_vec_preprocessor_flags",
    "get_vec_deps",
)
load(
    "@fbsource//xplat/executorch/kernels/portable:op_registration_util.bzl",
    "get_compiler_optimization_flags",
)

def _get_quantized_sdpa_deps():
    if runtime.is_oss:
        return []
    else:
        return ["//pytorch/ao/torchao/csrc/cpu/torch_free_kernels/interface:interface"]

def _get_quantized_preproc_flags():
    if runtime.is_oss:
        return []
    else:
        return ["-DENABLE_CUSTOM_QUANTIZED_SDPA"]

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    for mkl_dep in ["", "_mkl_noomp"]:
        runtime.cxx_library(
            name = "custom_ops" + mkl_dep,
            srcs = EXTENSION_LLM_CUSTOM_OPS_BUCK_SRCS,
            exported_headers = [
                "op_fallback.h",
                "op_fast_hadamard_transform.h",
                "op_sdpa.h",
                "op_update_cache.h",
            ],
            headers = [
                "op_sdpa_impl.h",
            ],
            exported_preprocessor_flags = get_vec_preprocessor_flags() +
                _get_quantized_preproc_flags(),
            exported_deps = [
                "//executorch/runtime/kernel:kernel_includes",
                "//executorch/kernels/portable/cpu:scalar_utils",
                "//executorch/kernels/optimized:libblas{}".format(mkl_dep),
                "//executorch/kernels/optimized:libvec",
                "//executorch/extension/kernel_util:kernel_util",
                "//executorch/extension/threadpool:threadpool",
            ],
            deps = [
                "//executorch/kernels/portable/cpu/util:reduce_util",
                "//executorch/extension/llm/custom_ops/spinquant:fast_hadamard_transform",
            ] + get_vec_deps() + _get_quantized_sdpa_deps(),
            compiler_flags = ["-Wno-missing-prototypes", "-Wno-global-constructors"] + get_compiler_optimization_flags() +
            select({
                "DEFAULT": [],
                "ovr_config//cpu:arm64": ["-march=armv8.2-a+dotprod"],
            }),
            visibility = [
                "//executorch/...",
                "//executorch/extension/llm/custom_ops/...",
                "@EXECUTORCH_CLIENTS",
            ],
            # @lint-ignore BUCKLINT link_whole
            link_whole = True,
            force_static = True,
        )

        runtime.cxx_library(
            name = "custom_ops_aot_lib" + mkl_dep,
            srcs = [
                "op_fast_hadamard_transform_aten.cpp",
                "op_sdpa_aot.cpp",
                "op_tile_crop.cpp",
                "op_tile_crop_aot.cpp",
            ],
            headers = ["op_tile_crop.h"],
            compiler_flags = ["-Wno-global-constructors"],
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
            external_deps = [
                "libtorch",
            ],
            deps = [
                ":custom_ops" + mkl_dep,
                "//executorch/extension/aten_util:aten_bridge",
            ],
        )

    runtime.python_library(
        name = "custom_ops_aot_py",
        srcs = [
            "custom_ops.py",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            "//caffe2:torch",
        ],
    )

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
            ":custom_ops",
        ],
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
            ":custom_ops",
        ],
    )

    ## For preprocess
    runtime.python_library(
        name = "preprocess_custom_ops_py",
        srcs = [
            "preprocess_custom_ops.py",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            "//caffe2:torch",
        ],
    )

    runtime.python_library(
        name = "model_sharding_py",
        srcs = [
            "model_sharding.py",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            "//caffe2:torch",
        ],
    )

    runtime.cxx_library(
        name = "op_tile_crop",
        srcs = ["op_tile_crop.cpp"],
        exported_headers = ["op_tile_crop.h"],
        exported_deps = [
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/extension/kernel_util:kernel_util",
        ],
        compiler_flags = ["-Wno-missing-prototypes", "-Wno-global-constructors"],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        force_static = True,
    )

    runtime.cxx_test(
        name = "op_tile_crop_test",
        srcs = [
            "op_tile_crop_test.cpp",
        ],
        visibility = ["//executorch/..."],
        deps = [
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
            "//executorch/kernels/test:test_util",
            ":op_tile_crop",
        ],
    )
