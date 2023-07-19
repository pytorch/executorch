load(
    "@fbsource//tools/build_defs:default_platform_defs.bzl",
    "ANDROID",
    "CXX",
)
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/pybindings:targets.bzl", "MODELS_ALL_OPS_ATEN_MODE_GENERATED_LIB", "MODELS_ALL_OPS_LEAN_MODE_GENERATED_LIB")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Test driver for models, uses all portable kernels.
    for aten_mode in (True, False):
        aten_suffix = ("_aten" if aten_mode else "")
        runtime.cxx_binary(
            name = "executor_runner" + aten_suffix,
            srcs = ["executor_runner.cpp"],
            deps = [
                "//executorch/runtime/executor/test:test_backend_compiler_lib" + aten_suffix,
                "//executorch/runtime/executor:executor" + aten_suffix,
                "//executorch/sdk/etdump:etdump",
                "//executorch/util:bundled_program_verification" + aten_suffix,
                "//executorch/util:embedded_data_loader",
                "//executorch/util:file_data_loader",
                "//executorch/util:util" + aten_suffix,
            ] + (MODELS_ALL_OPS_ATEN_MODE_GENERATED_LIB if aten_mode else [
                "//executorch/configurations:executor_cpu_optimized",
            ] + MODELS_ALL_OPS_LEAN_MODE_GENERATED_LIB),
            preprocessor_flags = ["-DUSE_ATEN_LIB"] if aten_mode else [],
            external_deps = [
                "gflags",
            ],
            platforms = [ANDROID, CXX],
            define_static_target = not aten_mode,
            visibility = [
                "@EXECUTORCH_CLIENTS",
            ],
        )
