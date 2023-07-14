load(
    "@fbsource//tools/build_defs:default_platform_defs.bzl",
    "ANDROID",
    "CXX",
)
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/pybindings:targets.bzl", "MODELS_ALL_OPS_ATEN_MODE_GENERATED_LIB", "MODELS_ALL_OPS_LEAN_MODE_GENERATED_LIB")

SIZE_TEST_SOURCES = [
    "size_test.cpp",
]

SIZE_TEST_DEPS = [
    "//executorch/runtime/executor:executor",
    "//executorch/util:file_data_loader",
    "//executorch/util:util",
]

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
            xplat_deps = [
                "//xplat/third-party/gflags:gflags",
            ],
            define_static_target = not aten_mode,
            visibility = [
                "@EXECUTORCH_CLIENTS",
            ],
        )

    # DO NOT MODIFY: This target `size_test_static` is being used on a per-diff
    # CI job to guard Executorch binary size. It doesn't contain any operators
    # or kernels thus shouldn't be used to run a model. Adding/removing dependencies
    # will likely result in inaccurate measure results.
    #
    # It's also best to build this with `-c executorch.enable_program_verification=false`
    # to remove ~30kB of optional flatbuffer verification code from the binary.
    runtime.cxx_binary(
        name = "size_test",
        srcs = SIZE_TEST_SOURCES,
        deps = SIZE_TEST_DEPS,
        define_static_target = True,
    )

    runtime.cxx_binary(
        name = "size_test_all_ops",
        srcs = SIZE_TEST_SOURCES,
        deps = SIZE_TEST_DEPS + [
            "//executorch/kernels/portable:generated_lib_all_ops",
        ],
        define_static_target = True,
    )

    runtime.export_file(
        name = "size_test.cpp",
        visibility = [
            "@EXECUTORCH_CLIENTS",
        ],
    )

    # Test binary that can create multiple Executor instances in the same
    # process.
    runtime.cxx_binary(
        name = "multi_runner",
        srcs = ["multi_runner.cpp"],
        deps = [
            "//executorch/runtime/core:core",
            "//executorch/kernels/portable:generated_lib_all_ops",
            "//executorch/runtime/executor:executor",
            "//executorch/runtime/executor/test:managed_memory_manager",
            "//executorch/util:embedded_data_loader",
            "//executorch/util:read_file",
            "//executorch/util:util",
        ],
        external_deps = [
            "gflags",
        ],
        xplat_deps = [
            "//xplat/third-party/gflags:gflags",
        ],
    )

    # Test binary that can create relocatable Executor instances.
    runtime.cxx_binary(
        name = "relocatable_runner",
        srcs = ["relocatable_runner.cpp"],
        deps = [
            "//executorch/kernels/portable:generated_lib_all_ops",
            "//executorch/runtime/executor:executor",
            "//executorch/configurations:executor_cpu_optimized",
            "//executorch/util:embedded_data_loader",
            "//executorch/util:read_file",
            "//executorch/util:util",
        ],
        external_deps = [
            "gflags",
        ],
        preprocessor_flags = [],
        define_static_target = True,
        xplat_deps = [
            "//xplat/third-party/gflags:gflags",
        ],
    )
