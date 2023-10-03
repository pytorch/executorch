load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_oss_build_kwargs", "runtime")

SIZE_TEST_SOURCES = [
    "size_test.cpp",
]

SIZE_TEST_DEPS = [
    "//executorch/runtime/executor:program",
    "//executorch/extension/data_loader:file_data_loader",
    "//executorch/util:util",
]

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # DO NOT MODIFY: This target `size_test_static` is being used on a per-diff
    # CI job to guard Executorch binary size. It doesn't contain any operators
    # or kernels thus shouldn't be used to run a model. Adding/removing dependencies
    # will likely result in inaccurate measure results.
    #
    # It's also best to build this with `-c executorch.enable_program_verification=false`
    # to remove ~30kB of optional flatbuffer verification code from the binary.
    # Building with `-c executorch.enable_et_log=0` removes ~15kB from the binary.
    runtime.cxx_binary(
        name = "size_test",
        srcs = SIZE_TEST_SOURCES,
        deps = SIZE_TEST_DEPS,
        define_static_target = True,
        **get_oss_build_kwargs()
    )

    runtime.cxx_binary(
        name = "size_test_all_ops",
        srcs = SIZE_TEST_SOURCES,
        deps = SIZE_TEST_DEPS + [
            "//executorch/kernels/portable:generated_lib_all_ops",
            "//executorch/runtime/executor/test:test_backend_compiler_lib",
        ],
        define_static_target = True,
        **get_oss_build_kwargs()
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
            "//executorch/runtime/executor:program",
            "//executorch/runtime/executor/test:managed_memory_manager",
            "//executorch/extension/data_loader:buffer_data_loader",
            "//executorch/util:read_file",
            "//executorch/util:util",
        ],
        external_deps = [
            "gflags",
        ],
        **get_oss_build_kwargs()
    )

    # Test binary that can create relocatable Executor instances.
    runtime.cxx_binary(
        name = "relocatable_runner",
        srcs = ["relocatable_runner.cpp"],
        deps = [
            "//executorch/kernels/portable:generated_lib_all_ops",
            "//executorch/runtime/executor:program",
            "//executorch/configurations:executor_cpu_optimized",
            "//executorch/extension/data_loader:buffer_data_loader",
            "//executorch/util:read_file",
            "//executorch/util:util",
        ],
        external_deps = [
            "gflags",
        ],
        preprocessor_flags = [],
        define_static_target = True,
    )
