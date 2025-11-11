load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_oss_build_kwargs", "runtime")

SIZE_TEST_SOURCES = [
    "size_test.cpp",
]

SIZE_TEST_DEPS = [
    "//executorch/runtime/executor:program",
    "//executorch/extension/data_loader:file_data_loader",
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
            "//executorch/kernels/portable:generated_lib",
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
