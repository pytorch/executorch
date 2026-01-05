load("@fbcode_macros//build_defs/lib:re_test_utils.bzl", "re_test_utils")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def get_backend_mode():
    """Get the supported backend mode of slimtensor."""
    return ["cuda", "cpu"]

def define_common_targets():
    """Define test targets for SlimTensor core module."""

    # GPU storage test with CUDA support
    for backend_mode in get_backend_mode():
        backend_suffix = "_" + backend_mode if backend_mode == "cuda" else ""

        backend_kwargs = {
            "external_deps": [("cuda", None, "cuda-lazy")],
            "preprocessor_flags": ["-DCUDA_AVAILABLE=1"],
            "keep_gpu_sections": True,
            "remote_execution": re_test_utils.remote_execution(
                platform = "gpu-remote-execution",
            ),
        } if backend_mode == "cuda" else {}

        runtime.cxx_test(
            name = "test_storage" + backend_suffix,
            srcs = [
                "test_storage.cpp",
            ],
            deps = [
                "//executorch/backends/aoti/slim/core:storage",
            ],
            **backend_kwargs
        )

    runtime.cxx_test(
        name = "test_slimtensor_basic",
        srcs = [
            "test_slimtensor_basic.cpp",
        ],
        deps = [
            "//executorch/backends/aoti/slim/core:slimtensor",
            "//executorch/backends/aoti/slim/core:storage",
        ],
    )

    runtime.cxx_test(
        name = "test_slimtensor_copy",
        srcs = [
            "test_slimtensor_copy.cpp",
        ],
        deps = [
            "//executorch/backends/aoti/slim/core:slimtensor",
            "//executorch/backends/aoti/slim/core:storage",
        ],
    )

    runtime.cxx_test(
        name = "test_slimtensor_dtypes",
        srcs = [
            "test_slimtensor_dtypes.cpp",
        ],
        deps = [
            "//executorch/backends/aoti/slim/factory:empty",
        ],
    )
