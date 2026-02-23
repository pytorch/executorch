load("@fbcode_macros//build_defs/lib:re_test_utils.bzl", "re_test_utils")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def get_backend_mode():
    """Get the supported backend mode of slimtensor."""
    return ["cuda", "cpu"]

def define_common_targets():
    """Define test targets for SlimTensor factory module."""

    # GPU empty test with CUDA support
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
            name = "test_empty" + backend_suffix,
            srcs = [
                "test_empty.cpp",
            ],
            deps = [
                "//executorch/backends/aoti/slim/factory:empty",
            ],
            **backend_kwargs
        )

        runtime.cxx_test(
            name = "test_from_blob" + backend_suffix,
            srcs = [
                "test_from_blob.cpp",
            ],
            deps = [
                "//executorch/backends/aoti/slim/core:storage",
                "//executorch/backends/aoti/slim/factory:from_blob",
                "//executorch/backends/aoti/slim/factory:empty",
            ],
            **backend_kwargs
        )

        runtime.cxx_test(
            name = "test_from_etensor" + backend_suffix,
            srcs = [
                "test_from_etensor.cpp",
            ],
            deps = [
                "//executorch/backends/aoti/slim/core:storage",
                "//executorch/backends/aoti/slim/factory:empty",
                "//executorch/backends/aoti/slim/factory:from_etensor",
                "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
            ],
            **backend_kwargs
        )
