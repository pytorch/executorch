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
