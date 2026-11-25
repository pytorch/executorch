load("@fbcode_macros//build_defs:python_unittest_remote_gpu.bzl", "python_unittest_remote_gpu")
load("@fbcode_macros//build_defs/lib:re_test_utils.bzl", "re_test_utils")

def define_common_targets(is_fbcode = False):
    if not is_fbcode:
        return

    python_unittest_remote_gpu(
        name = "test_lower_helion_cute",
        srcs = [
            "test_lower_helion_cute.py",
        ],
        env = {
            "RUN_CUDA_PROGRAM": "$(location //executorch/extension/pybindings/test:run_cuda_program)",
        },
        keep_gpu_sections = True,
        # Helion's CuTe backend targets Blackwell.
        remote_execution = re_test_utils.remote_execution(
            platform = "gpu-remote-execution",
            subplatform = "B200",
        ),
        deps = [
            "fbsource//third-party/pypi/nvidia-cutlass-dsl:nvidia-cutlass-dsl",
            "//caffe2:torch",
            "//executorch/backends/cuda/tests:test_lower_helion_lib",
            "//helion:helion",
        ],
    )
