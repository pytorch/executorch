load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode = False):
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    if not runtime.is_oss and is_fbcode:
        modules_env = {
           "ET_XNNPACK_GENERATED_ADD_LARGE_PTE_PATH": "$(location fbcode//executorch/test/models:exported_xnnp_delegated_programs[ModuleAddLarge.pte])",
           "ET_XNNPACK_GENERATED_SUB_LARGE_PTE_PATH": "$(location fbcode//executorch/test/models:exported_xnnp_delegated_programs[ModuleSubLarge.pte])",
        }

        runtime.cxx_test(
            name = "multi_method_delegate_test",
            srcs = [
                "multi_method_delegate_test.cpp",
            ],
            deps = [
                "//executorch/runtime/executor:program",
                "//executorch/extension/data_loader:file_data_loader",
                "//executorch/extension/memory_allocator:malloc_memory_allocator",
                "//executorch/kernels/portable:generated_lib",
                "//executorch/backends/xnnpack:xnnpack_backend",
                "//executorch/extension/runner_util:inputs",
            ],
            env = modules_env,
        )
