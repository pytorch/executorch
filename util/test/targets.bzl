load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    runtime.cxx_library(
        name = "temp_file",
        srcs = [],
        exported_headers = ["temp_file.h"],
        visibility = [],  # Private
    )

    runtime.cxx_test(
        name = "temp_file_test",
        srcs = [
            "temp_file_test.cpp",
        ],
        deps = [
            ":temp_file",
        ],
    )

    runtime.cxx_test(
        name = "ATenBridgeTest",
        srcs = ["ATenBridgeTest.cpp"],
        deps = [
            "//executorch/core:core",
            "//executorch/core/kernel_types:kernel_types",
            "//executorch/util:aten_bridge",
        ],
        fbcode_deps = [
            "//caffe2:ATen-core",
            "//caffe2:ATen-cpu",
            "//caffe2/c10:c10",
        ],
        xplat_deps = [
            "//xplat/caffe2:torch_mobile_core",
            "//xplat/caffe2/c10:c10",
            # Dont really like this but without this I dont have aten::empty
            # And havent figured out a more minimal target
            "//xplat/caffe2:torch_mobile_all_ops",
        ],
    )

    runtime.cxx_test(
        name = "dynamic_memory_allocator_test",
        srcs = [
            "DynamicMemoryAllocatorTest.cpp",
        ],
        deps = [
            "//executorch/util:dynamic_memory_allocator",
        ],
    )

    runtime.cxx_test(
        name = "embedded_data_loader_test",
        srcs = [
            "embedded_data_loader_test.cpp",
        ],
        deps = [
            "//executorch/util:embedded_data_loader",
        ],
    )

    runtime.cxx_test(
        name = "shared_ptr_data_loader_test",
        srcs = [
            "shared_ptr_data_loader_test.cpp",
        ],
        deps = [
            "//executorch/util:shared_ptr_data_loader",
        ],
    )

    runtime.cxx_test(
        name = "file_data_loader_test",
        srcs = [
            "file_data_loader_test.cpp",
        ],
        deps = [
            ":temp_file",
            "//executorch/util:file_data_loader",
        ],
    )

    runtime.cxx_test(
        name = "mmap_data_loader_test",
        srcs = [
            "mmap_data_loader_test.cpp",
        ],
        deps = [
            ":temp_file",
            "//executorch/util:mmap_data_loader",
        ],
    )

    runtime.cxx_test(
        name = "memory_utils_test",
        srcs = [
            "memory_utils_test.cpp",
        ],
        deps = [
            "//executorch/runtime/platform:compiler",
            "//executorch/util:memory_utils",
        ],
    )

    runtime.cxx_test(
        name = "ivalue_flatten_unflatten_test",
        srcs = ["IvalueFlattenUnflattenTest.cpp"],
        deps = ["//executorch/util:ivalue_flatten_unflatten"],
        fbcode_deps = [
            "//caffe2:torch-cpp",
        ],
        xplat_deps = [
            "//xplat/caffe2:torch_mobile_all_ops",
        ],
    )
