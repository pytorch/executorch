load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//tools/build_defs:default_platform_defs.bzl", "CXX")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "pt2_archive_data_map",
        srcs = [
            "pt2_archive_data_map.cpp",
        ],
        exported_headers = [
            "pt2_archive_data_map.h",
        ],
        deps = [
            "//executorch/runtime/core:core",
            "//executorch/runtime/core:evalue",
            "//executorch/runtime/core:named_data_map",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/runtime/platform:platform",
            "//executorch/extension/data_loader:mmap_data_loader",
        ],
        exported_deps = [
            "fbsource//xplat/caffe2:miniz",
            "fbsource//third-party/nlohmann-json:nlohmann-json",
        ],
        visibility = [
            "@EXECUTORCH_CLIENTS",
        ],
        # Not available for mobile with miniz and json dependencies.
        platforms = [CXX],
    )

    runtime.export_file(
        name = "linear",
        src = "test/linear.pt2",
    )

    runtime.cxx_test(
        name = "pt2_archive_data_map_test",
        srcs = [
            "test/pt2_archive_data_map_test.cpp",
        ],
        deps = [
            ":pt2_archive_data_map",
            "//executorch/extension/data_loader:mmap_data_loader",
            "//executorch/kernels/portable:generated_lib",
            "//executorch/runtime/core:core",
            "//executorch/runtime/core:evalue",
            "//executorch/runtime/core:named_data_map",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/runtime/executor:program",
            "//executorch/runtime/executor/test:managed_memory_manager",
            "//executorch/runtime/platform:platform",
        ],
        env = {
            "TEST_LINEAR_PT2": "$(location :linear)",
            "ET_MODULE_LINEAR_PATH": "$(location fbcode//executorch/test/models:exported_program_and_data[ModuleLinear.pte])",
        },
        # Not available for mobile with miniz and json dependencies.
        platforms = [CXX],
    )
