load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "dynamic_memory_allocator",
        exported_headers = [
            "DynamicMemoryAllocator.h",
        ],
        exported_deps = [
            "//executorch/runtime/core:memory_allocator",
        ],
        visibility = [
            "//executorch/util/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "system",
        exported_headers = [
            "system.h",
        ],
        visibility = [
            "//executorch/util/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "test_memory_config",
        srcs = [],
        exported_headers = ["TestMemoryConfig.h"],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            "//executorch/runtime/core:core",
            "//executorch/runtime/core:memory_allocator",
        ],
    )

    runtime.cxx_library(
        name = "read_file",
        srcs = ["read_file.cpp"],
        exported_headers = ["read_file.h"],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            ":system",
            "//executorch/runtime/core:core",
            "//executorch/runtime/platform:compiler",
        ],
    )

    runtime.cxx_library(
        name = "embedded_data_loader",
        srcs = [],
        exported_headers = ["embedded_data_loader.h"],
        visibility = [
            "//executorch/backends/test/...",
            "//executorch/runtime/executor/test/...",
            "//executorch/extension/pybindings/...",
            "//executorch/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/runtime/core:core",
        ],
    )

    runtime.cxx_library(
        name = "shared_ptr_data_loader",
        srcs = [],
        exported_headers = ["shared_ptr_data_loader.h"],
        visibility = [
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/runtime/core:core",
        ],
    )

    runtime.cxx_library(
        name = "file_data_loader",
        srcs = ["file_data_loader.cpp"],
        exported_headers = ["file_data_loader.h"],
        visibility = [
            "//executorch/test/...",
            "//executorch/runtime/executor/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/runtime/core:core",
        ],
    )

    runtime.cxx_library(
        name = "mmap_data_loader",
        srcs = ["mmap_data_loader.cpp"],
        exported_headers = ["mmap_data_loader.h"],
        visibility = [
            "//executorch/test/...",
            "//executorch/extension/pybindings/...",
            "//executorch/runtime/executor/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/runtime/core:core",
        ],
    )

    runtime.cxx_library(
        name = "memory_utils",
        srcs = ["memory_utils.cpp"],
        exported_headers = ["memory_utils.h"],
        visibility = [
            "//executorch/backends/...",
            "//executorch/runtime/backend/...",
            "//executorch/util/test/...",
        ],
        deps = [
            "//executorch/runtime/core:core",
        ],
        exported_deps = [
            ":system",
        ],
    )

    COMPILER_FLAGS = [
        "-frtti",
        "-fno-omit-frame-pointer",
        "-fexceptions",
        "-Wno-error",
        "-Wno-unused-local-typedef",
        "-Wno-self-assign-overloaded",
        "-Wno-global-constructors",
        "-Wno-unused-function",
    ]

    runtime.cxx_library(
        name = "aten_bridge",
        srcs = ["aten_bridge.cpp"],
        exported_headers = ["aten_bridge.h"],
        compiler_flags = COMPILER_FLAGS,
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            "//executorch/runtime/core:core",
            "//executorch/runtime/core/exec_aten:lib",
        ],
        fbcode_deps = [
            "//caffe2:ATen-core",
            "//caffe2:ATen-cpu",
            "//caffe2/c10:c10",
        ],
        xplat_deps = [
            "//xplat/caffe2:torch_mobile_core",
            "//xplat/caffe2/c10:c10",
        ],
    )

    runtime.cxx_library(
        name = "ivalue_flatten_unflatten",
        srcs = ["ivalue_flatten_unflatten.cpp"],
        exported_headers = ["ivalue_flatten_unflatten.h"],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
            "//executorch/pytree:pytree",
        ],
        compiler_flags = ["-Wno-missing-prototypes"],
        fbcode_deps = [
            "//caffe2:ATen-core",
            "//caffe2:ATen-cpu",
            "//caffe2/c10:c10",
        ],
        xplat_deps = [
            "//xplat/caffe2:torch_mobile_core",
            "//xplat/caffe2/c10:c10",
        ],
    )

    for aten_mode in (True, False):
        aten_suffix = ("_aten" if aten_mode else "")
        runtime.cxx_library(
            name = "bundled_program_verification" + aten_suffix,
            srcs = ["bundled_program_verification.cpp"],
            exported_headers = ["bundled_program_verification.h"],
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
            deps = [
                "//executorch/runtime/core/exec_aten/testing_util:tensor_util" + aten_suffix,
                "//executorch/runtime/executor:executor" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:dim_order_util" + aten_suffix,
                "//executorch/schema:schema",
                "//executorch/schema:bundled_program_schema",
            ],
        )

        runtime.cxx_library(
            name = "util" + aten_suffix,
            srcs = [],
            exported_headers = ["util.h"],
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
            deps = [
                "//executorch/runtime/executor:executor" + aten_suffix,
            ],
        )
