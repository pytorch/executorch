load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.python_library(
        name = "aoti_partitioner",
        srcs = [
            "aoti_partitioner.py",
        ],
        visibility = [
            "//executorch/...",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/exir/backend:partitioner",
            "//executorch/exir/backend:utils",
        ],
    )

    runtime.python_library(
        name = "aoti_backend",
        srcs = [
            "aoti_backend.py",
        ],
        visibility = [
            "//executorch/...",
        ],
        deps = [
            "//caffe2:torch",
            "//executorch/backends/aoti/passes:passes",
            "//executorch/exir/_serialize:lib",
            "//executorch/exir/backend:backend_details",
            "//executorch/exir/backend:compile_spec_schema",
        ],
    )

    # AOTI common shims functionality using ETensor
    # TODO(gasoonjia): Remove this after metal migration
    runtime.cxx_library(
        name = "common_shims",
        srcs = [
            "common_shims.cpp",
        ],
        headers = [
            "common_shims.h",
            "export.h",
            "utils.h",
        ],
        # @lint-ignore BUCKLINT: Avoid `link_whole=True` (https://fburl.com/avoid-link-whole)
        link_whole = True,
        supports_python_dlopen = True,
        # Constructor needed for backend registration.
        compiler_flags = ["-Wno-global-constructors"],
        visibility = ["PUBLIC"],
        deps = [
            "//executorch/runtime/core:core",
            "//executorch/runtime/core/exec_aten:lib",
        ],
    )

    # AOTI model container functionality
    runtime.cxx_library(
        name = "delegate_handle",
        headers = [
            "aoti_delegate_handle.h",
        ],
        # @lint-ignore BUCKLINT: Avoid `link_whole=True` (https://fburl.com/avoid-link-whole)
        link_whole = True,
        supports_python_dlopen = True,
        # Constructor needed for backend registration.
        compiler_flags = ["-Wno-global-constructors"],
        visibility = ["PUBLIC"],
        deps = [
            "//executorch/runtime/backend:interface",
            "//executorch/runtime/core:core",
        ],
    )

    # Common AOTI functionality (combining both common_shims and delegate_handle)
    runtime.cxx_library(
        name = "aoti_common",
        # @lint-ignore BUCKLINT: Avoid `link_whole=True` (https://fburl.com/avoid-link-whole)
        link_whole = True,
        supports_python_dlopen = True,
        visibility = ["PUBLIC"],
        exported_deps = [
            ":common_shims",
            ":delegate_handle",
        ],
    )

    # SlimTensor-based common shims library
    # Uses SlimTensor for all tensor operations
    # TODO(gasoonjia): Replace common_shims with this one after metal migration
    runtime.cxx_library(
        name = "common_shims_slim",
        srcs = [
            "common_shims_slim.cpp",
        ],
        headers = [
            "common_shims_slim.h",
            "export.h",
            "utils.h",
        ],
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            "//executorch/runtime/core:core",
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/backends/aoti/slim/core:slimtensor",
        ],
    )

    # Common AOTI functionality for SlimTensor-based backends (combining common_shims_slim and delegate_handle)
    # All CUDA backend code should depend on this target
    # TODO(gasoonjia): Replace aoti_common with this one after metal migration
    runtime.cxx_library(
        name = "aoti_common_slim",
        # @lint-ignore BUCKLINT: Avoid `link_whole=True` (https://fburl.com/avoid-link-whole)
        link_whole = True,
        supports_python_dlopen = True,
        visibility = ["PUBLIC"],
        exported_deps = [
            ":common_shims_slim",
            ":delegate_handle",
        ],
    )
