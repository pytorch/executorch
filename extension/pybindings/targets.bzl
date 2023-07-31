load("@fbsource//xplat/executorch/backends:backends.bzl", "get_all_cpu_backend_targets")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

# Aten ops with portable kernel
MODELS_ATEN_OPS_LEAN_MODE_GENERATED_LIB = [
    "//executorch/kernels/portable:generated_lib",
]

MODULE_DEPS = [
    "//caffe2:ATen",
    "//caffe2:torch",
    "//caffe2:torch_extension",
    "//executorch/runtime/kernel:operator_registry",
    "//executorch/runtime/executor:executor",
    "//executorch/schema:bundled_program_schema",
    "//executorch/schema:schema",
    "//executorch/extension/aten_util:aten_bridge",
    "//executorch/util:bundled_program_verification",
    "//executorch/extension/data_loader:buffer_data_loader",
    "//executorch/extension/data_loader:mmap_data_loader",
    "//executorch/util:test_memory_config",
    "//executorch/util:util",
    "//executorch/runtime/executor/test:test_backend_compiler_lib",
] + get_all_cpu_backend_targets()

# Generated lib for all ATen ops with aten kernel used by models in model inventory
MODELS_ATEN_OPS_ATEN_MODE_GENERATED_LIB = [
    "//executorch/kernels/quantized:generated_lib_aten",
    "//executorch/kernels/aten:generated_lib_aten",
]

def executorch_pybindings(python_module_name, srcs = [], cppdeps = [], visibility = ["//executorch/..."]):
    runtime.cxx_python_extension(
        name = python_module_name,
        srcs = [
            "//executorch/extension/pybindings:pybindings.cpp",
        ] + srcs,
        base_module = "executorch.extension.pybindings",
        preprocessor_flags = [
            "-DEXECUTORCH_PYTHON_MODULE_NAME={}".format(python_module_name),
        ],
        deps = [
            "//executorch/runtime/core:core",
            "//executorch/schema:schema",
            "//executorch/util:read_file",
        ] + cppdeps,
        external_deps = [
            "pybind11",
        ],
        use_static_deps = True,
        _is_external_target = bool(visibility != ["//executorch/..."]),
        visibility = visibility,
    )

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Export these so the internal fb/ subdir can create pybindings with custom internal deps
    # without forking the pybinding source.
    runtime.export_file(
        name = "pybindings.cpp",
        visibility = ["//executorch/extension/pybindings/..."],
    )

    runtime.export_file(
        name = "module.cpp",
        visibility = ["//executorch/extension/pybindings/..."],
    )

    executorch_pybindings(
        srcs = [
            "module_stub.cpp",
        ],
        python_module_name = "operator",
    )
