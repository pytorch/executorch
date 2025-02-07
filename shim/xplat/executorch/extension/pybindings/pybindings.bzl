load("@fbsource//xplat/executorch/backends:backends.bzl", "get_all_cpu_backend_targets")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

# Aten ops with portable kernel
MODELS_ATEN_OPS_LEAN_MODE_GENERATED_LIB = [
    "//executorch/kernels/portable:generated_lib",
    "//executorch/kernels/quantized:generated_lib",
]

PORTABLE_MODULE_DEPS = [
    "//executorch/runtime/kernel:operator_registry",
    "//executorch/runtime/executor:program",
    "//executorch/devtools/bundled_program/schema:bundled_program_schema_fbs",
    "//executorch/extension/aten_util:aten_bridge",
    "//executorch/devtools/bundled_program:runtime",
    "//executorch/extension/data_loader:buffer_data_loader",
    "//executorch/extension/data_loader:mmap_data_loader",
    "//executorch/extension/memory_allocator:malloc_memory_allocator",
    "//executorch/runtime/executor/test:test_backend_compiler_lib",
    "//executorch/devtools/etdump:etdump_flatcc",
] + get_all_cpu_backend_targets()

ATEN_MODULE_DEPS = [
    "//executorch/runtime/kernel:operator_registry",
    "//executorch/runtime/executor:program_aten",
    "//executorch/runtime/core/exec_aten:lib",
    "//executorch/devtools/bundled_program/schema:bundled_program_schema_fbs",
    "//executorch/extension/data_loader:buffer_data_loader",
    "//executorch/extension/data_loader:mmap_data_loader",
    "//executorch/extension/memory_allocator:malloc_memory_allocator",
    "//executorch/devtools/bundled_program:runtime_aten",
    "//executorch/runtime/executor/test:test_backend_compiler_lib_aten",
    "//executorch/devtools/etdump:etdump_flatcc",
]

# Generated lib for all ATen ops with aten kernel used by models in model inventory
MODELS_ATEN_OPS_ATEN_MODE_GENERATED_LIB = [
    "//executorch/kernels/quantized:generated_lib_aten",
    "//executorch/kernels/aten:generated_lib",
]

def executorch_pybindings(python_module_name, srcs = [], cppdeps = [], visibility = ["//executorch/..."], types = [], compiler_flags = []):
    runtime.cxx_python_extension(
        name = python_module_name,
        srcs = [
            "//executorch/extension/pybindings:pybindings.cpp",
        ] + srcs,
        types = types,
        base_module = "executorch.extension.pybindings",
        compiler_flags = compiler_flags,
        preprocessor_flags = [
            "-DEXECUTORCH_PYTHON_MODULE_NAME={}".format(python_module_name),
        ],
        deps = [
            "//executorch/exir:_warnings",
            "//executorch/runtime/core:core",
        ] + cppdeps,
        external_deps = [
            "pybind11",
            "libtorch_python",
        ],
        use_static_deps = True,
        _is_external_target = bool(visibility != ["//executorch/..."]),
        visibility = visibility,
    )
