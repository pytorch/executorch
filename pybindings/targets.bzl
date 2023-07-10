load("@fbsource//xplat/executorch/backends:backends.bzl", "get_all_cpu_backend_targets")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

# Aten ops with portable kernel
MODELS_ATEN_OPS_LEAN_MODE_GENERATED_LIB = [
    "//executorch/kernels/portable:generated_lib",
]

# Custom ops with portable kernel
MODELS_CUSTOM_OPS_LEAN_MODE_GENERATED_LIB = [
    "//executorch/kernels/quantized:generated_lib",
    "//pye/model_inventory/asr_models/runtime:generated_custom_op_lib_lean",
]

MODELS_ALL_OPS_LEAN_MODE_GENERATED_LIB = MODELS_ATEN_OPS_LEAN_MODE_GENERATED_LIB + MODELS_CUSTOM_OPS_LEAN_MODE_GENERATED_LIB

MODULE_DEPS = [
    "//caffe2:ATen",
    "//caffe2:torch",
    "//caffe2:torch_extension",
    "//executorch/runtime/kernel:operator_registry",
    "//executorch/executor:executor",
    "//executorch/schema:bundled_program_schema",
    "//executorch/schema:schema",
    "//executorch/util:aten_bridge",
    "//executorch/util:bundled_program_verification",
    "//executorch/util:embedded_data_loader",
    "//executorch/util:mmap_data_loader",
    "//executorch/util:test_memory_config",
    "//executorch/util:util",
    "//executorch/executor/test:test_backend_compiler_lib",
] + get_all_cpu_backend_targets()

# Generated lib for all ATen ops with aten kernel used by models in model inventory
MODELS_ATEN_OPS_ATEN_MODE_GENERATED_LIB = [
    "//executorch/kernels/quantized:generated_lib_aten",
    "//executorch/kernels/aten:generated_lib_aten",
]

# Generated libs for all ATen ops AND custom ops used by models in //pye/model_inventory
MODELS_ALL_OPS_ATEN_MODE_GENERATED_LIB = MODELS_ATEN_OPS_ATEN_MODE_GENERATED_LIB + [
    "//caffe2/fb/custom_ops/turing:turing_lib_aten",
    "//pye/model_inventory/asr_models/runtime:generated_lib_aten",
    "//pye/model_inventory/asr_models/runtime:custom_ops_generated_lib_aten",
    "//pye/model_inventory/fam_models/runtime:generated_lib_aten",
    "//pye/model_inventory/ocr_detection_model_non_quantized/runtime:generated_lib_aten",
    "//caffe2/fb/custom_ops/nimble/et_runtime:generated_lib_aten",
    "//pye/model_inventory/keyboard_tracking_model/runtime:generated_lib_aten",
]

def executorch_pybindings(python_module_name, srcs = [], cppdeps = [], visibility = ["//executorch/..."]):
    runtime.cxx_python_extension(
        name = python_module_name,
        srcs = [
            "pybindings.cpp",
        ] + srcs,
        base_module = "executorch.pybindings",
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
        xplat_deps = [
            "//arvr/third-party/pybind11:pybind11",
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

    executorch_pybindings(
        srcs = [
            "module_stub.cpp",
        ],
        python_module_name = "operator",
    )
