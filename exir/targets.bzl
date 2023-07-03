load("//bento:buck_macros.bzl", "bento_kernel")

executorch_bento_kernels_base_deps = [
    "//executorch/exir:bento_deps",
    "//pye/lib:eager_model_base",
]

executorch_portable_kernel_lib = ["//executorch/pybindings:portable"]
executorch_aten_mode_lib = ["//executorch/pybindings:aten_mode_lib"]

model_inventory_list = [
    {
        "deps": [
            "//on_device_ai/helios/arvr/nn:nn",
            "//on_device_ai/helios/compiler/utils:utils",
            "//on_device_ai/fx/tracers:leaf_modules",
            "//on_device_ai/helios/arch_params:arch_params",
            "//on_device_ai/helios/fx/passes:passes",
            "//on_device_ai/helios/pytorch/turing:turing_ops",
        ],
        "name": "executorch_helios",
    },
]

def load_executorch_bento_kernels():
    for entry in model_inventory_list:
        for suffix in ("", "_portable"):
            ops_lib = executorch_portable_kernel_lib if suffix else executorch_aten_mode_lib
            bento_kernel(
                name = entry["name"] + suffix,
                deps = executorch_bento_kernels_base_deps + entry["deps"] + ops_lib,
            )
