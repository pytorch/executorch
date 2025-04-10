# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
import torch

from executorch.exir import ExportRecipe
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig

def iphone_coreml_et_recipe(ios: int = 17, compute_unit: str = "CPU_ONLY") -> ExportRecipe:
    import coremltools as ct
    from coremltools.optimize.torch.quantization.quantization_config import (
        LinearQuantizerConfig,
        QuantizationScheme,
    )
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.backends.apple.coreml.quantizer import CoreMLQuantizer

    # TODO: Add compute precision, compute unit, and model type
    quantization_config = LinearQuantizerConfig.from_dict(
        {
            "global_config": {
                "quantization_scheme": QuantizationScheme.affine,
                "activation_dtype": torch.quint8,
                "weight_dtype": torch.qint8,
                "weight_per_channel": True,
            }
        }
    )
    minimum_deployment_target = {
        15: ct.target.iOS15,
        16: ct.target.iOS16,
        17: ct.target.iOS17,
        18: ct.target.iOS18,
    }[ios]
    compute_unit_types = ["CPU_ONLY", "CPU_AND_NE", "CPU_AND_GPU", "ALL"]
    assert (
        compute_unit in compute_unit_types
    ), f"Invalid compute unit: {compute_unit}, should be one of {compute_unit_types}"
    if compute_unit == "CPU_ONLY":
        compute_unit_specs = ct.ComputeUnit[ct.ComputeUnit.CPU_ONLY.name.upper()]
    elif compute_unit == "CPU_AND_NE":
        compute_unit_specs = ct.ComputeUnit[ct.ComputeUnit.CPU_AND_NE.name.upper()]
    elif compute_unit == "CPU_AND_GPU":
        compute_unit_specs = ct.ComputeUnit[ct.ComputeUnit.CPU_AND_GPU.name.upper()]
    else:  # compute_unit == "ALL"
        compute_unit_specs = ct.ComputeUnit[ct.ComputeUnit.ALL.name.upper()]
    compile_specs = CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=minimum_deployment_target,
        compute_precision=ct.precision(ct.precision.FLOAT16.value),
        compute_unit=compute_unit_specs,
        model_type=CoreMLBackend.MODEL_TYPE.MODEL,
    )
    take_over_mutable_buffer = minimum_deployment_target >= ct.target.iOS18
    partitioner = CoreMLPartitioner(
        compile_specs=compile_specs,
        take_over_mutable_buffer=take_over_mutable_buffer,
    )
    return ExportRecipe(
        "iphone_coreml",
        quantizer=CoreMLQuantizer(quantization_config),
        partitioners=[partitioner],
        edge_compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        edge_transform_passes=[],
        executorch_backend_config=ExecutorchBackendConfig(),
    )
