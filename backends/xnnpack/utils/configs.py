from typing import Optional

import executorch.exir as exir
from executorch.backends.canonical_partitioners.duplicate_dequant_node_pass import (
    DuplicateDequantNodePass,
)
from executorch.exir import CaptureConfig

### XNNPACK Configs ###
def get_xnnpack_edge_compile_config(additional_passes=None) -> exir.EdgeCompileConfig:
    additional_passes = additional_passes if additional_passes else []
    passes = additional_passes + [DuplicateDequantNodePass()]
    return exir.EdgeCompileConfig(
        passes=passes,
        _check_ir_validity=False,
        _use_edge_ops=True,
    )


def get_xnnpack_executorch_backend_config(
    additional_passes=None,
) -> exir.ExecutorchBackendConfig:
    additional_passes = additional_passes if additional_passes else []
    return exir.ExecutorchBackendConfig(
        passes=additional_passes,
        extract_segments=True,
    )


def get_xnnpack_capture_config(dynamic_shape=False, enable_aot: Optional[bool] = None):
    if enable_aot is None:
        return CaptureConfig(pt2_mode=True, enable_dynamic_shape=dynamic_shape)
    else:
        return CaptureConfig(
            pt2_mode=True, enable_dynamic_shape=dynamic_shape, enable_aot=enable_aot
        )
