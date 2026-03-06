# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile specification for TensorRT backend."""

import json
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from executorch.exir.backend.compile_spec_schema import CompileSpec


# Key used to identify TensorRT compile specs in the CompileSpec list
TENSORRT_COMPILE_SPEC_KEY = "tensorrt_compile_spec"


class TensorRTPrecision(Enum):
    """Supported precision modes for TensorRT."""

    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    BF16 = "bf16"


@dataclass
class TensorRTCompileSpec:
    """Configuration options for TensorRT compilation.

    This dataclass holds all the configuration options needed to compile
    a model with TensorRT. It can be serialized to/from CompileSpec format
    for use with the ExecuTorch backend interface.

    Attributes:
        workspace_size: Maximum GPU memory (in bytes) that TensorRT can use
            for temporary storage during engine building. Default is 1GB.
        precision: Target precision for the TensorRT engine.
        strict_type_constraints: If True, TensorRT will strictly follow the
            specified precision. If False, it may use higher precision where
            beneficial for accuracy.
        max_batch_size: Maximum batch size the engine will be optimized for.
        device_id: CUDA device ID to use for building the engine.
        dla_core: DLA (Deep Learning Accelerator) core to use, if available.
            Set to -1 to disable DLA.
        allow_gpu_fallback: If True, allows falling back to GPU for layers
            that cannot run on DLA.
        optimization_level: TRT builder optimization level (0-5). Higher
            values spend more time searching for better kernels. Default 3.
        timing_cache_path: Path to persist TRT timing cache across builds.
            Empty string disables caching. Reusing a cache speeds up
            repeated engine builds significantly.
    """

    workspace_size: int = 1 << 30  # 1GB default
    precision: TensorRTPrecision = TensorRTPrecision.FP32
    strict_type_constraints: bool = False
    max_batch_size: int = 1
    device_id: int = 0
    dla_core: int = -1  # -1 means DLA is disabled
    allow_gpu_fallback: bool = True
    optimization_level: int = 3  # 0-5, higher = slower build, better perf
    timing_cache_path: str = ""  # empty = no persistent timing cache

    def to_compile_specs(self) -> List[CompileSpec]:
        """Serialize this TensorRTCompileSpec to a list of CompileSpec.

        Returns:
            List containing a single CompileSpec with the serialized options.
        """
        options = {
            "workspace_size": self.workspace_size,
            "precision": self.precision.value,
            "strict_type_constraints": self.strict_type_constraints,
            "max_batch_size": self.max_batch_size,
            "device_id": self.device_id,
            "dla_core": self.dla_core,
            "allow_gpu_fallback": self.allow_gpu_fallback,
            "optimization_level": self.optimization_level,
            "timing_cache_path": self.timing_cache_path,
        }
        return [
            CompileSpec(
                key=TENSORRT_COMPILE_SPEC_KEY,
                value=json.dumps(options).encode("utf-8"),
            )
        ]

    @classmethod
    def from_compile_specs(
        cls, compile_specs: List[CompileSpec]
    ) -> Optional["TensorRTCompileSpec"]:
        """Deserialize a TensorRTCompileSpec from a list of CompileSpec.

        Args:
            compile_specs: List of CompileSpec to search for TensorRT options.

        Returns:
            TensorRTCompileSpec if found, None otherwise.
        """
        for spec in compile_specs:
            if spec.key == TENSORRT_COMPILE_SPEC_KEY:
                value = (
                    spec.value.decode("utf-8")
                    if isinstance(spec.value, (bytes, bytearray))
                    else spec.value
                )
                options = json.loads(value)
                return cls(
                    workspace_size=options.get("workspace_size", 1 << 30),
                    precision=TensorRTPrecision(options.get("precision", "fp32")),
                    strict_type_constraints=options.get(
                        "strict_type_constraints", False
                    ),
                    max_batch_size=options.get("max_batch_size", 1),
                    device_id=options.get("device_id", 0),
                    dla_core=options.get("dla_core", -1),
                    allow_gpu_fallback=options.get("allow_gpu_fallback", True),
                    optimization_level=options.get("optimization_level", 3),
                    timing_cache_path=options.get("timing_cache_path", ""),
                )
        return None

    def __repr__(self) -> str:
        return (
            f"TensorRTCompileSpec("
            f"workspace_size={self.workspace_size}, "
            f"precision={self.precision.value}, "
            f"max_batch_size={self.max_batch_size})"
        )
