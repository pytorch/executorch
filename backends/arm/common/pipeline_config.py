# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import auto, Enum
from typing import Any, cast


class SoftmaxDecompositionConfig(Enum):
    MASKED = auto()  # Stable softmax + masked fill decomposition
    STABLE = auto()  # Stable softmax, no masked fill decomposition


class LeakyReLULoweringConfig(Enum):
    TABLE = auto()  # Lower quantized leaky_relu with TOSA TABLE
    DECOMPOSE = auto()  # Lower leaky_relu into clamp, mul, and add


@dataclass
class QuantizeInfConfig:
    """Replacement values for infinities before quantization.

    Infinities cannot be quantized directly, so the Arm pipeline replaces them
    with finite values before running the quantization passes.

    Args:
        neg_inf (float): Value used for ``-inf``.
        pos_inf (float): Value used for ``inf``.

    """

    neg_inf: float = -256.0
    pos_inf: float = 255.0


@dataclass
class ArmPassPipelineConfig:
    """Options for tuning the Arm pass pipeline.

    Args:
        softmax (SoftmaxDecompositionConfig): Softmax decomposition mode.
        leaky_relu (LeakyReLULoweringConfig): Quantized leaky_relu lowering
            mode.
        quantize_inf (QuantizeInfConfig): Values used when replacing
            infinities before quantization.

    Example:
        compile_spec.set_pass_pipeline_config(
            ArmPassPipelineConfig(
                softmax=SoftmaxDecompositionConfig.STABLE,
                leaky_relu=LeakyReLULoweringConfig.DECOMPOSE,
                quantize_inf=QuantizeInfConfig(
                    neg_inf=-100.0,
                    pos_inf=100.0,
                ),
            )
        )

    """

    softmax: SoftmaxDecompositionConfig = SoftmaxDecompositionConfig.MASKED
    leaky_relu: LeakyReLULoweringConfig = LeakyReLULoweringConfig.DECOMPOSE
    quantize_inf: QuantizeInfConfig = field(default_factory=QuantizeInfConfig)

    def is_default(self) -> bool:
        return (
            self.softmax is SoftmaxDecompositionConfig.MASKED
            and self.leaky_relu is LeakyReLULoweringConfig.DECOMPOSE
            and self.quantize_inf == QuantizeInfConfig()
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if is_dataclass(value):
                data[f.name] = asdict(cast(Any, value))
            elif isinstance(value, Enum):
                data[f.name] = value.name
            else:
                raise AssertionError(f"Cannot serialize {f.name}")
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArmPassPipelineConfig":
        config = cls()
        for f in fields(cls):
            raw_value = data.get(f.name)
            if raw_value is None:
                continue

            if f.name == "quantize_inf":
                config.quantize_inf = QuantizeInfConfig(**raw_value)
            else:
                # The field is an enum
                enum_type = f.type
                setattr(config, f.name, enum_type[raw_value])
        return config

    def serialize(self) -> bytes:
        """Return a serialized representation of this config."""
        return json.dumps(self.to_dict()).encode()

    def __repr__(self):
        fields = ", ".join(f"{name}={value!r}" for name, value in self.__dict__.items())
        return f"({fields})"
