# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    TOSAQuantizer,
    VgfQuantizer,
)
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.backends.arm.vgf import VgfCompileSpec, VgfPartitioner
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.fx.passes.operator_support import OperatorSupportBase


def parse_compile_spec(compile_specs: list[CompileSpec]) -> ArmCompileSpec:
    output_format = None
    for spec in compile_specs:
        if spec.key == "output_format":
            output_format = spec.value.decode()
            break
    else:
        raise ValueError("Compile spec without output format.")
    if output_format == TosaCompileSpec.get_output_format():
        return TosaCompileSpec.from_list(compile_specs)
    if output_format == EthosUCompileSpec.get_output_format():
        return EthosUCompileSpec.from_list(compile_specs)
    if output_format == VgfCompileSpec.get_output_format():
        return VgfCompileSpec.from_list(compile_specs)
    raise ValueError(f"Unknown output format {output_format}")


def create_partitioner(
    compile_spec: ArmCompileSpec,
    additional_checks: list[OperatorSupportBase] | None = None,
):
    if isinstance(compile_spec, TosaCompileSpec):
        return TOSAPartitioner(compile_spec, additional_checks)
    elif isinstance(compile_spec, EthosUCompileSpec):
        return EthosUPartitioner(compile_spec, additional_checks)
    elif isinstance(compile_spec, VgfCompileSpec):
        return VgfPartitioner(compile_spec, additional_checks)
    else:
        raise ValueError("compile spec doesn't target any Arm Partitioner")


def create_quantizer(compile_spec: ArmCompileSpec):
    if isinstance(compile_spec, TosaCompileSpec):
        return TOSAQuantizer(compile_spec)
    elif isinstance(compile_spec, EthosUCompileSpec):
        return EthosUQuantizer(compile_spec)
    elif isinstance(compile_spec, VgfCompileSpec):
        return VgfQuantizer(compile_spec)
    else:
        raise ValueError("compile spec doesn't target any Arm Quantizer")
