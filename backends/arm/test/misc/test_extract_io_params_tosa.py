# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
from executorch.backends.arm.arm_backend import ArmCompileSpecBuilder
from executorch.backends.arm.quantizer import VgfQuantizer
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)

from executorch.backends.arm.test.common import SkipIfNoModelConverter
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.backends.arm.vgf import VgfPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.passes.quantize_io_pass import extract_io_quant_params
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class SimpleAdd(torch.nn.Module):
    def forward(self, x, y):
        return x + y


@pytest.mark.parametrize(
    "builder_method, quantizer_cls, partitioner_cls",
    [
        ("tosa_compile_spec", TOSAQuantizer, TOSAPartitioner),
        pytest.param(
            "vgf_compile_spec",
            VgfQuantizer,
            VgfPartitioner,
            marks=SkipIfNoModelConverter,
            id="VGF",
        ),
    ],
)
def test_roundtrip_extracts_io_params(builder_method, quantizer_cls, partitioner_cls):
    """
    Validates that IO quantization parameters round-trip for both flows.
    """
    example_inputs = (
        torch.ones(1, 5),
        torch.full((1, 5), 2.0),
    )
    mod = SimpleAdd().eval()

    base_spec = TosaSpecification.create_from_string("TOSA-1.0+INT")
    compile_spec = getattr(ArmCompileSpecBuilder(), builder_method)(
        tosa_spec=base_spec
    ).build()

    quantizer = quantizer_cls(compile_spec)
    operator_config = get_symmetric_quantization_config(is_qat=True)
    quantizer.set_global(operator_config)

    exported = torch.export.export(mod, copy.deepcopy(example_inputs), strict=True)
    prepared = prepare_pt2e(exported.module(), quantizer)
    _ = prepared(*example_inputs)

    converted = convert_pt2e(prepared)
    final_export = torch.export.export(converted, example_inputs, strict=True)
    partitioner = partitioner_cls(compile_spec)
    edge_prog = to_edge_transform_and_lower(final_export, partitioner=[partitioner])

    # Extract IO quantization parameters
    q = extract_io_quant_params(
        edge_prog,
        input_idxs=(0, 1),
        output_idxs=(0,),
    )

    assert "inputs" in q
    assert "outputs" in q
    assert len(q["inputs"]) == 2
    assert len(q["outputs"]) == 1

    for name, params in q["inputs"].items():
        assert isinstance(name, str)
        assert isinstance(params["scale"], float)
        assert isinstance(params["zero_point"], int)

    out_name, out_params = next(iter(q["outputs"].items()))
    assert isinstance(out_name, str)
    assert isinstance(out_params["scale"], float)
    assert isinstance(out_params["zero_point"], int)
