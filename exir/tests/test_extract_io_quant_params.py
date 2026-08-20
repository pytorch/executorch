# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.passes.quantize_io_pass import extract_io_quant_params

from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class SimpleAdd(torch.nn.Module):
    def forward(self, x, y):
        return x + y


class TestExtractIOQuantParamsPT2E(unittest.TestCase):
    def setUp(self):
        self.example_inputs = (
            torch.ones(1, 5),
            torch.full(
                (
                    1,
                    5,
                ),
                2.0,
            ),
        )
        self.mod = SimpleAdd().eval()

        # Setup XNNPACK quantizer for example
        self.quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config()
        self.quantizer.set_global(operator_config)

        exported = torch.export.export(
            self.mod,
            copy.deepcopy(self.example_inputs),
            strict=True,
        )
        prepared = prepare_pt2e(exported.module(), self.quantizer)

        # Call observers to calibrate
        _ = prepared(*self.example_inputs)

        converted = convert_pt2e(prepared)

        # Export again with quant parameters
        final_export = torch.export.export(
            converted,
            self.example_inputs,
            strict=True,
        )

        # Lower to EdgeProgramManager
        self.edge_prog = to_edge_transform_and_lower(final_export)

    def test_roundtrip_extracts_io_params(self):
        # Get dict with quant parameters
        q = extract_io_quant_params(
            self.edge_prog,
            input_idxs=(0, 1),
            output_idxs=(0,),
        )

        # Validate structure
        self.assertIn("inputs", q)
        self.assertIn("outputs", q)
        self.assertEqual(len(q["inputs"]), 2)
        self.assertEqual(len(q["outputs"]), 1)

        # Each entry must have a float 'scale' and int 'zero_point'
        for name, params in q["inputs"].items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(params["scale"], float)
            self.assertIsInstance(params["zero_point"], int)

        out_name, out_params = next(iter(q["outputs"].items()))
        self.assertIsInstance(out_name, str)
        self.assertIsInstance(out_params["scale"], float)
        self.assertIsInstance(out_params["zero_point"], int)


if __name__ == "__main__":
    unittest.main()
