# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import os
import sys
import unittest

import torch

from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.examples.models.llama.export_llama_lib import (
    build_args_parser,
    get_llama_model,
)


# Add project dir to sys path to workaround importlib.import_module() conditions in model_factory.py
this_files_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(this_files_dir, "../../../.."))
sys.path.append(project_dir)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestLlama(unittest.TestCase):
    """
    Test class of Llama models. Type of Llama model depends on command line parameters:
    --llama_inputs <path to .pt file> <path to json file>
    Example: --llama_inputs stories110M/stories110M.pt stories110M/params.json
    """

    def prepare_model(self):

        checkpoint = None
        params_file = None
        if conftest.is_option_enabled("llama_inputs"):
            param_list = conftest.get_option("llama_inputs")
            assert (
                isinstance(param_list, list) and len(param_list) == 2
            ), "invalid number of inputs for --llama_inputs"
            checkpoint = param_list[0]
            params_file = param_list[1]
            assert isinstance(checkpoint, str) and isinstance(
                params_file, str
            ), "invalid input for --llama_inputs"
        else:
            logging.warning(
                "Skipping Llama test because of lack of input. To run use --llama_inputs <.pt> <.json>"
            )
            return None, None, None

        assert os.path.isfile(checkpoint) and os.path.isfile(
            params_file
        ), "Invalid file paths"

        # TODO: Enable key value cache
        args = [
            "--disable_dynamic_shape",
            "-c",
            checkpoint,
            "-p",
            params_file,
            "--model",
            "stories110m",
        ]
        parser = build_args_parser()
        args = parser.parse_args(args)

        llama_model, llama_inputs, llama_meta = get_llama_model(args)

        # TODO: Remove workaround since attention mask should not be persistent,
        # it only works if input shape is always the same
        freqs_c = "freqs_cos"
        freqs_s = "freqs_sin"
        for i in range(llama_model.n_layers):
            val = llama_model.layers[i].attention.get_buffer("mask")
            llama_model.layers[i].attention.register_buffer(
                "mask", val, persistent=True
            )
            val = llama_model.layers[i].attention.rope.get_buffer(freqs_c)
            llama_model.layers[i].attention.rope.register_buffer(
                freqs_c, val, persistent=True
            )
            val = llama_model.layers[i].attention.rope.get_buffer(freqs_s)
            llama_model.layers[i].attention.rope.register_buffer(
                freqs_s, val, persistent=True
            )

        return llama_model, llama_inputs, llama_meta

    def test_llama_tosa_MI(self):
        llama_model, llama_inputs, llama_meta = self.prepare_model()

        if llama_model is None and llama_inputs is None and llama_meta is None:
            return

        with torch.no_grad():
            (
                ArmTester(
                    llama_model,
                    example_inputs=llama_inputs,
                    compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
                    constant_methods=llama_meta,
                )
                .export()
                .to_edge_transform_and_lower()
                .check_count({"torch.ops.higher_order.executorch_call_delegate": 14})
                .to_executorch()
                .run_method_and_compare_outputs(
                    inputs=llama_inputs, atol=1.8, rtol=0.01  # TODO: decrease tolerance
                )
            )
