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

import pytest
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


class TestLlama(unittest.TestCase):
    """
    Test class of Llama models. Type of Llama model depends on command line parameters:
    --llama_inputs <path to .pt file> <path to json file> <name of model variant>
    Example: --llama_inputs stories110M/stories110M.pt stories110M/params.json stories110m
    For more examples and info see examples/models/llama/README.md.
    """

    def prepare_model(self):

        checkpoint = None
        params_file = None
        usage = "To run use --llama_inputs <.pt/.pth> <.json> <name>"

        if conftest.is_option_enabled("llama_inputs"):
            param_list = conftest.get_option("llama_inputs")

            if not isinstance(param_list, list) or len(param_list) != 3:
                raise RuntimeError(
                    f"Invalid number of inputs for --llama_inputs. {usage}"
                )
            if not all(isinstance(param, str) for param in param_list):
                raise RuntimeError(
                    f"All --llama_inputs are expected to be strings. {usage}"
                )

            checkpoint = param_list[0]
            params_file = param_list[1]
            model_name = param_list[2]
        else:
            logger.warning(
                "Skipping Llama tests because of missing --llama_inputs. {usage}"
            )
            return None, None, None

        assert os.path.isfile(checkpoint) and os.path.isfile(
            params_file
        ), "Invalid file paths"

        logger.info("Running test_llama.py")

        # TODO: Enable key value cache
        args = [
            "--disable_dynamic_shape",
            "-c",
            checkpoint,
            "-p",
            params_file,
            "--model",
            model_name,
        ]
        parser = build_args_parser()
        args = parser.parse_args(args)

        llama_model, llama_inputs, llama_meta = get_llama_model(args)

        return llama_model, llama_inputs, llama_meta

    def test_llama_tosa_MI(self):
        llama_model, llama_inputs, llama_meta = self.prepare_model()

        if llama_model is None or llama_inputs is None:
            pytest.skip("Missing model and/or input files")

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
                .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
                .to_executorch()
                .run_method_and_compare_outputs(
                    inputs=llama_inputs,
                    atol=4.3,
                    rtol=1.1,  # TODO: MLETORCH-825 decrease tolerance
                )
            )

    def test_llama_tosa_BI(self):
        llama_model, llama_inputs, llama_meta = self.prepare_model()

        if llama_model is None or llama_inputs is None:
            pytest.skip("Missing model and/or input files")

        with torch.no_grad():
            (
                ArmTester(
                    llama_model,
                    example_inputs=llama_inputs,
                    compile_spec=common.get_tosa_compile_spec("TOSA-0.80+BI"),
                    constant_methods=llama_meta,
                )
                .quantize()
                .export()
                .to_edge_transform_and_lower()
                .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
                .to_executorch()
                .run_method_and_compare_outputs(
                    inputs=llama_inputs,
                    atol=9.9,
                    rtol=1.5,  # TODO: Tolerance needs to be updated after MLETORCH-907
                )
            )
