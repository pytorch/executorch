# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging

import os
import sys

from typing import Tuple

import pytest
import torch
from executorch.backends.arm._passes import InsertInt32CastsAfterInt64PlaceholdersPass
from executorch.backends.arm.quantizer import get_symmetric_quantization_config

from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from executorch.examples.models.llama.export_llama_lib import (
    build_args_parser,
    get_llama_model,
)
from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
    replace_kv_cache_with_static_quantized_kv_cache,
)

from executorch.extension.llm.export.config.llm_config import LlmConfig

input_t = Tuple[torch.Tensor, ...]

# Add project dir to sys path to workaround importlib.import_module() conditions in model_factory.py
this_files_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(this_files_dir, "../../../.."))
sys.path.append(project_dir)

logger = logging.getLogger(__name__)


class LlamaPositionalAdapter(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, input_pos):
        return self.model(tokens, {"input_pos": input_pos})


class TestLlama:
    """Test class of Llama models.

    Type of Llama model depends on command line parameters:
    --llama_inputs <path to .pt file> <path to json file> <name of model variant>
    Example: --llama_inputs stories110M/stories110M.pt stories110M/params.json stories110m
    For more examples and info see examples/models/llama/README.md.

    """

    def prepare_model(self, use_static_quantized_kv_cache: bool = True):
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

        args = [
            "--disable_dynamic_shape",
            "--max_seq_length",
            "4096",
            "--max_context_length",
            "4096",
            "-c",
            checkpoint,
            "-p",
            params_file,
            "--model",
            model_name,
            "--use_kv_cache",
        ]

        parser = build_args_parser()
        args = parser.parse_args(args)
        llm_config = LlmConfig.from_args(args)

        llama_model, llama_inputs, llama_meta = get_llama_model(llm_config)

        if llm_config.model.use_kv_cache:
            if use_static_quantized_kv_cache:
                replace_kv_cache_with_static_quantized_kv_cache(
                    llama_model,
                    use_custom_update_cache_op=False,
                    use_per_channel=False,
                )
            tokens, attn_options = llama_inputs
            llama_model = LlamaPositionalAdapter(llama_model).eval()
            llama_inputs = (tokens, attn_options["input_pos"])

        return llama_model, llama_inputs, llama_meta


def _use_partial_quantizer(pipeline, eps=2**-16):
    """Set the pipeline's quantizer to only include Linear layers."""
    pipeline.quantizer.set_global(None)
    pipeline.quantizer.set_module_type(
        torch.nn.Linear, get_symmetric_quantization_config(eps=eps)
    )


def test_llama_tosa_FP():
    llama_model, llama_inputs, llama_meta = TestLlama().prepare_model(
        use_static_quantized_kv_cache=False
    )

    if llama_model is None or llama_inputs is None:
        pytest.skip("Missing model and/or input files")

    with torch.no_grad():
        pipeline = TosaPipelineFP[input_t](
            llama_model,
            llama_inputs,
            aten_op=[],
            exir_op=[],
            custom_path="llama_tosa_fb",
            transform_passes=[InsertInt32CastsAfterInt64PlaceholdersPass()],
        )
        pipeline.run()


def test_llama_tosa_INT():
    llama_model, llama_inputs, llama_meta = TestLlama().prepare_model()

    if llama_model is None or llama_inputs is None:
        pytest.skip("Missing model and/or input files")

    with torch.no_grad():
        pipeline = TosaPipelineINT[input_t](
            llama_model,
            llama_inputs,
            aten_op=[],
            exir_op=[],
            custom_path="llama_tosa_fb_int",
            frobenius_threshold=None,
            cosine_threshold=None,
        )
        pipeline.run()


@common.SkipIfNoModelConverter
def test_llama_vgf_no_quant():
    llama_model, llama_inputs, llama_meta = TestLlama().prepare_model(
        use_static_quantized_kv_cache=False
    )

    if llama_model is None or llama_inputs is None:
        pytest.skip("Missing model and/or input files")

    with torch.no_grad():
        pipeline = VgfPipeline[input_t](
            llama_model,
            llama_inputs,
            aten_op=[],
            exir_op=[],
            transform_passes=[InsertInt32CastsAfterInt64PlaceholdersPass()],
            quantize=False,
        )
        pipeline.run()


@common.SkipIfNoModelConverter
def test_llama_vgf_quant():
    llama_model, llama_inputs, llama_meta = TestLlama().prepare_model()

    if llama_model is None or llama_inputs is None:
        pytest.skip("Missing model and/or input files")

    with torch.no_grad():
        pipeline = VgfPipeline[input_t](
            llama_model,
            llama_inputs,
            aten_op=[],
            exir_op=[],
            quantize=True,
        )
        pipeline.run()


def test_llama_tosa_INT_FP_partial_quant():
    llama_model, llama_inputs, llama_meta = TestLlama().prepare_model()

    if llama_model is None or llama_inputs is None:
        pytest.skip("Missing model and/or input files")

    with torch.no_grad():
        pipeline = TosaPipelineINT[input_t](
            llama_model,
            llama_inputs,
            aten_op=[],
            exir_op=[],
            tosa_extensions=["FP"],
            # Due to a few outliers, atol must be set high
            atol=1.1,
            qtol=1,
            frobenius_threshold=None,
            cosine_threshold=None,
        )
        _use_partial_quantizer(pipeline, eps=2**-12)
        pipeline.run()


@common.SkipIfNoModelConverter
def test_llama_vgf_quant_partial_quant():
    llama_model, llama_inputs, llama_meta = TestLlama().prepare_model()

    if llama_model is None or llama_inputs is None:
        pytest.skip("Missing model and/or input files")

    with torch.no_grad():
        pipeline = VgfPipeline[input_t](
            llama_model,
            llama_inputs,
            aten_op=[],
            exir_op=[],
            quantize=True,
            # Due to a few outliers, atol must be set high
            atol=1.1,
            qtol=1,
        )
        _use_partial_quantizer(pipeline, eps=2**-12)
        pipeline.run()
