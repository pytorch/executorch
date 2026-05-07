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

from executorch.extension.llm.export.config.llm_config import LlmConfig

from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM
from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM

input_t = Tuple[torch.Tensor]
input_th = Tuple[torch.Tensor, torch.Tensor]

# Add project dir to sys path to workaround importlib.import_module() conditions in model_factory.py
this_files_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(this_files_dir, "../../../.."))
sys.path.append(project_dir)

logger = logging.getLogger(__name__)


class HFPositionalAdapter(torch.nn.Module):
    def __init__(self, exportable):
        super().__init__()
        self.inner = exportable

    def forward(self, input_ids, cache_position):
        # HF StaticCache eager path requires int64 index tensors, but keeping
        # cache_position as int32 during export capture avoids adding an extra
        # int64->int32 cast node in the lowered graph.
        if torch._dynamo.is_compiling():
            cp = cache_position
        else:
            cp = cache_position.to(torch.long)
        return self.inner(input_ids=input_ids, cache_position=cp)


class TestLlama:
    """Test class of Llama models.

    Type of Llama model depends on command line parameters:
    --llama_inputs <path to .pt file> <path to json file> <name of model variant>
    Example: --llama_inputs stories110M/stories110M.pt stories110M/params.json stories110m
    For more examples and info see examples/models/llama/README.md.

    """

    def prepare_model_hf_static(self):
        """
        Build a tiny HF LLaMA wrapped with TorchExportableModuleForDecoderOnlyLM (StaticCache)
        See https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/executorch.py#L214C17-L214C53
        """
        # Tiny config
        cfg = LlamaConfig(
            vocab_size=32000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            use_cache=True,
        )
        base = LlamaForCausalLM(cfg).eval()

        # REQUIRED: generation_config must request a 'static' cache with batch_size & max_cache_len
        base.generation_config = GenerationConfig(
            use_cache=True,
            cache_implementation="static",
            cache_config={"batch_size": 1, "max_cache_len": 128},
        )

        exportable = TorchExportableModuleForDecoderOnlyLM(
            model=base, batch_size=1, max_cache_len=128
        )

        # Positional adapter so the pipeline can call module(*inputs)
        model_for_pipeline = HFPositionalAdapter(exportable).eval()

        # The tester will call model(*inputs). Provide (input_ids, cache_position)
        input_ids = torch.tensor([[0]], dtype=torch.long)  # shape [1, 1]
        cache_position = torch.tensor([0], dtype=torch.int32)  # shape [1]
        inputs = (input_ids, cache_position)

        return model_for_pipeline, inputs, None

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
        ]

        parser = build_args_parser()
        args = parser.parse_args(args)
        llm_config = LlmConfig.from_args(args)

        llama_model, llama_inputs, llama_meta = get_llama_model(llm_config)

        return llama_model, llama_inputs, llama_meta


def _use_partial_quantizer(pipeline, eps=2**-16):
    """Set the pipeline's quantizer to only include Linear layers."""
    pipeline.quantizer.set_global(None)
    pipeline.quantizer.set_module_type(
        torch.nn.Linear, get_symmetric_quantization_config(eps=eps)
    )


def test_llama_tosa_FP():
    llama_model, llama_inputs, llama_meta = TestLlama().prepare_model()

    if llama_model is None or llama_inputs is None:
        pytest.skip("Missing model and/or input files")

    with torch.no_grad():
        pipeline = TosaPipelineFP[input_t](
            llama_model,
            llama_inputs,
            aten_op=[],
            exir_op=[],
            custom_path="llama_tosa_fb",
            run_on_tosa_ref_model=True,  # Just want to write TOSA FB to disk
            use_to_edge_transform_and_lower=True,
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
            run_on_tosa_ref_model=True,  # Just want to write TOSA FB to disk
            use_to_edge_transform_and_lower=True,
            frobenius_threshold=None,
            cosine_threshold=None,
        )
        pipeline.run()


def test_llama_tosa_INT_static():
    llama_model, llama_inputs, _ = TestLlama().prepare_model_hf_static()
    if llama_model is None or llama_inputs is None:
        pytest.skip("Missing model and/or input files")

    with torch.no_grad():
        pipeline = TosaPipelineINT[input_th](
            llama_model,
            llama_inputs,
            aten_op=[],
            exir_op=[],
            custom_path="llama_tosa_hf_static_int",
            run_on_tosa_ref_model=True,
            use_to_edge_transform_and_lower=True,
            fold_quantize=False,
        )
        # NOTE: HF StaticCache INT currently keeps two delegated subgraphs
        # after partitioning on this path, so expect two delegate calls in EXIR.
        pipeline.change_args(
            "check_count.exir",
            {"torch.ops.higher_order.executorch_call_delegate": 2},
        )
        pipeline.run()


@common.SkipIfNoModelConverter
def test_llama_vgf_no_quant():
    llama_model, llama_inputs, llama_meta = TestLlama().prepare_model()

    if llama_model is None or llama_inputs is None:
        pytest.skip("Missing model and/or input files")

    with torch.no_grad():
        pipeline = VgfPipeline[input_t](
            llama_model,
            llama_inputs,
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            transform_passes=[InsertInt32CastsAfterInt64PlaceholdersPass()],
            run_on_vulkan_runtime=True,
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
            use_to_edge_transform_and_lower=True,
            run_on_vulkan_runtime=True,
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
