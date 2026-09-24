# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging

import tempfile
from functools import partial
from typing import Optional

import torch
from executorch.backends.qualcomm._passes import TagQuantIO
from executorch.backends.qualcomm._passes.build_quant_io import BuildQuantIo
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_qnn_pass_manager_cls,
)
from executorch.backends.qualcomm.builders.utils import is_graph_output
from executorch.backends.qualcomm.export_utils import get_backend_type, make_quantizer
from executorch.backends.qualcomm.hf_transformers.causal_lm.decoder_model_wrapper import (
    QnnCausalLMExportableModule,
)
from executorch.backends.qualcomm.hf_transformers.causal_lm.hf_llm_quant_recipe import (
    DefaultQuantRecipe,
    Granite_3_3_2B_Instruct_HFQuantRecipe,
    Llama3_2_1B_HFQuantRecipe,
    LLMQuantRecipe,
    Qwen2_5_0_5B_HFQuantRecipe,
    Qwen2_5_1_5B_HFQuantRecipe,
    Qwen3_0_6B_HFQuantRecipe,
    Smollm2_HFQuantRecipe,
)

from executorch.backends.qualcomm.utils.constants import (
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
)
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.devtools.backend_debug import print_delegation_info
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from pytorch_tokenizers import get_tokenizer
from torch.export.exported_program import ExportedProgram
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from transformers import AutoConfig, AutoTokenizer

from transformers.exporters import ExecutorchQnnConfig, ExecutorchQnnLlmConfig

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

KV_FORWARD = "kv_forward"

HUGGING_FACE_QUANT_RECIPES = {
    "NousResearch/Llama-3.2-1B": Llama3_2_1B_HFQuantRecipe,
    "Qwen/Qwen2.5-0.5B": Qwen2_5_0_5B_HFQuantRecipe,
    "Qwen/Qwen2.5-0.5B-Instruct": Qwen2_5_0_5B_HFQuantRecipe,
    "Qwen/Qwen2.5-1.5B-Instruct": Qwen2_5_1_5B_HFQuantRecipe,
    "Qwen/Qwen3-0.6B": Qwen3_0_6B_HFQuantRecipe,
    "HuggingFaceTB/SmolLM2-135M": Smollm2_HFQuantRecipe,
    "ibm-granite/granite-3.3-2b-instruct": Granite_3_3_2B_Instruct_HFQuantRecipe,
}


def prepare_for_qnn(model: torch.nn.Module, hf_config: ExecutorchQnnConfig):
    assert isinstance(
        hf_config, ExecutorchQnnLlmConfig
    ), "HF API currently only support LLM models with ExecutorchQnnLlmConfig. Other model types are not yet supported."
    return QnnCausalLMExportableModule(model, hf_config.max_seq_len)


def quantize_for_qnn(
    exported_program: ExportedProgram,
    hf_config: ExecutorchQnnConfig,
    sample_inputs: dict,
):
    backend_type = get_backend_type(hf_config.backend_hardware)
    quantizer = make_quantizer(backend=backend_type, soc_model=hf_config.soc_model)
    quant_recipe = _get_quant_recipe(hf_config.model_id)
    quantizer.set_recipe(quant_recipe.recipe)
    quantizer.set_convert_linear_to_conv2d(True)
    graph_module = prepare_pt2e(exported_program.module(), quantizer)

    tokenizer = AutoTokenizer.from_pretrained(hf_config.model_id)
    # Generalize tokenizer so it's consistent.
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer_json_path = tokenizer.save_pretrained(tmpdir)[-1]
        tokenizer = get_tokenizer(tokenizer_json_path)

    _pt2e_calibrate(
        graph_module=graph_module,
        calibration_dataset=hf_config.calibration_dataset,
        sample_inputs=sample_inputs,
        max_seq_len=hf_config.max_seq_len,
        ar_len=1,
        tokenizer=tokenizer,
    )
    qdq_module = convert_pt2e(graph_module)

    qdq_ep = torch.export.export(
        qdq_module,
        args=(),
        kwargs=copy.deepcopy(dict(sample_inputs)),
        strict=True,
    )

    return qdq_ep


def lower_for_qnn(
    exported_program: ExportedProgram,
    hf_config: ExecutorchQnnConfig,
    sample_inputs: dict,
):
    source_model_config = AutoConfig.from_pretrained(hf_config.model_id)
    backend_options = generate_htp_compiler_spec(use_fp16=hf_config.use_fp16)
    compiler_spec = generate_qnn_executorch_compiler_spec(
        soc_model=get_soc_to_chipset_map()[hf_config.soc_model],
        backend_options=backend_options,
        use_mha2sha=True,
    )

    passes_job = get_qnn_pass_manager_cls().get_capture_program_passes()

    if not hf_config.use_fp16:
        fixed_point_type = {}
        quant_recipe = _get_quant_recipe(hf_config.model_id, verbose=False)
        kv_bits = quant_recipe.get_kv_io_bit_width()
        if kv_bits == 8:
            fixed_point_type["kv_type"] = torch.uint8
        elif kv_bits == 16:
            fixed_point_type["kv_type"] = torch.uint16
        else:
            raise RuntimeError(f"unknown kv io bit width {kv_bits}")

        logits_bits = quant_recipe.get_logits_output_bit_width()
        if logits_bits == 16:
            fixed_point_type["io_type"] = torch.uint16
        else:
            raise ValueError("Only support uint16 logits output for quantized hf llm.")

        passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
        passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
            "get_quant_io_dtype_fn"
        ] = partial(
            _tag_ios,
            fixed_point_type=fixed_point_type,
            vocab_size=source_model_config.vocab_size,
            sample_inputs=sample_inputs,
        )

    constant_methods = {
        "get_bos_id": source_model_config.bos_token_id,
        "get_eos_ids": source_model_config.eos_token_id,
        "get_vocab_size": source_model_config.vocab_size,
        "get_max_seq_len": hf_config.max_seq_len,
        "get_n_layers": source_model_config.num_hidden_layers,
        "use_kv_cache": source_model_config.use_cache,
    }

    with torch.no_grad():
        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            {KV_FORWARD: exported_program.module()},
            {KV_FORWARD: sample_inputs},
            compiler_spec,
            constant_methods=constant_methods,
            passes_job=passes_job,
            convert_linear_to_conv2d=True,
        )

    print_delegation_info(edge_prog_mgr.exported_program(KV_FORWARD).graph_module)

    executorch_config = ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(
            alloc_graph_input=hf_config.alloc_graph_input,
            alloc_graph_output=hf_config.alloc_graph_output,
            alloc_mutable_buffers=hf_config.alloc_mutable_buffers,
        ),
        passes=[BuildQuantIo()],
    )
    exec_prog_mgr = edge_prog_mgr.to_executorch(config=executorch_config)

    return exec_prog_mgr


def _pt2e_calibrate(  # noqa: C901
    graph_module: torch.fx.GraphModule,
    calibration_dataset: list[list[int]] | list[str],
    sample_inputs: dict,
    max_seq_len: int,
    ar_len: int,
    tokenizer,
):
    def calibrate_template(
        module: torch.fx.GraphModule,
        prompt: list[int] | str,
        sample_inputs: dict,
        max_seq_len: int,
        ar_len: int,
        tokenizer,
    ):

        def _build_mask(n_past, past_len, context_len):
            mask = torch.full((1, 1, 1, context_len), -65535.0)
            mask[..., :n_past] = 0.0
            mask[..., past_len:] = 0.0
            return mask

        pos = 0

        token_list = prompt
        if isinstance(prompt, str):
            token_list = tokenizer.encode(prompt, bos=True, eos=False)
        past_k, past_v = [], []
        for _ in range(len(sample_inputs["past_k"])):
            past_k.append(torch.zeros(sample_inputs["past_k"][0].shape))
            past_v.append(torch.zeros(sample_inputs["past_v"][0].shape))
        past_len = max_seq_len - ar_len
        context_len = max_seq_len
        # The prefix buffer holds at most past_len slots, so we can advance
        # the position at most past_len times (matching the runner, whose
        # seq_len is clamped to context_len).
        max_seq_len = min(max_seq_len, past_len)

        with torch.no_grad():
            while token_list[-1] != tokenizer.eos_id and pos < max_seq_len:
                n_past = min(pos, past_len)
                atten_mask = _build_mask(n_past, past_len, context_len)
                input_pos = torch.tensor([[n_past]], dtype=torch.int32)
                logits, new_k, new_v = module(
                    torch.full((1, 1), token_list[pos], dtype=torch.int32),
                    atten_mask,
                    input_pos,
                    past_k,
                    past_v,
                )
                # Prefix append: write the new slot into buffer at slot n_past.
                for layer_idx in range(len(past_k)):
                    past_k[layer_idx][..., :, n_past] = new_k[layer_idx][..., :, 0]
                    past_v[layer_idx][..., n_past, :] = new_v[layer_idx][..., 0, :]
                pos += 1
                if pos >= len(token_list):
                    token_list.append(torch.argmax(logits, dim=-1).item())
        logging.info(
            f"Result of LLM with static cache:\n {tokenizer.decode(token_list)} \n\n\n"
        )

    for prompt in calibration_dataset:
        calibrate_template(
            module=graph_module,
            prompt=prompt,
            sample_inputs=sample_inputs,
            max_seq_len=max_seq_len,
            ar_len=ar_len,
            tokenizer=tokenizer,
        )

    logging.info("Calibration finish...")


def _tag_ios(node, fixed_point_type, vocab_size, sample_inputs):
    ar_len = sample_inputs["input_tokens"].shape[1]
    max_batch_size = 1
    logit_out_shape = {
        (
            max_batch_size,
            ar_len,
            vocab_size,
        )
    }
    past_k_shape = sample_inputs["past_k"][0].shape
    past_v_shape = sample_inputs["past_v"][0].shape
    kv_cache_shape = {past_k_shape, past_v_shape}
    kv_out_shape = {
        torch.Size([*past_k_shape[:3], ar_len]),
        torch.Size([*past_v_shape[:2], ar_len, *past_v_shape[3:]]),
    }
    atten_mask_shape = sample_inputs["atten_mask"].shape
    quant_io_type = None

    if node.op == "placeholder":
        if node.meta["val"].shape in kv_cache_shape:
            quant_io_type = fixed_point_type["kv_type"]
        elif node.meta["val"].shape == atten_mask_shape:
            quant_io_type = fixed_point_type["io_type"]
    if is_graph_output(node):
        if node.meta["val"].shape in kv_out_shape:
            quant_io_type = fixed_point_type["kv_type"]
        elif node.meta["val"].shape in logit_out_shape:
            quant_io_type = fixed_point_type["io_type"]

    return quant_io_type


def _get_quant_recipe(model_id, verbose=True):
    """
    model_id = The Hugging Face Model ID.
    verbose: If True, it will the quant config tagged for each node.
    """
    recipe_cls = HUGGING_FACE_QUANT_RECIPES.get(model_id, DefaultQuantRecipe)
    if recipe_cls == DefaultQuantRecipe:
        logging.warning(
            f"{model_id} does not have customized quant recipe using default quant recipe."
        )
    quant_recipe: Optional[LLMQuantRecipe] = recipe_cls(verbose) if recipe_cls else None
    return quant_recipe
