# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This is a PoC, so we just place the api under examples folder for now.

import copy

import json

import logging
from functools import partial
from typing import Optional

import torch
from executorch.backends.qualcomm._passes import TagQuantIO
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_qnn_pass_manager_cls,
)
from executorch.backends.qualcomm.builders.utils import is_graph_output
from executorch.backends.qualcomm.export_utils import get_backend_type, make_quantizer

from executorch.backends.qualcomm.utils.constants import (
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
    QCOM_QUANT_ATTRS_MAP,
)
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.devtools.backend_debug import print_delegation_info
from executorch.examples.qualcomm.oss_scripts.llm_utils.decoder_model_wrapper import (
    QnnCausalLMExportableModule,
)
from executorch.examples.qualcomm.oss_scripts.llm_utils.hf_llm_quant_recipe import (
    DefaultQuantRecipe,
    Granite_3_3_2B_Instruct_HFQuantRecipe,
    HFLLMQuantRecipe,
    Llama3_2_1B_HFQuantRecipe,
    Qwen2_5_0_5B_HFQuantRecipe,
    Qwen2_5_1_5B_HFQuantRecipe,
    Qwen3_0_6B_HFQuantRecipe,
    Smollm2_HFQuantRecipe,
)
from pytorch_tokenizers import get_tokenizer
from torch.export.exported_program import ExportedProgram
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from transformers import AutoTokenizer

# This import is only available once the patch is applied on hugging face.
from transformers.exporters import ExecutorchQnnLlmConfig

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


def get_qnn_causal_llm_exportable_module(
    model: torch.nn.Module, llm_config: ExecutorchQnnLlmConfig
):
    return QnnCausalLMExportableModule(model, llm_config)


def qnn_llm_quantize_pt2e(
    exported_program: ExportedProgram,
    llm_config: ExecutorchQnnLlmConfig,
    sample_inputs: dict,
):
    assert (
        llm_config.calibration_tasks is None
    ), "Task calibration in not yet supported."

    backend_type = get_backend_type(llm_config.backend_hardware)
    quantizer = make_quantizer(backend=backend_type, soc_model=llm_config.soc_model)
    quant_recipe = _get_quant_recipe(llm_config.model_id)
    quantizer.set_recipe(quant_recipe.recipe)
    quantizer.set_convert_linear_to_conv2d(True)
    graph_module = prepare_pt2e(exported_program.module(), quantizer)

    # For model like UTs where it has no model_id, ensure that the dataset passed in is already tokenized(int).
    # We can also change this behaviour in future and allow llm_config to store some dummy tokenizer_path for UT or models with no model_id
    if llm_config.model_id is not None:
        tokenizer = AutoTokenizer.from_pretrained(llm_config.model_id)
        tokenizer_json_path = tokenizer.save_pretrained(llm_config.artifact_dir)[-1]
        # Generalize tokenizer so it's consistent.
        tokenizer = get_tokenizer(tokenizer_json_path)
    else:
        tokenizer = None
        assert all(
            isinstance(x, int)
            for sublist in llm_config.calibration_dataset
            for x in sublist
        ), "Current flow with no model_id assumes no tokenizer, please pass in tokenized(integer) llm_config.calibration_dataset instead of string."

    _pt2e_calibrate(
        graph_module=graph_module,
        calibration_dataset=llm_config.calibration_dataset,
        calibration_tasks=llm_config.calibration_tasks,
        calibration_limit=llm_config.calibration_limit,
        sample_inputs=sample_inputs,
        max_seq_len=llm_config.max_seq_len,
        ar_len=llm_config.ar_len,
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


def qnn_llm_to_edge_transform_and_lower(
    exported_program: ExportedProgram,
    llm_config: ExecutorchQnnLlmConfig,
    sample_inputs: dict,
):
    backend_options = generate_htp_compiler_spec(use_fp16=llm_config.use_fp16)
    compiler_spec = generate_qnn_executorch_compiler_spec(
        soc_model=get_soc_to_chipset_map()[llm_config.soc_model],
        backend_options=backend_options,
        use_mha2sha=True,
    )

    passes_job = get_qnn_pass_manager_cls().get_capture_program_passes()

    if not llm_config.use_fp16:
        fixed_point_type = {}
        quant_recipe = _get_quant_recipe(llm_config.model_id, verbose=False)
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
            llm_config=llm_config,
        )

    config = llm_config.source_model_config
    constant_methods = {
        "get_bos_id": config.bos_token_id,
        "get_eos_ids": config.eos_token_id,
        "get_vocab_size": config.vocab_size,
        "get_max_seq_len": llm_config.max_seq_len,
        "get_n_layers": config.num_hidden_layers,
        "use_kv_cache": config.use_cache,
    }

    with torch.no_grad():
        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            {KV_FORWARD: exported_program.module()},
            {KV_FORWARD: sample_inputs},
            compiler_spec,
            constant_methods=constant_methods,
            passes_job=passes_job,
            skip_node_id_set=llm_config.skip_delegate_node_ids,
            skip_node_op_set=llm_config.skip_delegate_node_ops,
            convert_linear_to_conv2d=True,
        )

    print_delegation_info(edge_prog_mgr.exported_program(KV_FORWARD).graph_module)

    if not llm_config.use_fp16:
        logit_out_shape = {
            (
                llm_config.max_batch_size,
                llm_config.ar_len,
                llm_config.source_model_config.vocab_size,
            )
        }
        for n in edge_prog_mgr.exported_program(KV_FORWARD).graph.nodes:
            if n.op == "output":
                for node, output_encoding in n.meta[QCOM_QUANT_ATTRS_MAP].items():
                    if node.meta["val"].size() in logit_out_shape:
                        logits_quant_attrs = output_encoding
                        json.dump(
                            {
                                "scale": logits_quant_attrs["scale"],
                                "zero_point": logits_quant_attrs["zero_point"],
                            },
                            open(
                                f"{llm_config.artifact_dir}/logit_quant_attrs.txt", "w"
                            ),
                        )
    return edge_prog_mgr


def _pt2e_calibrate(  # noqa: C901
    graph_module: torch.fx.GraphModule,
    calibration_dataset: list[list[int]] | list[str],
    calibration_tasks: str,
    calibration_limit: int,
    sample_inputs: dict,
    max_seq_len: int,
    ar_len: int,
    tokenizer=None,
):
    def calibrate_template(
        module: torch.fx.GraphModule,
        prompt: list[int] | str,
        sample_inputs: dict,
        max_seq_len: int,
        ar_len: int,
        tokenizer=None,
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

    logging.info(
        f"Calibrating with tasks: {calibration_tasks}, limit: {calibration_limit}, calibration_data: {calibration_dataset}"
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
    if calibration_tasks is not None and calibration_limit is not None:
        # Import lazily so only import lm_eval when user use it.
        try:
            from executorch.examples.qualcomm.oss_scripts.llm_utils.eval_decoder_model_qnn import (
                GraphModuleCalibrationWrapper,
            )
            from lm_eval.evaluator import simple_evaluate
        except ImportError:
            raise ImportError(
                "Please install the llm eval dependency via examples/models/llama/install_requirements.sh"
            )

        eval_wrapper = GraphModuleCalibrationWrapper(
            model=graph_module,
            tokenizer=tokenizer,
            max_seq_length=max_seq_len,
            use_kv_cache=True,
            generate_full_logits=True,
            enable_dynamic_shape=False,
        )

        # Evaluate the model
        with torch.no_grad():
            eval_results = simple_evaluate(
                model=eval_wrapper,
                tasks=calibration_tasks,
                limit=calibration_limit,
            )

        for task, res in eval_results["results"].items():
            print(f"{task}: {res}")
    logging.info("Calibration finish...")


def _tag_ios(node, fixed_point_type, llm_config):
    model_config = llm_config.source_model_config
    if not hasattr(model_config, "head_dim"):
        model_config.head_dim = (
            model_config.hidden_size // model_config.num_attention_heads
        )

    # static_llama layout: K is transposed (seq last), V is seq-major.
    #   K in:  [B, H, head_dim, past_len]   K out: [B, H, head_dim, ar_len]
    #   V in:  [B, H, past_len, head_dim]   V out: [B, H, ar_len, head_dim]
    past_len = llm_config.max_seq_len - llm_config.ar_len
    kv_cache_shape = {
        # K (head_dim, seq)
        (model_config.head_dim, past_len),
        (model_config.head_dim, llm_config.ar_len),
        # V (seq, head_dim)
        (past_len, model_config.head_dim),
        (llm_config.ar_len, model_config.head_dim),
    }

    logit_out_shape = {
        (
            llm_config.max_batch_size,
            llm_config.ar_len,
            model_config.vocab_size,
        )
    }

    atten_mask_shape = {
        (
            llm_config.max_batch_size,
            1,
            llm_config.ar_len,
            llm_config.max_seq_len,
        )
    }

    quant_io_type = None

    if node.op == "placeholder":
        if (
            node.meta["val"].dim() == 4
            and node.meta["val"].size()[-2:] in kv_cache_shape
        ):
            quant_io_type = fixed_point_type["kv_type"]
        elif node.meta["val"].size() in atten_mask_shape:
            quant_io_type = fixed_point_type["io_type"]
    if is_graph_output(node):
        if (
            node.meta["val"].dim() == 4
            and node.meta["val"].size()[-2:] in kv_cache_shape
        ):
            quant_io_type = fixed_point_type["kv_type"]
        elif node.meta["val"].size() in logit_out_shape:
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
    quant_recipe: Optional[HFLLMQuantRecipe] = (
        recipe_cls(verbose) if recipe_cls else None
    )
    return quant_recipe
