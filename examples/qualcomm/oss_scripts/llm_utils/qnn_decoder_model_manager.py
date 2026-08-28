# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from functools import partial
from typing import Callable, List

import torch
from executorch.backends.qualcomm._passes import TagQuantIO
from executorch.backends.qualcomm._passes.build_quant_io import BuildQuantIo
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_qnn_pass_manager_cls,
)
from executorch.backends.qualcomm.builders.utils import is_graph_output
from executorch.backends.qualcomm.export_utils import make_quantizer
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
from executorch.examples.qualcomm.oss_scripts.llm_utils.llm_quant_recipe import (
    DefaultQuantRecipe,
    Granite_3_3_2B_Instruct_HFQuantRecipe,
    Llama3_2_1B_HFQuantRecipe,
    Qwen2_5_0_5B_HFQuantRecipe,
    Qwen2_5_1_5B_HFQuantRecipe,
    Qwen3_0_6B_HFQuantRecipe,
    Smollm2_HFQuantRecipe,
)
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from pytorch_tokenizers import get_tokenizer
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

# Method name the shared qnn_llama_runner expects for the KV path
# (see examples/qualcomm/oss_scripts/llama/runner/runner.cpp).
KV_FORWARD = "kv_forward"

HUGGING_FACE_REPO_IDS = {
    "llama3_2-1b": "NousResearch/Llama-3.2-1B",
    "qwen2_5-0_5b": "Qwen/Qwen2.5-0.5B",
    "qwen2_5-1_5b_instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2_5-0_5b_instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3-0_6b": "Qwen/Qwen3-0.6B",
    "smollm2_135m": "HuggingFaceTB/SmolLM2-135M",
    "granite-3_3-2b": "ibm-granite/granite-3.3-2b-instruct",
}

# TODO: This dict is temporary and will require a refactor later.
# Will create a file similar to executorch/examples/qualcomm/oss_scripts/llama/__init__.py
# and migrate model specific configs there.
HUGGING_FACE_QUANT_RECIPES = {
    "llama3_2-1b": Llama3_2_1B_HFQuantRecipe,
    "qwen2_5-0_5b": Qwen2_5_0_5B_HFQuantRecipe,
    "qwen2_5-0_5b_instruct": Qwen2_5_0_5B_HFQuantRecipe,
    "qwen2_5-1_5b_instruct": Qwen2_5_1_5B_HFQuantRecipe,
    "qwen3-0_6b": Qwen3_0_6B_HFQuantRecipe,
    "smollm2_135m": Smollm2_HFQuantRecipe,
    "granite-3_3-2b": Granite_3_3_2B_Instruct_HFQuantRecipe,
}


def get_qnn_llm_edge_manager(model_name, max_seq_len=128):
    model_id = HUGGING_FACE_REPO_IDS[model_name]
    config = AutoConfig.from_pretrained(model_id)
    device = "cpu"
    batch_size = 1
    dtype = "float32"
    cache_implementation = "static"
    attn_implementation = "eager"

    # Set configs
    config.max_seq_len = max_seq_len
    config.ar_len = 1  # kv mode
    config.max_batch_size = batch_size
    config.use_cache = True

    # Some config has head_dim provided that is different from equation below(e.g., qwen3)
    if not hasattr(config, "head_dim"):
        config.head_dim = config.hidden_size // config.num_attention_heads

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=dtype,
        config=config,
        attn_implementation=attn_implementation,
        generation_config=GenerationConfig(
            use_cache=True,
            cache_implementation=cache_implementation,
            max_length=max_seq_len,
            cache_config={
                "batch_size": batch_size,
                "max_cache_len": max_seq_len,
            },
        ),
    ).eval()
    model_wrapper = QnnCausalLMExportableModule(model)

    return QnnLLMEdgeManager(model_name, model_wrapper, config)


class QnnLLMEdgeManager:
    def __init__(self, model_name, model_wrapper, config, verbose=True) -> None:
        self.model_name = model_name
        self.model_wrapper = model_wrapper
        self.graph_module = model_wrapper
        self.config = config
        self.verbose = verbose
        self.use_fp16 = True
        self.passes_job = get_qnn_pass_manager_cls().get_capture_program_passes()
        self.edge_prog_mgr = None
        self.logits_quant_attrs = None
        recipe_cls = HUGGING_FACE_QUANT_RECIPES.get(model_name, DefaultQuantRecipe)
        if recipe_cls == DefaultQuantRecipe:
            logging.warning(
                f"{model_name} does not have customized quant recipe using default quant recipe."
            )
        self.quant_recipe = recipe_cls(verbose)

    def source_transform(
        self, transforms: List[Callable[[torch.nn.Module], torch.nn.Module]]
    ) -> "QnnLLMEdgeManager":
        """
        Apply source transforms to the model. The transforms are callables that
        takes nn.Module as input and returns nn.Module.
        Args:
            transforms (List[Callable[[torch.nn.Module], torch.nn.Module]]): A
                list of source transforms.
        """
        for transform in transforms:
            self.graph_module = transform(self.graph_module)

        if self.verbose:
            logging.info(f"Applied source transforms: {transforms}")
        logging.info(f"Model after source transforms: {self.graph_module}")
        return self

    def _tag_ios(self, node, fixed_point_type, config):
        # static_llama layout: K is transposed (seq last), V is seq-major.
        #   K in:  [B, H, head_dim, past_len]   K out: [B, H, head_dim, ar_len]
        #   V in:  [B, H, past_len, head_dim]   V out: [B, H, ar_len, head_dim]
        past_len = config.max_seq_len - config.ar_len
        kv_cache_shape = {
            # K (head_dim, seq)
            (config.head_dim, past_len),
            (config.head_dim, config.ar_len),
            # V (seq, head_dim)
            (past_len, config.head_dim),
            (config.ar_len, config.head_dim),
        }

        logit_out_shape = {
            (
                config.max_batch_size,
                config.ar_len,
                config.vocab_size,
            )
        }

        atten_mask_shape = {
            (
                config.max_batch_size,
                1,
                config.ar_len,
                config.max_seq_len,
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

    def export(self):
        with torch.no_grad():
            self.graph_module = torch.export.export(
                self.graph_module,
                args=self.model_wrapper.get_example_inputs(),
                strict=True,
            ).module()

    def pt2e_calibrate(
        self,
        calibration_tasks,
        calibration_limit,
        calibration_seq_length,
        calibration_data,
        tokenizer_path,
    ):
        assert calibration_tasks is None, "Task calibration is temporary unsupported."
        tokenizer = get_tokenizer(tokenizer_path)
        logging.info(
            f"Calibrating with tasks: {calibration_tasks}, limit: {calibration_limit}, calibration_data: {calibration_data}, tokenizer_path: {tokenizer_path}, seq_length: {self.config.max_seq_len}"
        )

        def _empty_past():
            past_k = [
                torch.zeros(
                    1,
                    self.model_wrapper.num_kv_heads,
                    self.model_wrapper.head_dim,
                    self.model_wrapper.past_len,
                )
                for _ in range(self.model_wrapper.num_layers)
            ]
            past_v = [
                torch.zeros(
                    1,
                    self.model_wrapper.num_kv_heads,
                    self.model_wrapper.past_len,
                    self.model_wrapper.head_dim,
                )
                for _ in range(self.model_wrapper.num_layers)
            ]
            return past_k, past_v

        def _build_mask(n_past, past_len, context_len):
            mask = torch.full((1, 1, 1, context_len), -65535.0)
            mask[..., :n_past] = 0.0
            mask[..., past_len:] = 0.0
            return mask

        def calibrate_template(
            module: torch.fx.GraphModule, tokenizer, prompts: str, max_len: int
        ):
            pos = 0
            token_list = tokenizer.encode(prompts, bos=True, eos=False)
            past_k, past_v = _empty_past()
            past_len = self.model_wrapper.past_len
            context_len = self.model_wrapper.max_seq_len
            # The prefix buffer holds at most past_len slots, so we can advance
            # the position at most past_len times (matching the runner, whose
            # seq_len is clamped to context_len).
            max_len = min(max_len, past_len)

            with torch.no_grad():
                while token_list[-1] != tokenizer.eos_id and pos < max_len:
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
                    for layer in range(self.model_wrapper.num_layers):
                        past_k[layer][..., :, n_past] = new_k[layer][..., :, 0]
                        past_v[layer][..., n_past, :] = new_v[layer][..., 0, :]
                    pos += 1
                    if pos >= len(token_list):
                        token_list.append(torch.argmax(logits, dim=-1).item())
            logging.info(
                f"Result of LLM with static cache:\n {tokenizer.decode(token_list)} \n\n\n"
            )

        calibrate_template(
            module=self.graph_module,
            tokenizer=tokenizer,
            prompts=calibration_data,
            max_len=calibration_seq_length,
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
                model=self.graph_module,
                tokenizer=tokenizer,
                max_seq_length=calibration_seq_length,
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

    def pt2e_quantize(
        self,
        fixed_point_type,
        calibration_tasks,
        calibration_limit,
        calibration_data,
        tokenizer_path,
        backend,
        soc_model,
    ):
        self.export()

        quantizer = make_quantizer(backend=backend, soc_model=soc_model)
        quantizer.set_recipe(self.quant_recipe.recipe)
        quantizer.set_convert_linear_to_conv2d(True)

        self.graph_module = prepare_pt2e(self.graph_module, quantizer)
        self.pt2e_calibrate(
            calibration_tasks,
            calibration_limit,
            self.config.max_seq_len,
            calibration_data,
            tokenizer_path,
        )
        self.graph_module = convert_pt2e(self.graph_module)

        self.passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
        self.passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
            "get_quant_io_dtype_fn"
        ] = partial(
            self._tag_ios, fixed_point_type=fixed_point_type, config=self.config
        )
        self.use_fp16 = False

    def to_edge_transform_and_lower_to_qnn(
        self, soc_model, skip_node_id_set, skip_node_op_set
    ):
        backend_options = generate_htp_compiler_spec(use_fp16=self.use_fp16)
        compiler_spec = generate_qnn_executorch_compiler_spec(
            soc_model=get_soc_to_chipset_map()[soc_model],
            backend_options=backend_options,
            use_mha2sha=True,
        )
        with torch.no_grad():
            self.edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                {KV_FORWARD: self.graph_module},
                {KV_FORWARD: self.model_wrapper.get_example_inputs()},
                compiler_spec,
                constant_methods=self.model_wrapper.get_metadata(),
                passes_job=self.passes_job,
                skip_node_id_set=skip_node_id_set,
                skip_node_op_set=skip_node_op_set,
                convert_linear_to_conv2d=True,
            )

        print_delegation_info(
            self.edge_prog_mgr.exported_program(KV_FORWARD).graph_module
        )
        if not self.use_fp16:
            logit_out_shape = {
                (
                    self.config.max_batch_size,
                    self.config.ar_len,
                    self.config.vocab_size,
                )
            }
            for n in self.edge_prog_mgr.exported_program(KV_FORWARD).graph.nodes:
                if n.op == "output":
                    for node, output_encoding in n.meta[QCOM_QUANT_ATTRS_MAP].items():
                        if node.meta["val"].size() in logit_out_shape:
                            self.logits_quant_attrs = output_encoding

    def get_logits_quant_attrs(self):
        return self.logits_quant_attrs

    def to_executorch(self, artifact, pte_filename):
        executorch_config = ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                alloc_graph_output=False,
            ),
            passes=[BuildQuantIo()],
        )
        exec_prog_mgr = self.edge_prog_mgr.to_executorch(config=executorch_config)
        with open(f"{artifact}/{pte_filename}.pte", "wb") as file:
            exec_prog_mgr.write_to_file(file)
        logging.info(f"Saved exported program to {artifact}/{pte_filename}.pte")
