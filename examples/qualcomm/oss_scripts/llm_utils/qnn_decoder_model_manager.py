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
    get_capture_program_passes,
)
from executorch.backends.qualcomm.builders.utils import is_graph_output
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
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
from executorch.examples.qualcomm.utils import make_quantizer
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from pytorch_tokenizers import get_tokenizer
from torchao.quantization.pt2e import MinMaxObserver
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

HUGGING_FACE_REPO_IDS = {
    "qwen2.5_0.5B": "Qwen/Qwen2.5-0.5B",
    "qwen2.5_1.5B_instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5_0.5B_instruct": "Qwen/Qwen2.5-0.5B-Instruct",
}


def get_qnn_llm_edge_manager(model_name, max_seq_len=128, enable_spinquant_r3=True):
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
    config.enable_spinquant_r3 = enable_spinquant_r3

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
        self.passes_job = get_capture_program_passes()
        self.edge_prog_mgr = None
        self.logits_quant_attrs = None

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
        # shape of k caches and v caches
        kv_cache_shape = {
            # single head, kv input
            (config.head_dim, config.max_seq_len),
            (config.max_seq_len, config.head_dim),
            # single head, kv output
            (config.head_dim, config.ar_len),
            (config.ar_len, config.head_dim),
        }

        logit_out_shape = {
            (
                config.max_batch_size,
                config.ar_len,
                config.vocab_size,
            )
        }

        quant_io_type = None

        if node.op == "placeholder":
            if (
                len(users := list(node.users)) == 1
                and users[0].meta["val"].size()[-2:] in kv_cache_shape
            ):
                quant_io_type = fixed_point_type["kv_type"]
        if is_graph_output(node):
            if node.meta["val"].size()[-2:] in kv_cache_shape:
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
        try:
            from executorch.examples.qualcomm.oss_scripts.llm_utils.eval_decoder_model_qnn import (
                GraphModuleCalibrationWrapper,
            )
            from lm_eval.evaluator import simple_evaluate
        except ImportError:
            raise ImportError(
                "Please install the llm eval dependency via examples/models/llama/install_requirements.sh"
            )

        tokenizer = get_tokenizer(tokenizer_path)
        logging.info(
            f"Calibrating with tasks: {calibration_tasks}, limit: {calibration_limit}, calibration_data: {calibration_data}, tokenizer_path: {tokenizer_path}, seq_length: {self.config.max_seq_len}"
        )

        def calibrate_template(
            module: torch.fx.GraphModule, tokenizer, prompts: str, max_len: int
        ):
            # TODO: change criteria & support batch inputs if necessary
            pos = 0
            token_list = tokenizer.encode(prompts, bos=True, eos=False)

            with torch.no_grad():
                while token_list[-1] != tokenizer.eos_id and pos < max_len:
                    cur_pos = torch.tensor([pos], dtype=torch.long)
                    logits = module(torch.full((1, 1), token_list[pos]), cur_pos)
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
        quant_dtype,
        fixed_point_type,
        calibration_tasks,
        calibration_limit,
        calibration_data,
        tokenizer_path,
    ):
        self.export()

        quantizer = make_quantizer(
            quant_dtype=quant_dtype,
            per_channel_linear=True,
            per_channel_conv=True,
            act_observer=MinMaxObserver,
        )
        if quant_dtype == QuantDtype.use_16a4w_block:

            def extract_linear_nodes(graph):
                linear_nodes = []
                for node in graph.nodes:
                    if node.target == torch.ops.aten.linear.default:
                        linear_nodes.append(node)  # linear node
                        linear_nodes.append(node.args[1])  # weight node
                return linear_nodes

            linear_nodes = extract_linear_nodes(self.graph_module.graph)
            block_size_map = {n.name: (1, 16) for n in linear_nodes}
            quantizer.set_block_size_map(block_size_map)
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
        )
        with torch.no_grad():
            self.edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                self.graph_module,
                self.model_wrapper.get_example_inputs(),
                compiler_spec,
                constant_methods=self.model_wrapper.get_metadata(),
                passes_job=self.passes_job,
                skip_node_id_set=skip_node_id_set,
                skip_node_op_set=skip_node_op_set,
                convert_linear_to_conv2d=True,
            )

        print_delegation_info(self.edge_prog_mgr.exported_program().graph_module)
        if not self.use_fp16:
            logit_out_shape = {
                (
                    self.config.max_batch_size,
                    self.config.ar_len,
                    self.config.vocab_size,
                )
            }
            for n in self.edge_prog_mgr.exported_program().graph.nodes:
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
            ),
            passes=[BuildQuantIo()],
        )
        exec_prog_mgr = self.edge_prog_mgr.to_executorch(config=executorch_config)
        with open(f"{artifact}/{pte_filename}.pte", "wb") as file:
            exec_prog_mgr.write_to_file(file)
        logging.info(f"Saved exported program to {artifact}/{pte_filename}.pte")
