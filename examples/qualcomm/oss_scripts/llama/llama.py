# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: reenable pyre after fixing the issues
# pyre-ignore-all-errors

import copy
import getpass
import json
import logging
import math
import os
import subprocess
import sys
import time
import types
from functools import partial
from multiprocessing.connection import Client

import torch
from executorch.backends.qualcomm._passes import FoldQDQ, TagQuantIO
from executorch.backends.qualcomm._passes.i64_to_i32 import I64toI32
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_capture_program_passes,
)
from executorch.backends.qualcomm._passes.utils import (
    get_passes_dependency_for_capture_program,
)

from executorch.backends.qualcomm.builders.utils import is_graph_output
from executorch.backends.qualcomm.quantizer.custom_annotation import (
    annotate_prefill_kv_output,
)

from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset

from executorch.backends.qualcomm.utils.constants import (
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
)
from executorch.backends.qualcomm.utils.utils import (
    convert_linear_to_conv2d,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_sdk_build_id,
    get_soc_to_chipset_map,
    is_qnn_sdk_version_less_than,
    to_edge_transform_and_lower_to_qnn,
    update_spill_fill_size,
)

from executorch.devtools.backend_debug import print_delegation_info

from executorch.examples.models.llama.hf_download import (
    download_and_convert_hf_checkpoint,
)
from executorch.examples.models.llama.source_transformation.quantize import (
    get_quant_embedding_transform,
)
from executorch.examples.qualcomm.oss_scripts.llama import (
    LLMModelConfig,
    SUPPORTED_LLM_MODELS,
)
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import EVAL_MODE
from executorch.examples.qualcomm.oss_scripts.llama.decoder_utils import (
    apply_prompt_template,
    graph_module_inference,
    QnnRunnerEvalWrapper,
    shift_pointer_updater,
    smart_mask_updater,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    LlamaModel,
    ModelArgs,
)
from executorch.examples.qualcomm.oss_scripts.llama.range_setting_pt2e import (
    compute_scales,
    make_custom_quantizer,
    reverse_quantize_module_swap,
    set_scales,
    WrappedLlamaModel,
)

from executorch.examples.qualcomm.utils import (
    make_output_dir,
    setup_common_args_and_variables,
    SimpleADB,
)
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.extension.llm.custom_ops import model_sharding
from executorch.extension.llm.export.builder import DType
from pytorch_tokenizers import get_tokenizer, TiktokenTokenizer
from pytorch_tokenizers.llama2c import Llama2cTokenizer as SentencePieceTokenizer

from torchao.prototype.spinquant import apply_spinquant

from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from transformers import AutoTokenizer

try:
    from lm_eval.evaluator import simple_evaluate
except ImportError:
    raise ImportError(
        "Please install the llm eval dependency via examples/models/llama/install_requirements.sh"
    )

sys.setrecursionlimit(4096)
FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.getLogger().setLevel(logging.INFO)
# Avoid the error message "Could not initialize NNPACK! Reason: Unsupported hardware."
torch.backends.nnpack.set_flags(False)


def next_power_of_two(n):
    if n == 0:
        return 1
    return 2 ** math.ceil(math.log2(n))


class SingleLlama:
    def __init__(
        self, decoder_model, decoder_model_config: LLMModelConfig, pte_filename
    ) -> None:
        super().__init__()
        self.decoder_model = decoder_model
        self.decoder_model_config = decoder_model_config
        self.passes_job = get_capture_program_passes()
        self.dep_table = get_passes_dependency_for_capture_program()
        self.quant_dtype = None
        self.llama_meta = self.decoder_model.get_metadata()
        self.has_quant_io = False
        self.pte_filename = pte_filename
        if self.llama_meta["get_use_kv_cache"]:
            tokens, atten_mask, pos_ids, k_caches, v_caches = self.get_example_inputs(
                use_kv_cache=True
            )
            self.inputs = (tokens, atten_mask, pos_ids, *k_caches, *v_caches)
        else:
            tokens, atten_mask = self.get_example_inputs(use_kv_cache=False)
            self.inputs = (tokens, atten_mask)
        self.llama_graph_module = decoder_model
        self.io_shape = {
            # logit output
            (
                self.llama_meta["get_max_batch_size"],
                self.llama_meta["get_ar_len"],
                self.llama_meta["get_vocab_size"],
            ),
        }

    def _tag_ios(self, node, fixed_point_type):
        if not self.has_quant_io:
            return

        # shape of k caches and v caches
        kv_cache_shape = {
            # single head, kv input
            (self.llama_meta["get_head_dim"], self.llama_meta["get_max_seq_len"]),
            (self.llama_meta["get_max_seq_len"], self.llama_meta["get_head_dim"]),
            # single head, kv output
            (self.llama_meta["get_head_dim"], self.llama_meta["get_ar_len"]),
            (self.llama_meta["get_ar_len"], self.llama_meta["get_head_dim"]),
        }

        atten_mask_shape = {
            (
                self.llama_meta["get_max_batch_size"],
                self.llama_meta["get_ar_len"],
                self.llama_meta["get_max_seq_len"],
            ),
        }

        freq_shape = {
            (self.llama_meta["get_ar_len"], self.llama_meta["get_head_dim"] // 2),
        }

        freq_op = {
            exir_ops.edge.aten.select.int,
        }
        quant_io_type = None

        if node.op == "placeholder":
            if (
                len(users := list(node.users)) == 1
                and users[0].meta["val"].size()[-2:] in kv_cache_shape
            ):
                quant_io_type = fixed_point_type["kv_type"]
            elif node.meta["val"].size() in self.io_shape:
                quant_io_type = fixed_point_type["io_type"]
            elif node.meta["val"].size() in atten_mask_shape:
                quant_io_type = fixed_point_type["io_type"]
        if is_graph_output(node):
            if node.meta["val"].size()[-2:] in kv_cache_shape:
                quant_io_type = fixed_point_type["kv_type"]
            elif node.meta["val"].size() in self.io_shape:
                quant_io_type = fixed_point_type["io_type"]

        # Tag sharding io
        if exir_ops.edge.llama.fallback.default in [
            u.target for u in list(node.users.keys())
        ] + [node.target]:
            quant_io_type = fixed_point_type["io_type"]

        # Tag select op as quantized tensors for freq_sin and freq_cos. It is caused by sharding
        if node.target in freq_op and node.meta["val"].size() in freq_shape:
            quant_io_type = fixed_point_type["io_type"]

        return quant_io_type

    def quantize(
        self,
        quant_dtype,
        args,
        tokenizer,
        custom_annotations=(),
        scales_state_dict=None,
        chat_template=None,
    ):
        self.quant_dtype = quant_dtype
        quantizer = make_custom_quantizer(
            quant_dtype, args.range_setting, custom_annotations
        )

        self.has_quant_io = True
        fx_graph_module = None
        with torch.no_grad():
            fx_graph_module = torch.export.export(
                self.llama_graph_module, self.inputs, strict=True
            ).module()

            if quant_dtype == QuantDtype.use_16a4w_block:
                if self.decoder_model_config.group_size is None:
                    raise ValueError(
                        "Group size is required when use quant_dtype 16a4w_block"
                    )
                conv_nodes = [
                    n for n in fx_graph_module.graph.nodes if "conv" in n.name
                ]
                block_size_map = {
                    n.name: (1, self.decoder_model_config.group_size, 1, 1)
                    for n in conv_nodes
                }
                quantizer.set_block_size_map(block_size_map)

            fx_graph_module = prepare_pt2e(fx_graph_module, quantizer)

        logging.info("Quantizing the model...")

        # Calibration
        if args.tasks is not None:
            graph_module_inference(
                use_kv_cache=self.llama_meta["get_use_kv_cache"],
                get_example_inputs=self.get_example_inputs,
                module=fx_graph_module,
                tokenizer=tokenizer,
                ar_len=self.llama_meta["get_ar_len"],
                max_seq_len=self.llama_meta["get_max_seq_len"],
                kv_updater=args.kv_updater,
                tasks=args.tasks,
                tasks_limit=args.limit,
                num_fewshot=args.num_fewshot,
                use_i64_token=args.embedding_quantize is not None,
                event_name="prepare_pt2e_tasks",
            )

        # Check user's prompt, helps calibrate special token
        prompt = (
            args.prompt[0]
            if chat_template is None
            else apply_prompt_template(
                chat_template, args.prompt[0], args.system_prompt
            )
        )
        graph_module_inference(
            use_kv_cache=self.llama_meta["get_use_kv_cache"],
            get_example_inputs=self.get_example_inputs,
            module=fx_graph_module,
            tokenizer=tokenizer,
            ar_len=self.llama_meta["get_ar_len"],
            max_seq_len=self.llama_meta["get_max_seq_len"],
            kv_updater=args.kv_updater,
            prompt=prompt,
            use_i64_token=args.embedding_quantize is not None,
            event_name="prepare_pt2e_prompt",
        )

        if scales_state_dict:
            set_scales(
                fx_graph_module, scales_state_dict, self.llama_graph_module.head_dim
            )

        self.llama_graph_module = convert_pt2e(fx_graph_module)

        if args.verbose:
            logging.info("Verifying the QDQ model...")
            # qdq cpu ppl evaluation is time consuming, only enable when eval_perplexity
            if args.eval_perplexity:
                # Check qdq cpu results
                graph_module_inference(
                    use_kv_cache=self.llama_meta["get_use_kv_cache"],
                    get_example_inputs=self.get_example_inputs,
                    module=self.llama_graph_module,
                    tokenizer=tokenizer,
                    ar_len=self.llama_meta["get_ar_len"],
                    max_seq_len=self.llama_meta["get_max_seq_len"],
                    kv_updater=args.kv_updater,
                    tasks=args.tasks,
                    tasks_limit=args.limit,
                    num_fewshot=args.num_fewshot,
                    use_i64_token=args.embedding_quantize is not None,
                    event_name="convert_pt2e_tasks",
                )
            # Check user's prompt
            prompt = (
                args.prompt[0]
                if chat_template is None
                else apply_prompt_template(
                    chat_template, args.prompt[0], args.system_prompt
                )
            )
            graph_module_inference(
                use_kv_cache=self.llama_meta["get_use_kv_cache"],
                get_example_inputs=self.get_example_inputs,
                module=self.llama_graph_module,
                tokenizer=tokenizer,
                ar_len=self.llama_meta["get_ar_len"],
                max_seq_len=self.llama_meta["get_max_seq_len"],
                kv_updater=args.kv_updater,
                prompt=prompt,
                use_i64_token=args.embedding_quantize is not None,
                event_name="convert_pt2e_prompt",
            )

    def save_logits_quant_attrs(self):
        for node in self.llama_graph_module.graph.nodes:
            if node.op == "output":
                for output_node in node.args[0]:
                    if (
                        output_node.target
                        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                    ):
                        source_node = output_node.args[0].args[0]
                        if source_node.meta["val"].size() in self.io_shape:
                            self.llama_meta["get_logits_scale"] = output_node.args[1]
                            self.llama_meta["get_logits_zero_point"] = output_node.args[
                                2
                            ]
                            break

    def lowering_modules(
        self,
        work_space,
        use_fp16=False,
        soc_model=QcomChipset.SM8650,
        shared_buffer=False,
        verbose=False,
    ):
        executorch_config = ExecutorchBackendConfig(
            # For shared buffer, user must pass the memory address
            # which is allocated by RPC memory to executor runner.
            # Therefore, won't want to pre-allocate
            # by memory manager in runtime.
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                alloc_graph_output=False,
            ),
            extract_delegate_segments=True,
        )
        with torch.no_grad():
            # backend option
            backend_options = generate_htp_compiler_spec(
                use_fp16=use_fp16,
                use_multi_contexts=self.decoder_model_config.num_sharding > 1,
            )
            compiler_specs = generate_qnn_executorch_compiler_spec(
                soc_model=soc_model,
                backend_options=backend_options,
                shared_buffer=shared_buffer,
            )
            skip_node_op_set = {"llama.fallback.default"}

            self.save_logits_quant_attrs()
            edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                self.llama_graph_module,
                self.inputs,
                compiler_specs,
                constant_methods=self.llama_meta,
                dep_table=self.dep_table,
                passes_job=self.passes_job,
                skip_node_op_set=skip_node_op_set,
            )

            if self.decoder_model_config.num_sharding > 1:
                update_spill_fill_size(edge_prog_mgr.exported_program())

            if verbose:
                print_delegation_info(edge_prog_mgr.exported_program().graph_module)

            exec_prog_mgr = edge_prog_mgr.to_executorch(config=executorch_config)
            with open(f"{work_space}/{self.pte_filename}.pte", "wb") as file:
                exec_prog_mgr.write_to_file(file)

    def get_example_inputs(self, use_kv_cache=True):
        return self.decoder_model.get_example_inputs(use_kv_cache)


def compile(
    args,
    decoder_model_config: LLMModelConfig,
    pte_filename: str,
    tokenizer,
    chat_template,
):
    os.makedirs(args.artifact, exist_ok=True)
    start_ts = time.time()

    kv_config, prefill_config = None, None
    if args.params:
        params_path = args.params
    else:
        params_path = decoder_model_config.params_path
    with open(params_path) as f:
        kv_config = ModelArgs(**json.load(f))

    # TODO: support batch inputs if necessary
    kv_config.max_batch_size = 1
    kv_config.max_seq_len = args.max_seq_len
    kv_config.use_kv_cache = True
    kv_config.enable_r3 = decoder_model_config.r3
    kv_config.kv_io_bit_width = decoder_model_config.get_kv_io_bit_width()
    if decoder_model_config.masked_softmax:
        if is_qnn_sdk_version_less_than("2.35"):
            logging.warning(
                f"Masked softmax is supported after QNN SDK 2.35. Given sdk version {get_sdk_build_id()} is lower the target version. Disabling the feature."
            )
            kv_config.enable_masked_softmax = False
        else:
            kv_config.enable_masked_softmax = True

    prefill_config = copy.copy(kv_config)
    prefill_config.use_kv_cache = (
        False if args.max_seq_len == args.prefill_ar_len else True
    )

    llama_instance_list = []
    use_i64_token = args.embedding_quantize is not None
    with torch.device("meta"):
        if args.model_mode == "kv":
            llama_instance_list.append(
                LlamaModel(
                    kv_config,
                    ar_len=1,
                    output_new_cache_only=True,
                    output_cache=True,
                    use_i64_token=use_i64_token,
                )
            )
        elif args.model_mode == "hybrid":
            llama_instance_list.append(
                LlamaModel(
                    kv_config,
                    ar_len=1,
                    output_new_cache_only=True,
                    output_cache=True,
                    use_i64_token=use_i64_token,
                )
            )
            llama_instance_list.append(
                LlamaModel(
                    prefill_config,
                    ar_len=args.prefill_ar_len,
                    output_new_cache_only=True,
                    output_cache=True,
                    use_i64_token=use_i64_token,
                )
            )
        elif args.model_mode == "lookahead":
            llama_instance_list.append(
                LlamaModel(
                    kv_config,
                    # To get better performance, we round up to the nearest power of 2.
                    ar_len=next_power_of_two(
                        (args.window + args.gcap) * (args.ngram - 1)
                    ),
                    output_new_cache_only=True,
                    output_cache=True,
                    use_i64_token=use_i64_token,
                )
            )
            llama_instance_list.append(
                LlamaModel(
                    prefill_config,
                    ar_len=args.prefill_ar_len,
                    output_new_cache_only=True,
                    output_cache=True,
                    use_i64_token=use_i64_token,
                )
            )
        else:
            raise RuntimeError(f"Unknown model_mode: {args.model_mode}.")

    if args.checkpoint is None:  # HF models
        checkpoint = download_and_convert_hf_checkpoint(
            decoder_model_config.repo_id,
            decoder_model_config.convert_weights.__func__,
        )
        state_dict = torch.load(
            checkpoint, weights_only=True, map_location="cpu", mmap=True
        )
    else:
        state_dict = torch.load(
            args.checkpoint, weights_only=True, map_location="cpu", mmap=True
        )

        if "model" in state_dict:
            state_dict = state_dict["model"]

        if args.decoder_model == "stories260k":
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    if decoder_model_config.transform_weight:
        # Change to HuggingFace weight to improve the performance of RoPE in HTP backend.
        def permute(w, heads):
            dim_0 = w.size(0)
            dim_1 = w.size(1)
            return (
                w.view(heads, dim_0 // heads // 2, 2, dim_1)
                .transpose(1, 2)
                .reshape(dim_0, dim_1)
            )

        n_heads = llama_instance_list[0].n_heads
        n_kv_heads = llama_instance_list[0].n_kv_heads
        n_layers = llama_instance_list[0].n_layers

        for layer_i in range(n_layers):
            state_dict[f"layers.{layer_i}.attention.wq.weight"] = permute(
                state_dict[f"layers.{layer_i}.attention.wq.weight"], n_heads
            )
            state_dict[f"layers.{layer_i}.attention.wk.weight"] = permute(
                state_dict[f"layers.{layer_i}.attention.wk.weight"], n_kv_heads
            )

    for llama_instance in llama_instance_list:
        llama_instance.load_state_dict(
            state_dict,
            strict=True,
            assign=True,
        )
    end_load_ts = time.time()
    logging.info(f"Time for loading checkpoint: {end_load_ts - start_ts}")

    if decoder_model_config.r1 or decoder_model_config.r2:
        config = types.SimpleNamespace(
            dim=prefill_config.dim,
            head_dim=prefill_config.dim // prefill_config.n_heads,
            n_local_heads=prefill_config.n_heads,
            intermediate_size=4 * prefill_config.dim,
        )
        for llama_instance in llama_instance_list:
            model = llama_instance
            model.config = config
            # Currently this script is on CPU: run with CUDA_VISIBLE_DEVICES=-1
            apply_spinquant(
                model,
                use_r1=decoder_model_config.r1,
                use_r2=decoder_model_config.r2,
                use_r4=False,
                pretrained_rotation_path=None,
                qkv_split=True,
            )
            logging.info("Applied SpinQuant to the model")

    scales_state_dict = dict()
    if args.range_setting == "mse_with_act_loss":
        try:
            scales_state_dict = torch.load(
                "scales_state_dict.pth", map_location=torch.device("cpu")
            )
            logging.info("Loaded scales_state_dict from file")
        except:
            logging.info("Computing scales using activation loss range setting")
            model = llama_instance_list[1]
            model.to(torch.float)
            ar_len, model.ar_len = model.ar_len, model.max_seq_len
            tokens, atten_mask = model.get_example_inputs(use_kv_cache=False)
            atten_mask.to(torch.float)
            wrapped_model = WrappedLlamaModel(
                model, atten_mask, model.use_kv_cache, args.max_seq_len, args.device
            )
            act_bits, weight_bits = {
                QuantDtype.use_8a8w: (8, 8),
                QuantDtype.use_16a4w: (16, 4),
                QuantDtype.use_16a4w_block: (16, 4),
            }[decoder_model_config.ptq]
            scales_state_dict = compute_scales(
                wrapped_model, tokens, weight_bits, act_bits, 1600
            )
            reverse_quantize_module_swap(wrapped_model)
            model.ar_len = ar_len

    for llama_instance in llama_instance_list:
        for layer in llama_instance.layers:
            if getattr(layer.attention, "prepare_sha", None):
                layer.attention.prepare_sha()
            if getattr(layer.feed_forward, "prepare_feedfoward_conv", None):
                layer.feed_forward.prepare_feedfoward_conv()

    use_fp16 = True
    # "io_type" here refers to logits output and "kv_type" refers to kv_cache input/output.
    fixed_point_type = {"kv_type": torch.float32, "io_type": torch.float32}
    if decoder_model_config.ptq:
        if decoder_model_config.get_kv_io_bit_width() == 8:
            fixed_point_type["kv_type"] = torch.uint8
        elif decoder_model_config.get_kv_io_bit_width() == 16:
            fixed_point_type["kv_type"] = torch.uint16
        else:
            raise RuntimeError(
                f"Unknown kv io bit width {decoder_model_config.get_kv_io_bit_width()}"
            )

        if decoder_model_config.get_logits_output_bit_width() == 16:
            fixed_point_type["io_type"] = torch.uint16
        else:
            raise RuntimeError(
                f"Unknown logits io bit width {decoder_model_config.get_logits_output_bit_width()}"
            )

    quant_dtype = decoder_model_config.ptq

    if args.dtype_override is not None:
        dtype_override = DType[args.dtype_override]
        for i in range(len(llama_instance_list)):
            llama_instance_list[i] = llama_instance_list[i].to(
                dtype_override.to_torch_dtype()
            )

    for i in range(len(llama_instance_list)):
        if args.embedding_quantize:
            llama_instance_list[i] = get_quant_embedding_transform(
                embedding_quantize=args.embedding_quantize
            )(llama_instance_list[i])
        llama_instance_list[i] = convert_linear_to_conv2d(llama_instance_list[i])
        llama_instance_list[i] = SingleLlama(
            llama_instance_list[i].eval(), decoder_model_config, pte_filename
        )

        if args.embedding_quantize:
            llama_instance_list[i].passes_job[I64toI32][
                QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY
            ]["skip_node"] = {"tokens"}
    if decoder_model_config.ptq:
        start_quantize_ts = time.time()
        custom_annotations = decoder_model_config.custom_annotation
        logging.info(f"Custom annotations applied: {custom_annotations}")
        kv_quant_attrs = {}
        for i, llama_instance in enumerate(llama_instance_list):
            llama_instance.quantize(
                quant_dtype=quant_dtype,
                args=args,
                tokenizer=tokenizer,
                custom_annotations=custom_annotations,
                scales_state_dict=scales_state_dict,
                chat_template=chat_template,
            )
            # If hybrid and lookahead mode, we store kv output quant_attrs and apply to prefill output quant_attrs later
            if i == 0 and args.model_mode in ["hybrid", "lookahead"]:
                output_indices = 0
                for node in llama_instance.llama_graph_module.graph.nodes:
                    if node.op == "output":
                        for output in node.args[0]:
                            kv_quant_attrs[output_indices] = output.args[1:]
                            output_indices += 1
                        break
                custom_annotations = custom_annotations + (
                    partial(
                        annotate_prefill_kv_output,
                        kv_quant_attrs=kv_quant_attrs,
                    ),
                )  # temporarily remove annotate_prefill_kv_output
            llama_instance.passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
            llama_instance.passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
                "get_quant_io_dtype_fn"
            ] = partial(llama_instance._tag_ios, fixed_point_type=fixed_point_type)
        end_quantize_ts = time.time()
        logging.info(f"Time for quantizing: {end_quantize_ts - start_quantize_ts}")

    start_lowering_ts = time.time()
    if decoder_model_config.num_sharding > 1:
        for llama_instance in llama_instance_list:
            SplitGraph, setting = model_sharding.get_split_graph_pass(
                llama_instance.llama_meta["get_n_layers"],
                shares=decoder_model_config.num_sharding,
            )
            llama_instance.passes_job[SplitGraph] = setting
            llama_instance.dep_table[SplitGraph] = [FoldQDQ]
            llama_instance.dep_table[TagQuantIO] = [SplitGraph]

    if args.model_mode in ["kv"]:
        llama_instance_list[0].lowering_modules(
            args.artifact,
            use_fp16=use_fp16,
            soc_model=get_soc_to_chipset_map()[args.model],
            shared_buffer=args.shared_buffer,
        )
    elif args.model_mode in ["hybrid", "lookahead"]:
        sample_inputs_list = [
            llama_instace.inputs for llama_instace in llama_instance_list
        ]
        backend_options = generate_htp_compiler_spec(
            use_fp16=use_fp16,
            use_multi_contexts=decoder_model_config.num_sharding > 1,
            use_weight_sharing=not args.enable_x86_64,  # x86 emulator does not support weight sharing
        )
        graph_names = ["kv_forward", "prefill_forward"]
        compiler_specs = [
            generate_qnn_executorch_compiler_spec(
                soc_model=get_soc_to_chipset_map()[args.model],
                backend_options=backend_options,
                shared_buffer=args.shared_buffer,
                graph_name=graph_name,
            )
            for graph_name in graph_names
        ]

        llama_instance_list[1].save_logits_quant_attrs()
        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            {
                graph_name: instance.llama_graph_module
                for graph_name, instance in zip(graph_names, llama_instance_list)
            },
            {
                graph_name: inputs
                for graph_name, inputs in zip(graph_names, sample_inputs_list)
            },
            {
                graph_name: compiler_spec
                for graph_name, compiler_spec in zip(graph_names, compiler_specs)
            },
            llama_instance_list[1].llama_meta,
            dep_table={
                graph_name: instance.dep_table
                for graph_name, instance in zip(graph_names, llama_instance_list)
            },
            passes_job={
                graph_name: instance.passes_job
                for graph_name, instance in zip(graph_names, llama_instance_list)
            },
            skip_node_op_set={"llama.fallback.default"},
        )

        if decoder_model_config.num_sharding > 1:
            # TODO: add arg parser of spill_fill_size since weight-sharing based
            #       context binaries cannot be opened in x86 host
            pass

        if args.verbose:
            for ep in edge_prog_mgr._edge_programs.values():
                print_delegation_info(ep.graph_module)

        executorch_config = ExecutorchBackendConfig(
            # For shared buffer, user must pass the memory address
            # which is allocated by RPC memory to executor runner.
            # Therefore, won't want to pre-allocate
            # by memory manager in runtime.
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                alloc_graph_output=False,
            ),
            extract_delegate_segments=True,
        )
        exec_prog_mgr = edge_prog_mgr.to_executorch(executorch_config)

        with open(f"{args.artifact}/{pte_filename}.pte", "wb") as file:
            exec_prog_mgr.write_to_file(file)

    end_lowering_ts = time.time()
    logging.info(f"Time for compiling: {end_lowering_ts - start_lowering_ts}")


def inference(
    args,
    decoder_model_config: LLMModelConfig,
    pte_filename,
    runtime_tokenizer_path,
    tokenizer,
):
    assert args.model_mode in EVAL_MODE, f"Unknown model_mode: {args.model_mode}."

    pte_path = (
        f"{args.pre_gen_pte}/{pte_filename}.pte"
        if args.pre_gen_pte
        else f"{args.artifact}/{pte_filename}.pte"
    )

    if args.eval_perplexity:
        # Generate the eval wrapper
        eval_wrapper = QnnRunnerEvalWrapper(
            args=args,
            pte_path=pte_path,
            tokenizer=tokenizer,
            runtime_tokenizer_path=runtime_tokenizer_path,
            max_seq_length=args.max_seq_len,
        )

        # Evaluate the model
        with torch.no_grad():
            eval_results = simple_evaluate(
                model=eval_wrapper,
                tasks=args.tasks,
                num_fewshot=args.num_fewshot,
                limit=args.limit,
            )

        if args.ip and args.port != -1:
            assert (
                len(args.tasks) == 1 and args.tasks[0] == "wikitext"
            ), "CI currently supports wikitext only"
            wiki_ppl = eval_results["results"][args.tasks[0]]["word_perplexity,none"]
            pte_size = os.path.getsize(pte_path)
            with Client((args.ip, args.port)) as conn:
                conn.send(
                    json.dumps(
                        {
                            "wiki_ppl": wiki_ppl,
                            "pte_size": pte_size,
                            "inference_speed": eval_wrapper.inference_speed,
                        }
                    )
                )
        else:
            for task, res in eval_results["results"].items():
                logging.info(f"{task}: {res}")
        return
    workspace = f"/data/local/tmp/{getpass.getuser()}/executorch/single_llama"

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)
    outputs = []

    def post_process():
        with open(f"{args.artifact}/outputs/outputs.txt", "r") as f:
            outputs.append(f.read())

    seq_len = args.max_seq_len
    multi_prompts = " ".join([f'--prompt "{prompt}"' for prompt in args.prompt])
    lookahead_args = " ".join(
        [
            f"--window {args.window}",
            f"--gcap {args.gcap}",
            f"--ngram {args.ngram}",
        ]
    )
    runner_args = " ".join(
        [
            multi_prompts,
            f"--eval_mode {EVAL_MODE[args.model_mode]}",
            f"--temperature {args.temperature}",
            f"--system_prompt '{args.system_prompt}'",
            lookahead_args if args.model_mode == "lookahead" else "",
        ]
    )

    runner_cmd = ""
    performance_output_path = "outputs/inference_speed.txt"
    if args.enable_x86_64:
        # x86 emulator is intended for CI and not performance. Check only the first few tokens.
        seq_len = min(seq_len, 16)

        if args.kv_updater == smart_mask_updater:
            logging.warning(
                "x86 only support ShiftPointer, overwrite kv_updater to ShiftPointer"
            )

        qnn_sdk = os.getenv("QNN_SDK_ROOT")
        target = "x86_64-linux-clang"
        runner_cmd = " ".join(
            [
                f"export LD_LIBRARY_PATH={qnn_sdk}/lib/{target}/:{args.build_folder}/lib &&",
                f"./{args.build_folder}/examples/qualcomm/oss_scripts/llama/qnn_llama_runner",
                f"--decoder_model_version {decoder_model_config.decoder_model_version}",
                f"--tokenizer_path {runtime_tokenizer_path}",
                f"--model_path {pte_path}",
                f"--seq_len {seq_len}",
                f"--output_path {args.artifact}/outputs/outputs.txt",
                f"--performance_output_path {args.artifact}/{performance_output_path}",
                f"--kv_updater ShiftPointer",
                runner_args,
            ]
        )
        subprocess.run(
            runner_cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
        )
        post_process()
    else:
        runner_cmd = " ".join(
            [
                f"cd {workspace} &&",
                f"./qnn_llama_runner",
                f"--decoder_model_version {decoder_model_config.decoder_model_version}",
                f"--tokenizer_path {os.path.basename(runtime_tokenizer_path)}",
                f"--model_path {pte_filename}.pte",
                f"--seq_len {seq_len}",
                "--output_path outputs/outputs.txt",
                f"--performance_output_path {performance_output_path}",
                f"--kv_updater {'SmartMask' if args.kv_updater == smart_mask_updater else 'ShiftPointer'}",
                runner_args,
            ]
        )

        adb = SimpleADB(
            qnn_sdk=os.getenv("QNN_SDK_ROOT"),
            build_path=f"{args.build_folder}",
            pte_path=pte_path,
            workspace=workspace,
            device_id=args.device,
            host_id=args.host,
            soc_model=args.model,
            shared_buffer=args.shared_buffer,
            runner=f"examples/qualcomm/oss_scripts/llama/qnn_llama_runner",
        )
        # No pregen inputs, input_list is not required
        adb.push(inputs=[], files=[runtime_tokenizer_path])
        adb.execute(custom_runner_cmd=runner_cmd)

        adb.pull(output_path=args.artifact, callback=post_process)
    if args.ip and args.port != -1:
        inference_speed = 0
        with open(
            f"{os.path.abspath(args.artifact)}/{performance_output_path}", "r"
        ) as f:
            inference_speed = float(f.read())

        pte_size = os.path.getsize(pte_path)
        with Client((args.ip, args.port)) as conn:
            conn.send(
                json.dumps(
                    {
                        "result": outputs,
                        "pte_size": pte_size,
                        "inference_speed": inference_speed,
                    }
                )
            )
    else:
        for idx, output in enumerate(outputs):
            logging.info(f"Results[{idx}]:\n{output}")


def _build_tasks_parser(parser):
    parser.add_argument(
        "--eval_perplexity",
        help="If enabled, this will use the tasks provided under args.tasks to calibrate the model",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        default=None,
        help="list of lm-eluther tasks to evaluate usage: --tasks task1 task2",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="number of samples to evalulate. If not set, evaluate all samples",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        metavar="N",
        help="Number of examples in few-shot context",
    )

    return parser


def _build_parser():
    parser = setup_common_args_and_variables()
    parser = _build_tasks_parser(parser)
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts and output by this example. Default ./llama_qnn",
        default="./llama_qnn",
        type=str,
    )

    parser.add_argument(
        "--decoder_model",
        choices=list(SUPPORTED_LLM_MODELS.keys()),
        help=f"The llm model to export. Current available options are: { SUPPORTED_LLM_MODELS.keys()}",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        help="Pass llama checkpoint.",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--params",
        help="Pass llama params json file.",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--tokenizer_bin",
        help="For Llama2. Pass Llama2 tokenizer binary.",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--tokenizer_model",
        help="Pass llama tokenizer model.",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--prompt",
        help="User prompts for Llama. When multiple prompts are entered, a multi-turn conversation will be initiated. Note that this feature is currently for testing purposes only.",
        required=True,
        type=str,
        nargs="+",
    )

    parser.add_argument(
        "--system_prompt",
        help="For Llama3. Tells the model what kind of assistant it should be. For example, You are a helpful AI assistant for travel tips and recommendations. Default is None",
        default="",
        type=str,
    )

    parser.add_argument(
        "--temperature",
        help="Sampling temperature for llama.",
        default=0.8,
        type=float,
    )

    parser.add_argument(
        "-d",
        "--dtype-override",
        default="fp32",
        type=str,
        choices=["fp32", "fp16"],
        help="Override the dtype of the model (default is the checkpoint dtype). Options: fp32",
    )

    parser.add_argument(
        "--pre_gen_pte",
        help="Run the pre-generated llama in the given directory.",
        type=str,
    )

    parser.add_argument(
        "--model_mode",
        help="Export and inference kv mode, hybrid mode, or lookahead decoding mode",
        default="hybrid",
        choices=["kv", "hybrid", "lookahead"],
        type=str,
    )

    parser.add_argument(
        "--max_seq_len",
        help="This refers to maximum number of tokens that the model can process & consider at once to generate predictions/responses.",
        default=512,
        type=int,
    )

    parser.add_argument(
        "--prefill_ar_len",
        help="The auto-regression (AR) length determines the number of tokens to consume and the number of logits to produce. Use this option to process the prompt and generate the key-value (kv) cache, which serves as a prompt processor for hybrid and lookahead mode.",
        default=32,
        type=int,
    )

    parser.add_argument(
        "--kv_updater",
        help="Choose how to update kv cache during runtime",
        choices=["smart_mask", "shift_pointer"],
        default="smart_mask",
        type=str,
    )

    parser.add_argument(
        "-E",
        "--embedding-quantize",
        default=None,
        type=str,
        help="Fallback to cpu embedding operator and type of embedding quantization, '<bitwidth>,<groupsize>', e.g., '4,32'.",
    )

    parser.add_argument(
        "--ngram",
        help="Represents the size of the n-grams used in the lookahead process.",
        default=5,
        type=int,
    )

    parser.add_argument(
        "--window",
        help="Determines how many future tokens the algorithm attempts to predict in each step.",
        default=8,
        type=int,
    )

    parser.add_argument(
        "--gcap",
        help="Represents the maximum number of speculations or candidate n-grams that the algorithm considers in each step for verification. It balances the trade-off between computation efficiency and exploring more possibilities.",
        default=8,
        type=int,
    )
    # TODO: remove mse_weight_only (doesn't help much), only keep mse_with_act_loss (=SeqMSE)
    parser.add_argument(
        "--range_setting",
        help="Choose which range setting method for weight quantization (e.g. mse_weight_only or mse_with_act_loss). If not specified, defaults to minmax",
        type=str,
    )

    parser.add_argument("-v", "--verbose", action="store_true")

    return parser


def export_llama(args) -> None:
    if args.compile_only and args.pre_gen_pte:
        raise RuntimeError("Cannot set both compile_only and pre_gen_pte as true")
    if args.eval_perplexity and args.model_mode != "kv":
        raise RuntimeError("Eval device perplexity is only supported for KV mode")
    if args.eval_perplexity and args.tasks is None:
        raise RuntimeError("Please provide --tasks to eval perplexity")
    assert (
        args.decoder_model in SUPPORTED_LLM_MODELS
    ), f"Unknown decoder_model: {args.decoder_model}."
    decoder_model_config = SUPPORTED_LLM_MODELS[args.decoder_model]

    if args.model_mode == "kv":
        pte_filename = "kv_llama_qnn"
    elif args.model_mode == "hybrid":
        assert (
            args.max_seq_len >= args.prefill_ar_len
        ), "Please ensure max_seq_len is >= prefill_ar_len"
        pte_filename = "hybrid_llama_qnn"
    elif args.model_mode == "lookahead":
        assert (
            args.max_seq_len >= args.prefill_ar_len
        ), "Please ensure max_seq_len is >= prefill_ar_len"
        assert args.max_seq_len > next_power_of_two(
            (args.window + args.gcap) * (args.ngram - 1)
        ), "Please ensure max_seq_len is > next_power_of_two((args.window + args.gcap) * (args.ngram - 1))"
        pte_filename = "lookahead_llama_qnn"
    else:
        raise RuntimeError(f"Unknown model_mode: {args.model_mode}.")

    if args.decoder_model == "stories260k":
        pte_filename = f"{args.decoder_model}_" + pte_filename

    tokenizer = None
    runtime_tokenizer_path = ""
    chat_template = None
    if args.decoder_model in {"stories110m", "stories260k"}:
        tokenizer = get_tokenizer(args.tokenizer_model)
        assert isinstance(
            tokenizer, SentencePieceTokenizer
        ), f"Wrong tokenizer provided for stories."
        assert (
            args.tokenizer_bin is not None
        ), "Please provide tokenizer_bin for stories."
        runtime_tokenizer_path = args.tokenizer_bin
    elif args.decoder_model == "llama3_2":
        tokenizer = get_tokenizer(args.tokenizer_model)
        assert isinstance(
            tokenizer, TiktokenTokenizer
        ), f"Wrong tokenizer provided for llama3_2."
        runtime_tokenizer_path = args.tokenizer_model
    elif args.decoder_model in SUPPORTED_LLM_MODELS:
        model_id = decoder_model_config.repo_id
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        chat_template = (
            tokenizer.apply_chat_template
            if hasattr(tokenizer, "apply_chat_template")
            and decoder_model_config.instruct_model
            else None
        )
        tokenizer_artifacts = tokenizer.save_pretrained(args.artifact)
        tokenizer_config = tokenizer_artifacts[0]
        runtime_tokenizer_path = tokenizer_artifacts[-1]
        tokenizer = get_tokenizer(runtime_tokenizer_path, tokenizer_config)

    # TODO: Remove this once error is resolved.
    if args.decoder_model == "phi_4_mini":
        with open(runtime_tokenizer_path, "r+") as file:
            data = json.load(file)
            # TODO: Encountered the following error during runtime, so switched behavior for now.
            # Error: libc++abi: terminating due to uncaught exception of type std::runtime_error: invert=true is not supported for Split PreTokenizer. Only invert=false is supported.
            data["pre_tokenizer"]["pretokenizers"][-2]["invert"] = False
            file.seek(0)
            json.dump(data, file, indent=4)
            file.truncate()

    if args.kv_updater == "smart_mask":
        args.shared_buffer = True
        args.kv_updater = smart_mask_updater
    elif args.kv_updater == "shift_pointer":
        args.kv_updater = shift_pointer_updater
    else:
        raise RuntimeError(f"Using an unknown kv update {args.kv_updater}")

    if args.pre_gen_pte:
        inference(
            args, decoder_model_config, pte_filename, runtime_tokenizer_path, tokenizer
        )
        print(f"Finish the running pre_gen_pte from {args.pre_gen_pte}")
        return

    if args.compile_only:
        compile(args, decoder_model_config, pte_filename, tokenizer, chat_template)

        if args.ip and args.port != -1:
            pte_path = f"{args.artifact}/{pte_filename}.pte"
            pte_size = os.path.getsize(pte_path)
            with Client((args.ip, args.port)) as conn:
                conn.send(
                    json.dumps(
                        {
                            "pte_size": pte_size,
                        }
                    )
                )
        print(f"Finish compile_only and save to {args.artifact}")
        return

    compile(args, decoder_model_config, pte_filename, tokenizer, chat_template)
    inference(
        args, decoder_model_config, pte_filename, runtime_tokenizer_path, tokenizer
    )


def main():
    parser = _build_parser()
    args = parser.parse_args()
    try:
        export_llama(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)


# flake8: noqa: C901
if __name__ == "__main__":
    main()
