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
import os
import subprocess
import sys
import time
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
from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
from executorch.backends.qualcomm.quantizer.custom_annotation import (
    annotate_linear_16a8w_in_affine_layer,
    annotate_matmul_16a8w,
    annotate_prefill_kv_output,
)

from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset

from executorch.backends.qualcomm.utils.constants import (
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
    QCOM_QUANT_ATTRS_MAP,
)
from executorch.backends.qualcomm.utils.utils import (
    capture_program,
    convert_linear_to_conv2d,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,
    to_edge_transform_and_lower_to_qnn,
    update_spill_fill_size,
)

from executorch.devtools.backend_debug import print_delegation_info
from executorch.examples.models.llama.source_transformation.quantize import (
    get_quant_embedding_transform,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    LlamaModel,
    ModelArgs,
)
from executorch.examples.qualcomm.utils import (
    make_output_dir,
    make_quantizer,
    setup_common_args_and_variables,
    SimpleADB,
)
from executorch.exir import EdgeProgramManager
from executorch.exir.backend.backend_api import (
    MethodProgramsPartitionerSpec,
    to_backend,
)
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.extension.llm.custom_ops import model_sharding
from executorch.extension.llm.export.builder import DType
from pytorch_tokenizers import get_tokenizer, TiktokenTokenizer
from pytorch_tokenizers.llama2c import Llama2cTokenizer as SentencePieceTokenizer

from torchao.quantization.pt2e import MinMaxObserver
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

sys.setrecursionlimit(4096)
FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.getLogger().setLevel(logging.INFO)


def smart_mask_updater(
    ar_len, atten_mask, pos, k_caches, v_caches, new_k_caches, new_v_caches
):
    # Update the KV cache input for the next inference when the position exceeds the autoregressive length.
    if pos >= ar_len:
        for i, k_cache in enumerate(k_caches):
            k_cache[:, :, pos - ar_len] = new_k_caches[i][:, :, 0]

        for i, v_cache in enumerate(v_caches):
            v_cache[:, pos - ar_len, :] = new_v_caches[i][:, 0, :]
        atten_mask[:, :, pos - ar_len] = 0

    pos += 1
    return (atten_mask, pos, k_caches, v_caches)


def shift_pointer_updater(
    ar_len, atten_mask, pos, k_caches, v_caches, new_k_caches, new_v_caches
):
    # Update the KV cache input for the next inference when the position exceeds the autoregressive length.
    if pos >= ar_len:
        k_caches = [
            torch.cat([k_cache[:, :, 1:], new_k_caches[i][:, :, :1]], dim=-1)
            for i, k_cache in enumerate(k_caches)
        ]
        v_caches = [
            torch.cat([v_cache[:, 1:, :], new_v_caches[i][:, :1, :]], dim=1)
            for i, v_cache in enumerate(v_caches)
        ]
        atten_mask[:, :, -pos - 1] = 0

    pos += 1
    return (atten_mask, pos, k_caches, v_caches)


def _kv_calibrate(
    example_inputs,
    user_prompts,
    module: torch.fx.GraphModule,
    tokenizer,
    ar_len=1,
    max_seq_len=512,
    updater=smart_mask_updater,
    use_i64_token=False,
):
    _, atten_mask, _, k_caches, v_caches = example_inputs

    # TODO: change criteria & support batch inputs if necessary
    all_pos = torch.arange(0, max_seq_len, 1, dtype=torch.int32).unsqueeze(0)

    token_list = []
    # Llama2 tokenizer has no special tokens
    if isinstance(tokenizer, SentencePieceTokenizer):
        token_list = tokenizer.encode(user_prompts, bos=True, eos=False)
    elif isinstance(tokenizer, TiktokenTokenizer):
        token_list = tokenizer.encode(
            user_prompts, bos=True, eos=False, allowed_special="all"
        )
    else:
        raise RuntimeError("Unkown tokenizer")

    pos = len(token_list) if len(token_list) < ar_len else ar_len
    dtype = torch.int64 if use_i64_token else torch.int32

    with torch.no_grad():
        while token_list[-1] != tokenizer.eos_id and pos < max_seq_len:
            tmp_token_list = torch.tensor(
                token_list[pos - ar_len : pos], dtype=dtype
            ).reshape(1, -1)
            tmp_pos = all_pos[:, pos - ar_len : pos]
            tmp_atten_mask = atten_mask
            if pos < ar_len:
                tmp_token_list = torch.cat(
                    [
                        torch.zeros((1, ar_len - pos), dtype=dtype),
                        torch.tensor(token_list, dtype=dtype).reshape(1, -1),
                    ],
                    dim=1,
                )
                tmp_pos = torch.cat(
                    [
                        torch.zeros((1, ar_len - pos), dtype=torch.int32),
                        all_pos[:, :pos],
                    ],
                    dim=1,
                )
                tmp_atten_mask = torch.cat(
                    [
                        torch.ones(1, ar_len, max_seq_len - pos) * -255.0,
                        atten_mask[:, :, -pos:],
                    ],
                    dim=-1,
                )

            logits, new_k_caches, new_v_caches = module(
                tmp_token_list,
                tmp_atten_mask,
                tmp_pos,
                *k_caches,
                *v_caches,
            )
            atten_mask, pos, k_caches, v_caches = updater(
                ar_len, atten_mask, pos, k_caches, v_caches, new_k_caches, new_v_caches
            )
            if pos > len(token_list):
                token_list.append(torch.argmax(logits[:, -1], dim=-1).item())

    print(f"kv calibration data:\n{tokenizer.decode(token_list)}")


def _prefill_calibrate(
    example_inputs,
    user_prompts,
    module: torch.fx.GraphModule,
    tokenizer,
    max_seq_len=512,
    use_i64_token=False,
):
    _, atten_mask = example_inputs

    # TODO: change criteria & support batch inputs if necessary

    token_list = []
    # Llama2 tokenizer has no special tokens
    if isinstance(tokenizer, SentencePieceTokenizer):
        token_list = tokenizer.encode(user_prompts, bos=True, eos=False)
    elif isinstance(tokenizer, TiktokenTokenizer):
        token_list = tokenizer.encode(
            user_prompts, bos=True, eos=False, allowed_special="all"
        )
    else:
        raise RuntimeError("Unkown tokenizer")

    pos = len(token_list)
    dtype = torch.int64 if use_i64_token else torch.int32

    with torch.no_grad():
        while token_list[-1] != tokenizer.eos_id and pos < max_seq_len:
            tmp_token_list = torch.tensor(token_list, dtype=dtype).reshape(1, -1)
            if pos < max_seq_len:
                tmp_token_list = torch.cat(
                    [
                        tmp_token_list,
                        torch.zeros((1, max_seq_len - pos), dtype=dtype),
                    ],
                    dim=1,
                )
            results = module(
                tmp_token_list,
                atten_mask,
            )
            if len(results) == 3:
                logits, new_k_caches, new_v_caches = results
            elif len(results) == 1:
                logits = results
            token_list.append(torch.argmax(logits[:, pos - 1], dim=-1).item())
            pos += 1

    print(f"prefill calibration data:\n{tokenizer.decode(token_list)}")


def calibrate(
    example_inputs,
    user_prompts,
    module: torch.fx.GraphModule,
    tokenizer,
    ar_len=1,
    max_seq_len=512,
    kv_updater=smart_mask_updater,
    use_i64_token=False,
):
    if len(example_inputs) == 2:
        _prefill_calibrate(
            example_inputs,
            user_prompts,
            module,
            tokenizer,
            max_seq_len,
            use_i64_token,
        )
    elif len(example_inputs) == 5:
        _kv_calibrate(
            example_inputs,
            user_prompts,
            module,
            tokenizer,
            ar_len,
            max_seq_len,
            updater=kv_updater,
            use_i64_token=use_i64_token,
        )
    else:
        raise RuntimeError("Get wrong inputs")


class SingleLlama:
    def __init__(self, llama_model, pte_filename) -> None:
        super().__init__()
        self.llama_model = llama_model
        self.passes_job = get_capture_program_passes()
        self.dep_table = get_passes_dependency_for_capture_program()
        self.quant_attrs = None
        self.quant_dtype = None
        self.llama_meta = self.llama_model.get_metadata()
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
        self.llama_graph_module = llama_model
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

    def quantize(self, quant_dtype, args, tokenizer, custom_annotations=()):
        self.quant_dtype = quant_dtype
        quantizer = make_quantizer(
            quant_dtype=quant_dtype,
            per_channel_conv=True,
            per_channel_linear=True,
            act_observer=MinMaxObserver,
        )
        quantizer.add_custom_quant_annotations(custom_annotations)

        self.has_quant_io = True
        fx_graph_module = None

        with torch.no_grad():
            fx_graph_module = torch.export.export(
                self.llama_graph_module, self.inputs, strict=True
            ).module()

            if QuantDtype == QuantDtype.use_16a4w_block:
                conv_nodes = [
                    n for n in fx_graph_module.graph.nodes if "conv" in n.name
                ]
                block_size_map = {n.name: (1, 64, 1, 1) for n in conv_nodes}
                quantizer.set_block_size_map(block_size_map)

            fx_graph_module = prepare_pt2e(fx_graph_module, quantizer)

        logging.info("Quantizing the model...")
        calibrate(
            self.get_example_inputs(self.llama_meta["get_use_kv_cache"]),
            args.prompt[0],
            fx_graph_module,
            tokenizer=tokenizer,
            ar_len=self.llama_meta["get_ar_len"],
            max_seq_len=self.llama_meta["get_max_seq_len"],
            kv_updater=args.kv_updater,
            use_i64_token=args.embedding_quantize is not None,
        )

        self.llama_graph_module = convert_pt2e(fx_graph_module)

    def lowering_modules(
        self,
        work_space,
        use_fp16=False,
        soc_model=QcomChipset.SM8650,
        num_sharding=1,
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
                use_fp16=use_fp16, use_multi_contexts=num_sharding > 1
            )
            compiler_specs = generate_qnn_executorch_compiler_spec(
                soc_model=soc_model,
                backend_options=backend_options,
                shared_buffer=shared_buffer,
            )
            skip_node_op_set = {"llama.fallback.default"}
            edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                self.llama_graph_module,
                self.inputs,
                compiler_specs,
                constant_methods=self.llama_meta,
                dep_table=self.dep_table,
                passes_job=self.passes_job,
                skip_node_op_set=skip_node_op_set,
            )

            for n in edge_prog_mgr.exported_program().graph.nodes:
                if n.op == "output":
                    for node, output_encoding in n.meta[QCOM_QUANT_ATTRS_MAP].items():
                        if node.meta["val"].size() in self.io_shape:
                            self.quant_attrs = output_encoding

            if num_sharding > 1:
                update_spill_fill_size(edge_prog_mgr.exported_program())

            if verbose:
                print_delegation_info(edge_prog_mgr.exported_program().graph_module)

            exec_prog_mgr = edge_prog_mgr.to_executorch(config=executorch_config)
            with open(f"{work_space}/{self.pte_filename}.pte", "wb") as file:
                exec_prog_mgr.write_to_file(file)

    def get_example_inputs(self, use_kv_cache=True):
        return self.llama_model.get_example_inputs(use_kv_cache)

    def get_quant_attrs(self):
        return self.quant_attrs


def compile(args, pte_filename, tokenizer):
    os.makedirs(args.artifact, exist_ok=True)
    start_ts = time.time()

    with open(args.params) as f:
        kv_config = ModelArgs(**json.load(f))
        # TODO: support batch inputs if necessary
        kv_config.max_batch_size = 1
        kv_config.max_seq_len = args.max_seq_len
        kv_config.use_kv_cache = True

        prefill_config = copy.copy(kv_config)
        prefill_config.max_seq_len = args.max_seq_len
        prefill_config.use_kv_cache = (
            False if args.max_seq_len == args.prefill_ar_len else True
        )

    state_dict = torch.load(
        args.checkpoint, weights_only=True, map_location="cpu", mmap=True
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
        else:
            raise RuntimeError(f"Unknown model_mode: {args.model_mode}.")

    if "model" in state_dict:
        state_dict = state_dict["model"]

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
            strict=False,
            assign=True,
        )
    end_load_ts = time.time()
    logging.info(f"Time for loading checkpoint: {end_load_ts - start_ts}")

    for llama_instance in llama_instance_list:
        for layer in llama_instance.layers:
            if getattr(layer.attention, "prepare_sha", None):
                layer.attention.prepare_sha()
            if getattr(layer.feed_forward, "prepare_feedfoward_conv", None):
                layer.feed_forward.prepare_feedfoward_conv()

    use_fp16 = True
    fixed_point_type = {"kv_type": torch.float32, "io_type": torch.float32}
    if args.ptq:
        use_fp16 = False
        fixed_point_type["kv_type"] = torch.uint8
        if args.ptq == "8a8w":
            fixed_point_type["io_type"] = torch.uint8
        elif args.ptq in ("16a4w", "16a4w_block"):
            fixed_point_type["io_type"] = torch.uint16
        else:
            assert args.ptq in [
                "8a8w",
                "16a4w",
                "16a4w_block",
            ], f"No support for quant type {args.ptq}. Support 8a8w, 16a4w and 16a4w_block."
        quant_dtype = getattr(QuantDtype, f"use_{args.ptq}")

    assert args.tokenizer_model is not None, "Need tokenizer model for calibration"

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
            llama_instance_list[i].eval(), pte_filename
        )
        if args.embedding_quantize:
            llama_instance_list[i].passes_job[I64toI32][
                QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY
            ]["skip_node"] = {"tokens"}

    if args.ptq:
        start_quantize_ts = time.time()
        custom_annotations = (annotate_matmul_16a8w,)
        if args.llama_model == "stories110m":
            custom_annotations = custom_annotations + (
                annotate_linear_16a8w_in_affine_layer,
            )
        kv_quant_attrs = {}
        for i, llama_instance in enumerate(llama_instance_list):
            llama_instance.quantize(
                quant_dtype=quant_dtype,
                args=args,
                tokenizer=tokenizer,
                custom_annotations=custom_annotations,
            )
            # If hybrid mode, we store kv output quant_attrs and apply to prefill output quant_attrs later
            if i == 0 and args.model_mode == "hybrid":
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
                )
            llama_instance.passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
            llama_instance.passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
                "get_quant_io_dtype_fn"
            ] = partial(llama_instance._tag_ios, fixed_point_type=fixed_point_type)
        end_quantize_ts = time.time()
        logging.info(f"Time for quantizing: {end_quantize_ts - start_quantize_ts}")

    start_lowering_ts = time.time()
    quant_attrs = None
    if args.num_sharding > 1:
        for llama_instance in llama_instance_list:
            SplitGraph, setting = model_sharding.get_split_graph_pass(
                llama_instance.llama_meta["get_n_layers"],
                shares=args.num_sharding,
            )
            llama_instance.passes_job[SplitGraph] = setting
            llama_instance.dep_table[SplitGraph] = [FoldQDQ]
            llama_instance.dep_table[TagQuantIO] = [SplitGraph]

    if args.model_mode in ["kv"]:
        llama_instance_list[0].lowering_modules(
            args.artifact,
            use_fp16=use_fp16,
            soc_model=get_soc_to_chipset_map()[args.model],
            num_sharding=args.num_sharding,
            shared_buffer=args.shared_buffer,
        )
        quant_attrs = llama_instance_list[0].get_quant_attrs()
    elif args.model_mode == "hybrid":
        sample_inputs_list = [
            llama_instace.inputs for llama_instace in llama_instance_list
        ]
        backend_options = generate_htp_compiler_spec(
            use_fp16=use_fp16,
            use_multi_contexts=args.num_sharding > 1,
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

        # TODO: retire capture_program once we figure out how to extract
        #       intermediate graph from official lowering API
        edge_progs = {
            graph_name: capture_program(
                module=llama_instance.llama_graph_module,
                inputs=sample_input,
                dep_table=llama_instance.dep_table,
                passes_job=llama_instance.passes_job,
            ).exported_program
            for graph_name, llama_instance, sample_input in zip(
                graph_names, llama_instance_list, sample_inputs_list
            )
        }
        for n in edge_progs[graph_names[0]].graph.nodes:
            if n.op == "output":
                for node, output_encoding in n.meta[QCOM_QUANT_ATTRS_MAP].items():
                    if node.meta["val"].size() in llama_instance_list[0].io_shape:
                        quant_attrs = output_encoding

        partitioners = {
            graph_name: QnnPartitioner(
                compiler_spec, skip_node_op_set={"llama.fallback.default"}
            )
            for graph_name, compiler_spec in zip(graph_names, compiler_specs)
        }

        lowered_ep_dict = to_backend(
            MethodProgramsPartitionerSpec(edge_progs, partitioners)
        )

        if args.num_sharding > 1:
            # TODO: add arg parser of spill_fill_size since weight-sharing based
            #       context binaries cannot be opened in x86 host
            pass

        if args.verbose:
            for ep in lowered_ep_dict.values():
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
        exec_prog_mgr = EdgeProgramManager(
            edge_programs=lowered_ep_dict,
            constant_methods=llama_instance_list[1].llama_meta,
        ).to_executorch(executorch_config)

        with open(f"{args.artifact}/{pte_filename}.pte", "wb") as file:
            exec_prog_mgr.write_to_file(file)

    end_lowering_ts = time.time()
    logging.info(f"Time for compiling: {end_lowering_ts - start_lowering_ts}")
    return quant_attrs


def inference(args, pte_filename, runtime_tokenizer_path, pre_gen_pte=""):
    workspace = f"/data/local/tmp/{getpass.getuser()}/executorch/single_llama"

    if args.model_mode == "kv":
        eval_mode = 0
    elif args.model_mode == "hybrid":
        eval_mode = 1
    else:
        raise RuntimeError(f"Unknown model_mode: {args.model_mode}.")

    pte_path = (
        f"{pre_gen_pte}/{pte_filename}.pte"
        if pre_gen_pte
        else f"{args.artifact}/{pte_filename}.pte"
    )

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)
    outputs = []

    def post_process():
        with open(f"{args.artifact}/outputs/outputs.txt", "r") as f:
            outputs.append(f.read())

    seq_len = args.max_seq_len
    multi_prompts = " ".join([f'--prompt "{prompt}"' for prompt in args.prompt])
    runner_args = " ".join(
        [
            multi_prompts,
            f"--eval_mode {eval_mode}",
            f"--temperature {args.temperature}",
            f"--system_prompt '{args.system_prompt}'",
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
                f"--tokenizer_path {runtime_tokenizer_path}",
                f"--model_path {pte_path}",
                f"--seq_len {seq_len}",
                f"--output_path {args.artifact}/outputs/outputs.txt",
                f"--performance_output_path {performance_output_path}",
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
        adb.push(inputs=[], input_list="", files=[runtime_tokenizer_path])
        adb.execute(custom_runner_cmd=runner_cmd)

        adb.pull(output_path=args.artifact, callback=post_process)
    if args.ip and args.port != -1:
        inference_speed = 0
        with open(f"{args.artifact}/{performance_output_path}", "r") as f:
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


def _build_parser():
    parser = setup_common_args_and_variables()
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts and output by this example. Default ./llama_qnn",
        default="./llama_qnn",
        type=str,
    )

    parser.add_argument(
        "-P",
        "--ptq",
        help="If specified, will do PTQ quantization. default is 16bits activation and 4bits weight. Support 8a8w, 16a4w and 16a4w_block.",
        type=str,
    )

    parser.add_argument(
        "--llama_model",
        choices=["stories110m", "llama3_2"],
        help="The Llama model to export. Current available options are: [stories110m, llama3_2]",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        help="Pass llama checkpoint.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--params",
        help="Pass llama params json file.",
        required=True,
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
        "--num_sharding",
        type=int,
        default=1,
        help="Specify the number of splits by inserting the fallback custom op. The graph will be split evenly by layers.",
    )

    parser.add_argument(
        "--model_mode",
        help="Export and inference kv mode or hybrid mode",
        default="kv",
        choices=["kv", "hybrid"],
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
        help="The auto-regression (AR) length determines the number of tokens to consume and the number of logits to produce. Use this option to process the prompt and generate the key-value (kv) cache, which serves as a prompt processor for hybrid mode.",
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

    parser.add_argument("-v", "--verbose", action="store_true")

    return parser


def export_llama(args) -> None:
    if args.compile_only and args.pre_gen_pte:
        raise RuntimeError("Cannot set both compile_only and pre_gen_pte as true")

    if args.model_mode == "kv":
        pte_filename = "kv_llama_qnn"
    elif args.model_mode == "hybrid":
        assert (
            args.max_seq_len >= args.prefill_ar_len
        ), "Please ensure max_seq_len is >= prefill_ar_len"
        pte_filename = "hybrid_llama_qnn"
    else:
        raise RuntimeError(f"Unknown model_mode: {args.model_mode}.")

    tokenizer = get_tokenizer(args.tokenizer_model)
    runtime_tokenizer_path = ""
    if args.llama_model == "stories110m":
        assert isinstance(
            tokenizer, SentencePieceTokenizer
        ), f"Wrong tokenizer provided for stories110m."
        assert (
            args.tokenizer_bin is not None
        ), "Please provide tokenizer_bin for stories110m."
        runtime_tokenizer_path = args.tokenizer_bin
    elif args.llama_model == "llama3_2":
        assert isinstance(
            tokenizer, TiktokenTokenizer
        ), f"Wrong tokenizer provided for llama3_2."
        runtime_tokenizer_path = args.tokenizer_model
    else:
        raise RuntimeError(f"Unknown llama_model: {args.llama_model}.")

    if args.kv_updater == "smart_mask":
        args.shared_buffer = True
        args.kv_updater = smart_mask_updater
    elif args.kv_updater == "shift_pointer":
        args.kv_updater = shift_pointer_updater
    else:
        raise RuntimeError(f"Using an unknown kv update {args.kv_updater}")

    if args.pre_gen_pte:
        inference(args, pte_filename, runtime_tokenizer_path, args.pre_gen_pte)
        print(f"Finish the running pre_gen_pte from {args.pre_gen_pte}")
        return

    if args.compile_only:
        compile(args, pte_filename, tokenizer)

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

    compile(args, pte_filename, tokenizer)
    inference(args, pte_filename, runtime_tokenizer_path)


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
